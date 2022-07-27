from time import perf_counter
import warnings
from dataclasses import dataclass
from typing import ClassVar

import torch
import pandas as pd
from torch import nn
from torchdiffeq import odeint
from scipy.optimize import root_scalar
from xitorch.optimize import rootfinder
from xitorch._utils.exceptions import ConvergenceWarning

from biomechanical_models.base_classes import (
    ODEBase, 
    CardiacDriver, 
    Valve, 
    BloodVessel, 
    PressureVolume, 
    InertialValve, 
    RespiratoryPatternGenerator
)
from biomechanical_models.unit_conversions import convert
from biomechanical_models import parameters

USE_XITORCH = False
warnings.simplefilter('error', ConvergenceWarning)
# pd.options.plotting.backend = 'plotly'


@dataclass
class PassiveRespiratorySystem(ODEBase):
    """Jallon passive mechanical respiratory system"""

    state_names: ClassVar[list[str]] = ['p_mus', 'v_alv']

    def __init__(
        self, 
        e_alv: float = convert(5, 'cmH2O/l'),
        e_cw: float = convert(4, 'cmH2O/l'),
        r_ua: float = convert(5, 'cmH2O s/l'),  # Wrong units in Jallon paper (cmH2O/l)
        r_ca: float = convert(1, 'cmH2O s/l'),  # Wrong units in Jallon paper (cmH2O/l)
        v_th0: float = convert(2, 'l'),  # Wrong units in Jallon paper (1/l)
        lam: float = convert(1.5, 'mmHg'),  # Units not given in Jallon; actually mmHg/s
        mu: float = convert(1.0, 'mmHg'),  # Units not given in Jallon; actually mmHg/s ***1.08504***
        beta: float = 0.1,  # Manually added parameter to control drift (1/s)
    ):
        super().__init__()
        self.e_alv = nn.Parameter(torch.as_tensor(e_alv), requires_grad=False)
        self.e_cw = nn.Parameter(torch.as_tensor(e_cw), requires_grad=False)
        self.r_ua = nn.Parameter(torch.as_tensor(r_ua), requires_grad=False)
        self.r_ca = nn.Parameter(torch.as_tensor(r_ca), requires_grad=False)
        self.v_th0 = nn.Parameter(torch.as_tensor(v_th0), requires_grad=False)
        self.lam = nn.Parameter(torch.as_tensor(lam), requires_grad=False)
        self.mu = nn.Parameter(torch.as_tensor(mu), requires_grad=False)
        self.beta = nn.Parameter(torch.as_tensor(beta), requires_grad=False)

    def model(self, t: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:

        # These are not states of this model, but states of the overall system
        y = states['y']
        v_pcd = states['v_lv'] + states['v_rv']
        v_pu = states['v_pu']
        v_pa = states['v_pa']

        v_bth = v_pcd + v_pu + v_pa
        v_th = v_bth + states['v_alv']
        p_pl = states['p_mus'] + self.e_cw * (v_th - self.v_th0)

        dp_mus_dt = self.lam * y + self.mu - self.beta * states['p_mus']
        dv_alv_dt = -(p_pl + self.e_alv * states['v_alv']) / (self.r_ca + self.r_ua)

        outputs = {
            'p_pl': p_pl,
            'v_th': v_th,
            'v_bth': v_bth,
            'dp_mus_dt': dp_mus_dt, 
            'dv_alv_dt': dv_alv_dt,
        }

        return outputs


class SmithCardioVascularSystem(ODEBase):
    """Smith CVS model with no inertia and Heaviside valve law"""

    state_names: ClassVar[list[str]] = ['v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Valves
        self.mt = Valve(convert(0.06, 'kPa s/l'))  # Mitral valve
        self.tc = Valve(convert(0.18, 'kPa s/l'))  # Tricuspid valve
        self.av = Valve(convert(1.4, 'kPa s/l'))  # Aortic valve
        self.pv = Valve(convert(0.48, 'kPa s/l'))  # Pulmonary valve

        # Circulation resistance
        self.pul = BloodVessel(convert(19, 'kPa s/l'))  # Pulmonary circulation
        self.sys = BloodVessel(convert(140, 'kPa s/l'))  # Systematic circulation

        # Pleural pressure
        self.p_pl = nn.Parameter(torch.tensor(-4.0), requires_grad=False)

        # Pressure-volume relationships
        # Left ventricle free wall
        self.lvf = PressureVolume(convert(454, 'kPa/l'), convert(0.005, 'l'), convert(0.005, 'l'), convert(15, '1/l'), convert(0.17, 'kPa'))
        # Right ventricle free wall
        self.rvf = PressureVolume(convert(87, 'kPa/l'), convert(0.005, 'l'), convert(0.005, 'l'), convert(15, '1/l'), convert(0.16, 'kPa'))
        # Septum free wall
        self.spt = PressureVolume(convert(6500, 'kPa/l'), convert(0.002, 'l'), convert(0.002, 'l'), convert(435, '1/l'), convert(0.148, 'kPa'))
        # Pericardium
        self.pcd = PressureVolume(None, None, convert(0.2, 'l'), convert(30, '1/l'), convert(0.0667, 'kPa'))
        # Vena-cava
        self.vc = PressureVolume(convert(1.5, 'kPa/l'), convert(2.83, 'l'), None, None, None)
        # Pulmonary artery
        self.pa = PressureVolume(convert(45, 'kPa/l'), convert(0.16, 'l'), None, None, None)
        # Pulmonary vein
        self.pu = PressureVolume(convert(0.8, 'kPa/l'), convert(0.2, 'l'), None, None, None)
        # Aorta
        self.ao = PressureVolume(convert(94, 'kPa/l'), convert(0.8, 'l'), None, None, None)
    
        # Cardiac pattern generator
        self.e = CardiacDriver(hr=80)

        # Jallon 2009 modification
        self.p_pl_affects_pu_and_pa = nn.Parameter(torch.tensor(False), requires_grad=False)

        # First initial guess for v_spt
        self._v_spt_old = torch.tensor(convert(0.0055, 'l'))
        self._v_spt_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        super().callback_accept_step(t, x, dt)
        self._v_spt_old = self.trajectory[-1][3]['v_spt']

    def model(self, t: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:

        # t1 = perf_counter()

        outputs = self.pressures_volumes(t, states)
        self.flow_rates(outputs)
        self.derivatives(states, outputs)

        # t2 = perf_counter()
        # print(f'total: {t2-t1:.2e}s')

        return outputs

    def flow_rates(self, outputs: dict[str, torch.Tensor]):
        outputs['q_mt'] = self.mt.flow_rate(outputs['p_pu'], outputs['p_lv'])
        outputs['q_av'] = self.av.flow_rate(outputs['p_lv'], outputs['p_ao'])
        outputs['q_tc'] = self.tc.flow_rate(outputs['p_vc'], outputs['p_rv'])
        outputs['q_pv'] = self.pv.flow_rate(outputs['p_rv'], outputs['p_pa'])
        outputs['q_pul'] = self.pul.flow_rate(outputs['p_pa'], outputs['p_pu'])
        outputs['q_sys'] = self.sys.flow_rate(outputs['p_ao'], outputs['p_vc'])

    def derivatives(self, states: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):

        # Chamber volume changes
        # Explicitly defined in Smith 2007, Fig. 1, C17-C22
        outputs['dv_pa_dt'] = outputs['q_pv'] - outputs['q_pul']
        outputs['dv_pu_dt'] = outputs['q_pul'] - outputs['q_mt']
        outputs['dv_lv_dt'] = outputs['q_mt'] - outputs['q_av']
        outputs['dv_ao_dt'] = outputs['q_av'] - outputs['q_sys']
        outputs['dv_vc_dt'] = outputs['q_sys'] - outputs['q_tc']
        outputs['dv_rv_dt'] = outputs['q_tc'] - outputs['q_pv']

    def pressures_volumes(self, t: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Equation numbers from Smith 2004 (or 2007 if explicitly stated).

        Laid out as per Fig. 1, Smith 2007
        """

        # Pericardium pressure-volume relationship
        # Eq. 11, 18, 14. Note p_pl (pleural cavity) is p_th (thoracic cavity)
        v_pcd = states['v_lv'] + states['v_rv']
        p_pcd = self.pcd.p_ed(v_pcd)
        p_peri = p_pcd + self.p_pl

        # Evaluate model driving function
        e_t = self.e(t)

        # Ventricular pressure-volume relationship
        # t1 = perf_counter()
        v_spt = self.solve_v_spt(states['v_lv'], states['v_rv'], e_t)
        # t2 = perf_counter()
        # print(f'solve: {t2-t1:.2e}s')

        # t1 = perf_counter()
        # Eq. 9, 10, 16
        v_lvf = states['v_lv'] - v_spt
        v_rvf = states['v_rv'] + v_spt
        p_lvf = e_t * self.lvf.p_es(v_lvf) + (1 - e_t) * self.lvf.p_ed(v_lvf)
        p_rvf = e_t * self.rvf.p_es(v_rvf) + (1 - e_t) * self.rvf.p_ed(v_rvf)
        p_spt = e_t * self.spt.p_es(v_spt) + (1 - e_t) * self.spt.p_ed(v_spt)

        # Eq. 12, 13
        p_lv = p_lvf + p_peri
        p_rv = p_rvf + p_peri

        # Peripheral chamber pressure-volume relationships
        p_pa = self.pa.p_es(states['v_pa'])
        p_pu = self.pu.p_es(states['v_pu'])
        p_ao = self.ao.p_es(states['v_ao'])
        p_vc = self.vc.p_es(states['v_vc'])

        if self.p_pl_affects_pu_and_pa:
            p_pa = p_pa + self.p_pl
            p_pu = p_pu + self.p_pl
            
        # t2 = perf_counter()
        # print(f'pv: {t2-t1:.2e}s')

        outputs = {
            'v_pa': states['v_pa'], 
            'v_pu': states['v_pu'], 
            'v_lv': states['v_lv'],
            'v_ao': states['v_ao'],
            'v_vc': states['v_vc'], 
            'v_rv': states['v_rv'],
            'e_t': e_t,
            'v_pcd': v_pcd,
            'p_pcd': p_pcd,
            'p_peri': p_peri,
            'v_spt': v_spt,
            'v_lvf': v_lvf,
            'v_rvf': v_rvf,
            'p_lvf': p_lvf,
            'p_rvf': p_rvf,
            'p_lv': p_lv,
            'p_rv': p_rv,
            'p_pa': p_pa,
            'p_pu': p_pu,
            'p_ao': p_ao,
            'p_vc': p_vc,
            'p_spt': p_spt,
        }
        

        return outputs

    def simulate(
        self, 
        t_final: float, 
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        super().simulate(t_final, resolution)
        
        states = {
            'v_pa': torch.tensor(convert(0.185, 'l')),
            'v_pu': torch.tensor(convert(0.90, 'l')),
            'v_lv': torch.tensor(convert(0.135, 'l')),
            'v_ao': torch.tensor(convert(0.95, 'l')),
            'v_vc': torch.tensor(convert(3.19, 'l')),
            'v_rv': torch.tensor(convert(0.14, 'l')),
        }
        x_0 = self.ode_state_tensor(states)
        v_tot = convert(x_0.sum(), to='l')
        print(f'Total blood volume: {v_tot.item():.4f}l')

        t = torch.linspace(0, t_final, t_final*resolution + 1)
        sol = odeint(
            self, 
            x_0, 
            t, 
            method='dopri5', 
            rtol=1e-6, 
            atol=1e-6, 
            options={'max_step': 1e-2},
        )

        return t, sol

    def solve_v_spt_linearised(
        self, 
        v_lv: torch.Tensor, 
        v_rv: torch.Tensor, 
        e_t: torch.Tensor, 
        method: str = 'jallon',
    ):
        # Linearised solutions - not used
        raise NotImplementedError

        if method == 'jallon':
            # Jallon method
            num = e_t * (
                + self.spt.e_es * self.spt.v_d 
                + self.lvf.e_es * (v_lv - self.lvf.v_d)
                - self.rvf.e_es * (v_rv - self.rvf.v_d)
            ) + (1 - e_t) * (
                + self.spt.p_0 * self.spt.lam * self.spt.v_0
                + self.lvf.p_0 * self.lvf.lam * (v_lv - self.lvf.v_0)
                - self.rvf.p_0 * self.rvf.lam * (v_rv - self.rvf.v_0)
            )
            den = e_t * (
                self.spt.e_es + self.lvf.e_es + self.rvf.e_es
            ) + (1 - e_t) * (
                + self.spt.p_0 * self.spt.lam
                + self.lvf.p_0 * self.lvf.lam
                + self.rvf.p_0 * self.rvf.lam
            )
            v_spt = num / den

        elif method == 'smith':
            # Smith 2005 method
            del_v_spt = convert(0.1, 'ml')
            x_1 = self._v_spt_old + del_v_spt
            x_2 = self._v_spt_old - del_v_spt

            def a_b(pv):
                a = (torch.exp(pv.lam * x_2) - torch.exp(pv.lam * x_1)) / (x_2 - x_1)
                b = torch.exp(pv.lam * x_1) - (torch.exp(pv.lam * x_2) - torch.exp(pv.lam * x_1) * x_1 / (x_2 - x_1))
                return a, b

            a_spt, b_spt = a_b(self.spt)
            a_lvf, b_lvf = a_b(self.lvf)
            a_rvf, b_rvf = a_b(self.rvf)

            v_spt_lin = (
                (1 - e_t) * (
                    self.spt.p_0 * (torch.exp(-self.spt.lam * self.spt.v_0) * b_spt - 1)
                    - self.lvf.p_0 * (torch.exp(self.lvf.lam * v_lv) * b_lvf - 1)
                    + self.rvf.p_0 * (torch.exp(self.rvf.lam * v_rv) * b_rvf - 1)
                ) - e_t * (
                    self.spt.e_es * self.spt.v_d + self.lvf.e_es * v_lv - self.rvf.e_es * v_rv
                )
            ) / (
                (1 - e_t) * (
                    - self.spt.p_0 * torch.exp(-self.spt.lam * self.spt.v_0) * a_spt
                    + self.lvf.p_0 * torch.exp(self.lvf.lam * v_lv) * a_lvf
                    - self.rvf.p_0 * torch.exp(self.rvf.lam * v_rv) * a_rvf
                ) - e_t * (
                    self.spt.e_es + self.lvf.e_es + self.rvf.e_es
                )
            )

        # self._v_spt_old = v_spt
        return v_spt

    def solve_v_spt(
        self, 
        v_lv: torch.Tensor, 
        v_rv: torch.Tensor, 
        e_t: torch.Tensor,
    ) -> torch.Tensor:

        # No explicit solution for v_spt, need to use root finder
        if USE_XITORCH:
            x_tol = 1e-3
            f_tol = 1e-5
            try:
                self._v_spt_old = rootfinder(
                    self.v_spt_residual, 
                    torch.tensor([self._v_spt_old]), 
                    params=(v_lv, v_rv, e_t, self), 
                    method='broyden1', 
                    x_tol=x_tol, 
                    f_tol=f_tol,
                )
            except (ValueError, ConvergenceWarning):
                # Retry with a different method
                # TODO: fix xitorch
                self._v_spt_old = rootfinder(
                    self.v_spt_residual, 
                    torch.tensor([self._v_spt_old]), 
                    params=(v_lv, v_rv, e_t, self), 
                    method='linearmixing', 
                    x_tol=x_tol, 
                    f_tol=f_tol,
                )
        else:
            # t1 = perf_counter()
            # sol = root_scalar(
            #     self.v_spt_residual, 
            #     (v_lv, v_rv, e_t, self), 
            #     x0=self._v_spt_old, 
            #     method='newton', 
            #     xtol=1e-5,
            #     rtol=1e-3,
            #     fprime=True,
            # )
            # if not sol.converged:
            #     raise Exception

            v_spt = self._v_spt_old
            i = 0
            while True:
                i += 1
                res, grad = self.v_spt_residual_analytical(v_spt, v_lv, v_rv, e_t, self)
                dv = res / grad
                v_spt = v_spt - dv
                step_abs = dv.abs()
                if step_abs < 1e-5 * v_spt.abs():
                    # Rel tol
                    break
                if step_abs < 1e-6:
                    # Abs tol
                    break
                
            # print(f'Iterations: {i}')
            self._v_spt_old = v_spt
            # t2 = perf_counter()
            # print(f'solve inner: {t2-t1:.2e}s, {sol.function_calls} calls')

        # v_spt = self._v_spt_old / self._v_spt_scale
        return v_spt

    @staticmethod
    def v_spt_residual_analytical(
        v_spt_in: torch.Tensor,
        v_lv: torch.Tensor,
        v_rv: torch.Tensor,
        e_t: torch.Tensor,
        cvs: "SmithCardioVascularSystem"
    ) -> torch.Tensor:

        # Implemented as static method as xitorch requires a pure function
        # TODO: if we don't use xitorch, can use normal method

        v_spt = v_spt_in/cvs._v_spt_scale

        # Free wall volumes v_(lvf/rvf/spt) are not physical volumes, but 
        # defined to capture deflection of cardiac free walls relative to 
        # ventricle volumes
        # Eq. 9, 10
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        # Eq. 16, 17
        p_lvf = e_t * cvs.lvf.p_es(v_lvf) + (1 - e_t) * cvs.lvf.p_ed(v_lvf)
        p_rvf = e_t * cvs.rvf.p_es(v_rvf) + (1 - e_t) * cvs.rvf.p_ed(v_rvf)
        p_spt = e_t * cvs.spt.p_es(v_spt) + (1 - e_t) * cvs.spt.p_ed(v_spt)

        # Eq. 15
        p_spt_rhs = p_lvf - p_rvf

        # Residual between Eq. 15 and 17
        res = p_spt - p_spt_rhs

        dp_dv_lvf = e_t * cvs.lvf.dp_es_dv(v_lvf) + (1 - e_t) * cvs.lvf.dp_ed_dv(v_lvf)
        dp_dv_rvf = e_t * cvs.rvf.dp_es_dv(v_rvf) + (1 - e_t) * cvs.rvf.dp_ed_dv(v_rvf)
        dp_dv_spt = e_t * cvs.spt.dp_es_dv(v_spt) + (1 - e_t) * cvs.spt.dp_ed_dv(v_spt)

        grad = dp_dv_lvf + dp_dv_rvf + dp_dv_spt

        return res, grad

    @staticmethod
    def v_spt_residual(
        v_spt_in: float,
        v_lv: torch.Tensor,
        v_rv: torch.Tensor,
        e_t: torch.Tensor,
        cvs: "SmithCardioVascularSystem", 
        verbose: bool = False, 
        deriv: bool = True,
    ) -> float:
        """Broken-down implementation of Eq. 20 (Smith 2004)"""

        # t1 = perf_counter()

        v_spt_in = torch.as_tensor(v_spt_in)
        v_spt_in.requires_grad_(deriv)

        if deriv and v_spt_in.grad is not None:
            v_spt_in.grad.zero_()

        v_spt = v_spt_in/cvs._v_spt_scale

        # Free wall volumes v_(lvf/rvf/spt) are not physical volumes, but 
        # defined to capture deflection of cardiac free walls relative to 
        # ventricle volumes
        # Eq. 9, 10
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        # Eq. 16, 17
        p_lvf = e_t * cvs.lvf.p_es(v_lvf) + (1 - e_t) * cvs.lvf.p_ed(v_lvf)
        p_rvf = e_t * cvs.rvf.p_es(v_rvf) + (1 - e_t) * cvs.rvf.p_ed(v_rvf)
        p_spt = e_t * cvs.spt.p_es(v_spt) + (1 - e_t) * cvs.spt.p_ed(v_spt)

        # Eq. 15
        p_spt_rhs = p_lvf - p_rvf

        # Residual between Eq. 15 and 17
        res = p_spt - p_spt_rhs

        if deriv:
            res.backward()
            grad = v_spt_in.grad.detach()
            v_spt_in.requires_grad_(False)
        
            if verbose:
                print(f'x={v_spt*1e3}, res={res}, grad={grad}')

            out = (res.detach().item(), grad.detach().item())

            # t2 = perf_counter()
            # print(f'solve fn: {t2-t1:.2e}s')
            
            return out
        else:
            
            if verbose:
                print(f'x={v_spt*1e3}, res={res}')

            return res.detach().item()

class InertialSmithCVS(SmithCardioVascularSystem):
    
    state_names: ClassVar[list[str]] = ['v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv', 'q_mt', 'q_av', 'q_tc', 'q_pv']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mitral valve
        self.mt = InertialValve(convert(0.0158, 'mmHg s/ml'), convert(7.6967e-5, 'mmHg s^2/ml'))
        # Tricuspid valve
        self.tc = InertialValve(convert(0.0237, 'mmHg s/ml'), convert(8.0093e-5, 'mmHg s^2/ml'))
        # Aortic valve
        self.av = InertialValve(convert(0.0180, 'mmHg s/ml'), convert(1.2189e-4, 'mmHg s^2/ml'))
        # Pulmonary valve
        self.pv = InertialValve(convert(0.0055, 'mmHg s/ml'), convert(1.4868e-4, 'mmHg s^2/ml'))

        # Set Paeme 2011 parameters
        with torch.no_grad():
            self.load_state_dict(parameters.paeme_2011_cvs)

    def derivatives(self, states: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
        super().derivatives(states, outputs)
        outputs['dq_mt_dt'] = self.mt.flow_rate_deriv(outputs['p_pu'], outputs['p_lv'], states['q_mt'])
        outputs['dq_av_dt'] = self.av.flow_rate_deriv(outputs['p_lv'], outputs['p_ao'], states['q_av'])
        outputs['dq_tc_dt'] = self.tc.flow_rate_deriv(outputs['p_vc'], outputs['p_rv'], states['q_tc'])
        outputs['dq_pv_dt'] = self.pv.flow_rate_deriv(outputs['p_rv'], outputs['p_pa'], states['q_pv'])

    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        outputs = self.pressures_volumes(t, states)
        self.flow_rates(outputs)
        outputs['q_mt'] = torch.clamp(states['q_mt'], min=0.0)
        outputs['q_av'] = torch.clamp(states['q_av'], min=0.0)
        outputs['q_tc'] = torch.clamp(states['q_tc'], min=0.0)
        outputs['q_pv'] = torch.clamp(states['q_pv'], min=0.0)
        self.derivatives(states, outputs)

        return outputs

    def flow_rates(self, outputs: dict[str, torch.Tensor], static: bool = False):
        if static:
            super().flow_rates(outputs)
        else:
            outputs['q_pul'] = self.pul.flow_rate(outputs['p_pa'], outputs['p_pu'])
            outputs['q_sys'] = self.sys.flow_rate(outputs['p_ao'], outputs['p_vc'])

    def simulate(self, t_final: float, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:

        super().simulate(t_final, resolution)

        # Matlab
        states = {
            'v_pa': torch.tensor(convert(0.043, 'l')),
            'v_pu': torch.tensor(convert(0.808, 'l')),
            'v_lv': torch.tensor(convert(0.094, 'l')),
            'v_ao': torch.tensor(convert(0.133, 'l')),
            'v_vc': torch.tensor(convert(0.330, 'l')),
            'v_rv': torch.tensor(convert(0.090, 'l')),
        }

        init_outputs = self.pressures_volumes(torch.tensor(0.), states)
        self.flow_rates(init_outputs, static=True)
        x_0 = self.ode_state_tensor(init_outputs)

        # sol = solve_ivp(
        #     self, [0, 2], x_0, method='RK45', dense_output=True,
        #     max_step=1e-4,
        #     )
        t = torch.linspace(0, t_final, int(t_final*resolution) + 1)
        sol = odeint(
            self, 
            x_0, 
            t, 
            method='dopri5', 
            rtol=1e-6, 
            atol=1e-6, 
            options={'max_step': 1e-2},
        )

        return t, sol

class JallonHeartLungs(ODEBase):

    state_names: ClassVar[list[str]] = (
        SmithCardioVascularSystem.state_names + 
        PassiveRespiratorySystem.state_names + 
        RespiratoryPatternGenerator.state_names
    )

    def __init__(self):
        super().__init__()
        self.resp_pattern = RespiratoryPatternGenerator()
        self.resp = PassiveRespiratorySystem()
        self.cvs = SmithCardioVascularSystem()

        self.store = []

        # Jallon CVS model modifications
        with torch.no_grad():
            # Table 2 of Jallon 2009
            self.cvs.spt.e_es.copy_(torch.tensor(convert(3750, 'mmHg s/l')))  # Wrong units in paper
            self.cvs.spt.lam.copy_(torch.tensor(convert(35, '1/l')))
            self.cvs.vc.e_es.copy_(torch.tensor(convert(2, 'mmHg s/l')))
            # HR = 54bpm
            self.cvs.e.hr.copy_(torch.tensor(54.0))
            # Eq 2.9, 2.10
            self.cvs.p_pl_affects_pu_and_pa.copy_(torch.tensor(True))

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        super().callback_accept_step(t, x, dt)
        self.cvs._v_spt_old = self.trajectory[-1][3]['v_spt']

    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # print(t)
        # print(states)
        
        # Run respiratory model
        resp_outputs = self.resp.model(t, states)

        # Update parameters of cardiovascular model
        with torch.no_grad():
            self.cvs.p_pl.copy_(resp_outputs['p_pl'])

        cvs_outputs = self.cvs.model(t, states)

        # Update parameters of respiratory pattern generator
        with torch.no_grad():
            self.resp_pattern.dv_alv_dt.copy_(resp_outputs['dv_alv_dt'])

        resp_pattern_outputs = self.resp_pattern.model(t, states)

        all_outputs = states | resp_outputs | cvs_outputs | resp_pattern_outputs

        self.store.append((t, states, all_outputs))

        return all_outputs

    def simulate(self, t_final: float, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        super().simulate(t_final, resolution)

        states = {
            # Blood volume: should total 5.5
            'v_pa': torch.tensor(convert(0.1875, 'l')),
            'v_pu': torch.tensor(convert(0.80, 'l')),
            'v_lv': torch.tensor(convert(0.135, 'l')),
            'v_ao': torch.tensor(convert(0.9, 'l')),
            'v_vc': torch.tensor(convert(3.35, 'l')),
            'v_rv': torch.tensor(convert(0.1275, 'l')),
            # Jallon 2009, Table 1:
            'v_alv': torch.tensor(convert(0.5, 'l')),
            'p_mus': torch.tensor(0.0),
            'x': torch.tensor(-0.6),
            'y': torch.tensor(0.0),
        }

        # Commented out as not using inertial valves
        # init_outputs = self.cvs.pressures_volumes(torch.tensor(0.), states)
        # self.cvs.flow_rates(init_outputs, static=True)

        x_0 = self.ode_state_tensor(states)
        
        v_tot = convert(x_0[:6].sum(), to='l')
        print(f'Total blood volume: {v_tot.item():.3f}l')

        t = torch.linspace(0, t_final, int(t_final*resolution) + 1)
        sol = odeint(
            self, 
            x_0, 
            t, 
            method='dopri5', 
            rtol=1e-9,
            atol=1e-6,
            options={'max_step': 1e-2},
        )

        return t, sol
