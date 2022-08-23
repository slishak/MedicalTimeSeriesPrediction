from time import perf_counter
import warnings
from dataclasses import dataclass
from typing import ClassVar, Union, Optional, Callable, Type

import torch
from torch import nn
from scipy.optimize import root_scalar
from xitorch.optimize import rootfinder
from xitorch._utils.exceptions import ConvergenceWarning

from biophysical_models.base_classes import (
    ODEBase, 
    CardiacDriver, 
    DynamicCardiacDriver,
    Valve, 
    BloodVessel, 
    PressureVolume, 
    InertialValve, 
    RespiratoryPatternGenerator
)
from biophysical_models.unit_conversions import convert
from biophysical_models import parameters

warnings.simplefilter('error', ConvergenceWarning)
# pd.options.plotting.backend = 'plotly'


@dataclass
class PassiveRespiratorySystem(ODEBase):
    """Passive mechanical respiratory system. Note that this model cannot be 
    simulated in isolation as it requires additional states from the 
    respiratory pattern generator and the cardiovascular model.
    
    (Jallon, 2009)"""

    state_names: ClassVar[list[str]] = ['v_alv']

    def __init__(
        self, 
        e_alv: float = convert(5, 'cmH2O/l'),
        e_cw: float = convert(4, 'cmH2O/l'),
        r_ua: float = convert(5, 'cmH2O s/l'),
        r_ca: float = convert(1, 'cmH2O s/l'),
        v_th0: float = convert(2, 'l'),
    ):
        """Initialise

        Args:
            e_alv (float, optional): Alveolar elastance. Defaults to 5 cmH2O/l
            e_cw (float, optional): Chest wall elastance. Defaults to 4 cmH2O/l
            r_ua (float, optional): Upper airways resistance. Defaults to 
                5 cmH2O s/l. Note: Jallon gives incorrect units of cmH2O/l
            r_ca (float, optional): Central airways resistance. Defaults to 
                1 cmH2O s/l. Note: Jallon gives incorrect units of cmH2O/l
            v_th0 (float, optional): Rib cage or intrathoracic volume at zero
                pressure. Defaults to 2 l. Note: Jallon gives incorrect units 
                of 1/l.
        """
        super().__init__()
        self.e_alv = nn.Parameter(torch.as_tensor(e_alv), requires_grad=False)
        self.e_cw = nn.Parameter(torch.as_tensor(e_cw), requires_grad=False)
        self.r_ua = nn.Parameter(torch.as_tensor(r_ua), requires_grad=False)
        self.r_ca = nn.Parameter(torch.as_tensor(r_ca), requires_grad=False)
        self.v_th0 = nn.Parameter(torch.as_tensor(v_th0), requires_grad=False)

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Passive respiratory model implementation

        Args:
            t (torch.Tensor): Time (s)
            states (torch.Tensor): Model states

        Returns:
            dict[str, torch.Tensor]: Model outputs
        """

        # These are not states of this model, but states of the overall system
        v_pcd = states['v_lv'] + states['v_rv']
        v_pu = states['v_pu']
        v_pa = states['v_pa']

        v_bth = v_pcd + v_pu + v_pa
        v_th = v_bth + states['v_alv']
        p_pl = states['p_mus'] + self.e_cw * (v_th - self.v_th0)

        dv_alv_dt = -(p_pl + self.e_alv * states['v_alv']) / (self.r_ca + self.r_ua)

        outputs = {
            'p_pl': p_pl,
            'v_th': v_th,
            'v_bth': v_bth,
            'dv_alv_dt': dv_alv_dt,
        }

        return outputs

    def init_states(self) -> dict[str, torch.Tensor]:
        """Return initial values of ODE states.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """

        # Jallon 2009, Table 1:
        return {
            'v_alv': torch.tensor(convert(0.5, 'l')),
        }

class SmithCardioVascularSystem(ODEBase):
    """Smith CVS model with no inertia and Heaviside valve law
    
    (Smith, 2004) and (Hann, 2004)"""

    state_names: ClassVar[list[str]] = ['v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv']

    def __init__(
        self, 
        *args, 
        p_pl_is_input: bool = False, 
        f_hr: Optional[Callable] = None, 
        **kwargs,
    ):
        """Initialise. All parameters passed to nn.Module. 

        Default parameter values from Smith, 2005.

        Args:
            p_pl_is_input (bool, optional): Get pleural pressure from input,
                otherwise create a parameter. Defaults to False.
            f_hr (Callable, optional): Heart rate as a function of time. 
                Defaults to None, in which case a constant heart rate model is
                used.
        """

        super().__init__(*args, **kwargs)

        self._p_pl_is_input = p_pl_is_input

        # Valves
        self.mt = Valve(convert(0.06, 'kPa s/l'))  # Mitral valve
        self.tc = Valve(convert(0.18, 'kPa s/l'))  # Tricuspid valve
        self.av = Valve(convert(1.4, 'kPa s/l'))  # Aortic valve
        self.pv = Valve(convert(0.48, 'kPa s/l'))  # Pulmonary valve

        # Circulation resistance
        self.pul = BloodVessel(convert(19, 'kPa s/l'))  # Pulmonary circulation
        self.sys = BloodVessel(convert(140, 'kPa s/l'))  # Systematic circulation

        # Pleural pressure
        if not p_pl_is_input:
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
        if f_hr is None:
            self.e = CardiacDriver(hr=80.0)
            self.dynamic_hr = False
        else:
            self.e = DynamicCardiacDriver(hr=f_hr)
            self.dynamic_hr = True
            self.state_names = self.state_names + self.e.state_names

        # Total blood volume
        self.v_tot = nn.Parameter(torch.tensor(convert(5.5, 'l')), requires_grad=False)

        # Jallon 2009 modification
        self.p_pl_affects_pu_and_pa = nn.Parameter(torch.tensor(False), requires_grad=False)

        # First initial guess for v_spt
        self._v_spt_old = torch.tensor(convert(0.0055, 'l'))
        self._v_spt_scale = 1.0

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        """Called by torchdiffeq at the end of a successful step. Used to 
        build outputs (irregularly-spaced grid) in self.trajectory.

        Stores last value of v_spt for use as initial guess in next step.

        Args:
            t (torch.Tensor): Time (s)
            x (torch.Tensor): ODE state tensor
            dt (torch.Tensor): Time step (s)
        """
        super().callback_accept_step(t, x, dt)
        self._v_spt_old = self.trajectory[-1][3]['v_spt']

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Model implementation

        Args:
            t (torch.Tensor): Time (s)
            states (dict[str, torch.Tensor]): Model states
            p_pl (torch.Tensor, optional): Optional input to ODE. Defaults to 
                None.

        Returns:
            dict[str, torch.Tensor]: Model outputs
        """

        # t1 = perf_counter()

        if self._p_pl_is_input:
            assert p_pl is not None, "p_pl not passed to CVS model"
        else:
            p_pl = self.p_pl

        outputs = self.pressures_volumes(t, states, p_pl)
        self.flow_rates(outputs)
        self.derivatives(states, outputs)

        # t2 = perf_counter()
        # print(f'total: {t2-t1:.2e}s')

        return outputs

    def flow_rates(self, outputs: dict[str, torch.Tensor]):
        """Compute flow rates from pressures and add to output dict.

        Args:
            outputs (dict[str, torch.Tensor]): Partial model outputs containing
                pressures
        """
        outputs['q_mt'] = self.mt.flow_rate(outputs['p_pu'], outputs['p_lv'])
        outputs['q_av'] = self.av.flow_rate(outputs['p_lv'], outputs['p_ao'])
        outputs['q_tc'] = self.tc.flow_rate(outputs['p_vc'], outputs['p_rv'])
        outputs['q_pv'] = self.pv.flow_rate(outputs['p_rv'], outputs['p_pa'])
        outputs['q_pul'] = self.pul.flow_rate(outputs['p_pa'], outputs['p_pu'])
        outputs['q_sys'] = self.sys.flow_rate(outputs['p_ao'], outputs['p_vc'])

    def derivatives(self, states: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]):
        """Compute volume derivatives from flow rates and add to output dict.

        Args:
            states (dict[str, torch.Tensor]): Model states (unused in this 
                model)
            outputs (dict[str, torch.Tensor]): Partial model outputs containing
                flow rates
        """

        # Chamber volume changes
        # Explicitly defined in Smith 2007, Fig. 1, C17-C22
        outputs['dv_pa_dt'] = outputs['q_pv'] - outputs['q_pul']
        outputs['dv_pu_dt'] = outputs['q_pul'] - outputs['q_mt']
        outputs['dv_lv_dt'] = outputs['q_mt'] - outputs['q_av']
        outputs['dv_ao_dt'] = outputs['q_av'] - outputs['q_sys']
        outputs['dv_vc_dt'] = outputs['q_sys'] - outputs['q_tc']
        outputs['dv_rv_dt'] = outputs['q_tc'] - outputs['q_pv']

    def pressures_volumes(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute pressures and volumes of cardiovascular model.

        Pleural pressure either comes from a parameter or an external input.
        
        Equation numbers from Smith 2004 (or 2007 if explicitly stated).

        See Smith 2007, Fig. 1 for a general overview.

        Args:
            t (torch.Tensor): Time (s)
            states (dict[str, torch.Tensor]): Model states
            p_pl (torch.Tensor, Optional): Pleural pressure

        Returns:
            dict[str, torch.Tensor]: Pressure/volume outputs
        """

        if self._p_pl_is_input:
            assert p_pl is not None, "p_pl not passed to CVS model"
        else:
            p_pl = self.p_pl

        # Evaluate model driving function
        if self.dynamic_hr:
            cardiac_driver_output = self.e.model(t, states)
            e_t = cardiac_driver_output['e_t']
        else:
            e_t = self.e(t)
            cardiac_driver_output = {'e_t': e_t}
            
        # Ventricular pressure-volume relationship
        # t1 = perf_counter()
        v_spt = self.solve_v_spt(states['v_lv'], states['v_rv'], e_t)
        # t2 = perf_counter()
        # print(f'solve: {t2-t1:.2e}s')

        # t1 = perf_counter()
        # Eq. 9, 10, 16
        v_lvf = states['v_lv'] - v_spt
        v_rvf = states['v_rv'] + v_spt
        p_lvf = self.lvf.p(v_lvf, e_t) 
        p_rvf = self.rvf.p(v_rvf, e_t)
        p_spt = self.spt.p(v_spt, e_t)

        # Pericardium pressure-volume relationship
        # Eq. 11, 18, 14. Note p_pl (pleural cavity) is p_th (thoracic cavity)
        v_pcd = states['v_lv'] + states['v_rv']
        p_pcd = self.pcd.p_ed(v_pcd)
        p_peri = p_pcd + p_pl

        # Eq. 12, 13
        p_lv = p_lvf + p_peri
        p_rv = p_rvf + p_peri

        # Peripheral chamber pressure-volume relationships
        p_pa = self.pa.p_es(states['v_pa'])
        p_pu = self.pu.p_es(states['v_pu'])
        p_ao = self.ao.p_es(states['v_ao'])
        p_vc = self.vc.p_es(states['v_vc'])

        if self.p_pl_affects_pu_and_pa:
            p_pa = p_pa + p_pl
            p_pu = p_pu + p_pl
            
        # t2 = perf_counter()
        # print(f'pv: {t2-t1:.2e}s')

        outputs = {
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
        } | cardiac_driver_output | states
        
        return outputs

    def init_states(
        self,
        r_pa: float = 0.034,
        r_pu: float = 0.164,
        r_lv: float = 0.025,
        r_ao: float = 0.173,
        r_rv: float = 0.024,
    ) -> dict[str, torch.Tensor]:
        """Return initial values of ODE states by setting initial proportions 
        of the total blood volume in each of the six compartments. The volume
        of blood in the vena cava is automatically calculated as whatever is 
        left over.

        Args:
            r_pa (float, optional): Proportion of blood initially in pulmonary
                artery. Defaults to 0.034.
            r_pu (float, optional): Proportion of blood initially in pulmonary 
                vein. Defaults to 0.164.
            r_lv (float, optional): Proportion of blood initially in left 
                ventricle. Defaults to 0.025.
            r_ao (float, optional): Proportion of blood initially in aorta. 
                Defaults to 0.173.
            r_rv (float, optional): Proportion of blood initially in right 
                ventricle. Defaults to 0.024.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """

        r_vc = 1 - r_pa - r_pu - r_lv - r_ao - r_rv

        assert r_vc > 0, "Initial v_vc must not be negative"

        states = {
            'v_pa': r_pa * self.v_tot,
            'v_pu': r_pu * self.v_tot,
            'v_lv': r_lv * self.v_tot,
            'v_ao': r_ao * self.v_tot,
            'v_vc': r_vc * self.v_tot,
            'v_rv': r_rv * self.v_tot,
        }

        if self.dynamic_hr:
            states['s'] = torch.tensor(0.)

        return states

    def solve_v_spt(
        self, 
        v_lv: torch.Tensor, 
        v_rv: torch.Tensor, 
        e_t: torch.Tensor,
        method: str = 'newton',
        rtol: float = 1e-5,
        xtol: float = 1e-6,
    ) -> torch.Tensor:
        """Find value for v_spt using root finding algorithm.

        TODO: Remove all algorithms other than Newton-Raphson.

        Args:
            v_lv (torch.Tensor): Left ventricle volume
            v_rv (torch.Tensor): Right ventricle volume
            e_t (torch.Tensor): Cardiac driver function
            method (str): Root finding algorithm. Defaults to 'newton', which 
                is implemented in-line for simplicity. Also available: 
                'xitorch', 'scipy'.
            xtol (float): Absolute tolerance for v_spt. Defaults to 1e-5.
            rtol (float): Relative tolerance for v_spt. Defaults to 1e-6.

        Returns:
            torch.Tensor: v_spt solution
        """

        # No explicit solution for v_spt, need to use root finder
        if method == 'xitorch':
            f_tol = 1e-5
            try:
                v_spt_scaled = rootfinder(
                    self.v_spt_residual, 
                    torch.tensor([self._v_spt_old]), 
                    params=(v_lv, v_rv, e_t, self), 
                    method='broyden1', 
                    x_tol=xtol, 
                    f_tol=f_tol,
                )
            except (ValueError, ConvergenceWarning):
                # Retry with a different method
                # TODO: fix xitorch
                v_spt_scaled = rootfinder(
                    self.v_spt_residual, 
                    torch.tensor([self._v_spt_old]), 
                    params=(v_lv, v_rv, e_t, self), 
                    method='linearmixing', 
                    x_tol=xtol, 
                    f_tol=f_tol,
                )
        elif method == 'newton':
            v_spt = self._v_spt_old
            i = 0
            while True:
                i += 1
                res, grad = self.v_spt_residual_analytical(v_spt, v_lv, v_rv, e_t, self)
                dv = res / grad
                v_spt = v_spt - dv
                step_abs = dv.abs()
                if step_abs < rtol * v_spt.abs():
                    # Rel tol
                    break
                if step_abs < xtol:
                    # Abs tol
                    break
                
            # print(f'Iterations: {i}')
            v_spt_scaled = v_spt
        elif method == 'scipy':

            sol = root_scalar(
                self.v_spt_residual, 
                (v_lv, v_rv, e_t, self), 
                x0=self._v_spt_old, 
                method='newton', 
                xtol=xtol,
                rtol=rtol,
                fprime=True,
            )

            if not sol.converged:
                raise Exception

            v_spt_scaled = sol.root
        else:
            raise NotImplementedError(method)
            
        v_spt = v_spt_scaled / self._v_spt_scale
        return v_spt

    @staticmethod
    def v_spt_residual_analytical(
        v_spt_in: torch.Tensor,
        v_lv: torch.Tensor,
        v_rv: torch.Tensor,
        e_t: torch.Tensor,
        cvs: "SmithCardioVascularSystem"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual function for v_spt, with analytic derivative.
        
        See Smith 2004, Eq. 20

        Implemented as static method as xitorch requires a pure function

        TODO: if we don't use xitorch, can use normal method

        Args:
            v_spt_in (torch.Tensor): Current scaled value of v_spt from 
                root finding algorithm
            v_lv (torch.Tensor): Left ventricle volume
            v_rv (torch.Tensor): Right ventricle volume
            e_t (torch.Tensor): Cardiac driver function
            cvs (SmithCardioVascularSystem): self

        Returns:
            Tuple containing:
            - res (torch.Tensor): Residual
            - grad (torch.Tensor): Gradient of residual wrt scaled v_spt
        """

        v_spt = v_spt_in/cvs._v_spt_scale

        # Free wall volumes v_(lvf/rvf/spt) are not physical volumes, but 
        # defined to capture deflection of cardiac free walls relative to 
        # ventricle volumes
        # Eq. 9, 10
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        # Eq. 15, 16, 17
        res = cvs.spt.p(v_spt, e_t) - cvs.lvf.p(v_lvf, e_t) + cvs.rvf.p(v_rvf, e_t)

        # Analytical gradient of residual wrt v_spt
        grad = cvs.lvf.dp_dv(v_lvf, e_t) + cvs.rvf.dp_dv(v_rvf, e_t) + cvs.spt.dp_dv(v_spt, e_t)

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
    ) -> Union[float, tuple[float, float]]:
        """Residual function for v_spt, with derivative computed by PyTorch.

        Returns floats rather than torch.Tensor (c.f. v_spt_residual_analytical)

        # TODO: Delete this method

        See Smith 2004, Eq. 20

        Args:
            v_spt_in (torch.Tensor): Current scaled value of v_spt from 
                root finding algorithm
            v_lv (torch.Tensor): Left ventricle volume
            v_rv (torch.Tensor): Right ventricle volume
            e_t (torch.Tensor): Cardiac driver function
            cvs (SmithCardioVascularSystem): self
            verbose (bool, optional): Print progress. Defaults to False.
            deriv (bool, optional): Enable derivative output. Defaults to True.

        Returns:
            If deriv=False, then a single float (residual)
            Otherwise, a tuple containing:
            - res (float): Residual
            - grad (float): Gradient of residual wrt scaled v_spt
        """

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
        p_lvf = cvs.lvf.p(v_lvf, e_t)
        p_rvf = cvs.rvf.p(v_rvf, e_t)
        p_spt = cvs.spt.p(v_spt, e_t)

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
    """Smith CVS model with inertia and Heaviside valve law
    
    (Smith, 2004), (Hann, 2004) and (Paeme, 2011)"""
    
    state_names: ClassVar[list[str]] = [
        'v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv', 
        'q_mt', 'q_av', 'q_tc', 'q_pv',
    ]

    def __init__(self, *args, **kwargs):
        """Initialise. All parameters passed to nn.Module. 

        Default parameter values from Paeme, 2011.
        """
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
        """Compute volume derivatives from flow rates and add to output dict.

        Args:
            states (dict[str, torch.Tensor]): Model states (unused in this 
                model)
            outputs (dict[str, torch.Tensor]): Partial model outputs containing
                flow rates
        """
        super().derivatives(states, outputs)
        outputs['dq_mt_dt'] = self.mt.flow_rate_deriv(outputs['p_pu'], outputs['p_lv'], states['q_mt'])
        outputs['dq_av_dt'] = self.av.flow_rate_deriv(outputs['p_lv'], outputs['p_ao'], states['q_av'])
        outputs['dq_tc_dt'] = self.tc.flow_rate_deriv(outputs['p_vc'], outputs['p_rv'], states['q_tc'])
        outputs['dq_pv_dt'] = self.pv.flow_rate_deriv(outputs['p_rv'], outputs['p_pa'], states['q_pv'])

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Model implementation

        Args:
            t (torch.Tensor): Time (s)
            states (dict[str, torch.Tensor]): Model states
            p_pl (torch.Tensor, optional): Optional input to ODE. Defaults to 
                None.

        Returns:
            dict[str, torch.Tensor]: Model outputs
        """

        outputs = self.pressures_volumes(t, states, p_pl)
        self.flow_rates(outputs)
        outputs['q_mt'] = torch.clamp(states['q_mt'], min=0.0)
        outputs['q_av'] = torch.clamp(states['q_av'], min=0.0)
        outputs['q_tc'] = torch.clamp(states['q_tc'], min=0.0)
        outputs['q_pv'] = torch.clamp(states['q_pv'], min=0.0)
        self.derivatives(states, outputs)

        return outputs

    def flow_rates(self, outputs: dict[str, torch.Tensor], static: bool = False):
        """Compute flow rates from pressures and add to output dict.

        Args:
            outputs (dict[str, torch.Tensor]): Partial model outputs containing
                pressures
            static (bool): Don't consider inertia in valve laws, just compute
                static solution. Defaults to False.
        """
        if static:
            super().flow_rates(outputs)
        else:
            outputs['q_pul'] = self.pul.flow_rate(outputs['p_pa'], outputs['p_pu'])
            outputs['q_sys'] = self.sys.flow_rate(outputs['p_ao'], outputs['p_vc'])

    def init_states(
        self, 
        r_pa: float = 0.029,
        r_pu: float = 0.539,
        r_lv: float = 0.063,
        r_ao: float = 0.089,
        r_rv: float = 0.061,
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return initial values of ODE states.

        Note that the default blood volume proportions here are quite different
        to those in SmithCardioVascularSystem, as the default parameters from 
        Paeme (2011) seem to use a total blood volume of 1.5 litres rather than
        5.5 litres as stated - or in other words, the unstressed 4 litres are 
        ignored and only the stressed 1.5 litres are simulated (there is only 
        2ml of dead space in total in the system).

        Args:
            r_pa (float, optional): Proportion of blood initially in pulmonary
                artery. Defaults to 0.029.
            r_pu (float, optional): Proportion of blood initially in pulmonary 
                vein. Defaults to 0.539.
            r_lv (float, optional): Proportion of blood initially in left 
                ventricle. Defaults to 0.063.
            r_ao (float, optional): Proportion of blood initially in aorta. 
                Defaults to 0.089.
            r_rv (float, optional): Proportion of blood initially in right 
                ventricle. Defaults to 0.061.
            p_pl (torch.Tensor, optional): Optional input to ODE. Defaults to 
                None.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """

        states = super().init_states(
            r_pa=r_pa,
            r_pu=r_pu,
            r_lv=r_lv,
            r_ao=r_ao,
            r_rv=r_rv,
        )

        # Also compute initial flow rates assuming quasi-steady state
        init_outputs = self.pressures_volumes(torch.tensor(0.), states, p_pl)
        self.flow_rates(init_outputs, static=True)

        states = {
            key: val for key, val in init_outputs.items() 
            if key in self.state_names
        }

        return states

class JallonHeartLungs(ODEBase):
    """Jallon model of heart and lungs (combined cardiovascular and 
    respiratory model). Uses simple valve law with no inertia.
    
    (Jallon, 2009)"""

    state_names: ClassVar[list[str]] = (
        SmithCardioVascularSystem.state_names + 
        PassiveRespiratorySystem.state_names + 
        RespiratoryPatternGenerator.state_names
    )

    def __init__(
        self, 
        *args, 
        f_hr: Optional[Callable] = None, 
        **kwargs,
    ):
        """Initialise. 

        Args:
            f_hr (Callable, optional): Heart rate as a function of time. 
                Defaults to None, in which case a constant heart rate model is
                used.

        All other parameters passed to nn.Module. 

        Default parameters from Smith (2007) with modifications from 
        Jallon (2009)
        """

        super().__init__(*args, **kwargs)
        self.resp_pattern = RespiratoryPatternGenerator()
        self.resp = PassiveRespiratorySystem()
        self.cvs = SmithCardioVascularSystem(p_pl_is_input=True, f_hr=f_hr)

        # Jallon CVS model modifications
        with torch.no_grad():
            # Table 2 of Jallon 2009
            # self.cvs.spt.e_es.copy_(torch.tensor(convert(3750, 'mmHg s/l')))  # Wrong units in paper
            # self.cvs.spt.lam.copy_(torch.tensor(convert(35, '1/l')))
            # self.cvs.vc.e_es.copy_(torch.tensor(convert(2, 'mmHg s/l')))
            # # HR = 54bpm
            # self.cvs.e.hr.copy_(torch.tensor(54.0))
            # Eq 2.9, 2.10
            self.cvs.p_pl_affects_pu_and_pa.copy_(torch.tensor(True))

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        """Called by torchdiffeq at the end of a successful step. Used to 
        build outputs (irregularly-spaced grid) in self.trajectory.

        Stores last value of v_spt for use as initial guess in next step.

        Args:
            t (torch.Tensor): Time (s)
            x (torch.Tensor): ODE state tensor
            dt (torch.Tensor): Time step (s)
        """
        super().callback_accept_step(t, x, dt)
        self.cvs._v_spt_old = self.trajectory[-1][3]['v_spt']

    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Model implementation. Just consists of calling the lower-level 
        models and passing parameters between. See Figure 5 from Jallon (2009)
        for schematic.

        Args:
            t (torch.Tensor): Time (s)
            states (torch.Tensor): Model states

        Returns:
            dict[str, torch.Tensor]: Model outputs
        """

        # Run respiratory model
        resp_outputs = self.resp.model(t, states)
        # Run cardiovascular model
        cvs_outputs = self.cvs.model(t, states, resp_outputs['p_pl'])
        # Run respiratory pattern generator
        resp_pattern_outputs = self.resp_pattern.model(t, states, resp_outputs['dv_alv_dt'])

        all_outputs = states | resp_outputs | cvs_outputs | resp_pattern_outputs

        return all_outputs

    def init_states(self) -> dict[str, torch.Tensor]:
        """Return initial values of ODE states.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """

        cvs_states = self.cvs.init_states(
            r_pa=0.034,
            r_pu=0.145,
            r_lv=0.025,
            r_ao=0.164,
            r_rv=0.023,
        )
        resp_states = self.resp.init_states()
        resp_pattern_states = self.resp_pattern.init_states()

        init_states = cvs_states | resp_states | resp_pattern_states

        return init_states
