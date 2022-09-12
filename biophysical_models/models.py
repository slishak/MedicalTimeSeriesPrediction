import warnings
from typing import Optional, Callable, Type

import torch
from torch import nn
from torchdiffeq import RejectStepError
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
from biophysical_models.misc import newton_raphson

# pd.options.plotting.backend = 'plotly'


class PassiveRespiratorySystem(ODEBase):
    """Passive mechanical respiratory system.
    
    Note that this model cannot be simulated in isolation as it requires 
    additional states from the respiratory pattern generator and the 
    cardiovascular model.
    
    (Jallon, 2009)
    """

    def __init__(
        self, 
        e_alv: float = convert(5, 'cmH2O/l'),
        e_cw: float = convert(4, 'cmH2O/l'),
        r_ua: float = convert(5, 'cmH2O s/l'),
        r_ca: float = convert(1, 'cmH2O s/l'),
        v_th0: float = convert(2, 'l'),
    ):
        """Initialise.

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
        super().__init__(state_names=['v_alv'])
        self.e_alv = nn.Parameter(torch.as_tensor(e_alv))
        self.e_cw = nn.Parameter(torch.as_tensor(e_cw))
        self.r_ua = nn.Parameter(torch.as_tensor(r_ua))
        self.r_ca = nn.Parameter(torch.as_tensor(r_ca))
        self.v_th0 = nn.Parameter(torch.as_tensor(v_th0))

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Passive respiratory model implementation.

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

    def init_states(self, device='cpu') -> dict[str, torch.Tensor]:
        """Return initial values of ODE states.

        Args:
            device (str, optional): PyTorch device. Defaults to 'cpu'.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """
        # Jallon 2009, Table 1:
        return {
            'v_alv': torch.tensor(convert(0.5, 'l'), device=device),
        }


class SmithCardioVascularSystem(ODEBase):
    """Smith CVS model with no inertia and Heaviside valve law.
    
    (Smith, 2004) and (Hann, 2004)
    """

    def __init__(
        self, 
        p_pl_is_input: bool = False, 
        f_hr: Optional[Callable] = None, 
        volume_ratios: bool = False,
        v_spt_method: str = 'xitorch',
    ):
        """Initialise.

        Default parameter values from Smith, 2005.

        Args:
            p_pl_is_input (bool, optional): Get pleural pressure from input,
                otherwise create a parameter. Defaults to False.
            f_hr (Callable, optional): Heart rate as a function of time. 
                Defaults to None, in which case a constant heart rate model is
                used.
            volume_ratios (bool, optional): Use blood volume proportions as ODE
                states. Otherwise, use raw volumes. Allows v_tot parameter 
                sensitivities to be found correctly. Defaults to False.
            v_spt_method (str, optional): either 'xitorch', 'newton' or 
                'jallon'. Defaults to 'xitorch'.
        """
        super().__init__(state_names=['v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv'])

        self._p_pl_is_input = p_pl_is_input
        self.volume_ratios = volume_ratios

        if self.volume_ratios:
            # Calculate v_vc from other states
            self.state_names.remove('v_vc')

        # Cardiac pattern generator
        if f_hr is None:
            self.e = CardiacDriver(hr=80.0)
            self.dynamic_hr = False
        else:
            self.e = DynamicCardiacDriver(hr=f_hr)
            self.dynamic_hr = True
            self.state_names.extend(self.e.state_names)

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
            self.p_pl = nn.Parameter(torch.tensor(-4.0))

        # Pressure-volume relationships
        # Left ventricle free wall
        self.lvf = PressureVolume(
            convert(454, 'kPa/l'), 
            convert(0.005, 'l'), 
            convert(0.005, 'l'), 
            convert(15, '1/l'), 
            convert(0.17, 'kPa'),
        )
        # Right ventricle free wall
        self.rvf = PressureVolume(
            convert(87, 'kPa/l'), 
            convert(0.005, 'l'), 
            convert(0.005, 'l'), 
            convert(15, '1/l'), 
            convert(0.16, 'kPa'),
        )
        # Septum free wall
        self.spt = PressureVolume(
            convert(6500, 'kPa/l'), 
            convert(0.002, 'l'), 
            convert(0.002, 'l'), 
            convert(435, '1/l'), 
            convert(0.148, 'kPa'),
        )
        # Pericardium
        self.pcd = PressureVolume(
            None, 
            None, 
            convert(0.2, 'l'), 
            convert(30, '1/l'), 
            convert(0.0667, 'kPa'),
        )
        # Vena-cava
        self.vc = PressureVolume(convert(1.5, 'kPa/l'), convert(2.83, 'l'), None, None, None)
        # Pulmonary artery
        self.pa = PressureVolume(convert(45, 'kPa/l'), convert(0.16, 'l'), None, None, None)
        # Pulmonary vein
        self.pu = PressureVolume(convert(0.8, 'kPa/l'), convert(0.2, 'l'), None, None, None)
        # Aorta
        self.ao = PressureVolume(convert(94, 'kPa/l'), convert(0.8, 'l'), None, None, None)
    
        # Total blood volume
        self.v_tot = nn.Parameter(torch.tensor(convert(5.5, 'l')))

        # Jallon 2009 modification
        self.p_pl_affects_pu_and_pa = nn.Parameter(torch.tensor(False), requires_grad=False)

        self._v_spt_method = v_spt_method
        # First initial guess for v_spt
        self._v_spt_old = self._default_v_spt()
        self._v_spt_last = self._v_spt_old

    def _default_v_spt(self):
        return torch.tensor(convert(0.0055, 'l'))

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
        self._v_spt_old = self._v_spt_last

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Model implementation.

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
        if self.volume_ratios:
            deriv_scale = self.v_tot
        else:
            deriv_scale = 1

        # Chamber volume changes
        # Explicitly defined in Smith 2007, Fig. 1, C17-C22
        outputs['dv_pa_dt'] = (outputs['q_pv'] - outputs['q_pul']) / deriv_scale
        outputs['dv_pu_dt'] = (outputs['q_pul'] - outputs['q_mt']) / deriv_scale
        outputs['dv_lv_dt'] = (outputs['q_mt'] - outputs['q_av']) / deriv_scale
        outputs['dv_ao_dt'] = (outputs['q_av'] - outputs['q_sys']) / deriv_scale
        outputs['dv_vc_dt'] = (outputs['q_sys'] - outputs['q_tc']) / deriv_scale
        outputs['dv_rv_dt'] = (outputs['q_tc'] - outputs['q_pv']) / deriv_scale

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
        if self.volume_ratios:
            states = {
                key: val * self.v_tot if key in self.state_names[:5] else val
                for key, val in states.items()
            }
            states['v_vc'] = self.v_tot - torch.stack([
                states[state] for state in self.state_names[:5]
            ]).sum(axis=0)

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
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            try:
                v_spt = self.solve_v_spt(states['v_lv'], states['v_rv'], e_t)
            except ConvergenceWarning as ex:
                # print(f"v_spt didn't converge with previous initial guess, resetting")
                # self._v_spt_old = self._default_v_spt()
                # v_spt = self.solve_v_spt(states['v_lv'], states['v_rv'], e_t)
                print('Rejecting step')
                raise RejectStepError(ex)
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
        device: str = 'cpu',
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
            device (str, optional): PyTorch device. Defaults to 'cpu'.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """
        r_vc = 1 - r_pa - r_pu - r_lv - r_ao - r_rv

        assert r_vc > 0, "Initial v_vc must not be negative"

        states = {
            'v_pa': torch.tensor(r_pa, device=device),
            'v_pu': torch.tensor(r_pu, device=device),
            'v_lv': torch.tensor(r_lv, device=device),
            'v_ao': torch.tensor(r_ao, device=device),
            'v_vc': torch.tensor(r_vc, device=device),
            'v_rv': torch.tensor(r_rv, device=device),
        }
        if not self.volume_ratios:
            for key in states:
                states[key] *= self.v_tot

        if self.dynamic_hr:
            states['s'] = torch.tensor(0., device=device)

        # Store initial v_spt
        # If p_pl is an input, it needs to be passed, but the value doesn't matter as we only
        # need v_spt which doesn't depend on it
        p_pl = torch.tensor(0.0, device=device)
        init_outputs = self.pressures_volumes(torch.tensor(0., device=device), states, p_pl)
        self._v_spt_old = init_outputs['v_spt'].detach()
        
        return states

    def solve_v_spt(
        self, 
        v_lv: torch.Tensor, 
        v_rv: torch.Tensor, 
        e_t: torch.Tensor,
        rtol: float = 1e-5,
        xtol: float = 1e-6,
    ) -> torch.Tensor:
        """Find value for v_spt using root finding algorithm.

        Args:
            v_lv (torch.Tensor): Left ventricle volume
            v_rv (torch.Tensor): Right ventricle volume
            e_t (torch.Tensor): Cardiac driver function
            xtol (float): Absolute tolerance for v_spt. Defaults to 1e-5.
            rtol (float): Relative tolerance for v_spt. Defaults to 1e-6.

        Returns:
            torch.Tensor: v_spt solution
        """
        # No explicit solution for v_spt, need to use root finder
        if self._v_spt_method == 'xitorch':
            v_spt = rootfinder(
                self.v_spt_residual_analytical, 
                self._v_spt_old, 
                params=(v_lv, v_rv, e_t, self), 
                method=newton_raphson, 
                f_tol=xtol,
                f_rtol=rtol,
                maxiter=100,
            )
        elif self._v_spt_method == 'newton':
            v_spt = newton_raphson(
                self.v_spt_residual_analytical, 
                self._v_spt_old, 
                params=(v_lv, v_rv, e_t, self), 
                f_tol=xtol,
                f_rtol=rtol,
                maxiter=100,
            )
        elif self._v_spt_method == 'jallon':
            # Linearisation from Jallon 2009
            num = e_t * (
                self.lvf.p_es(v_lv) - self.rvf.p_es(v_rv) + 
                self.spt.e_es * self.spt.v_d
            ) + (1 - e_t) * (
                self.lvf.p_ed_linear(v_lv) - self.rvf.p_ed_linear(v_rv) + 
                self.spt.lam * self.spt.p_0 * self.spt.v_0
            )
            den = e_t * (
                self.lvf.e_es + self.rvf.e_es + self.spt.e_es
            ) + (1 - e_t) * (
                self.lvf.lam * self.lvf.p_0 + 
                self.rvf.lam * self.rvf.p_0 + 
                self.spt.lam * self.spt.p_0)
            v_spt = num / den
        else:
            raise NotImplementedError(self._v_spt_method)
            
        self._v_spt_last = v_spt.detach()

        return v_spt

    @staticmethod
    def v_spt_residual_analytical(
        v_spt: torch.Tensor,
        v_lv: torch.Tensor,
        v_rv: torch.Tensor,
        e_t: torch.Tensor,
        cvs: "SmithCardioVascularSystem",
        return_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual function for v_spt, with analytic derivative.
        
        See Smith 2004, Eq. 20

        Implemented as static method as xitorch requires a pure function

        Args:
            v_spt (torch.Tensor): Current value of v_spt from 
                root finding algorithm
            v_lv (torch.Tensor): Left ventricle volume
            v_rv (torch.Tensor): Right ventricle volume
            e_t (torch.Tensor): Cardiac driver function
            cvs (SmithCardioVascularSystem): self
            return_grad (bool, Optional): return grad. Defaults to False

        Returns:
            Tuple containing:
            - res (torch.Tensor): Residual
            - grad (torch.Tensor): Gradient of residual wrt v_spt
                (only if return_grad)
        """
        # Free wall volumes v_(lvf/rvf/spt) are not physical volumes, but 
        # defined to capture deflection of cardiac free walls relative to 
        # ventricle volumes
        # Eq. 9, 10
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        # Eq. 15, 16, 17
        res = cvs.spt.p(v_spt, e_t) - cvs.lvf.p(v_lvf, e_t) + cvs.rvf.p(v_rvf, e_t)

        if not return_grad:
            return res

        # Analytical gradient of residual wrt v_spt
        grad = cvs.lvf.dp_dv(v_lvf, e_t) + cvs.rvf.dp_dv(v_rvf, e_t) + cvs.spt.dp_dv(v_spt, e_t)

        return res, grad


class InertialSmithCVS(SmithCardioVascularSystem):
    """Smith CVS model with inertia and Heaviside valve law.
    
    (Smith, 2004), (Hann, 2004) and (Paeme, 2011)
    """
    
    def __init__(self):
        """Initialise.

        Default parameter values from Paeme, 2011.
        """
        super().__init__(state_names=[
            'v_pa', 'v_pu', 'v_lv', 'v_ao', 'v_vc', 'v_rv', 
            'q_mt', 'q_av', 'q_tc', 'q_pv',
        ])
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
        outputs['dq_mt_dt'] = self.mt.flow_rate_deriv(
            outputs['p_pu'], outputs['p_lv'], states['q_mt'])
        outputs['dq_av_dt'] = self.av.flow_rate_deriv(
            outputs['p_lv'], outputs['p_ao'], states['q_av'])
        outputs['dq_tc_dt'] = self.tc.flow_rate_deriv(
            outputs['p_vc'], outputs['p_rv'], states['q_tc'])
        outputs['dq_pv_dt'] = self.pv.flow_rate_deriv(
            outputs['p_rv'], outputs['p_pa'], states['q_pv'])

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        p_pl: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Model implementation.

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
        device: str = 'cpu',
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
            device (str, optional): PyTorch device. Defaults to 'cpu'.

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
        init_outputs = self.pressures_volumes(torch.tensor(0., device=device), states, p_pl)
        self.flow_rates(init_outputs, static=True)

        states = {
            key: val for key, val in init_outputs.items() 
            if key in self.state_names
        }

        return states


class JallonHeartLungs(ODEBase):
    """Jallon model of heart and lungs (combined cardiovascular and 
    respiratory model). Uses simple valve law with no inertia.
    
    (Jallon, 2009)
    """

    def __init__(
        self, 
        f_hr: Optional[Callable] = None, 
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
        state_names = (
            SmithCardioVascularSystem.state_names + 
            PassiveRespiratorySystem.state_names + 
            RespiratoryPatternGenerator.state_names
        )
        super().__init__(state_names=state_names)
        self.resp_pattern = RespiratoryPatternGenerator()
        self.resp = PassiveRespiratorySystem()
        self.cvs = SmithCardioVascularSystem(p_pl_is_input=True, f_hr=f_hr, v_spt_method='jallon')

        # Jallon CVS model modifications
        with torch.no_grad():
            # Table 2 of Jallon 2009
            self.cvs.spt.e_es.copy_(torch.tensor(convert(3750, 'mmHg/l')))  # Wrong units in paper
            self.cvs.spt.lam.copy_(torch.tensor(convert(35, '1/l')))
            self.cvs.vc.e_es.copy_(torch.tensor(convert(2, 'mmHg/l')))
            # # HR = 54bpm
            self.cvs.e.hr.copy_(torch.tensor(54.0))
            # Eq 2.9, 2.10
            self.cvs.p_pl_affects_pu_and_pa.copy_(torch.tensor(True))

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        """Called by torchdiffeq at the end of a successful step. 
        
        Used to build outputs (irregularly-spaced grid) in self.trajectory.

        Stores last value of v_spt for use as initial guess in next step.

        Args:
            t (torch.Tensor): Time (s)
            x (torch.Tensor): ODE state tensor
            dt (torch.Tensor): Time step (s)
        """
        super().callback_accept_step(t, x, dt)
        self.cvs._v_spt_old = self.trajectory[-1][3]['v_spt']

    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Model implementation. 
        
        Just consists of calling the lower-level models and passing parameters
        between. See Figure 5 from Jallon (2009) for schematic.

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

    def init_states(self, device: str = 'cpu') -> dict[str, torch.Tensor]:
        """Return initial values of ODE states.

        Args:
            device (str, optional): PyTorch device. Defaults to 'cpu'.

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """
        cvs_states = self.cvs.init_states(
            r_pa=0.034,
            r_pu=0.145,
            r_lv=0.025,
            r_ao=0.164,
            r_rv=0.023,
            device=device
        )
        resp_states = self.resp.init_states(device=device)
        resp_pattern_states = self.resp_pattern.init_states(device=device)

        init_states = cvs_states | resp_states | resp_pattern_states

        return init_states


def add_bp_metrics(cls: Type[ODEBase]) -> Type[ODEBase]:
    """Add blood pressure metrics model to a cardiovascular ODE model.

    Args:
        cls (Type[ODEBase]): Class definition of cardiovascular model

    Returns:
        Type[ODEBase]: New model with extra states for blood pressure metrics.
    """
    class BloodPressureMetrics(cls):

        def __init__(self, *args, **kwargs):
            """Initialise cls and add new states and parameters."""
            assert kwargs.get('f_hr') is not None, "Must use dynamic HR for BP metrics"
            super().__init__(*args, **kwargs)

            self.state_names.extend([
                'p_aod', 'p_aos', 'p_aom', 'p_vcm', 'p_pad', 'p_pas', 'p_pam',
            ])

            self.moving_avg_weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.moving_avg_weight_s = nn.Parameter(torch.tensor(0.01), requires_grad=False)
            self.moving_avg_weight_d = nn.Parameter(torch.tensor(0.05), requires_grad=False)
            self.moving_avg_power_s = nn.Parameter(torch.tensor(6), requires_grad=False)
            # self.moving_avg_power_d = nn.Parameter(torch.tensor(2), requires_grad=False)
            self.s_d = nn.Parameter(torch.tensor(0.3), requires_grad=False)
            self.b_d = nn.Parameter(torch.tensor(200), requires_grad=False)

        def model(
            self, 
            t: torch.Tensor, 
            states: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            """Call model, and then update blood pressure metric states.

            Args:
                t (torch.Tensor): Time (s)
                states (torch.Tensor): Model states

            Returns:
                dict[str, torch.Tensor]: Model outputs
            """
            outputs = super().model(t, states)

            # Moving averages
            outputs['dp_aom_dt'] = (outputs['p_ao'] - states['p_aom']) / self.moving_avg_weight
            outputs['dp_pam_dt'] = (outputs['p_pa'] - states['p_pam']) / self.moving_avg_weight
            outputs['dp_vcm_dt'] = (outputs['p_vc'] - states['p_vcm']) / self.moving_avg_weight

            # Systolic moving average
            dp_aos_dt = (
                (outputs['p_ao'] - states['p_aos']) * outputs['e_t']**self.moving_avg_power_s /
                self.moving_avg_weight_s
            )
            dp_pas_dt = (
                (outputs['p_pa'] - states['p_pas']) * outputs['e_t']**self.moving_avg_power_s /
                self.moving_avg_weight_s
            )
            # Only update systolic moving average when volume is increasing
            outputs['dp_aos_dt'] = torch.where(
                outputs['dv_ao_dt'] > 0,
                dp_aos_dt,
                0.0
            )
            outputs['dp_pas_dt'] = torch.where(
                outputs['dv_pa_dt'] > 0,
                dp_pas_dt,
                0.0
            )

            # Find end of diastole (just as e(t) starts to increase)
            e_diastole = torch.exp(-self.b_d * (outputs['s_wrapped'] - self.s_d)**2)

            # Diastolic moving average
            dp_aod_dt = (outputs['p_ao'] - states['p_aod']) * e_diastole / self.moving_avg_weight_d
            dp_pad_dt = (outputs['p_pa'] - states['p_pad']) * e_diastole / self.moving_avg_weight_d

            # Only update diastolic moving average when volume is decreasing
            outputs['dp_aod_dt'] = torch.where(
                outputs['dv_ao_dt'] < 0,
                dp_aod_dt,
                0.0
            )
            outputs['dp_pad_dt'] = torch.where(
                outputs['dv_pa_dt'] < 0,
                dp_pad_dt,
                0.0
            )

            return outputs

        def init_states(self, device='cpu') -> dict[str, torch.Tensor]:
            """Return initial values of ODE states.

            Args:
                device (str, optional): PyTorch device. Defaults to 'cpu'.

            Returns:
                dict[str, torch.Tensor]: Initial ODE states
            """
            init_states = super().init_states(device=device)
            init_outputs = super().model(torch.tensor(0., device=device), init_states)
            init_states['p_aom'] = init_outputs['p_ao']
            init_states['p_aos'] = init_outputs['p_ao']
            init_states['p_aod'] = init_outputs['p_ao']
            init_states['p_pam'] = init_outputs['p_pa']
            init_states['p_pas'] = init_outputs['p_pa']
            init_states['p_pad'] = init_outputs['p_pa']
            init_states['p_vcm'] = init_outputs['p_vc']
            
            return init_states

    return BloodPressureMetrics
