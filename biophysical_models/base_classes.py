from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from biophysical_models.unit_conversions import convert


class ODEBase(ABC, nn.Module):
    state_names: ClassVar[list[str]]

    """Base class for ODE problems.
    
    state_names must be set in the child class definition: a list of strings
    to denote state names
    """

    def __init__(self, *args, verbose=True, **kwargs):
        """Initialises nn.Module

        Args:
            verbose (bool, optional): Print integration progress after every
                accepted step. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.trajectory = []
        self.verbose = verbose

    def ode_state_dict(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a tensor of ODE states to a dict

        Args:
            states (torch.Tensor): ODE state tensor

        Returns:
            dict[str, torch.Tensor]: ODE states formatted as dict
        """
        return {name: states[i] for i, name in enumerate(self.state_names)}

    def ode_state_tensor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert a dict of ODE states to a tensor

        Args:
            states (torch.Tensor): ODE states formatted as dict

        Returns:
            dict[str, torch.Tensor]: ODE state tensor
        """
        state = torch.stack([
            outputs[state] for state in self.state_names
        ])

        return state

    def ode_deriv_tensor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert a dict of ODE outputs to a derivative tensor

        Args:
            outputs (torch.Tensor): ODE outputs formatted as dict

        Returns:
            dict[str, torch.Tensor]: ODE state derivative tensor
        """

        deriv = torch.stack([
            outputs[f'd{state}_dt'] for state in self.state_names
        ])

        return deriv

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ODE function (forward pass)

        Args:
            t (torch.Tensor): Time (s)
            x (torch.Tensor): ODE state tensor

        Returns:
            torch.Tensor: ODE derivative tensor
        """

        states = self.ode_state_dict(x)
        outputs = self.model(t, states)
        deriv = self.ode_deriv_tensor(outputs)

        return deriv

    @abstractmethod
    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """To be implemented in subclass. Implementation of ODE model. 
        Should output a dict of tensors, at least containing a "d{state}_dt"
        key for every state in 

        Args:
            t (torch.Tensor): Time (s)
            states (dict[str, torch.Tensor]): ODE states formatted as dict

        Returns:
            dict[str, torch.Tensor]: ODE outputs formatted as dict
        """
        pass

    @abstractmethod
    def init_states(self) -> dict[str, torch.Tensor]:
        """To be implemented in subclass. Initial states of ODE.
        """
        pass

    def simulate(
        self,
        t_final: float, 
        resolution: int,
        adjoint: bool = False,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_step: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate model. Returns regularly spaced time and state tensors. 
        Outputs at integration steps available in self.trajectory.

        Args:
            t_final (float): Final time (s)
            resolution (int): State output resolution (Hz)
            adjoint (bool): Use adjoint integrator
            rtol (float): Relative tolerance. Defaults to 1e-6.
            atol (float): Absolute tolerance. Defaults to 1e-6.
            max_step (float): Maximum ODE step. Defaults to 1e-2.

        Returns:
            Tuple containing:
            - t (torch.Tensor): Time tensor
            - sol (torch.Tensor): State tensor
        """

        self.trajectory = []
        states = self.init_states()
        x_0 = self.ode_state_tensor(states)

        t = torch.linspace(0, t_final, int(t_final*resolution + 1))
        if adjoint:
            solver = odeint_adjoint
        else:
            solver = odeint
        sol = solver(
            self, 
            x_0, 
            t, 
            method='dopri5', 
            rtol=rtol, 
            atol=atol, 
            options={'max_step': max_step},
        )

        return t, sol

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        """Called by torchdiffeq at the end of a successful step. Used to 
        build outputs (irregularly-spaced grid) in self.trajectory

        NOTE: Only works with torchdiffeq from master branch

        Args:
            t (torch.Tensor): Time (s)
            x (torch.Tensor): ODE state tensor
            dt (torch.Tensor): Time step (s)
        """
        if self.verbose:
            print(f't={t:.4f}s, dt={dt:.4f}')
        states = self.ode_state_dict(x)
        outputs = self.model(t, states)
        deriv = self.ode_deriv_tensor(outputs)
        self.trajectory.append((t, x, deriv, outputs))


class PressureVolume(nn.Module):
    """Model of a pressure-volume relationship, with end systolic and end 
    diastolic pressure-volume relationships (ESPVR/EDPVR).
    
    (Smith, 2004)"""

    def __init__(
        self, 
        e_es: Optional[float], 
        v_d: Optional[float], 
        v_0: Optional[float], 
        lam: Optional[float], 
        p_0: Optional[float],
    ):
        """Initialise

        Args:
            e_es (Optional[float]): End-systolic elastance (if None, no
                ESPVR available. If not None, must also define v_d)
            v_d (Optional[float]): Volume at zero pressure
            v_0 (Optional[float]): EDPVR volume offest (if None, no EDPVR 
                available. If not None, must also define lam and p_0)
            lam (Optional[float]): EDPVR exponential gain
            p_0 (Optional[float]): EDPVR gain
        """
        super().__init__()
        if e_es is not None:
            self.e_es = nn.Parameter(torch.as_tensor(e_es), requires_grad=False)
            self.v_d = nn.Parameter(torch.as_tensor(v_d), requires_grad=False)
        if v_0 is not None:
            self.v_0 = nn.Parameter(torch.as_tensor(v_0), requires_grad=False)
            self.lam = nn.Parameter(torch.as_tensor(lam), requires_grad=False)
            self.p_0 = nn.Parameter(torch.as_tensor(p_0), requires_grad=False)

    def p(self, v: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """Chamber pressure due to volume and time-varying cardiac driving
        function

        Args:
            v (torch.Tensor): Volume
            e_t (torch.Tensor): Cardiac driving function, normalised between 0
                (EDPVR) and 1 (ESPVR).

        Returns:
            torch.Tensor: Chamber pressure
        """
        return e_t * self.p_es(v) + (1 - e_t) * self.p_ed(v)

    def dp_dv(self, v: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """Derivative of chamber pressure wrt volume

        Args:
            v (torch.Tensor): Volume
            e_t (torch.Tensor): Cardiac driving function, normalised between 0
                (EDPVR) and 1 (ESPVR).

        Returns:
            torch.Tensor: dp/dv
        """
        return e_t * self.dp_es_dv(v) + (1 - e_t) * self.dp_ed_dv(v)

    def p_es(self, v: torch.Tensor) -> torch.Tensor:
        """Calculate end systolic pressure

        Args:
            v (torch.Tensor): Volume

        Returns:
            torch.Tensor: End systolic pressure
        """
        return self.e_es * (v - self.v_d)

    def dp_es_dv(self, v: torch.Tensor) -> torch.Tensor:
        """Derivative of end-systolic pressure wrt volume

        Args:
            v (torch.Tensor): Volume

        Returns:
            torch.Tensor: d(p_es)/dv
        """
        return self.e_es

    def p_ed(self, v: torch.Tensor) -> torch.Tensor:
        """Calculate end diastolic pressure

        Args:
            v (torch.Tensor): Volume

        Returns:
            torch.Tensor: End diastolic pressure
        """
        return self.p_0 * (torch.exp(self.lam * (v - self.v_0)) - 1)

    def dp_ed_dv(self, v: torch.Tensor) -> torch.Tensor:
        """Derivative of end-diastolic pressure wrt volume

        Args:
            v (torch.Tensor): Volume

        Returns:
            torch.Tensor: d(p_ed)/dv
        """
        return self.lam * self.p_0 * torch.exp(self.lam * (v - self.v_0))
            

class BloodVessel(nn.Module):
    """Blood vessel with resistance according to Poiseuille's equation
    
    (Smith, 2004)"""

    def __init__(self, r: float):
        """Initialise 

        Args:
            r (float): Resistance parameter
        """
        super().__init__()
        self.r = nn.Parameter(torch.as_tensor(r), requires_grad=False)

    def flow_rate(self, p_upstream: torch.Tensor, p_downstream: torch.Tensor) -> torch.Tensor:
        """Volume flow rate due to pressure differential

        Args:
            p_upstream (torch.Tensor): Pressure upstream of vessel
            p_downstream (torch.Tensor): Pressure downstream of vessel

        Returns:
            torch.Tensor: Volume flow rate
        """
        q_flow = (p_upstream - p_downstream) / self.r
        return q_flow


# TODO: Valve methods are a bit confusing - maybe avoid having to calculate the
# flow rate through a closed valve and pass it into the valve open function?

class Valve(BloodVessel):
    """Simple non-inertial valve
    
    (Smith, 2004)"""

    def open(
        self, 
        p_upstream: torch.Tensor, 
        p_downstream: torch.Tensor, 
        q_flow: torch.Tensor
    ) -> torch.Tensor:
        """Return status of non-inertial valve: 0 for closed, 1 for open.
        Simple valve law: valve opens on positive flow, and closes to prevent
        flow reversing

        Args:
            p_upstream (torch.Tensor): Pressure upstream of valve
            p_downstream (torch.Tensor): Pressure downstream of valve
            q_flow (torch.Tensor): Volume flow rate through valve (if open)

        Returns:
            torch.Tensor: Valve status
        """
        return torch.gt(q_flow, 0.0)

    def flow_rate(self, p_upstream: torch.Tensor, p_downstream: torch.Tensor) -> torch.Tensor:
        """Volume flow rate through valve, given pressure before and after the 
        valve

        Args:
            p_upstream (torch.Tensor): Pressure upstream of valve
            p_downstream (torch.Tensor): Pressure downstream of valve

        Returns:
            torch.Tensor: Volume flow rate
        """
        q_flow = super().flow_rate(p_upstream, p_downstream)
        return torch.where(self.open(p_upstream, p_downstream, q_flow), q_flow, 0.0)


class InertialValve(Valve):
    """Inertial valve
    
    (Smith, 2004)"""

    def __init__(self, r: float, l: float):
        """Initialise

        Args:
            r (float): Resistance of open valve
            l (float): Inertance of flow through valve
        """
        super().__init__(r)
        self.l = nn.Parameter(torch.as_tensor(l), requires_grad=False)

    def open(
        self, 
        p_upstream: torch.Tensor, 
        p_downstream: torch.Tensor, 
        q_flow: torch.Tensor
    ) -> torch.Tensor:
        """Return status of inertial valve: 0 for closed, 1 for open. Inertial
        valve law: open on positive pressure gradient, close on reversed flow.

        Args:
            p_upstream (torch.Tensor): Pressure upstream of valve
            p_downstream (torch.Tensor): Pressure downstream of valve
            q_flow (torch.Tensor): Volume flow rate through valve (if open)

        Returns:
            torch.Tensor: Valve status
        """
        return torch.logical_or(
            p_upstream > p_downstream,
            q_flow > 0.0
        )

    def flow_rate_deriv(
        self, 
        p_upstream: torch.Tensor,
        p_downstream: torch.Tensor,
        q_flow: torch.Tensor,
    ) -> torch.Tensor:
        """Volume flow rate derivative function, taking into account 
        valve status, resistance, pressure gradient and flow inertia.

        Args:
            p_upstream (torch.Tensor): Pressure upstream of valve
            p_downstream (torch.Tensor): Pressure downstream of valve
            q_flow (torch.Tensor): Volume flow rate through valve (if open)

        Returns:
            torch.Tensor: Volume flow rate derivative
        """

        d_q_dt = (p_upstream - p_downstream - q_flow * self.r) / self.l
        return torch.where(self.open(p_upstream, p_downstream, q_flow), d_q_dt, 0.0)


class CardiacDriver(nn.Module):
    """Cardiac driver function
    
    (Smith, 2004)"""

    def __init__(
        self, 
        a: float = 1.0, 
        b: float = 80.0, 
        hr: float = 80.0,
    ):
        """Initialise

        Args:
            a (float, optional): Scale parameter. Defaults to 1.0
            b (float, optional): Slope parameter. Defaults to 80.0.
            hr (float, optional): Heart rate (bpm). Defaults to 80.0.
        """
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b), requires_grad=False)
        self.hr = nn.Parameter(torch.as_tensor(hr), requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate cardiac driver function at a given time point.

        Args:
            t (torch.Tensor): Time (s)

        Returns:
            torch.Tensor: e(t)
        """

        t_wrapped = torch.remainder(t, 60/self.hr)
        e = self.a * torch.exp(-self.b * (t_wrapped - 30/self.hr)**2)

        return e


class RespiratoryPatternGenerator(ODEBase):
    """Lienard system of ODEs
    
    (Jallon, 2009)"""

    state_names: ClassVar[list[str]] = ['x', 'y', 'p_mus']

    def __init__(
        self, 
        hb: float = 1.0,  # Jallon: units not given
        a: float = -0.8, 
        b: float = -3, 
        alpha: float = 1,
        lam: float = convert(1.0, 'mmHg'),
        mu: float = convert(1.0, 'mmHg'),  # 1.08504
        beta: float = 0.1,
    ):
        """Initialise model.

        dv_alv_dt should be set by the top level model, from the 
        cardiovascular system model.

        Args:
            hb (float, optional): Hering-Breur reflex parameter. Defaults to 
                1 per litre.
            a (float, optional): Lienard parameter. Defaults to -0.8.
            b (float, optional): Lienard parameter. Defaults to -3.
            alpha (float, optional): Parameter to cover higher range of 
                respiratory frequencies. Defaults to 1.
            lam (float, optional): Positive constant to allow pleural pressure
                to be set at physiological values. Defaults to 1.5 mmHg/s. 
                Note: units not given in Jallon, assumed mmHg/s.
            mu (float, optional): Offset to respiratory muscle pressure 
                derivative with time. Defaults to 1 mmHg/s. Note that with this
                implementation, a value of 1.08504 gives steady state
                behaviour.
            beta (float, optional): Integral control on respiratory muscle 
                pressure to keep it at constant mean (prevents drift). 
                Modification original in this work. Defaults to 0.1/s
        """
        super().__init__()
        self.hb = nn.Parameter(torch.as_tensor(hb), requires_grad=False)
        self.a = nn.Parameter(torch.as_tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b), requires_grad=False)
        self.alpha = nn.Parameter(torch.as_tensor(alpha), requires_grad=False)
        self.lam = nn.Parameter(torch.as_tensor(lam), requires_grad=False)
        self.mu = nn.Parameter(torch.as_tensor(mu), requires_grad=False)
        self.beta = nn.Parameter(torch.as_tensor(beta), requires_grad=False)

    def model(
        self, 
        t: torch.Tensor, 
        states: dict[str, torch.Tensor],
        dv_alv_dt: torch.Tensor = torch.tensor(0.0),
    ) -> dict[str, torch.Tensor]:
        """Derivatives of central respiratory pattern generator

        Args:
            t (torch.Tensor): Time (s)
            states (torch.Tensor):  ODE states formatted as dict
            dv_alv_dt (torch.Tensor, optional): Input to ODE. Defaults to 0.0.

        Returns:
            dict[str, torch.Tensor]:  ODE derivatives formatted as dict
        """

        dx_dt = self.alpha * (self.lienard(states['x'], states['y']) - self.hb * dv_alv_dt)
        dy_dt = self.alpha * states['x']
        dp_mus_dt = self.lam * states['y'] + self.mu - self.beta * states['p_mus']
        
        return {'dx_dt': dx_dt, 'dy_dt': dy_dt, 'dp_mus_dt': dp_mus_dt}
    
    def lienard(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Lienard function

        Args:
            x (torch.Tensor): x value
            y (torch.Tensor): y value

        Returns:
            torch.Tensor: f(x, y)
        """
        f = (self.a * y**2 + self.b * y) * (x + y)
        return f

    def init_states(self) -> dict[str, torch.Tensor]:
        """Initial states of Lienard system

        Returns:
            dict[str, torch.Tensor]: Initial ODE states
        """

        return {
            'x': torch.tensor(-0.6),
            'y': torch.tensor(0.0),
            'p_mus': torch.tensor(0.0),
        }
