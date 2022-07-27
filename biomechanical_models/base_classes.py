from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import torch
from torch import nn

from biomechanical_models.unit_conversions import convert


class ODEBase(ABC, nn.Module):
    state_names: ClassVar[list[str]]
    input_names: ClassVar[list[str]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = []

    def ode_state_dict(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: states[i] for i, name in enumerate(self.state_names)}

    def ode_state_tensor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = torch.tensor([
            outputs[state] for state in self.state_names
        ])

        return state

    def ode_deriv_tensor(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        deriv = torch.tensor([
            outputs[f'd{state}_dt'] for state in self.state_names
        ])

        return deriv

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        # print(f'forward {t}, states ({x})')

        states = self.ode_state_dict(x)
        outputs = self.model(t, states)
        deriv = self.ode_deriv_tensor(outputs)

        return deriv

    @abstractmethod
    def model(self, t: torch.Tensor, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass

    def simulate(self):
        self.trajectory = []

    def callback_accept_step(self, t: torch.Tensor, x: torch.Tensor, dt: torch.Tensor):
        print(f't={t:.4f}s, dt={dt:.4f}')
        states = self.ode_state_dict(x)
        outputs = self.model(t, states)
        deriv = self.ode_deriv_tensor(outputs)
        self.trajectory.append((t, x, deriv, outputs))


class PressureVolume(nn.Module):
    def __init__(
        self, 
        e_es: Optional[float], 
        v_d: Optional[float], 
        v_0: Optional[float], 
        lam: Optional[float], 
        p_0: Optional[float],
    ):
        super().__init__()
        if e_es is not None:
            self.e_es = nn.Parameter(torch.as_tensor(e_es), requires_grad=False)
            self.v_d = nn.Parameter(torch.as_tensor(v_d), requires_grad=False)
        if v_0 is not None:
            self.v_0 = nn.Parameter(torch.as_tensor(v_0), requires_grad=False)
            self.lam = nn.Parameter(torch.as_tensor(lam), requires_grad=False)
            self.p_0 = nn.Parameter(torch.as_tensor(p_0), requires_grad=False)

    def p_es(self, v: torch.Tensor) -> torch.Tensor:
        """End systolic pressure"""
        return self.e_es * (v - self.v_d)

    def dp_es_dv(self, v: torch.Tensor) -> torch.Tensor:
        """Derivative of p_es wrt v"""
        return self.e_es

    def p_ed(self, v: torch.Tensor) -> torch.Tensor:
        """End diastolic pressure"""
        return self.p_0 * (torch.exp(self.lam * (v - self.v_0)) - 1)

    def dp_ed_dv(self, v: torch.Tensor) -> torch.Tensor:
        """Derivative of p_ed wrt v"""
        return self.lam * self.p_0 * torch.exp(self.lam * (v - self.v_0))
            

class BloodVessel(nn.Module):
    """Blood vessel with resistance according to Poiseuille's equation"""

    def __init__(self, r: float):
        super().__init__()
        self.r = nn.Parameter(torch.as_tensor(r), requires_grad=False)

    def flow_rate(self, p_upstream: torch.Tensor, p_downstream: torch.Tensor) -> torch.Tensor:
        q_flow = (p_upstream - p_downstream) / self.r
        return q_flow


class Valve(BloodVessel):
    """Non-inertial valve"""

    def open(
        self, 
        p_upstream: torch.Tensor, 
        p_downstream: torch.Tensor, 
        q_flow: torch.Tensor
    ) -> torch.Tensor:
        return torch.gt(q_flow, 0.0)

    def flow_rate(self, p_upstream: torch.Tensor, p_downstream: torch.Tensor) -> torch.Tensor:
        q_flow = super().flow_rate(p_upstream, p_downstream)
        return torch.where(self.open(p_upstream, p_downstream, q_flow), q_flow, 0.0)


class InertialValve(Valve):
    """Inertial valve"""

    def __init__(self, r: float, l: float):
        super().__init__(r)
        self.l = nn.Parameter(torch.as_tensor(l), requires_grad=False)

    def open(
        self, 
        p_upstream: torch.Tensor, 
        p_downstream: torch.Tensor, 
        q_flow: torch.Tensor
    ) -> torch.Tensor:
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
        d_q_dt = (p_upstream - p_downstream - q_flow * self.r) / self.l
        return torch.where(self.open(p_upstream, p_downstream, q_flow), d_q_dt, 0.0)


class CardiacDriver(nn.Module):

    def __init__(
        self, 
        a: float = 1., 
        b: float = 80.0, 
        c: float = 0.375, 
        hr: float = 80.0,
    ):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b), requires_grad=False)
        self.c = nn.Parameter(torch.as_tensor(c), requires_grad=False)
        self.hr = nn.Parameter(torch.as_tensor(hr), requires_grad=False)

    def forward(self, t: torch.Tensor, n_beats: int = 500) -> torch.Tensor:

        t_beats = torch.arange(n_beats)[:, None]*60/self.hr
        e_beats = self.a * torch.exp(-self.b * (t - self.c - t_beats)**2)
        e = e_beats.sum(0).reshape_as(t)

        return e


class RespiratoryPatternGenerator(ODEBase):
    """Lienard system"""

    state_names: ClassVar[list[str]] = ['x', 'y']
    input_names: ClassVar[list[str]] = ['dv_alv_dt']

    def __init__(
        self, 
        hb: float = convert(1, '1/l'),  # Jallon: units not given
        a: float = -0.8, 
        b: float = -3, 
        alpha: float = 1,
    ):
        super().__init__()
        self.hb = nn.Parameter(torch.as_tensor(hb), requires_grad=False)
        self.a = nn.Parameter(torch.as_tensor(a), requires_grad=False)
        self.b = nn.Parameter(torch.as_tensor(b), requires_grad=False)
        self.alpha = nn.Parameter(torch.as_tensor(alpha), requires_grad=False)

        # Inputs
        self.dv_alv_dt = nn.Parameter(torch.as_tensor(0.0), requires_grad=False)

    def model(self, t: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
        dx_dt = self.alpha * (self.lienard(states['x'], states['y']) - self.hb * self.dv_alv_dt)
        dy_dt = self.alpha * states['x']
        
        return {'dx_dt': dx_dt, 'dy_dt': dy_dt}
    
    def lienard(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f = (self.a * y**2 + self.b * y) * (x + y)
        return f
