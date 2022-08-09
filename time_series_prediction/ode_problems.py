import numpy as np
import torch
from scipy.integrate import solve_ivp
from numpy.typing import ArrayLike

from biophysical_models.models import SmithCardioVascularSystem, InertialSmithCVS, JallonHeartLungs

def vdp(
    t: float, 
    state: ArrayLike, 
    mu: float = 2.0, 
    A: float = 1.0, 
    B: float = 0.5, 
    omega_1: float = 0.2, 
    omega_2: float = 0.3,
) -> np.ndarray:
    """Forced Van der Pol oscillator

    d2x/dt2 - mu(1-x^2) dx/dt  + x - A sin(omega_1*t) + B sin(omega_2*t)

    Args:
        t (float): Time
        state (ArrayLike): ODE states
        mu (float, optional): VdP parameter. Defaults to 2.0.
        A (float, optional): Forcing parameter. Defaults to 1.0.
        B (float, optional): Forcing parameter. Defaults to 0.5.
        omega_1 (float, optional): Forcing parameter. Defaults to 0.2.
        omega_2 (float, optional): Forcing parameter. Defaults to 0.3.

    Returns:
        np.ndarray: ODE derivatives
    """
    dx_dt, x = state

    d2x_dt2 = mu*(1-x**2)*dx_dt - x + A*np.sin(omega_1 * t) - B*np.sin(omega_2 * t)

    return np.array([d2x_dt2, dx_dt])


def lorenz(
    t: float, 
    state: ArrayLike, 
    rho: float = 28.0, 
    sigma: float = 10.0, 
    beta: float = 8/3,
) -> np.ndarray:
    """Lorenz attractor

    dx/dt = sigma*(y-x)
    dy/dt = x*(rho-z) - y
    dz/dt = x*y - beta*z

    Args:
        t (float): Time
        state (ArrayLike): ODE states
        rho (float, optional): Lorenz parameter. Defaults to 28.0.
        sigma (float, optional): Lorenz parameter. Defaults to 10.0.
        beta (float, optional): Lorenz parameter. Defaults to 8/3.

    Returns:
        np.ndarray: ODE derivatives
    """
    x, y, z = state

    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return np.array([dx_dt, dy_dt, dz_dt])


def rossler(
    t: float, 
    state: ArrayLike, 
    a: float = 0.2, 
    b: float = 0.2, 
    c: float = 5.7,
) -> np.ndarray:
    """Rössler attractor

    dx/dt = -y-z
    dy/dt = x + a*y
    dz/dt = b + z*(x-c)

    Args:
        t (float): Time
        state (ArrayLike): ODE states
        a (float, optional): Rössler parameter. Defaults to 0.2.
        b (float, optional): Rössler parameter. Defaults to 0.2.
        c (float, optional): Rössler parameter. Defaults to 5.7.

    Returns:
        np.ndarray: ODE derivatives
    """
    x, y, z = state

    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)

    return np.array([dx_dt, dy_dt, dz_dt])


def generate_data(
    source: str,
    n: int = 2000,
    resolution: int = 10,
    noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate data from an ODE, with an empty input vector.

    Supported sources:
    - vdp: Van der Pol oscillator
    - lorenz: Lorenz attractor
    - rossler: Rössler attractor
    - sin: Sin wave
    - cvs: Cardiovascular system
    - inertial_cvs: Cardiovascular system with inertial valve law
    - jallon: Jallon heart/lung model

    Args:
        source (str): One of the above sources
        n (int, optional): Number of data points. Defaults to 2000.
        resolution (int, optional): Number of points per second. Defaults to 
            10.
        noise (float, optional): Output noise covariance. Defaults to 0.0.

    Returns:
        Tuple containing
        - torch.Tensor: ODE outputs [n * n_states]
        - torch.Tensor: ODE inputs [n * 0]
        - torch.Tensor: Time vector [n]
    """
    t_range = (0, (n-1)/resolution)

    if source == 'vdp':
        state_0 = np.array([1, 0.3])
        sol = solve_ivp(
            vdp, t_range, state_0, method='RK45', dense_output=True,
            # args=(2.0, 0.8, 0.3, 0.2, 0.5),
            args=(2.0, 0, 0, 0.2, 0.5),
            )
        
        t_out = np.linspace(t_range[0], t_range[1], n)
        x_out = sol.sol(t_out)
        y_full = torch.tensor(
            x_out, 
            # device=settings.device,
            dtype=torch.float32
        ).T

    elif source == 'lorenz':
        state_0 = np.array([0., 2., 20.])
        sol = solve_ivp(
            lorenz, t_range, state_0, method='RK45', dense_output=True,
            # args=(2.0, 0.8, 0.3, 0.2, 0.5),
            args=(28.0, 10.0, 8/3),
            )
        
        t_out = np.linspace(t_range[0], t_range[1], n)
        x_out = sol.sol(t_out)
        y_full = torch.tensor(
            x_out, 
            # device=settings.device,
            dtype=torch.float32
        ).T

    elif source == 'rossler':
        state_0 = np.array([1e-4, 1e-4, 1e-4])
        sol = solve_ivp(
            rossler, t_range, state_0, method='RK45', dense_output=True,
            # args=(2.0, 0.8, 0.3, 0.2, 0.5),
            args=(0.2, 0.2, 5.7),
            )
        
        t_out = np.linspace(t_range[0], t_range[1], n)
        x_out = sol.sol(t_out)
        y_full = torch.tensor(
            x_out, 
            # device=settings.device,
            dtype=torch.float32
        ).T

    elif source == 'sin':
        
        y_full = torch.sin(torch.linspace(
            t_range[0], 
            t_range[1], 
            n, 
            # device=settings.device,
        ))
        y_full = y_full[:, None]

    else:
        models = {
            'cvs': SmithCardioVascularSystem,
            'inertial_cvs': InertialSmithCVS,
            'jallon': JallonHeartLungs,
        }
        model = models[source]()
        t_out, y_full = model.simulate(int((n-1)/resolution), resolution)

    y_full = y_full + torch.randn_like(y_full) * noise
    u_full = torch.zeros(
        (n, 0), 
        # device=settings.device,
    )
    t_out = torch.as_tensor(t_out, dtype=torch.float32)


    return y_full, u_full, t_out