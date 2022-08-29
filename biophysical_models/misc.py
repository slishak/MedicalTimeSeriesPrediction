import warnings

from typing import Callable, Iterable
import torch
from xitorch._utils.exceptions import ConvergenceWarning


def newton_raphson(
    fcn: Callable, 
    x0: torch.Tensor, 
    params: Iterable, 
    f_tol: float = 1e-5, 
    f_rtol: float = 1e-6, 
    maxiter: int = torch.inf,
    **unused,
) -> torch.Tensor:
    """Newton-Raphon method with infinity-norm convergence criteria, for 
    functions returning residual and Jacobian.

    y, dy_dx = f(x, *params)

    Args:
        fcn (Callable): Function returning residual and Jacobian
        x0 (torch.Tensor): Initial guess for x
        params (Iterable): Arguments for fcn
        f_tol (float, optional): Absolute tolerance. Defaults to 1e-5.
        f_rtol (float, optional): Relative tolerance. Defaults to 1e-6.
        maxiter (int, optional): Max iterations. Defaults to torch.inf.

    Returns:
        torch.Tensor: Best value of x
    """

    x = x0

    best_x = None
    best_ynorm = torch.inf
    best_dxnorm = torch.inf
    best_iter = None

    # Add return_grad argument to end of params
    extended_params = list(params)
    extended_params.append(True)

    with torch.no_grad():
        i = 0
        while True:
            i += 1
            y, dy_dx = fcn(x, *extended_params)
            dx = y / dy_dx
            x = x - dx
            step_abs = dx.abs()

            if (step_abs < f_rtol * x.abs()).all():
                # Rel tol
                break
            if (step_abs < f_tol).all():
                # Abs tol
                break

            ynorm = y.abs().max()
            if ynorm < best_ynorm:
                best_x = x
                best_ynorm = ynorm
                best_dxnorm = step_abs.max()
                best_iter = i

            if i == maxiter:
                msg = (
                    "The rootfinder does not converge after %d iterations. "
                    "Best |dx|=%.3e, |f|=%.3e at iter %d") % (
                        maxiter, best_dxnorm, best_ynorm, best_iter
                    )
                warnings.warn(ConvergenceWarning(msg))
                x = best_x

        return x