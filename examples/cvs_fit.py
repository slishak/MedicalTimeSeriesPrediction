from typing import Type

import pandas as pd
import torch
from torch import nn, distributions
from torchinterp1d import Interp1d

from biophysical_models import models
from time_series_prediction import kalman, settings

# settings.switch_device('cuda')

def load_data(cvs_cls: Type[models.ODEBase], path='ZW0064.json', save_traj=False, verbose=False):
    df = pd.read_json(path)

    output_map = {
        'p_aom': ['ABPm', 'UAPm', 'ARTm', 'AoM'],
        'p_aos': ['ABPs', 'UAPs', 'ARTs', 'AoS'],
        'p_aod': ['ABPd', 'UAPd', 'ARTd', 'AoD'],
        'p_vcm': ['CVPm', 'UVPm'],
        'p_pam': ['PAPm'],
        'p_pas': ['PAPs'],
        'p_pad': ['PAPd'],
    }

    obs_cols = {}
    for state, cols in output_map.items():
        for col in cols:
            if col in df.columns:
                obs_cols[state] = col
                break

    t = torch.tensor(df.index.array, dtype=torch.float32, device=settings.device)
    hr = torch.tensor(df['Pulse'].to_numpy(), dtype=torch.float32, device=settings.device)
    f_hr = lambda t_i: Interp1d()(t, hr, t_i[None])[0, 0]
    
    # Using v_spt_method='jallon' substantially speeds up backward pass
    cvs = cvs_cls(
        f_hr=f_hr, 
        save_traj=save_traj, 
        verbose=verbose, 
        volume_ratios=True, 
        # v_spt_method='jallon',
    )


    y = []
    obs_matrix = torch.zeros((len(obs_cols), len(cvs.state_names)), device=settings.device)
    i_output = 0
    for key, val in obs_cols.items():
        i_state = cvs.state_names.index(key)
        obs_matrix[i_output, i_state] = 1
        y.append(torch.tensor(df[val].to_numpy(), dtype=torch.float32, device=settings.device))
        i_output += 1

    y = torch.stack(y, dim=1)

    return y, obs_matrix, cvs


def get_enkf(
    obs_matrix, 
    cvs, 
    n_particles=100, 
    init_proc_noise=1e-3, 
    obs_noise=5.0, 
    init_dist_var=1e-9,
    rtol=1e-6,
    atol=1e-7,
    max_step=1e-2,
    adjoint=False,
):
    
    # cvs_cls = models.add_bp_metrics(models.SmithCardioVascularSystem)

    # t, y, hr, obs_matrix = load_data(cvs_cls)
    n_outputs, n_states = obs_matrix.shape

    obs_noise = kalman.ScalarNoise(
        torch.tensor([obs_noise], device=settings.device), 
        n_outputs,
    ).to(settings.device)

    proc_noise = kalman.ScalarNoise(
        torch.tensor([init_proc_noise], device=settings.device), 
        n_states,
    ).to(settings.device)

    scales = torch.ones(n_states, device=settings.device)
    for state in ['p_aod', 'p_aos', 'p_aom', 'p_vcm', 'p_pad', 'p_pas', 'p_pam', 's']:
        scales[cvs.state_names.index(state)] = 0

    # proc_noise = kalman.ScaledDiagonalNoise(
    #     torch.tensor([init_proc_noise], device=settings.device), 
    #     scales,
    # ).to(settings.device)

    cvs.e.a.requires_grad_(False)

    # ode = EnsembleNeuralODE(cvs).to(settings.device)
    ode = cvs.to(settings.device)
    init_states = cvs.init_states(device=settings.device)
    init_loc = cvs.ode_state_tensor(init_states)

    init_state_dist = distributions.MultivariateNormal(
        init_loc,
        init_dist_var * torch.eye(len(cvs.state_names), device=settings.device),
    )

    kf = kalman.AD_EnKF(
        ode, obs_matrix, obs_noise, proc_noise, n_particles, neural_ode=True,
        init_state_distribution=init_state_dist,
        odeint_kwargs={
            'method': 'dopri5',
            'rtol': rtol,
            'atol': atol,
            'options': {'max_step': max_step, 'min_step': 1e-5},
        },
        adjoint=adjoint
    )

    return kf


if __name__ == '__main__':
    cvs_cls = models.add_bp_metrics(models.SmithCardioVascularSystem)
    y, obs_matrix, cvs = load_data(cvs_cls)

    for param in cvs.parameters():
        param.requires_grad_(False)
    cvs.v_tot.requires_grad_(True)
    cvs.mt.r.requires_grad_(True)
    cvs.tc.r.requires_grad_(True)
    cvs.av.r.requires_grad_(True)
    cvs.pv.r.requires_grad_(True)
    cvs.pul.r.requires_grad_(True)
    cvs.sys.r.requires_grad_(True)

    kf = get_enkf(obs_matrix, cvs, n_particles=100)

    kf.train(y[:100, :], 10, dt=1, subseq_len=10, print_timing=True)