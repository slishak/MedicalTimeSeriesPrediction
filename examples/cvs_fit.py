from typing import Type

import pandas as pd
import torch
from torch import nn, distributions
from torchinterp1d import Interp1d

from biophysical_models import models
from time_series_prediction import kalman, settings

# settings.switch_device('cuda')

def load_data(cvs_cls: Type[models.ODEBase], path='ZW0064.json'):
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

    states = cvs_cls.state_names + ['s']

    obs_cols = {}
    for state, cols in output_map.items():
        for col in cols:
            if col in df.columns:
                obs_cols[state] = col
                break

    t = torch.tensor(df.index.array, dtype=torch.float32, device=settings.device)
    hr = torch.tensor(df['Pulse'].to_numpy(), dtype=torch.float32, device=settings.device)
    y = []
    obs_matrix = torch.zeros((len(obs_cols), len(states)), device=settings.device)
    i_output = 0
    for key, val in obs_cols.items():
        i_state = states.index(key)
        obs_matrix[i_output, i_state] = 1
        y.append(torch.tensor(df[val].to_numpy(), dtype=torch.float32, device=settings.device))
        i_output += 1

    y = torch.stack(y, dim=1)

    return t, y, hr, obs_matrix


def get_enkf(obs_matrix, cvs, n_particles=100, taper_radius=None):
    
    # cvs_cls = models.add_bp_metrics(models.SmithCardioVascularSystem)

    # t, y, hr, obs_matrix = load_data(cvs_cls)
    n_outputs, n_states = obs_matrix.shape

    obs_noise = kalman.ScalarNoise(
        torch.tensor([5.], device=settings.device), 
        n_outputs,
    ).to(settings.device)

    proc_noise = kalman.ScalarNoise(
        torch.tensor([1e-5], device=settings.device), 
        n_states,
    ).to(settings.device)

    cvs.e.a.requires_grad_(False)

    # ode = EnsembleNeuralODE(cvs).to(settings.device)
    ode = cvs.to(settings.device)
    init_states = cvs.init_states(device=settings.device)
    init_loc = cvs.ode_state_tensor(init_states)

    init_state_dist = distributions.MultivariateNormal(
        init_loc,
        1e-6 * torch.eye(len(cvs.state_names), device=settings.device),
    )

    kf = kalman.AD_EnKF(
        ode, obs_matrix, obs_noise, proc_noise, n_particles, neural_ode=True,
        init_state_distribution=init_state_dist,
        taper_radius=taper_radius,
        odeint_kwargs={
            'method': 'dopri5',
            'rtol': 1e-6,
            'atol': 1e-7,
            'options': {'max_step': 1e-2},
        }
    )

    return kf


if __name__ == '__main__':
    cvs_cls = models.add_bp_metrics(models.SmithCardioVascularSystem)
    t, y, hr, obs_matrix = load_data(cvs_cls)
    f_hr = lambda t_i: Interp1d()(t, hr, t_i[None])[0, 0]
    cvs = cvs_cls(f_hr=f_hr)
    kf = get_enkf(obs_matrix, cvs, n_particles=10)
    kf.train(y[:100, :], 10, dt=1, subseq_len=20)