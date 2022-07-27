from typing import Optional

import torch
import numpy as np
from plotly import graph_objects as go
from plotly import subplots
from torch import nn
from IPython.display import display

from time_series_prediction import settings


class NoiseModel(nn.Module):
    ...


class ScalarNoise(NoiseModel):
    def __init__(self, param: torch.Tensor, dim: int):
        super().__init__()
        param_trans = self._softplus_inv(param)
        self.param = nn.Parameter(param_trans)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        param_untrans = self._softplus(self.param)
        x_out = x + torch.randn_like(x) * param_untrans
        return x_out

    def cov_mat(self) -> torch.Tensor:
        return torch.eye(self.dim, device=settings.device) * self._softplus(self.param)

    @staticmethod
    def _softplus(t):
        return torch.log(1.0 + torch.exp(t))

    @staticmethod
    def _softplus_inv(t):
        return torch.log(-1.0 + torch.exp(t))


class AD_EnKF:
    def __init__(
        self,
        transition_function: nn.Module,
        observation_matrix: torch.Tensor,
        observation_noise: NoiseModel,
        process_noise: NoiseModel,
        n_particles: int,
        init_state_distribution: Optional[torch.distributions.Distribution] = None,
    ):

        self.transition_function = transition_function
        self.observation_matrix = observation_matrix
        self.observation_noise = observation_noise
        self.process_noise = process_noise
        self.n_observations, self.n_states = observation_matrix.shape
        self.n_particles = n_particles

        if init_state_distribution is None:
            init_state_distribution = torch.distributions.MultivariateNormal(
                torch.zeros(self.n_states), torch.eye(self.n_states)
            )

        self.init_state_distribution = init_state_distribution
        self._opt = None
        self._scheduler = None
        self._fig = None

    def log_likelihood(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        n_steps = obs.shape[0]
        x = torch.zeros((n_steps, self.n_particles, self.n_states), device=settings.device)
        
        ll = torch.zeros(n_steps, device=settings.device)

        # Draw initial particles from initial state distribution
        x[0, :, :] = self.init_state_distribution.sample((self.n_particles,)).to(settings.device)

        r = self.observation_noise.cov_mat()

        for t in range(1, n_steps):
            # Forecast step
            # TODO: Wire up input vector
            x_hat = self.process_noise(self.transition_function(t, x[t-1, :, :].clone(), None))
            
            # Mean state
            m_hat = x_hat.mean(0)

            # Empirical covariance
            x_centered = x_hat - m_hat
            c_hat = (x_centered.T @ x_centered) / (self.n_particles + 1)

            # Kalman gain
            c_ht = torch.mm(c_hat, self.observation_matrix.T)
            h_c_ht = torch.mm(self.observation_matrix, c_ht)
            # k_hat = torch.mm(c_ht, torch.linalg.inv(h_c_ht + r))
            k_hat = torch.linalg.solve(h_c_ht + r, c_ht).T

            # Analysis step
            # TODO: These are both reversed compared to paper - does it matter?
            h_xhat = torch.mm(x_hat, self.observation_matrix)
            x[t, :, :] = x_hat + torch.mm(self.observation_noise(obs[t-1, :].repeat(self.n_particles, 1)) - h_xhat, k_hat)
            # for n in range(self.n_particles):
            #     h_xhat = torch.mm(self.observation_matrix, x[t, n, :])
            #     x[t, n, :] += torch.mm(k_hat, obs[t-1, :] + observation_noise_dist.sample((self.n_particles,)) - h_xhat)

            # Likelihood
            h_mhat = self.observation_matrix @ m_hat
            likelihood_dist = torch.distributions.MultivariateNormal(
                h_mhat, h_c_ht + r
            )
            ll[t] = likelihood_dist.log_prob(obs[t-1, :])

        return ll.sum(), x

    def train(
        self, 
        obs: torch.Tensor, 
        n: int = 50, 
        lr_decay: int = 1, 
        lr_hold: int = 10, 
        show_fig: bool = True, 
        display_fig: bool = True,
    ):

        lambda1 = lambda2 = lambda epoch: (epoch+1-lr_hold)**(-lr_decay) if epoch >= lr_hold else 1

        if self._opt is None:
            alpha = self.transition_function.parameters()
            beta = self.process_noise.parameters()
            self._opt = torch.optim.Adam([
                {'params': alpha, 'lr': 1e-2}, 
                {'params': beta, 'lr': 1e-1},
            ])
            # From https://github.com/ymchen0/torchEnKF/blob/016b4f8412310c195671c81790d372bd6cd9dc95/examples/l96_NN_demo.py
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, lr_lambda=[lambda1, lambda2])

        if show_fig:
            if self._fig is None:
                self._fig = go.FigureWidget(subplots.make_subplots(rows=3, cols=1, shared_xaxes=True))
                self._fig.update_layout(
                    title_text=f'Training: n={n}, lr_decay={lr_decay}, lr_hold={lr_hold}'
                )
                self._fig.update_yaxes(row=1, title_text='Log likelihood')
                self._fig.update_yaxes(row=2, title_text='LR')
                self._fig.update_yaxes(row=3, title_text='Process noise')
                self._fig.add_scatter(row=1, col=1, x=[], y=[])
                self._fig.add_scatter(row=2, col=1, x=[], y=[])
                self._fig.add_scatter(row=3, col=1, x=[], y=[])
            if display_fig:
                display(self._fig)
            # i_list = []
            # ll_list = []
            # lambda_list = []
            # beta_list = []


        for i in range(n):
            self._opt.zero_grad()
            ll, _ = self.log_likelihood(obs)
            (-ll).backward()
            with torch.no_grad():
                if show_fig:
                    i_list = list(self._fig.data[0].x)
                    ll_list = list(self._fig.data[0].y)
                    lambda_list = list(self._fig.data[1].y)
                    beta_list = list(self._fig.data[2].y)

                    i_list.append(i_list[-1] + 1)
                    ll_list.append(ll.detach().cpu().numpy())
                    lambda_list.append(self._scheduler.get_last_lr()[0])
                    beta_list.append(self.process_noise.param.detach().cpu().numpy()[0])

                    self._fig.data[0].x = i_list
                    self._fig.data[1].x = i_list
                    self._fig.data[2].x = i_list
                    self._fig.data[0].y = ll_list
                    self._fig.data[1].y = lambda_list
                    self._fig.data[2].y = beta_list
                else:
                    print(i, ll.detach().cpu().numpy(), self.process_noise.param.detach().cpu().numpy()[0])
            self._opt.step()
            self._scheduler.step()

    def forward(self, x: torch.Tensor, include_process_noise: bool = False) -> torch.Tensor:
        # TODO: Wire up t, u
        x_next = self.transition_function(None, x, None)
        if include_process_noise:
            x_next = self.process_noise(x_next)

        return x_next

    def predict(
        self,
        x_i: torch.Tensor, 
        n: int, 
        include_process_noise: bool = False, 
        include_observation_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_kf = []
        y_kf = []

        for i in range(n):
            y_i = torch.mm(x_i, self.observation_matrix)
            if include_observation_noise:
                y_i = self.observation_noise(y_i)

            x_kf.append(x_i.detach().cpu().numpy())
            y_kf.append(y_i.detach().cpu().numpy())
            
            x_i = self.forward(x_i, include_process_noise)

        return np.array(x_kf), np.array(y_kf)


class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dims[0])
        self.last_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.hidden_layers = nn.ModuleList()
        for dim_1, dim_2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.hidden_layers.append(nn.Linear(dim_1, dim_2))

    def forward(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.first_layer(x))
        for layer in self.hidden_layers:
            y = torch.relu(layer(y))
        y = self.last_layer(y)
        return y


class ResNet(NeuralNet):
    def forward(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = super().forward(t, x, u)
        return x + y


class EulerStepNet(NeuralNet):
    def __init__(self, *args, dt: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = super().forward(t, x, u)
        return x + y * self.dt


# From https://github.com/ymchen0/torchEnKF/blob/master/torchEnKF/nn_templates.py
class L96_ODE_Net_2(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.x_dim = x_dim
        self.layer1 = nn.Conv1d(1, 72, 5, padding=2, padding_mode='circular')
        # self.layer1b = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
        # self.layer1c = nn.Conv1d(1, 24, 5, padding=2, padding_mode='circular')
        self.layer2 = nn.Conv1d(48, 37, 5, padding=2, padding_mode='circular')
        self.layer3 = nn.Conv1d(37, 1, 1)

        # self.layer1 = nn,Conv1d(1,6,5)

    def forward(self, u):
        bs = u.shape[:-1] # (*bs, x_dim)
        out = torch.relu(self.layer1(u.view(-1, self.x_dim).unsqueeze(-2))) # (bs, 1, x_dim) -> (bs, 72, x_dim)
        out = torch.cat((out[...,:24,:], out[...,24:48,:] * out[...,48:,:]), dim=-2) # (bs, 72, x_dim) -> (bs, 48, x_dim)
        out = torch.relu(self.layer2(out)) # (bs, 48, x_dim) -> (bs, 37, x_dim)
        out = self.layer3(out).squeeze(-2).view(*bs, self.x_dim) # (bs, 37, x_dim) -> (bs, 1, x_dim) -> (*bs, x_dim)
        return out