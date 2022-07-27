from typing import Callable

import torch
import numpy as np

from time_series_prediction import settings

# For details:
# https://arxiv.org/pdf/2012.02974.pdf

class ESN:
    def __init__(
        self,
        w_in: float = 1, 
        sparsity: float = 0.1, 
        spectral_radius: float = 0.99, 
        n_neurons: int = 100, 
        n_outputs: int = 1,
        n_inputs: int = 1,
        f_activation: Callable = torch.special.expit,
    ):
        self.w_in = w_in
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.f_activation = f_activation

        self._generate()

    def _generate(self):
        
        # For now, saple input/output/backprop layers from same distribution
        layer_weight_dist = torch.distributions.Uniform(-self.w_in, self.w_in)
        self.input_weights = layer_weight_dist.sample([self.n_neurons, self.n_inputs]).to(settings.device)
        self.backprop_weights = layer_weight_dist.sample([self.n_neurons, self.n_outputs]).to(settings.device)
        self.output_weights = layer_weight_dist.sample([self.n_outputs, self.n_neurons]).to(settings.device)

        # Reservoir
        sparsity_dist = torch.distributions.Bernoulli(self.sparsity)
        weight_dist = torch.distributions.Uniform(-1, 1)
        s = sparsity_dist.sample([self.n_neurons, self.n_neurons]).to(settings.device)
        self.w = weight_dist.sample([self.n_neurons, self.n_neurons]).to(settings.device)
        self.w *= s
        
        L, V = torch.linalg.eig(self.w)
        max_eig = L.abs().max()
        self.w *= self.spectral_radius / max_eig

    def forwards(
        self, 
        inp: torch.Tensor, 
        prev_state: torch.Tensor, 
        prev_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.f_activation(
            self.input_weights @ inp
            + self.w @ prev_state
            + self.backprop_weights @ prev_obs
            )

        y = x @ self.output_weights
        return x, y

    def predict(
        self, 
        u: torch.Tensor, 
        x_i: torch.Tensor, 
        y_i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_esn = []
        y_esn = []

        # x_dist = torch.distributions.Uniform(-self.w_in, self.w_in)
        # x_i = x_dist.sample([self.n_neurons]) / 100

        # x_i = torch.zeros((self.n_neurons))
        # y_i = torch.zeros([self.n_outputs])
        for i in range(u.shape[0]):
            x_i, y_i = self.forwards(u[i, :], x_i, y_i)
            x_esn.append(x_i.detach().cpu().numpy())
            y_esn.append(y_i.detach().cpu().numpy())

        return np.array(x_esn), np.array(y_esn)

    def train(
        self, 
        inp: torch.Tensor, 
        obs: torch.Tensor, 
        n_discard: int = 100, 
        k_l2: float = 0,
    ) -> torch.Tensor:
        
        n_steps = obs.shape[0]

        # Initialise with zero weights
        x = torch.zeros((n_steps, self.n_neurons), device=settings.device)

        # Drive network with training data
        for i in range(1, n_steps):
            x[i, :] = self.f_activation(
                self.input_weights @ inp[i, :]
                + self.w @ x[i-1, :]
                + self.backprop_weights @ obs[i-1, :]
            )

        phi = torch.concat((x, inp), 1)[n_discard:, :]
        y = obs[n_discard:, :]
        #self.output_weights = torch.pinverse(phi[n_discard:, :]) @ torch.tensor(obs[n_discard:, :])

        self.output_weights = torch.linalg.solve(
            torch.matmul(phi.T, phi) + k_l2 * torch.eye(self.n_neurons, device=settings.device), 
            torch.matmul(phi.T, y))
        # self.output_weights, res, rank, sing = torch.linalg.lstsq(phi, y)

        return x
