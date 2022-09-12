from typing import Callable

import torch

from time_series_prediction import settings


class ESN:
    """Echo state network"""
    def __init__(
        self,
        w_in: float = 1, 
        sparsity: float = 0.1, 
        spectral_radius: float = 0.99, 
        leaking_rate: float = 1.0,
        n_neurons: int = 100, 
        n_outputs: int = 1,
        n_inputs: int = 1,
        f_activation: Callable = torch.special.expit,
        bias: bool = True,
    ):
        """Initialise echo state network.

        See https://arxiv.org/pdf/2012.02974.pdf for more details.

        Args:
            w_in (float, optional): Input scaling parameter. Defaults to 1.
            sparsity (float, optional): Sparsity of reservoir. Defaults to 0.1.
            spectral_radius (float, optional): Spectral radius of reservoir. 
                Defaults to 0.99.
            leaking_rate (float, optional): Neuron leaking rate. Defaults to 
                1.0.
            n_neurons (int, optional): Size of reservoir. Defaults to 100.
            n_outputs (int, optional): Number of ESN outputs. Defaults to 1.
            n_inputs (int, optional): Number of ESN inputs. Defaults to 1.
            f_activation (Callable, optional): Activation function. Defaults to
                torch.special.expit.
            bias (bool, optional): Whether to use a bias term. Defaults to 
                True.
        """
        self.w_in = w_in
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.f_activation = f_activation
        self.bias = bias

        self._generate()

    def _generate(self, simple_scaling=False):
        """Generate reservoir matrices/vectors based on the given parameters"""
        
        # For now, saple input/output/backprop layers from same distribution
        layer_weight_dist = torch.distributions.Uniform(-self.w_in, self.w_in)
        self.input_weights = layer_weight_dist.sample(
            [self.n_neurons, self.n_inputs + self.bias]).to(settings.device)
        self.backprop_weights = layer_weight_dist.sample(
            [self.n_neurons, self.n_outputs]).to(settings.device)
        self.output_weights = None

        # Reservoir
        sparsity_dist = torch.distributions.Bernoulli(self.sparsity)
        if simple_scaling:
            weight_dist = torch.distributions.Uniform(-1, 1)
        else:
            weight_dist = torch.distributions.Uniform(0, 1)
        s = sparsity_dist.sample([self.n_neurons, self.n_neurons]).to(settings.device)
        self.w = weight_dist.sample([self.n_neurons, self.n_neurons]).to(settings.device)
        self.w *= s
        
        L, V = torch.linalg.eig(self.w)
        max_eig = L.abs().max()
        self.w = self.w * self.spectral_radius / max_eig
        if not simple_scaling:
            sign_dist = torch.distributions.Bernoulli(0.5)
            signs = sign_dist.sample([self.n_neurons, self.n_neurons]).to(settings.device) * 2 - 1
            self.w = self.w * signs

    def forward(
        self, 
        inp: torch.Tensor, 
        prev_state: torch.Tensor, 
        prev_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute next state and output given the current state and output.

        Args:
            inp (torch.Tensor): Next input
            prev_state (torch.Tensor): Current state
            prev_output (torch.Tensor): Current output

        Returns:
            Tuple containing:
            - torch.Tensor: next state
            - torch.Tensor: next (predicted) output
        """

        if self.bias:
            inp = torch.cat([torch.tensor([1.0], device=settings.device), inp])

        x = self.leaking_rate * self.f_activation(
            self.input_weights @ inp +
            self.w @ prev_state +
            self.backprop_weights @ prev_output
        ) + (1 - self.leaking_rate) * prev_state

        y = torch.cat([x, inp]) @ self.output_weights
        return x, y

    def predict(
        self, 
        u: torch.Tensor, 
        x_0: torch.Tensor, 
        y_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forwards function repeatedly and output a sequence of states
        and outputs

        Args:
            u (torch.Tensor): Inputs to model. If the model has no inputs 
                (self.n_inputs=0), then this must be a tensor of shape [n, 0]
                where n is the length of the required output sequence
            x_0 (torch.Tensor): Initial reservoir state of shape 
                [self.n_neurons]
            y_0 (torch.Tensor): Initial output of shape [self.n_outputs]

        Returns:
            Tuple containing:
            - torch.Tensor: ESN states
            - torch.Tensor: ESN outputs
        """
        x_esn = []
        y_esn = []
        x_i = x_0
        y_i = y_0

        for i in range(u.shape[0]):
            x_i, y_i = self.forward(u[i, :], x_i, y_i)
            x_esn.append(x_i)
            y_esn.append(y_i)

        return torch.stack(x_esn), torch.stack(y_esn)

    def train(
        self, 
        inputs: torch.Tensor, 
        outputs: torch.Tensor, 
        n_discard: int = 100, 
        k_l2: float = 0,
    ) -> torch.Tensor:
        """Generate output layer weights by driving the network with a sequence
        of inputs and outputs to a process, and performing ridge (L2) 
        regression on the reservoir states such that the ESN outputs 
        approximate the process outputs.

        Args:
            inputs (torch.Tensor): Inputs to process, of shape 
                [n, self.n_inputs]. Note that the first input is not used in
                training as the reservoir is initialised with zeros. If there
                are no inputs this must still be of shape [n, 0] 
            outputs (torch.Tensor): Outputs of process, of shape 
                [n, self.n_outputs]
            n_discard (int, optional): Number of initial reservoir states to 
                discard when training the output layer. Used to make sure the
                arbitrary choice of initial conditions does not affect the
                solution. Consider analogous to MCMC burn-in. Defaults to 100.
            k_l2 (float, optional): L2 regression strength. Defaults to 0.

        Returns:
            torch.Tensor: Reservoir states driven by the inputs and outputs, of
                shape [n, self.n_neurons]
        """
        
        n_steps = outputs.shape[0]

        # Initialise with zeros
        x = torch.zeros((n_steps, self.n_neurons), device=settings.device)

        if self.bias:
            inputs = torch.cat([torch.ones((n_steps, 1), device=settings.device), inputs], 1)

        # Drive network with training data
        for i in range(1, n_steps):
            x[i, :] = self.leaking_rate * self.f_activation(
                self.input_weights @ inputs[i, :] +
                self.w @ x[i-1, :] +
                self.backprop_weights @ outputs[i-1, :]
            ) + (1 - self.leaking_rate) * x[i-1, :]

        # Ridge regression: minimise ||y - phi * w||^2_2 + alpha * ||w||^2_2
        phi = torch.cat((x, inputs), 1)[n_discard:, :]
        y = outputs[n_discard:, :]

        self.output_weights = torch.linalg.solve(
            torch.matmul(phi.T, phi) +
            k_l2 * torch.eye(
                self.n_neurons + self.n_inputs + self.bias, device=settings.device), 
            torch.matmul(phi.T, y))

        # self.y_train = torch.cat((x, inputs), 1) @ self.output_weights

        return x
