from typing import Optional, Type
from time import perf_counter

import torch
import numpy as np
from plotly import graph_objects as go
from plotly import subplots
from torch import nn
from IPython.display import display
from torchdiffeq import odeint

from time_series_prediction import settings


class NoiseModel(nn.Module):
    """Base class for noise generation models"""
    ...


class ScalarNoise(NoiseModel):
    """Scalar Gaussian noise with a parameterised standard deviation

    Internally represents the parameter with an inverse softplus transform
    such that it is forced to be positive.

    For a more complete implementation, see:
    https://github.com/ymchen0/torchEnKF/blob/master/torchEnKF/noise.py
    """
    def __init__(self, param: torch.Tensor, dim: int):
        """Initialise

        Args:
            param (torch.Tensor): Standard deviation of Gaussian noise
            dim (int): Number of dimensions (standard deviation is same in all
                dimensions)
        """
        super().__init__()
        param_trans = self._softplus_inv(param)
        self.param = nn.Parameter(param_trans)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise to a tensor

        Args:
            x (torch.Tensor): Tensor (one axis should have length self.dim)

        Returns:
            torch.Tensor: x with added noise
        """
        param_untrans = self._softplus(self.param)
        x_out = x + torch.randn_like(x) * param_untrans
        return x_out

    def cov_mat(self) -> torch.Tensor:
        """Covariance matrix of multivariate Gaussian. Just a diagonal matrix.

        Returns:
            torch.Tensor: Covariance matrix of shape [self.dim, self.dim]
        """
        return torch.eye(self.dim, device=settings.device) * self._softplus(self.param)

    @staticmethod
    def _softplus(t: torch.Tensor) -> torch.Tensor:
        """Softplus transformation:

        y = log(1 + e^t)

        Args:
            t (torch.Tensor): Tensor to be transformed

        Returns:
            torch.Tensor: Transformed tensor
        """
        return torch.log(1.0 + torch.exp(t))

    @staticmethod
    def _softplus_inv(t: torch.Tensor) -> torch.Tensor:
        """Inverse softplus transformation:

        y = log(e^t - 1)

        Args:
            t (torch.Tensor): Tensor to be transformed

        Returns:
            torch.Tensor: Transformed tensor
        """
        return torch.log(torch.exp(t) - 1.0)


class AD_EnKF:
    """Auto-Differentiable Ensemble Kalman Filter (AD-EnKF)"""
    def __init__(
        self,
        transition_function: nn.Module,
        observation_matrix: torch.Tensor,
        observation_noise: NoiseModel,
        process_noise: NoiseModel,
        n_particles: int,
        init_state_distribution: Optional[torch.distributions.Distribution] = None,
        taper_radius: Optional[float] = None,
        neural_ode: bool = False,
        odeint_kwargs: Optional[dict] = None,
    ):
        """Learn the dynamics of a dataset using AD-EnKF, with linear state
        observations. The number of states and observations are defined by 
        the shape of observation_matrix.

        Args:
            transition_function (nn.Module): Function to be trained to 
                approximate the transition function: x_{i+1} = f(x_i)
            observation_matrix (torch.Tensor): Observation matrix that 
                transforms states into outputs: y = Ax. Has shape 
                [n_observations, n_states]
            observation_noise (NoiseModel): Function that adds noise to the
                outputs to simulate measurement noise
            process_noise (NoiseModel): Function that adds noise to the states
                to simulate uncertainty/stochasticity in the process dynamics
            n_particles (int): Number of particles to use in the ensemble 
                Kalman filter
            init_state_distribution (torch.distributions.Distribution, 
                optional): Distribution of size [n_states, n_states]. Defaults 
                to a MultivariateNormal distribution.
            taper_radius (float, optional): Tapering radius. Defaults to None.
            neural_ode (bool, optional): Assume transition_function is a neural
                ODE. Defaults to False, in which case transition_function 
                computes the next discrete state
            odeint_kwargs (dict, optional): kwargs to pass to odeint when 
                transition function is a neural ODE.
        """

        self.transition_function = transition_function
        self.observation_matrix = observation_matrix
        self.observation_noise = observation_noise
        self.process_noise = process_noise
        self.n_observations, self.n_states = observation_matrix.shape
        self.n_particles = n_particles
        self.taper_radius = taper_radius
        self.neural_ode = neural_ode
        if odeint_kwargs is None:
            odeint_kwargs = {'method': 'rk4', 'options': {'step_size': 0.01}}
        self.odeint_kwargs = odeint_kwargs

        if init_state_distribution is None:
            init_state_distribution = torch.distributions.MultivariateNormal(
                torch.zeros(self.n_states), torch.eye(self.n_states)
            )

        self.init_state_distribution = init_state_distribution
        self._opt = None
        self._scheduler = None
        self._fig = None

    def taper_rho(self) -> torch.Tensor:
        """Covariance tapering matrix - defined in equation SM4.3 of AD-EnKF 
        paper. Also see Gaspari and Cohn, 1999

        Returns: 
            torch.Tensor: Tapering matrix
        """

        def f1(x):
            return (
                1 
                - (5/3) * x**2
                + (5/8) * x**3
                + (1/2) * x**4
                - (1/4) * x**5
            )

        def f2(x):
            return (
                4 
                - 5 * x
                + (5/3) * x**2
                + (5/8) * x**3
                - (1/2) * x**4
                + (1/12) * x**5
                - 2/(3*x)
            )

        if self.taper_radius is None:
            return torch.ones((self.n_states, self.n_states), device=settings.device)

        inds = torch.arange(0, self.n_states, device=settings.device)
        i, j = torch.meshgrid(inds, inds, indexing='ij')
        z = torch.abs(i - j) / self.taper_radius
        rho = torch.zeros_like(z, dtype=torch.float)
        z_lt1 = z < 1
        z_gt1_lt2 = torch.logical_and(z >= 1, z < 2)
        rho[z_lt1] = f1(z[z_lt1])
        rho[z_gt1_lt2] = f2(z[z_gt1_lt2])
        return rho

    def log_likelihood(
        self, 
        obs: torch.Tensor, 
        x_0: Optional[torch.Tensor] = None, 
        dt: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log likelihood of the current model generating the observations

        Args:
            obs (torch.Tensor): Observations to compute log likelihood of, 
                with shape [n_steps, self.n_observations]
            x_0 (torch.Tensor, optional): Initial state. Defaults to None,
                in which case drawn from self.init_state_distribution.
            dt (float, optional): Timestep between observations (obs_train, 
                obs_test). Required if training a neural ODE. Defaults to 
                None.

        Returns:
            Tuple containing
            - torch.Tensor: Log likelihood of data
            - torch.Tensor: Estimated states, with shape 
                [n_steps, self.n_particles, self.n_states]
        """

        n_steps = obs.shape[0]
        x = []
        ll = 0

        # Draw initial particles from initial state distribution
        if x_0 is None:
            x_i = self.init_state_distribution.sample((self.n_particles,)).to(settings.device)
        else:
            x_i = x_0

        if self.neural_ode and dt is None:
            raise Exception('Must set dt if using neural ODE')

        r = self.observation_noise.cov_mat()

        # Tapering
        rho = self.taper_rho()

        for i_t in range(n_steps):
            # Forecast step
            # TODO: Wire up t and input vector
            if self.neural_ode:
                t = i_t * dt
                t_ode = torch.tensor([t, t+dt], dtype=torch.float32, device=settings.device)
                x_hat = odeint(self.transition_function, x_i, t_ode, **self.odeint_kwargs)[-1]
            else:
                x_hat = self.transition_function(None, x_i)
            x_hat = self.process_noise(x_hat)
            
            # Mean state
            m_hat = x_hat.mean(0)

            # Empirical covariance
            x_centered = x_hat - m_hat
            c = (x_centered.T @ x_centered) / (self.n_particles + 1)
            c = c * rho

            # Kalman gain
            c_ht = torch.mm(c, self.observation_matrix.T)
            h_c_ht = torch.mm(self.observation_matrix, c_ht)
            # k_hat = torch.mm(c_ht, torch.linalg.inv(h_c_ht + r))
            k_hat = torch.linalg.solve(h_c_ht + r, c_ht.T).T

            # Analysis step
            h_xhat = torch.mm(self.observation_matrix, x_hat.T)
            x_i = x_hat + torch.mm(
                k_hat, 
                self.observation_noise(obs[i_t, :].repeat(self.n_particles, 1)).T - h_xhat,
            ).T

            # Likelihood
            h_mhat = self.observation_matrix @ m_hat
            likelihood_dist = torch.distributions.MultivariateNormal(
                h_mhat, h_c_ht + r
            )
            ll = ll + likelihood_dist.log_prob(obs[i_t, :])
            x.append(x_i.detach())

        x = torch.stack(x)

        return ll, x

    def train(
        self, 
        obs_train: torch.Tensor, 
        n_epochs: int = 50, 
        lr_decay: float = 1.0, 
        lr_hold: int = 10, 
        progress_fig: bool = True, 
        display_fig: bool = True,
        lr_alpha: float = 1e-2,
        lr_beta: float = 1e-1,
        subseq_len: Optional[int] = None,
        obs_test: Optional[torch.Tensor] = None,
        print_timing: bool = False,
        dt: Optional[float] = False,
    ):
        """Train the transition function and the process noise models using
        AD-EnKF.

        In a notebook context, supports creating a Plotly figure that is 
        updated after every epoch to show the training progress.

        If this method is called multiple times, subsequent calls will warm-
        start the training. If progress_fig is True, the previous figure will
        be re-used; setting display_fig to True will also display the figure
        again below the current cell.

        Args:
            obs_train (torch.Tensor): Observations of process for training.
            n_epochs (int, optional): Number of training epochs. Defaults to 
                50.
            lr_decay (float, optional): Polynomial decay rate of learning rate. 
                Defaults to 1.0.
            lr_hold (int, optional): Number of epochs before beginning to decay
                learning rate. Defaults to 10.
            progress_fig (bool, optional): In a notebook context, create a 
                Plotly figure that is updated every epoch to describe the 
                training progress. Defaults to True.
            display_fig (bool, optional): In a notebook context, display the 
                figure created (if progress_fig is True) below this cell. 
                Defaults to True.
            lr_alpha (float, optional): Learning rate of transition function
                parameters. Defaults to 1e-2.
            lr_beta (float, optional): Learning rate of process noise scale.
                Defaults to 1e-1.
            subseq_len (float, optional): Subsequence length for truncated
                backprop. Defaults to None (no truncation).
            print_timing (bool, optional): Print timing info at each iteration.
                Defaults to False.
            dt (float, optional): Timestep between observations (obs_train, 
                obs_test). Required if training a neural ODE. Defaults to 
                None.
        """

        if self._opt is None:
            lambda1 = lambda2 = lambda epoch: (epoch+1-lr_hold)**(-lr_decay) if epoch >= lr_hold else 1
            alpha = self.transition_function.parameters()
            beta = self.process_noise.parameters()
            self._opt = torch.optim.Adam([
                {'params': alpha, 'lr': lr_alpha},
                {'params': beta, 'lr': lr_beta},
            ])
            # From https://github.com/ymchen0/torchEnKF/blob/016b4f8412310c195671c81790d372bd6cd9dc95/examples/l96_NN_demo.py
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._opt, lr_lambda=[lambda1, lambda2])
            self._epoch = 0

        if subseq_len is None:
            subseq_len = obs_train.shape[0]

        if progress_fig:
            if self._fig is None:
                self._fig = go.FigureWidget(
                    subplots.make_subplots(rows=3, cols=1, shared_xaxes=True))
                self._fig.update_layout(
                    title_text=f'Training: n={n_epochs}, lr_decay={lr_decay}, lr_hold={lr_hold}'
                )
                self._fig.update_yaxes(row=1, title_text='LL')
                self._fig.update_yaxes(row=2, title_text='LR', type='log')
                self._fig.update_yaxes(row=3, title_text='Process noise')
                self._fig.update_xaxes(row=3, title_text='Epochs')
                self._fig.add_scatter(row=1, col=1, x=[], y=[], showlegend=False)
                self._fig.add_scatter(row=1, col=1, x=[], y=[], showlegend=False, line_dash='dot')
                self._fig.add_scatter(row=2, col=1, x=[], y=[], showlegend=False, line_dash='dash')
                self._fig.add_scatter(row=2, col=1, x=[], y=[], showlegend=False)
                self._fig.add_scatter(row=3, col=1, x=[], y=[], showlegend=False)
                self._lines = self._fig.data
            if display_fig:
                display(self._fig)

        for i in range(n_epochs):

            t1 = perf_counter()

            # Testing
            if obs_test is not None:
                with torch.no_grad():
                    ll_test, _ = self.log_likelihood(obs_test, dt=dt)
                if print_timing:
                    t2 = perf_counter()
                    print(f'Test step: {t2-t1:.3f}s, ll={ll_test}')
            else:
                t2 = t1

            # Training
            x_0 = None
            ll_sum = 0
            for j in range(0, obs_train.shape[0], subseq_len):
                self._opt.zero_grad()
                ll, x = self.log_likelihood(obs_train[j:j+subseq_len, :], x_0, dt=dt)
                (-ll).backward()
                ll_sum = ll_sum + ll.detach()
                x_0 = x[-1, :, :].detach()
                self._opt.step()

            if print_timing:
                t3 = perf_counter()
                print(f'Train step: {t3-t2:.3f}s, ll={ll_sum}')

            self._scheduler.step()

            self._epoch += 1

            if progress_fig:
                i_list = list(self._lines[0].x)
                ll_list = list(self._lines[0].y)
                ll_test_list = list(self._lines[1].y)
                lambda_alpha_list = list(self._lines[2].y)
                lambda_beta_list = list(self._lines[3].y)
                beta_list = list(self._lines[4].y)

                i_list.append(self._epoch)
                ll_list.append(ll_sum.cpu().numpy())
                ll_test_list.append(ll_test.cpu().numpy() * obs_train.shape[0] / obs_test.shape[0])
                lambda_alpha_list.append(self._scheduler.get_last_lr()[0])
                lambda_beta_list.append(self._scheduler.get_last_lr()[1])
                beta_list.append(self.process_noise.param.detach().cpu().numpy()[0])

                for line in self._lines:
                    line.x = i_list
                
                self._lines[0].y = ll_list
                self._lines[1].y = ll_test_list
                self._lines[2].y = lambda_alpha_list
                self._lines[3].y = lambda_beta_list
                self._lines[4].y = beta_list
            else:
                t4 = perf_counter()
                print(
                    self._epoch, 
                    ll_sum.cpu().numpy(), 
                    self.process_noise.param.detach().cpu().numpy()[0],
                    t4-t1,
                )
                    
            if print_timing:
                t5 = perf_counter()
                print(f'Total epoch: {t5-t1:.3f}s')

    def forward(
        self, 
        x: torch.Tensor, 
        include_process_noise: bool = False, 
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """Call the transition function to predict the next state of a 
        sequence, given the current state. Optionally add process noise to the
        new state.

        Args:
            x (torch.Tensor): Current state
            include_process_noise (bool, optional): Add process noise to 
                transition. Defaults to False.
            dt (float, optional): Timestep between observations. Required 
                if transition_function is a neural ODE. Defaults to None.

        Returns:
            torch.Tensor: Next state
        """
        # TODO: Wire up t
        x_next = self.transition_function(None, x)
        if include_process_noise:
            x_next = self.process_noise(x_next)

        return x_next

    def predict(
        self,
        x_0: torch.Tensor, 
        n: int, 
        include_process_noise: bool = False, 
        include_observation_noise: bool = False,
        dt: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the state and output of a sequence a number of steps into
        the future, given an initial state.

        Args:
            x_i (torch.Tensor): Initial state
            n (int): Number of steps to predict
            include_process_noise (bool, optional): Add process noise to 
                transition. Defaults to False.
            include_observation_noise (bool, optional): Add observation noise.
                Defaults to False.
            dt (float, optional): Timestep between observations. Required 
                if transition_function is a neural ODE. Defaults to None.

        Returns:
            Tuple containing:
            - torch.Tensor: AD-EnKF states
            - torch.Tensor: AD-EnKF outputs
        """

        if self.neural_ode:
            if dt is None:
                raise Exception('Must set dt if using neural ODE')

            if not include_observation_noise:
                t = torch.arange(0, (n+1)*dt, dt, device=settings.device)
                x_kf = odeint(self.transition_function, x_0, t)
                y_kf = torch.mm(self.observation_matrix, x_kf.T).T
            else:
                raise NotImplementedError('todo')
        else:
            x_kf = []
            y_kf = []

            x_i = x_0

            for i in range(n):
                x_i = self.forward(x_i, include_process_noise, dt)
                y_i = torch.mm(self.observation_matrix, x_i.T).T
                if include_observation_noise:
                    y_i = self.observation_noise(y_i)

                x_kf.append(x_i)
                y_kf.append(y_i)
                
        return torch.stack(x_kf), torch.stack(y_kf)


class NeuralNet(nn.Module):
    """Simple neural network"""
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: list[int], 
        activation_function: Type[nn.Module] = nn.ReLU,
    ):
        """Initialise

        Args:
            input_dim (int): Dimension of NN input
            output_dim (int): Dimension of NN output
            hidden_dims (list[int]): List of hidden layer dimensions
        """
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dims[0])
        self.last_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.activation_function = activation_function()
        self.hidden_layers = nn.ModuleList()
        for dim_1, dim_2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.hidden_layers.append(nn.Linear(dim_1, dim_2))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through neural network.

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state

        Returns:
            torch.Tensor: Next state
        """
        y = self.activation_function(self.first_layer(x))
        for layer in self.hidden_layers:
            y = self.activation_function(layer(y))
        y = self.last_layer(y)
        return y


class ResNet(NeuralNet):
    """Residual neural network"""
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through neural network.

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state

        Returns:
            torch.Tensor: Next state
        """
        y = super().forward(t, x)
        return x + y


class EulerStepNet(NeuralNet):
    """Euler step network. Half-way between a ResNet and a Neural ODE, just
    multiply the residual by a timestep such that the network approximately
    learns the derivative.
    """
    def __init__(self, *args, dt: float = 0.1, **kwargs):
        """Initialise

        Args:
            dt (float, optional): Fixed timestep. Defaults to 0.1.
        """
        super().__init__(*args, **kwargs)
        self.dt = dt

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through neural network.

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state

        Returns:
            torch.Tensor: Next state
        """

        y = super().forward(t, x)
        return x + y * self.dt
