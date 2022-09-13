import os
from tempfile import mkdtemp, mkstemp
from time import perf_counter
from typing import Iterable, Optional, Any
from itertools import product

import torch
import numpy as np
from plotly import subplots, colors, graph_objects as go
from torch import nn
from IPython.display import display
from dask.distributed import Client, progress

from time_series_prediction import kalman, echo_state_network, settings, ode_problems


T_LYAPUNOV = {
    'lorenz': 1.104,
    'rossler': 14.0,
}


def param_sweep(params: list[tuple[str, list]]) -> list[dict[str, Any]]:
    """Get list of parameter dicts.

    Uses itertools.product to get all combinations of parameters.

    Args:
        params (list[tuple[str, list]]): List of (name, values).

    Returns:
        list[dict[str, Any]]: List of parameter dicts.
    """
    keys, vals = zip(*params)
    d = [{k: v for k, v in zip(keys, val)} for val in product(*vals)]
    print(f'{len(d)} simulations')
    return d


class Sweep:
    """Parameter sweep base class."""

    def __init__(
        self,
        n_warmup: int = 1000,
        n_train: int = 4000, 
        n_test: int = 4000, 
        n_test_err: int = 1000,
        noise: float = 1e-2, 
        init_noise: float = 0.0,
        source: Optional[str] = None,
    ):
        """Initialise.

        Args:
            n_warmup (int, optional): Number of initial points to ignore. 
                Defaults to 1000.
            n_train (int, optional): Size of training set. Defaults to 4000.
            n_test (int, optional): Size of test set. Defaults to 4000.
            n_test_err (int, optional): Number of points for test error. 
                Defaults to 1000.
            noise (float, optional): Isotropic noise added to standardised
                data. Defaults to 1e-2.
            init_noise (float, optional): Isotropic noise added to initial
                states. Defaults to 0.0.
            source (Optional[str], optional): ODE data source. Defaults to
                None.
        """
        self.n_warmup = n_warmup
        self.n_train = n_train
        self.n_test = n_test
        self.n_test_err = n_test_err
        self.noise = noise
        self.init_noise = init_noise
        self.source = source

        if self.source is not None and init_noise == 0:
            y = self.generate_data()
            self.n_outputs = y.shape[1]
            self.y_train, self.y_test, self.y_mean, self.y_std = self.split_train_test(y)
        else:
            self.y_train = None
            self.y_test = None
            self.y_mean = None
            self.y_std = None
            self.n_outputs = None

    @staticmethod
    def get_resolution(source: str) -> int:
        """Get resolution for a given source.

        Args:
            source (str): Name of ODE source.

        Returns:
            int: Resolution (Hz).
        """
        if source in ('vdp', 'rossler'):
            return 10
        else:
            return 50

    def generate_data(self, source: Optional[str] = None) -> torch.Tensor:
        """Generate data from an ODE.

        Args:
            source (Optional[str], optional): ODE source. If None, use 
                self.source. Defaults to None.

        Returns:
            torch.Tensor: Simulated states.
        """
        if source is None:
            source = self.source
        y, _, _ = ode_problems.generate_data(
            source, 
            self.n_warmup+self.n_train+self.n_test, 
            self.get_resolution(source),
            init_noise=self.init_noise,
        )
        y = y.to(settings.device)

        return y

    def unscale(
        self, 
        y: torch.Tensor, 
        y_mean: Optional[torch.Tensor] = None, 
        y_std: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Undo standardisation of data.

        Args:
            y (torch.Tensor): Standardised states.
            y_mean (Optional[torch.Tensor], optional): Mean state value. If 
                None, use self.y_mean. Defaults to None.
            y_std (Optional[torch.Tensor], optional): Std of state values. If
                None, use self.y_std. Defaults to None.

        Returns:
            torch.Tensor: Un-standardised states.
        """
        if y_mean is None:
            y_mean = self.y_mean
            y_std = self.y_std

        y_unscaled = (y * y_std) + y_mean

        return y_unscaled

    def split_train_test(
        self, 
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split train and test data.

        Args:
            y (torch.Tensor): Simulated states

        Returns:
            tuple containing:
            - torch.Tensor: Training data.
            - torch.Tensor: Test data.
            - torch.Tensor: Mean of training data.
            - torch.Tensor: Std of training data.
        """
        y_full = y[self.n_warmup:, :]

        y_train_raw = y_full[:self.n_train, :]
        y_test_raw = y_full[self.n_train:, :]

        y_mean = y_train_raw.mean(axis=0)
        y_std = y_train_raw.std(axis=0)
        
        y_train = (y_train_raw - y_mean) / y_std
        y_test = (y_test_raw - y_mean) / y_std

        y_train = y_train + torch.randn_like(y_train) * self.noise
        y_test = y_test + torch.randn_like(y_test) * self.noise

        return y_train, y_test, y_mean, y_std

    def calc_err(
        self, 
        y_pred: torch.Tensor, 
        y_test: Optional[torch.Tensor] = None,
        source: Optional[str] = None
    ) -> tuple[torch.Tensor]:
        """Calculate error on test set.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_test (Optional[torch.Tensor], optional): Test values.  If None, 
                use self.y_test. Defaults to None.
            source (Optional[str], optional): Name of data source. If None, use
                self.source. Defaults to None.

        Returns:
            tuple containing:
            - torch.Tensor: Average mean squared error over first 
                self.n_test_err points
            - torch.Tensor: Average mean absolute error over first
                self.n_test_err points
            - torch.Tensor: Square error over all dimensions for each point.
            - torch.Tensor: Absolute error for each point.
            - torch.Tensor: Square error over all dimensions averaged over each
                Lyapunov time.
            - torch.Tensor: Absolute error over all dimensions averaged over 
                each Lyapunov time.
        """
        if y_test is None:
            y_test = self.y_test
        if source is None:
            source = self.source

        sq_err = (y_pred - y_test)**2
        
        # Prevent NaN error
        sq_err[torch.isnan(sq_err) | (torch.abs(sq_err) > 1e10)] = 1e10

        # Sum over all output dimensions
        sq_err = sq_err.sum(axis=-1)

        # Absolute error
        abs_err = torch.sqrt(sq_err)
        
        # Metrics
        mse = torch.mean(sq_err[:self.n_test_err])
        mae = torch.mean(abs_err[:self.n_test_err])

        # Error over first 10 Lyapunov times
        n_lyap = 10
        res = self.get_resolution(source)
        try:
            t_lyap = T_LYAPUNOV[source]
        except KeyError:
            mse_lyap = None
            mae_lyap = None
        else:
            mse_lyap = torch.zeros(n_lyap)
            mae_lyap = torch.zeros(n_lyap)
            for i in range(n_lyap):
                lyap_window = [int(i*res*t_lyap), int((i+1)*res*t_lyap)]

                mse_lyap[i] = torch.mean(sq_err[lyap_window[0]:lyap_window[1]])
                mae_lyap[i] = torch.mean(abs_err[lyap_window[0]:lyap_window[1]])

        return mse, mae, sq_err, abs_err, mse_lyap, mae_lyap


class SweepESN(Sweep):
    """Echo state network sweep."""

    def __init__(
        self,
        param_dicts: list[dict],
        esn_kwargs: dict = {}, 
        n_ensemble: int = 10,
        **kwargs
    ):
        """Initialise.

        Args:
            param_dicts (list[dict]): List of parameter dicts.
            esn_kwargs (dict, optional): Constant arguments to ESN initialiser.
                Defaults to {}.
            n_ensemble (int, optional): Number of ESNs to train for each set
                of parameters. Defaults to 10.
        """
        super().__init__(**kwargs)
        self.param_dicts = param_dicts
        self.esn_kwargs = esn_kwargs
        self.n_ensemble = n_ensemble

        self.plot_folder = mkdtemp(prefix='esn-ensemble-')

    def submit_jobs(self, client: Client, show_progress: bool = True):
        """Submit jobs to a dask.distributed.Client.

        Args:
            client (Client): Client to submit jobs to.
            show_progress (bool, optional): Show a progress bar in a notebook 
                context. Defaults to True.
        """
        self.jobs = []
        for d in self.param_dicts:
            esn_kwargs = self.esn_kwargs | d
            self.jobs.append(client.submit(self.run_ensemble, esn_kwargs))

        if show_progress:
            return progress(self.jobs)

    def get_results(self) -> list[dict[str, Any]]:
        """Get results from dask.distributed.Client.

        Returns:
            list[dict[str, Any]]: List of result dicts.
        """
        rows = []
        for task, d in zip(self.jobs, self.param_dicts):
            mae, mse, mae_lyap, mse_lyap, fig_path = task.result()
            row = {
                'MAE': mae, 
                'MSE': mse, 
                'MAE Lyapunov': mae_lyap,
                'MSE Lyapunov': mse_lyap,
                '_fig_path': fig_path,
            }
            row.update(d)
            row['Min MAE'] = torch.min(mae)
            row['Max MAE'] = torch.max(mae)
            row['Median MAE'] = torch.median(mae)
            row['Mean MAE'] = torch.mean(mae)
            row['Min MSE'] = torch.min(mse)
            row['Max MSE'] = torch.max(mse)
            row['Median MSE'] = torch.median(mse)
            row['Mean MSE'] = torch.mean(mse)

            rows.append(row)

        return rows
            
    def train_esn(
        self, 
        source: str = None, 
        n_discard: int = 100, 
        k_l2: float = 0, 
        **esn_kwargs,
    ) -> tuple:
        """Train an ESN.

        Args:
            source (str, optional): ODE source. If None, use self.source. 
                Defaults to None.
            n_discard (int, optional): Number of initial points to discard in 
                linear regression. Defaults to 100.
            k_l2 (float, optional): L2 regularisation weight. Defaults to 0.

        Returns:
            tuple containing:
            - torch.Tensor: Average mean squared error over first 
                self.n_test_err points
            - torch.Tensor: Average mean absolute error over first
                self.n_test_err points
            - torch.Tensor: Square error over all dimensions for each point.
            - torch.Tensor: Absolute error for each point.
            - torch.Tensor: Square error over all dimensions averaged over each
                Lyapunov time.
            - torch.Tensor: Absolute error over all dimensions averaged over 
                each Lyapunov time.
            - torch.Tensor: Predictions on test data.
            - torch.Tensor: Test data.
            - torch.Tensor: Timing data (train time, generation+train time, 
                generation+train+test time)
        """
        if self.y_train is None:
            y_full = self.generate_data(self.source or source)
            y_train, y_test, y_mean, y_std = self.split_train_test(y_full)
            n_outputs = y_test.shape[1]
        else:
            y_train = self.y_train
            y_test = self.y_test
            y_mean = self.y_mean
            y_std = self.y_std
            n_outputs = self.n_outputs

        u_train = torch.empty((self.n_train, 0), device=settings.device)
        u_test = torch.empty((self.n_test, 0), device=settings.device)

        t1 = perf_counter()

        esn = echo_state_network.ESN(
            n_inputs=0, 
            n_outputs=n_outputs,
            **esn_kwargs
        )

        t2 = perf_counter()

        x_train = esn.train(
            u_train, 
            y_train,
            n_discard,
            k_l2,
        )

        t3 = perf_counter()

        x_init = x_train[-1, :]
        y_init = y_train[-1, :]
        _, y_test_esn = esn.predict(
            u_test, 
            x_init, 
            y_init,
        )

        mse, mae, sq_err, abs_err, mse_lyap, mae_lyap = self.calc_err(
            y_test_esn, y_test, source=self.source or source)

        y_esn_unscaled = (y_test_esn * y_std) + y_mean
        y_test_unscaled = (y_test * y_std) + y_mean

        t4 = perf_counter()

        dt = [t3 - t2, t3 - t1, t4 - t1]
        # print(f't={dt:.3f}s')

        return mse, mae, sq_err, abs_err, mse_lyap, mae_lyap, y_esn_unscaled, y_test_unscaled, dt
        
    def run_ensemble(self, esn_kwargs: dict, return_timing: bool = False) -> tuple:
        """Fit an ensemble of ESNs.

        Args:
            esn_kwargs (dict, optional): Constant arguments to ESN initialiser.
            return_timing (bool, optional): Return timing data. Defaults to 
                False.

        Returns:
            tuple containing:
            - torch.Tensor: Average mean squared error over first 
                self.n_test_err points for each ESN
            - torch.Tensor: Average mean absolute error over first
                self.n_test_err points for each ESN.
            - torch.Tensor: Square error over all dimensions averaged over each
                Lyapunov time for each ESN.
            - torch.Tensor: Absolute error over all dimensions averaged over 
                each Lyapunov time for each ESN.
            - str: Path to a Plotly figure json showing performance on test 
                data. Will be generated in a temporary folder.
            - list: If return_timing is True, list of timing data for each ESN.
            
        if return_timing:
            return mae, mse, mae_lyap, mse_lyap, fig_path, timing
        else:
            return mae, mse, mae_lyap, mse_lyap, fig_path
        """
        y_esn_list = []
        y_test_list = []
        abs_err_list = []
        sq_err_list = []
        mae_list = []
        mse_list = []
        mse_lyap_list = []
        mae_lyap_list = []
        timing = []

        for i in range(self.n_ensemble):
            # print(f'{i+1}/{self.n_ensemble}')

            mse, mae, sq_err, abs_err, mse_lyap, mae_lyap, y_esn_unscaled, y_test_unscaled, dt = self.train_esn(  # noqa: 273
                **esn_kwargs
            )

            y_esn_list.append(y_esn_unscaled.cpu())
            y_test_list.append(y_test_unscaled.cpu())
            abs_err_list.append(abs_err.cpu())
            sq_err_list.append(sq_err.cpu())
            mae_list.append(mae.cpu())
            mse_list.append(mse.cpu())
            mae_lyap_list.append(mae_lyap)
            mse_lyap_list.append(mse_lyap)
            timing.append(dt)

        mae = torch.stack(mae_list)
        mse = torch.stack(mse_list)
        if mae_lyap_list[0] is not None:
            mae_lyap = torch.stack(mae_lyap_list).cpu()
            mse_lyap = torch.stack(mse_lyap_list).cpu()
        else:
            mae_lyap = None
            mse_lyap = None
        
        # print(f'MAE {torch.mean(mae)} ± {torch.std(mae)}')
        # print(f'MSE {torch.mean(mse)} ± {torch.std(mse)}')

        fig = self.plot_ensemble(
            y_esn_list, 
            y_test_list, 
            abs_err_list, 
            sq_err_list, 
            self.source or esn_kwargs.get('source'),
        )

        fd, fig_path = mkstemp('.json', 'esn-', self.plot_folder)
        fig.write_json(fig_path)
        os.close(fd)

        if return_timing:
            return mae, mse, mae_lyap, mse_lyap, fig_path, timing
        else:
            return mae, mse, mae_lyap, mse_lyap, fig_path

    def plot_ensemble(
        self, 
        y_esn_list: list[torch.Tensor], 
        y_test_list: list[torch.Tensor], 
        abs_err_list: list[torch.Tensor], 
        sq_err_list: list[torch.Tensor], 
        source: Optional[str] = None,
    ) -> go.Figure:
        """Plot results of an ESN ensemble.

        Args:
            y_esn_list (list[torch.Tensor]): List of ESN predictions.
            y_test_list (list[torch.Tensor]): List of test data.
            abs_err_list (list[torch.Tensor]): List of absolute errors.
            sq_err_list (list[torch.Tensor]): List of squared errors.
            source (Optional[str], optional): ODE source name. If None, use
                self.source. Defaults to None.

        Returns:
            go.Figure: Plotly figure.
        """
        source = self.source or source
        res = self.get_resolution(source)
        t_lyap = T_LYAPUNOV.get(source)

        y_esn = torch.stack(y_esn_list, 1)
        y_test = torch.stack(y_test_list, 1)

        y_esn_mean = torch.mean(y_esn, 1)
        y_esn_std = torch.std(y_esn, 1)

        t_esn = torch.arange(0, y_test.shape[0]) / res

        n_outputs = y_esn.shape[2]
        
        c = colors.sample_colorscale(
            colors.get_colorscale('Plotly3'), 
            np.linspace(0, 0.9, self.n_ensemble),
        )
        if n_outputs <= 3:
            specs = [[{}, {'rowspan': n_outputs+2}]]
            if n_outputs > 2:
                specs[0][1]['type'] = 'scene'
            specs.extend([[{}, None]]*(n_outputs+1))
            cols = 2
        else:
            specs = None
            cols = 1
        fig = subplots.make_subplots(
            rows=n_outputs+2, cols=cols, shared_xaxes=True, specs=specs,
        )

        for i in range(self.n_ensemble):
            for j in range(n_outputs):
                fig.add_scatter(
                    x=t_esn,
                    y=y_esn[:, i, j],
                    row=j+1,
                    col=1,
                    line_color=c[i],
                    name=f'ESN {i}',
                    legendgroup=f'ESN {i}',
                    showlegend=False,
                )
                if self.init_noise != 0:
                    fig.add_scatter(
                        x=t_esn,
                        y=y_test[:, i, j],
                        row=j+1,
                        col=1,
                        line_color=c[i],
                        line_dash='dash',
                        name=f'System data {i}',
                        legendgroup=f'ESN {i}',
                        showlegend=False,
                    )
                # fig.add_scatter(
                #     # x=t_esn,
                #     y=sq_err_list[i][:, j],
                #     row=j+1,
                #     col=2,
                #     line_color=c[i],
                #     name=f'ESN {i}',
                #     legendgroup=f'ESN {i}',
                #     showlegend=False,
                # )

            fig.add_scatter(
                x=t_esn,
                y=abs_err_list[i],
                row=n_outputs+1,
                col=1,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=True,
            )
            fig.add_scatter(
                x=t_esn,
                y=sq_err_list[i],
                row=n_outputs+2,
                col=1,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=False,
            )

            if i == 0 and t_lyap is not None:
                t = t_lyap
                while t < t_esn[-1]:
                    fig.add_vline(x=t, row=n_outputs+1, col=1, line_color='black', line_dash='dot')
                    fig.add_vline(x=t, row=n_outputs+2, col=1, line_color='black', line_dash='dot')
                    t += t_lyap

            if n_outputs == 3:
                fig.add_scatter3d(
                    x=y_esn[:, i, 0],
                    y=y_esn[:, i, 1],
                    z=y_esn[:, i, 2],
                    row=1,
                    col=2,
                    line_color=c[i],
                    name=f'ESN {i}',
                    legendgroup=f'ESN {i}',
                    showlegend=False,
                    mode='lines',
                    opacity=0.3,
                )
                if self.init_noise != 0:
                    fig.add_scatter3d(
                        x=y_test[:, i, 0],
                        y=y_test[:, i, 1],
                        z=y_test[:, i, 2],
                        row=1,
                        col=2,
                        line_color=c[i],
                        line_dash='dash',
                        name=f'System data {i}',
                        legendgroup=f'ESN {i}',
                        showlegend=False,
                        mode='lines',
                    )
            elif n_outputs == 2:
                fig.add_scatter(
                    x=y_esn[:, i, 0],
                    y=y_esn[:, i, 1],
                    row=1,
                    col=2,
                    line_color=c[i],
                    name=f'ESN {i}',
                    legendgroup=f'ESN {i}',
                    showlegend=False,
                    opacity=0.3,
                )
                if self.init_noise != 0:
                    fig.add_scatter(
                        x=y_test[:, i, 0],
                        y=y_test[:, i, 1],
                        row=1,
                        col=2,
                        line_color=c[i],
                        line_dash='dash',
                        name=f'System data {i}',
                        legendgroup=f'ESN {i}',
                        showlegend=False,
                    )

            # fig.add_scatter(
            #     y=np.cumsum(abs_err_list[i]) / np.arange(1, len(abs_err_list[i])+1),
            #     row=n_outputs+1,
            #     col=2,
            #     line_color=c[i],
            #     name=f'ESN {i}',
            #     legendgroup=f'ESN {i}',
            #     showlegend=False,
            # )
            # fig.add_scatter(
            #     y=np.cumsum(sq_err_list[i]) / np.arange(1, len(sq_err_list[i])+1),
            #     row=n_outputs+2,
            #     col=2,
            #     line_color=c[i],
            #     name=f'ESN {i}',
            #     legendgroup=f'ESN {i}',
            #     showlegend=False,
            # )

        for i in range(n_outputs):
            fig.update_yaxes(
                row=i+1,
                col=1,
                title_text=f'Dimension {i+1}'
            )
            fig.update_yaxes(
                row=i+1,
                col=2,
                title_text='Square error'
            )

            if self.init_noise == 0:
                fig.add_scatter(
                    x=t_esn,
                    y=y_test[:, 0, i],
                    row=i+1,
                    col=1,
                    line_color='black',
                    # line_dash='dash',
                    name='System data',
                    legendgroup='Test',
                    showlegend=i == 0,
                )

                fig.add_scatter(
                    x=t_esn,
                    y=y_esn_mean[:, i] - y_esn_std[:, i], 
                    row=i+1, 
                    col=1, 
                    line_color='red', 
                    showlegend=False, 
                    opacity=0.9, 
                    line_width=1,
                    legendgroup='esn-fill', 
                )

                fig.add_scatter(
                    x=t_esn,
                    y=y_esn_mean[:, i] + y_esn_std[:, i], 
                    row=i+1, 
                    col=1, 
                    line_color='red', 
                    showlegend=i == 0, 
                    opacity=0.9, 
                    legendgroup='esn-fill', 
                    name='Ensemble mean ±σ', 
                    line_width=1,
                    fillcolor='rgba(255, 0, 0, 0.2)', fill='tonexty')

                if n_outputs == 3:
                    fig.add_scatter3d(
                        x=y_test[:, 0, 0],
                        y=y_test[:, 0, 1],
                        z=y_test[:, 0, 2],
                        row=1,
                        col=2,
                        line_color='black',
                        # line_dash='dash',
                        name=f'System data {i}',
                        legendgroup='Test',
                        showlegend=False,
                        mode='lines',
                    )
                elif n_outputs == 2:
                    fig.add_scatter(
                        x=y_test[:, 0, 0],
                        y=y_test[:, 0, 1],
                        row=1,
                        col=2,
                        line_color='black',
                        # line_dash='dash',
                        name=f'System data {i}',
                        legendgroup='Test',
                        showlegend=False,
                    )

            if n_outputs == 2:
                fig.update_xaxes(title_text=r'$\frac{dx}{dt}$', col=2)
                fig.update_yaxes(title_text='x', col=2)

        fig.update_yaxes(
            row=n_outputs+1,
            col=1,
            title_text='Absolute error',
        )
        # fig.update_yaxes(
        #     row=n_outputs+1,
        #     col=2,
        #     title_text='Absolute error (cumulative mean)',
        # )
        fig.update_yaxes(
            row=n_outputs+2,
            col=1,
            title_text='Square error',
        )
        # fig.update_yaxes(
        #     row=n_outputs+2,
        #     col=2,
        #     title_text='Square error (cumulative mean)',
        # )

        return fig


class SweepEnKF(Sweep):
    """AD-EnKF sweep."""

    def __init__(
        self,
        source: str,
        sweep_name: str, 
        sweep_vals: Iterable, 
        enkf_kwargs: dict = {}, 
        train_kwargs: dict = {}, 
        hidden_layers: list[int] = [6, 10, 6], 
        train_sweep: bool = True,
        notebook_display: bool = True,
        additional_states: int = 0,
        continuous_colour: bool = True,
        **kwargs
    ):
        """Initialise.

        Args:
            source (str): ODE source name.
            sweep_name (str): Name of single parameter to sweep.
            sweep_vals (Iterable): Values for swept parameter.
            enkf_kwargs (dict, optional): kwargs to pass to AD_EnKF 
                initialiser. Defaults to {}.
            train_kwargs (dict, optional): kwargs to pass to AD_EnKF.train. 
                Defaults to {}.
            hidden_layers (list[int], optional): Size of hidden RNN layers. 
                Defaults to [6, 10, 6].
            train_sweep (bool, optional): Add swept parameter to train_kwargs.
                Otherwise, add to enkf_kwargs. Defaults to True.
            notebook_display (bool, optional): When running in a notebook 
                context, show a live updating figure. Defaults to True.
            additional_states (int, optional): Number of latent states to use.
                Defaults to 0.
            continuous_colour (bool, optional): Use a different colour for 
                each EnKF in the plot. Otherwise, use same colours for networks
                with same hyperparameters. Defaults to True.
        """
        super().__init__(source=source, **kwargs)
        self.sweep_name = sweep_name
        self.sweep_vals = sweep_vals
        self.hidden_layers = hidden_layers
        self.train_sweep = train_sweep
        self.notebook_display = notebook_display
        self.additional_states = additional_states

        self.resolution = self.get_resolution(self.source)
        
        self.kfs = []

        self.enkf_kwargs = []
        self.train_kwargs = []

        enkf_kwargs_in = enkf_kwargs
        train_kwargs_in = train_kwargs

        for i, val in enumerate(sweep_vals):
            print(f'Creating AD-EnKF {i}')
            kwargs = {sweep_name: val}

            if train_sweep:
                train_kwargs_in = train_kwargs | kwargs
            else:
                enkf_kwargs_in = enkf_kwargs | kwargs

            self.enkf_kwargs.append(enkf_kwargs_in)
            self.train_kwargs.append(train_kwargs_in)

            kf = self.prepare_enkf(enkf_kwargs_in)
            kf.train(
                self.y_train, 
                1, 
                progress_fig=self.notebook_display, 
                display_fig=False, 
                obs_test=self.y_test, 
                dt=1/self.resolution, 
                **train_kwargs_in,
            )
            self.kfs.append(kf)

        if self.notebook_display:
            self.train_fig = self.combine_figs(
                self.kfs, 
                [f'{sweep_name}={val}' for val in self.sweep_vals],
                continuous_colour
            )
            display(self.train_fig)

    def next_kf_to_train(self, i_kfs: Optional[list[int]] = None) -> int:
        """Find next AD_EnKF to train, based on minimum number of epochs.

        Args:
            i_kfs (list[int], optional): List of AD_EnKF indices to select 
                from. If None, use all. Defaults to None.

        Returns:
            int: Index of AD_EnKF with lowest number of epochs.
        """
        if i_kfs is None:
            return np.argmin([kf._epoch for kf in self.kfs])
        else:
            i_train = np.argmin([kf._epoch for i, kf in enumerate(self.kfs) if i in i_kfs])
            return i_kfs[i_train]

    def train(self, to_epochs: int, i_kfs: Optional[list[int]] = None):
        """Train AD_EnKFs up to a set number of epochs.

        Args:
            to_epochs (int): Maximum number of epochs to train to.
            i_kfs (list[int], optional): List of AD_EnKF indices to select 
                from. If None, use all. Defaults to None.
        """
        while True:
            i = self.next_kf_to_train(i_kfs)
            if self.kfs[i]._epoch >= to_epochs:
                break
            self.kfs[i].train(
                self.y_train, 
                1, 
                progress_fig=self.notebook_display, 
                display_fig=False, 
                obs_test=self.y_test, 
                dt=1/self.resolution,
                **self.train_kwargs[i],
            )

    @staticmethod
    def combine_figs(
        kfs: list[kalman.AD_EnKF], 
        names: list[str], 
        continuous_colour: bool = True,
    ) -> go.FigureWidget:
        """Utility function to combine training figures from multiple AD_EnKFs.

        Args:
            kfs (list[kalman.AD_EnKF]): List of AD_EnKFs
            names (list[str]): List of display names.
            continuous_colour (bool, optional): Use a different colour for 
                each EnKF in the plot. Otherwise, use same colours for networks
                with same hyperparameters. Defaults to True.

        Returns:
            go.FigureWidget: Combined plot.
        """
        fig = go.FigureWidget(
            subplots.make_subplots(rows=3, cols=1, shared_xaxes=True))
        if continuous_colour:
            c = colors.sample_colorscale(
                colors.get_colorscale('Plasma'), 
                np.linspace(0, 0.9, len(kfs)),
            )
        else:
            _, inds = np.unique(names, return_inverse=True)
            c = [colors.qualitative.Plotly[i] for i in inds]

        for i, kf in enumerate(kfs):
            kf._lines[0].update(showlegend=True)
            kf._fig.update_traces(name=names[i], legendgroup=names[i], line_color=c[i])
            fig.add_traces(kf._lines, rows=[1, 1, 2, 2, 3], cols=1)
            kf._lines = fig.data[-len(kf._lines):]
            
            if i == 0:
                fig.layout = kf._fig.layout
                fig.update_layout(title_text='', height=600)

        return fig

    def plot_test(self, i: int, show_ensemble: bool = True) -> go.Figure:
        """Generate plot of AD_EnKF test performance.

        Args:
            i (int): Index of AD_EnKF to test.
            show_ensemble (bool, optional): Show trajectory of all particles. 
                Defaults to True.

        Returns:
            go.Figure: Plot of test performance.
        """
        kf = self.kfs[i]

        with torch.no_grad():
            ll, x = kf.log_likelihood(self.y_train, dt=1/self.resolution)
            x_enkf, y_enkf_raw = kf.predict(x[-1, :, :], self.n_test, True, False)
            x_0 = x[-1, :, :].mean(axis=0)
            x_enkf_no_noise, y_enkf_no_noise_raw = kf.predict(
                x_0[None, :], self.n_test, False, False)

        t_enkf = torch.arange(0, self.n_test) / self.resolution

        x_enkf_no_noise = x_enkf_no_noise[:, 0, :]
        y_enkf_no_noise_raw = y_enkf_no_noise_raw[:, 0, :]

        mse, mae, sq_err, abs_err, mse_lyap, mae_lyap = self.calc_err(y_enkf_no_noise_raw)
        
        y_test = (self.y_test * self.y_std) + self.y_mean
        y_enkf = (y_enkf_raw * self.y_std) + self.y_mean
        y_enkf_no_noise = (y_enkf_no_noise_raw * self.y_std) + self.y_mean

        y_test = y_test.cpu()
        x_enkf = x_enkf.cpu()
        y_enkf = y_enkf.cpu()
        x_enkf_no_noise = x_enkf_no_noise.cpu()
        y_enkf_no_noise = y_enkf_no_noise.cpu()

        x_enkf_mean = x_enkf.mean(axis=1)
        x_enkf_std = x_enkf.std(axis=1)
        y_enkf_mean = y_enkf.mean(axis=1)
        y_enkf_std = y_enkf.std(axis=1)

        n_ensemble = x.shape[1]
        n_outputs = y_enkf.shape[2]
        n_states = x_enkf.shape[2]
        n_subplots = max(n_outputs, n_states) + 2

        c = colors.sample_colorscale(
            colors.get_colorscale('Plotly3'),
            np.linspace(0, 0.9, n_ensemble),
        )

        if n_outputs <= 3:
            specs = [[{}, {'rowspan': n_subplots}]]
            if n_outputs > 2:
                specs[0][1]['type'] = 'scene'
            specs.extend([[{}, None]]*(n_subplots-1))
            cols = 2
        else:
            specs = None
            cols = 1
        fig = subplots.make_subplots(rows=n_subplots, cols=cols, shared_xaxes=True, specs=specs)

        y_max = y_test.max(axis=0).values
        y_min = y_test.min(axis=0).values
        # y_std = y_test.std(axis=0).numpy()
        y_upper = y_max + 0.5 * self.y_std
        y_lower = y_min - 0.5 * self.y_std

        fig.add_scatter(
            x=t_enkf,
            y=abs_err,
            row=n_subplots-1,
            col=1,
            line_color='black',
            name='Abs error',
            showlegend=False,
        )
        fig.add_scatter(
            x=t_enkf,
            y=sq_err,
            row=n_subplots,
            col=1,
            line_color='black',
            name='Square error',
            showlegend=False,
        )

        t_lyap = T_LYAPUNOV.get(self.source)
        if t_lyap is not None:
            t = t_lyap
            while t < t_enkf[-1]:
                fig.add_vline(x=t, row=n_outputs+1, col=1, line_color='black', line_dash='dot')
                fig.add_vline(x=t, row=n_outputs+2, col=1, line_color='black', line_dash='dot')
                t += t_lyap

        for j in range(n_subplots - 2):
            if j >= n_outputs:
                plot_enkf = x_enkf
                plot_enkf_mean = x_enkf_mean
                plot_enkf_std = x_enkf_std
                plot_enkf_no_noise = x_enkf_no_noise
            else:
                plot_enkf = y_enkf
                plot_enkf_mean = y_enkf_mean
                plot_enkf_std = y_enkf_std
                plot_enkf_no_noise = y_enkf_no_noise
                fig.update_yaxes(row=j+1, col=1, range=[y_lower[j], y_upper[j]])

            if show_ensemble:
                for i in range(kf.n_particles):
                    fig.add_scatter(
                        x=t_enkf,
                        y=plot_enkf[:, i, j], 
                        name='Ensemble mean ±σ', 
                        row=j+1, 
                        col=1, 
                        legendgroup='EnKF', 
                        showlegend=False, 
                        line_color=c[i],
                        opacity=0.3,
                    )

            if j < n_outputs:
                fig.add_scatter(
                    x=t_enkf,
                    y=y_test[:, j], 
                    name='System data', 
                    row=j+1,
                    col=1, 
                    line_color='black', 
                    legendgroup='system data', 
                    showlegend=False,
                )

            fig.add_scatter(
                x=t_enkf,
                y=plot_enkf_mean[:, j] - plot_enkf_std[:, j], 
                row=j+1, 
                col=1, 
                line_color='red', 
                showlegend=False, 
                opacity=0.9, 
                line_width=1,
                legendgroup='enkf-fill', 
            )

            fig.add_scatter(
                x=t_enkf,
                y=plot_enkf_mean[:, j] + plot_enkf_std[:, j],
                row=j+1, 
                col=1, 
                line_color='red', 
                showlegend=j == 0, 
                opacity=0.9, 
                legendgroup='enkf-fill', 
                name='Ensemble mean ±σ', 
                line_width=1,
                fillcolor='rgba(255, 0, 0, 0.2)', fill='tonexty')

            fig.add_scatter(
                x=t_enkf,
                y=plot_enkf_no_noise[:, j], 
                name='EnKF (no noise)', 
                row=j+1, 
                col=1, 
                legendgroup='EnKF (no noise)', 
                showlegend=False, 
                line_color='green',
            )

            # fig.add_vline(x=n_train, row=j+1, col=1)

        fig.update_layout(hovermode='x', height=800)
        fig.update_xaxes(title_text='t', col=1)

        if n_outputs == 3:
            if show_ensemble:
                for i in range(x.shape[1]):
                    fig.add_scatter3d(
                        x=y_enkf[:, i, 0], 
                        y=y_enkf[:, i, 1], 
                        z=y_enkf[:, i, 2], 
                        mode='lines',
                        name='EnKF', 
                        row=1, 
                        col=2, 
                        legendgroup='EnKF', 
                        showlegend=i == 1, 
                        opacity=0.3,
                        line_color=c[i],
                    )
            fig.add_scatter3d(
                x=y_test[:, 0],
                y=y_test[:, 1],
                z=y_test[:, 2],
                name='System data',
                line_color='black',
                # opacity=0.4,
                mode='lines',
                legendgroup='system data',
                showlegend=True,
                opacity=0.5,
                row=1,
                col=2,
            )
            fig.add_scatter3d(
                x=y_enkf_no_noise[:, 0], 
                y=y_enkf_no_noise[:, 1], 
                z=y_enkf_no_noise[:, 2], 
                mode='lines',
                name='EnKF (no noise)', 
                row=1, 
                col=2, 
                legendgroup='EnKF (no noise)', 
                showlegend=True, 
                line_color='green',
            )
            fig.update_scenes(
                row=1, col=2, 
                xaxis_range=[y_lower[0], y_upper[0]], 
                yaxis_range=[y_lower[1], y_upper[1]],
                zaxis_range=[y_lower[2], y_upper[2]],
            )

        elif n_outputs == 2:
            fig.update_xaxes(row=1, col=2, range=[y_lower[0], y_upper[0]])
            fig.update_yaxes(row=1, col=2, range=[y_lower[1], y_upper[1]])

            fig.add_scatter(
                x=y_test[:, 0],
                y=y_test[:, 1],
                name='System data',
                line_color='black',
                # opacity=0.4,
                mode='lines',
                legendgroup='system data',
                showlegend=True,
                row=1,
                col=2,
            )

            if show_ensemble:
                for i in range(x.shape[1]):
                    fig.add_scatter(
                        x=y_enkf[:, i, 0], 
                        y=y_enkf[:, i, 1], 
                        name='EnKF', 
                        row=1, 
                        col=2, 
                        legendgroup='EnKF', 
                        showlegend=i == 0, 
                        line_color=c[i],
                        opacity=0.3,
                    )
            fig.add_scatter(
                x=y_enkf_no_noise[:, 0], 
                y=y_enkf_no_noise[:, 1], 
                name='EnKF (no noise)', 
                row=1, col=2, 
                legendgroup='EnKF (no noise)', 
                showlegend=True, 
                line_color='green',
            )

            if self.source == 'vdp':
                fig.update_xaxes(title_text=r'$\frac{dx}{dt}$', col=2)
                fig.update_yaxes(title_text='x', col=2)
            else:
                fig.update_xaxes(title_text='x', col=2)
                fig.update_yaxes(title_text='y', col=2)

        fig.update_yaxes(
            row=n_subplots-1,
            col=1,
            title_text='Absolute error',
        )
        fig.update_yaxes(
            row=n_subplots,
            col=1,
            title_text='Square error',
        )

        # fig.add_vline(x=self.n_test_err, row=n_subplots-1, col=1)
        # fig.add_vline(x=self.n_test_err, row=n_subplots, col=1)

        return fig

    def prepare_enkf(self, kwargs: dict) -> kalman.AD_EnKF:
        """Generate an AD_EnKF with the parameterised arguments.

        Args:
            kwargs (dict): Extra rguments for AD_EnKF.__init__.

        Returns:
            kalman.AD_EnKF: Initialised AD_EnKF. 
        """
        return prepare_enkf(
            self.n_outputs, 
            self.resolution, 
            self.noise, 
            self.hidden_layers, 
            additional_states=self.additional_states, 
            **kwargs,
        )
    
    def enkf_err(self, i: int) -> tuple:
        """Calcualte test error on an AD_EnKF.

        Args:
            i (int): Index of AD_EnKF.

        Returns:
            tuple containing:
            - torch.Tensor: Average mean squared error over first 
                self.n_test_err points
            - torch.Tensor: Average mean absolute error over first
                self.n_test_err points
            - torch.Tensor: Square error over all dimensions for each point.
            - torch.Tensor: Absolute error for each point.
            - torch.Tensor: Square error over all dimensions averaged over each
                Lyapunov time.
            - torch.Tensor: Absolute error over all dimensions averaged over 
                each Lyapunov time.
        """
        kf = self.kfs[i]
        ll, x = kf.log_likelihood(self.y_train, dt=1/self.resolution)
        x_0 = x[-1, :, :].mean(axis=0)
        _, y_enkf_no_noise = kf.predict(x_0[None, :], self.n_test, False, False)
        return self.calc_err(y_enkf_no_noise[:, 0, :])


def prepare_enkf(
    n_outputs: int, 
    resolution: int, 
    noise: float, 
    hidden_layers: list[int], 
    n_particles: int = 20, 
    additional_states: int = 0, 
    activation_function: str = 'ReLU', 
    **kwargs
) -> kalman.AD_EnKF:
    """Generate an AD_EnKF with the parameterised arguments.

    Args:
        n_outputs (int): Number of outputs
        resolution (int): Resolution of data
        noise (float): Noise level of data
        hidden_layers (list[int]): Number of hidden layers of RNN.
        n_particles (int, optional): Number of AD-EnKF particles. Defaults to 
            20.
        additional_states (int, optional): Number of latent states. Defaults to
            0.
        activation_function (str, optional): Either ReLU, ELU or Tanh. Defaults
            to 'ReLU'.

    Returns:
        kalman.AD_EnKF: Initialised AD_EnKF. 
    """
    n_states = n_outputs + additional_states

    activation_functions = {
        'ELU': nn.ELU,
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
    }
    f = activation_functions[activation_function]

    n = kalman.EulerStepNet(
        n_states, 
        n_states, 
        hidden_layers, 
        dt=1/resolution,
        activation_function=f,
    ).to(settings.device)

    obs_noise = kalman.ScalarNoise(
        torch.tensor([noise], device=settings.device), 
        n_outputs,
    ).to(settings.device)
    proc_noise = kalman.ScalarNoise(
        torch.tensor([1], device=settings.device), 
        n_states,
    ).to(settings.device)
    obs_matrix = torch.eye(n_states, device=settings.device)[:n_outputs, :]

    kf = kalman.AD_EnKF(
        n, obs_matrix, obs_noise, proc_noise, n_particles, **kwargs
    )

    return kf
