import os
from tempfile import mkdtemp, mkstemp
from time import perf_counter
from typing import Iterable, Optional
from itertools import product

import torch
import numpy as np
from plotly import subplots, colors, graph_objects as go
from torch import nn
from IPython.display import display
from dask.distributed import Client, progress

from time_series_prediction import kalman, echo_state_network, settings, ode_problems


def param_sweep(params: list[tuple[str, list]]):
    keys, vals = zip(*params)
    d = [{k: v for k, v in zip(keys, val)} for val in product(*vals)]
    print(f'{len(d)} simulations')
    return d

class Sweep:
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
    def set_resolution(source):
        if source in ('vdp', 'rossler'):
            return 10
        else:
            return 50

    def generate_data(self, source=None):
        if source is None:
            source = self.source
        y, _, _ = ode_problems.generate_data(
            source, 
            self.n_warmup+self.n_train+self.n_test, 
            self.set_resolution(source),
            init_noise=self.init_noise,
        )
        y = y.to(settings.device)

        return y

    def unscale(self, y, y_mean=None, y_std=None):
        if y_mean is None:
            y_mean = self.y_mean
            y_std = self.y_std

        y_unscaled = (y * y_std) + y_mean

        return y_unscaled

    def split_train_test(self, y):
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

    def calc_err(self, y_pred, y_test=None):
        if y_test is None:
            y_test = self.y_test
        sq_err = (y_pred - y_test)**2
        
        # Prevent NaN error
        sq_err[torch.isnan(sq_err) | (torch.abs(sq_err) > 1e10)] = 1e10

        # Sum over all output dimensions
        sq_err = torch.sqrt(sq_err.sum(axis=1))

        # Absolute error
        abs_err = torch.sqrt(sq_err)
        
        # Metrics
        mse = torch.mean(sq_err[:self.n_test_err])
        mae = torch.mean(abs_err[:self.n_test_err])

        return mse, mae, sq_err, abs_err


class SweepESN(Sweep):
    def __init__(
        self,
        param_dicts: list[dict],
        esn_kwargs: dict = {}, 
        train_kwargs: dict = {},
        n_ensemble: int = 10,
        train_sweep: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.param_dicts = param_dicts
        self.esn_kwargs = esn_kwargs
        self.train_kwargs = train_kwargs
        self.n_ensemble = n_ensemble
        self.train_sweep = train_sweep

        self.plot_folder = mkdtemp(prefix='esn-ensemble-')

    def submit_jobs(self, client: Client, show_progress=True):
        self.jobs = []
        esn_kwargs = self.esn_kwargs
        train_kwargs = self.train_kwargs
        for d in self.param_dicts:
            if self.train_sweep:
                train_kwargs = train_kwargs | d
            else:
                esn_kwargs = esn_kwargs | d
            self.jobs.append(client.submit(self.run_ensemble, esn_kwargs, train_kwargs))

        if show_progress:
            progress(self.jobs)

    def get_results(self):
        rows = []
        for task, d in zip(self.jobs, self.param_dicts):
            mae, mse, fig_path = task.result()
            row = {'MAE': mae, 'MSE': mse, '_fig_path': fig_path}
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
            
    def train_esn(self, esn_kwargs, train_kwargs):
        t1 = perf_counter()

        if self.y_train is None:
            y_full = self.generate_data(self.source or esn_kwargs['source'])
            y_train, y_test, y_mean, y_std = self.split_train_test(y_full)
        else:
            y_train = self.y_train
            y_test = self.y_test
            y_mean = self.y_mean
            y_std = self.y_std
            n_outputs = self.n_outputs

        esn = echo_state_network.ESN(
            n_inputs=0, 
            n_outputs=n_outputs,
            **esn_kwargs
        )

        u_train = torch.empty((0, self.n_train), device=settings.device)
        u_test = torch.empty((0, self.n_test), device=settings.device)

        x_train = esn.train(
            u_train, 
            y_train,
            **train_kwargs
        )
        x_init = x_train[-1, :]
        y_init = y_train[-1, :]
        _, y_test_esn = esn.predict(
            u_test, 
            x_init, 
            y_init,
        )

        mse, mae, sq_err, abs_err = self.calc_err(y_test_esn, y_test)

        y_esn_unscaled = (y_test_esn * y_std) + y_mean
        y_test_unscaled = (y_test * y_std) + y_mean

        t2 = perf_counter()
        print(f't={t2-t1:.3f}s')

        return mse, mae, sq_err, abs_err, y_esn_unscaled, y_test_unscaled
        
    def run_ensemble(self, esn_kwargs, train_kwargs):
        y_esn_list = []
        y_test_list = []
        abs_err_list = []
        sq_err_list = []
        mae_list = []
        mse_list = []

        for i in range(self.n_ensemble):
            print(f'{i+1}/{self.n_ensemble}')

            mse, mae, sq_err, abs_err, y_esn_unscaled, y_test_unscaled = self.train_esn(
                esn_kwargs, train_kwargs,
            )

            y_esn_list.append(y_esn_unscaled)
            y_test_list.append(y_test_unscaled)
            abs_err_list.append(abs_err)
            sq_err_list.append(sq_err)
            mae_list.append(mae)
            mse_list.append(mse)

        mae = torch.stack(mae_list)
        mse = torch.stack(mse_list)
        
        print(f'MAE {torch.mean(mae)} ± {torch.std(mae)}')
        print(f'MSE {torch.mean(mse)} ± {torch.std(mse)}')

        fig = self.plot_ensemble(y_esn_list, y_test_list, abs_err_list, sq_err_list)

        fd, fig_path = mkstemp('.json', 'esn-', self.plot_folder)
        fig.write_json(fig_path)
        os.close(fd)

        return mae, mse, fig_path

    def plot_ensemble(self, y_esn_list, y_test_list, abs_err_list, sq_err_list):
        
        c = colors.sample_colorscale(
            colors.get_colorscale('Plotly3'), 
            np.linspace(0, 0.9, self.n_ensemble),
        )
        fig = subplots.make_subplots(
            rows=self.n_outputs+2, cols=2, shared_xaxes='all',
        )

        for i in range(self.n_ensemble):
            for j in range(self.n_outputs):
                fig.add_scatter(
                    # x=t_out,
                    y=y_esn_list[i][:, j],
                    row=j+1,
                    col=1,
                    line_color=c[i],
                    name=f'ESN {i}',
                    legendgroup=f'ESN {i}',
                    showlegend=False,
                )
                fig.add_scatter(
                    # x=t_out,
                    y=sq_err_list[i][:, j],
                    row=j+1,
                    col=2,
                    line_color=c[i],
                    name=f'ESN {i}',
                    legendgroup=f'ESN {i}',
                    showlegend=False,
                )
                if self.init_noise != 0:
                    fig.add_scatter(
                        # x=t_out,
                        y=y_test_list[i][:, j],
                        row=j+1,
                        col=1,
                        line_color=c[i],
                        line_dash='dash',
                        name=f'test data {i}',
                        legendgroup=f'ESN {i}',
                        showlegend=False,
                    )

            fig.add_scatter(
                y=abs_err_list[i],
                row=self.n_outputs+1,
                col=1,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=True,
            )
            fig.add_scatter(
                y=sq_err_list[i],
                row=self.n_outputs+2,
                col=1,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=False,
            )
            fig.add_scatter(
                y=np.cumsum(abs_err_list[i]) / np.arange(1, len(abs_err_list[i])+1),
                row=self.n_outputs+1,
                col=2,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=False,
            )
            fig.add_scatter(
                y=np.cumsum(sq_err_list[i]) / np.arange(1, len(sq_err_list[i])+1),
                row=self.n_outputs+2,
                col=2,
                line_color=c[i],
                name=f'ESN {i}',
                legendgroup=f'ESN {i}',
                showlegend=False,
            )

        for i in range(self.n_outputs):
            fig.update_yaxes(
                row=i+1,
                col=1,
                title_text=f'Dimension {i+1}'
            )
            fig.update_yaxes(
                row=i+1,
                col=2,
                title_text=f'Square error'
            )

            if self.init_noise == 0:
                fig.add_scatter(
                    # x=t_out,
                    y=y_test_list[0][:, i],
                    row=i+1,
                    col=1,
                    line_color='black',
                    line_dash='dash',
                    name=f'test data',
                    legendgroup=f'Test',
                    showlegend=i==0,
                )

        fig.update_yaxes(
            row=self.n_outputs+1,
            col=1,
            title_text='Absolute error',
        )
        fig.update_yaxes(
            row=self.n_outputs+1,
            col=2,
            title_text='Absolute error (cumulative mean)',
        )
        fig.update_yaxes(
            row=self.n_outputs+2,
            col=1,
            title_text='Square error',
        )
        fig.update_yaxes(
            row=self.n_outputs+2,
            col=2,
            title_text='Square error (cumulative mean)',
        )

        fig.add_vline(x=self.n_test_err, row=self.n_outputs+1, col=2)
        fig.add_vline(x=self.n_test_err, row=self.n_outputs+2, col=2)

        return fig


class SweepEnKF(Sweep):
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

        super().__init__(source=source, **kwargs)
        self.sweep_name = sweep_name
        self.sweep_vals = sweep_vals
        self.enkf_kwargs = enkf_kwargs
        self.train_kwargs = train_kwargs
        self.hidden_layers = hidden_layers
        self.train_sweep = train_sweep
        self.notebook_display = notebook_display
        self.additional_states = additional_states

        self.resolution = self.set_resolution(self.source)
        
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

    def next_kf_to_train(self, i_kfs=None):
        if i_kfs is None:
            return np.argmin([kf._epoch for kf in self.kfs])
        else:
            return np.argmin([kf._epoch for i, kf in enumerate(self.kfs) if i in i_kfs])

    def train(self, to_epochs, i_kfs=None):
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
    def combine_figs(kfs, names, continuous_colour=True):
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

    def plot_test(self, i, show_ensemble=True):

        kf = self.kfs[i]

        with torch.no_grad():
            ll, x = kf.log_likelihood(self.y_train, dt=1/self.resolution)
            x_enkf, y_enkf_raw = kf.predict(x[-1, :, :], self.n_test, True, False)
            x_0 = x[-1, :, :].mean(axis=0)
            x_enkf_no_noise, y_enkf_no_noise_raw = kf.predict(x_0[None, :], self.n_test, False, False)

        x_enkf_no_noise = x_enkf_no_noise[:, 0, :]
        y_enkf_no_noise_raw = y_enkf_no_noise_raw[:, 0, :]

        mse, mae, sq_err, abs_err = self.calc_err(y_enkf_no_noise_raw)
        
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
                    y=y_test[:, j], 
                    name='system data', 
                    row=j+1,
                    col=1, 
                    line_color='black', 
                    legendgroup='system data', 
                    showlegend=False,
                )

            fig.add_scatter(
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
                y=plot_enkf_mean[:, j] + plot_enkf_std[:, j],
                row=j+1, 
                col=1, 
                line_color='red', 
                showlegend=(j==0), 
                opacity=0.9, 
                legendgroup='enkf-fill', 
                name='Ensemble mean ±σ', 
                line_width=1,
                fillcolor='rgba(255, 0, 0, 0.2)', fill='tonexty')

            fig.add_scatter(
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
                        showlegend=(i==1), 
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
                        showlegend=(i==0), 
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

        fig.add_scatter(
            y=abs_err,
            row=n_subplots-1,
            col=1,
            line_color='black',
            name=f'Abs error',
            showlegend=False,
        )
        fig.add_scatter(
            y=sq_err,
            row=n_subplots,
            col=1,
            line_color='black',
            name=f'Square error',
            showlegend=False,
        )

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

    def prepare_enkf(self, kwargs):
        return prepare_enkf(
            self.n_outputs, 
            self.resolution, 
            self.noise, 
            self.hidden_layers, 
            additional_states=self.additional_states, 
            **kwargs,
        )
    
    def enkf_err(self, i):
        kf = self.kfs[i]
        ll, x = kf.log_likelihood(self.y_train, dt=1/self.resolution)
        x_0 = x[-1, :, :].mean(axis=0)
        _, y_enkf_no_noise = kf.predict(x_0[None, :], self.n_test_err, False, False)
        return self.calc_err(y_enkf_no_noise)

def prepare_enkf(
    n_outputs, 
    resolution, 
    noise, 
    hidden_layers, 
    n_particles=20, 
    additional_states=0, 
    activation_function='relu', 
    **kwargs
):
    
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
