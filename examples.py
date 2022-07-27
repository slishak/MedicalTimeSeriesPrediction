import torch
import numpy as np
from plotly import subplots

from time_series_prediction import echo_state_network, ode_problems, settings

def lorenz_esn():
    source = 'lorenz'
    n_train = 3000
    n_test = 2000
    resolution = 50
    noise = 1e-1

    y_full, u_full, t_out = ode_problems.generate_data(source, n_train+n_test, resolution, noise)

    y_train = y_full[:n_train, :]
    u_train = u_full[:n_train, :]

    y_test = y_full[n_train:, :]
    u_test = u_full[n_train:, :]

    # torch.manual_seed(5)
    esn = echo_state_network.ESN(
        n_inputs=0, 
        n_outputs=y_train.shape[1],
        n_neurons=1000,
        spectral_radius=0.99,
        f_activation=torch.tanh,
        )
    x_train = esn.train(
        u_train.to(settings.device), 
        y_train[:, :].to(settings.device), 
        n_discard=500, 
        k_l2=1e-1,
    )

    x_init = x_train[-1, :]
    y_init = y_train[-1, :]
    x_test_esn, y_test_esn = esn.predict(
        u_test.to(settings.device), 
        x_init, 
        y_init.to(settings.device),
    )

    x_esn = np.concatenate([x_train.cpu(), x_test_esn])
    y_esn = np.concatenate([(x_train @ esn.output_weights).cpu(), y_test_esn])

    rows = esn.n_outputs # + 1
    specs = [[{}, {"rowspan": rows, "type": "scene"}]]
    specs.extend([[{}, None]]*(rows-1))
    fig = subplots.make_subplots(
        rows=rows, cols=2, 
        specs=specs,
        shared_xaxes=True,
        )
    for i in range(esn.n_outputs):
        fig.add_scatter(y=y_full[:, i], name='system data', row=i+1, col=1, line_color='black', legendgroup='systemdata', showlegend=i==0)
        fig.add_scatter(y=y_esn[:, i], name='ESN', row=i+1, col=1, line_color='red', legendgroup='esn', showlegend=i==0)
        fig.add_vline(x=n_train, row=i+1, col=1)
    if rows > esn.n_outputs:
        for x_neuron in x_esn.T:
            fig.add_scatter(y=x_neuron, showlegend=False, row=3, col=1)
    fig.update_layout(hovermode='x')
    fig.update_xaxes(title_text='n')

    fig.add_scatter3d(
        x=y_full[:, 0],
        y=y_full[:, 1],
        z=y_full[:, 2],
        name='system data',
        line_color='black',
        opacity=0.4,
        mode='lines',
        legendgroup='systemdata',
        showlegend=False,
        row=1,
        col=2,
    )
    fig.add_scatter3d(
        x=y_esn[n_train:, 0],
        y=y_esn[n_train:, 1],
        z=y_esn[n_train:, 2],
        name='ESN',
        line_color='red',
        mode='lines',
        legendgroup='esn',
        showlegend=False,
        row=1,
        col=2,
    )
    fig.update_layout(height=800)


    return fig

if __name__ == '__main__':
    fig = lorenz_esn()
    fig.write_html('Lorenz ESN.html', auto_open=True)