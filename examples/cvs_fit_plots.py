import pickle as pkl

import torch
import pandas as pd
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from biophysical_models import models
from cvs_example import latex
import cvs_fit

def get_trajectory_df(cvs):
    t, x, dx_dt, outputs = zip(*cvs.trajectory)
    t = torch.tensor(t)
    x = torch.stack(x)
    df = pd.DataFrame(outputs).apply(lambda s: [float(x) for x in s])
    df['t'] = t

    # Downsample
    df = df.iloc[::50]

    return df

def generate_plot(plot_data):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)

    hr_plotted = False
    for color, name, df_ in plot_data:
        fig.add_scatter(
            x=df_['t'], y=df_['p_aom'], name=f"{name}, mean", row=1, col=1, line_color=color, legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_aos'], name=f"{name}, systolic", row=1, col=1, line_color=color, line_dash='dot', legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_aod'], name=f"{name}, diastolic", row=1, col=1, line_color=color, line_dash='dash', legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_pam'], showlegend=False, row=2, col=1, line_color=color, legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_pas'], showlegend=False, row=2, col=1, line_color=color, line_dash='dot', legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_pad'], showlegend=False, row=2, col=1, line_color=color, line_dash='dash', legendgroup=name,
        )
        fig.add_scatter(
            x=df_['t'], y=df_['p_vcm'], showlegend=False, row=3, col=1, line_color=color, legendgroup=name,
        )
        if not hr_plotted:
            try:
                fig.add_scatter(
                    x=df_['t'], y=df_['hr'], showlegend=False, row=4, col=1, line_color='black', legendgroup='Patient',
                )
            except KeyError:
                pass
            else:
                hr_plotted = True

    fig.update_yaxes(
        row=1, title_text=latex('P_{ao}')
    )
    fig.update_yaxes(
        row=2, title_text=latex('P_{pa}')
    )
    fig.update_yaxes(
        row=3, title_text=latex('P_{vc}')
    )
    fig.update_yaxes(
        row=4, title_text='Heart rate (bpm)'
    )
    fig.update_xaxes(
        row=4, title_text='Time (s)'
    )

    return fig


def simulate(cvs):
    with torch.no_grad():
        cvs.simulate(500, 50)
    df = get_trajectory_df(cvs)
    return df


if __name__ == '__main__':

    try:
        # Open existing data - delete file to re-run
        with open('plot_data.pkl', 'rb') as f:
            plot_data = pkl.load(f)
    except FileNotFoundError:
        # Run simulations

        # Instantiate CVS model and load patient data
        cvs_cls = models.add_bp_metrics(models.SmithCardioVascularSystem)
        y, obs_matrix, cvs = cvs_fit.load_data(cvs_cls)
        out_inds = torch.where(obs_matrix)[1]
        out_names = [cvs.state_names[x] for x in out_inds]
        cvs.save_traj = True
        cvs.verbose = True

        # Dataframe of patient data
        df_patient = pd.DataFrame([
            {name: y[i+1, j]
            for j, name in enumerate(out_names)
            }
            for i in range(500)
        ])
        df_patient['t'] = torch.arange(1, 501)

        # Load CVS parameters after 20 epochs
        params2 = torch.load('params20.to')

        # With initial parameters
        df = simulate(cvs)

        # After 20 epochs of parameter estimation
        cvs.load_state_dict(params2)
        df2 = simulate(cvs)

        # Collate plot data: list of (colour, name, df)
        plot_data = [
            ('black', 'Patient', df_patient),
            (DEFAULT_PLOTLY_COLORS[0], 'Before training', df),
            (DEFAULT_PLOTLY_COLORS[1], 'After 20 epochs', df2),
        ]

        # Save for refernece (expensive to generate)
        with open('plot_data.pkl', 'wb') as f:
            pkl.dump(plot_data, f)

    fig = generate_plot(plot_data)
    fig.write_html('comparison.html', auto_open=True, include_mathjax='cdn')