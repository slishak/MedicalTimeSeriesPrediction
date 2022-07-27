
import pandas as pd
import torch
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid

from biomechanical_models.models import SmithCardioVascularSystem, JallonHeartLungs, InertialSmithCVS
from biomechanical_models.unit_conversions import convert


def plot_states(cvs, t_sol, sol, t, x, dx_dt):
    fig = make_subplots(len(cvs.state_names), 2, column_titles=['states', 'derivatives'], shared_xaxes='all')
    for i, name in enumerate(cvs.state_names):
        fig.add_scatter(x=t_sol, y=sol[:, i], line_color='black', name='grid', row=i+1, col=1, showlegend=False)
        fig.add_scatter(x=t, y=x[:, i], line_color='red', name='steps', row=i+1, col=1, showlegend=False)
        fig.add_scatter(x=t, y=df[f'd{name}_dt'], line_color='blue', name='derivative', row=i+1, col=2, showlegend=False)
        fig.update_yaxes(title_text=name, row=i+1, col=1)
        fig.update_yaxes(title_text=f'd{name}_dt', row=i+1, col=2)
    return fig


def plot_outputs(df):
    specs = [
        [{'colspan': 2}, None, {'colspan': 2}, None],
        [{'colspan': 2}, None, {'colspan': 2}, None],
        [{'colspan': 2}, None, {}, {}],
        [{'colspan': 2}, None, {'colspan': 2}, None],
        [{'colspan': 2}, None, {'colspan': 2}, None],
    ]
    fig = make_subplots(len(specs), 4, specs=specs)
    fig.update_layout(hovermode='x')
    fig.update_xaxes(matches='x1')
    fig.update_xaxes(row=3, col=3, matches=None)
    fig.update_xaxes(row=3, col=4, matches=None)

    fig.update_yaxes(row=1, col=1, title_text='lvf/lv/ao/pu pressures (mmHg)')
    for col in ['p_lvf', 'p_lv', 'p_ao', 'p_pu']:
        fig.add_scatter(x=df['t'], y=convert(df[col], to='mmHg'), name=col, row=1, col=1)

    fig.update_yaxes(row=1, col=3, title_text='lvf/lv/ao/pu volumes (ml)')
    for col in ['v_lvf', 'v_lv', 'v_ao', 'v_pu']:
        fig.add_scatter(x=df['t'], y=convert(df[col], 'l', 'ml'), name=col, row=1, col=3)

    fig.update_yaxes(row=2, col=1, title_text='rvf/rv/pa/vc pressures (mmHg)')
    for col in ['p_rvf', 'p_rv', 'p_pa', 'p_vc']:
        fig.add_scatter(x=df['t'], y=convert(df[col], to='mmHg'), name=col, row=2, col=1)

    fig.update_yaxes(row=2, col=3, title_text='rvf/rv/pa/vc volumes (ml)')
    for col in ['v_rvf', 'v_rv', 'v_pa', 'v_vc']:
        fig.add_scatter(x=df['t'], y=convert(df[col], to='ml'), name=col, row=2, col=3)

    fig.update_yaxes(row=3, col=1, title_text='Flow rates (l/s)')
    for col in ['q_mt', 'q_av', 'q_tc', 'q_pv', 'q_pul', 'q_sys']:
        fig.add_scatter(x=df['t'], y=df[col], name=col, row=3, col=1)

    fig.update_xaxes(row=3, col=3, title_text='v_lv')
    fig.update_yaxes(row=3, col=3, title_text='p_lv')
    fig.update_xaxes(row=3, col=4, title_text='v_rv')
    fig.update_yaxes(row=3, col=4, title_text='p_rv')
    fig.add_scatter(x=convert(df['v_lv'], to='ml'), y=convert(df['p_lv'], to='mmHg'), name='lv', row=3, col=3)
    fig.add_scatter(x=convert(df['v_rv'], to='ml'), y=convert(df['p_rv'], to='mmHg'), name='rv', row=3, col=4)
    #fig.add_scatter(x=df['v_spt'], y=df['p_spt'], name='spt', row=1, col=2)
    
    fig.update_yaxes(row=4, col=1, title_text='Pericardium pressures (mmHg)')
    for col in ['p_pcd', 'p_peri']:
        fig.add_scatter(x=df['t'], y=convert(df[col], to='mmHg'), name=col, row=4, col=1)

    fig.update_yaxes(row=4, col=3, title_text='Pericardium volume (ml)')
    fig.add_scatter(x=df['t'], y=convert(df['v_pcd'], to='ml'), name='v_pcd', row=4, col=3)

    fig.update_yaxes(row=5, col=1, title_text='Cardiac driver')
    fig.add_scatter(x=df['t'], y=df['e_t'], name='e_t', row=5, col=1)

    fig.update_yaxes(row=5, col=3, title_text='Septum volume (ml)')
    fig.add_scatter(x=df['t'], y=convert(df['v_spt'], to='ml'), name='v_spt', row=5, col=3)
    return fig

if __name__ == '__main__':
    # cvs = SmithCardioVascularSystem()
    cvs = JallonHeartLungs()
    t_sol, sol = cvs.simulate()

    t, x, dx_dt, outputs = zip(*cvs.trajectory)
    t = torch.tensor(t)
    x = torch.stack(x)
    dx_dt = torch.stack(dx_dt)
    df = pd.DataFrame(outputs).apply(lambda s: [float(x) for x in s])
    df['t'] = t

    fig_states = plot_states()
    fig_states.write_html('states.html', auto_open=True)

    fig = plot_outputs(df)
    fig.write_html('cvs.html', auto_open=True)

    # Respiratory system plot
    # if isinstance(cvs, JallonHeartLungs):
    #     fig_r = make_subplots(rows=4, cols=1)
    #     fig_r.add_scatter(x=df['t'], y=df['x'], name='x', row=1, col=1)
    #     fig_r.add_scatter(x=df['t'], y=df['y'], name='y', row=1, col=1)
        
    #     fig_r.add_scatter(x=df['t'], y=df['p_mus'], name='p_mus', row=3, col=1)
    #     fig_r.add_scatter(x=df['t'], y=cumulative_trapezoid(df['p_mus'], df['t']), name='int(p_mus)', row=3, col=1)


    #     fig_r.add_scatter(x=df['t'], y=df['dv_alv_dt'], name='dv_alv_dt', row=4, col=1)
    #     fig_r.add_scatter(x=df['t'], y=-df['p_pl']/(cvs.resp.r_ca.item() + cvs.resp.r_ua.item()), name='p_pl contrib', row=4, col=1)
    #     fig_r.add_scatter(x=df['t'], y=-df['v_alv']*cvs.resp.e_alv.item()/(cvs.resp.r_ca.item() + cvs.resp.r_ua.item()), name='v_alv contrib', row=4, col=1)
    #     fig_r.write_html('resp.html', auto_open=True)




