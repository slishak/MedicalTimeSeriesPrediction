import pandas as pd
import torch
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly import colors

from biophysical_models.models import SmithCardioVascularSystem, JallonHeartLungs, InertialSmithCVS, add_bp_metrics
from biophysical_models.unit_conversions import convert
import cvs_fit

pio.templates.default = "plotly_white"

X_RANGE = [6,9]
# X_RANGE = [30, 50]

def plot_states(cvs, df, t, x):
    fig = make_subplots(
        len(cvs.state_names), 2, 
        # column_titles=['states', 'derivatives'], 
        shared_xaxes='all'
    )
    fig.update_xaxes(range=X_RANGE)
    fig.update_xaxes(row=len(cvs.state_names), title_text='Time (s)')
    showlegend_grid = False
    showlegend_steps = False
    for i, name in enumerate(cvs.state_names):
        # fig.add_scatter(
        #     x=t_sol, 
        #     y=sol[:, i], 
        #     line_color='red', 
        #     name='grid', 
        #     row=i+1, col=1, 
        #     showlegend=showlegend_grid, legendgroup='grid',
        # )
        fig.add_scatter(
            x=t, 
            y=x[:, i], 
            line_color='black', 
            name='steps', 
            row=i+1, col=1, 
            showlegend=showlegend_steps, legendgroup='steps',
        )
        showlegend_grid = False
        showlegend_steps = False
        fig.add_scatter(
            x=t, 
            y=df[f'd{name}_dt'], 
            line_color='black', 
            name='steps', 
            row=i+1, col=2, 
            showlegend=showlegend_steps, legendgroup='steps',
        )
        latex_name = name.replace('v_', 'V_').replace('p_', 'P_').replace('q_', 'Q_')
        if '_' in latex_name:
            latex_name = latex_name.replace('_', '_{') + '}'
        units = {
            'v': 'l',
            'q': 'l/s',
            'p': 'mmHg',
        }
        unit = units.get(name[0])
        state_title = latex(latex_name)
        deriv_title = latex(rf'\frac{{d{latex_name}}}{{dt}}')
        if unit is not None:
            state_title = state_title + f' ({unit})'
            deriv_title = deriv_title + f' ({unit}/s)'
        fig.update_yaxes(title_text=state_title, row=i+1, col=1)
        fig.update_yaxes(title_text=deriv_title, row=i+1, col=2)
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
    for col in ['p_lvf', 'p_lv', 'p_ao', 'p_pu', 'p_aom', 'p_aos', 'p_aod']:
        try:
            fig.add_scatter(x=df['t'], y=convert(df[col], to='mmHg'), name=col, row=1, col=1)
        except KeyError:
            pass

    fig.update_yaxes(row=1, col=3, title_text='lvf/lv/ao/pu volumes (ml)')
    for col in ['v_lvf', 'v_lv', 'v_ao', 'v_pu']:
        fig.add_scatter(x=df['t'], y=convert(df[col], 'l', 'ml'), name=col, row=1, col=3)

    fig.update_yaxes(row=2, col=1, title_text='rvf/rv/pa/vc pressures (mmHg)')
    for col in ['p_rvf', 'p_rv', 'p_pa', 'p_vc', 'p_vcm']:
        try:
            fig.add_scatter(x=df['t'], y=convert(df[col], to='mmHg'), name=col, row=2, col=1)
        except KeyError:
            pass

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


def plot_lv_pressures(df):

    # df = df.iloc[200:]

    fig = make_subplots(2, 1, shared_xaxes=True)
    fig.update_xaxes(range=X_RANGE)
    fig.update_layout(hovermode='x')
    fig.update_yaxes(row=1, col=1, title_text='Pressure (mmHg)')
    fig.update_yaxes(row=2, col=1, title_text='Flow rates (l/s)')
    fig.update_xaxes(row=2, col=1, title_text='Time (s)')

    fig.add_scatter(x=df['t'], y=convert(df['p_lv'], to='mmHg'), name=latex('P_{lv}'), row=1, col=1)
    fig.add_scatter(x=df['t'], y=convert(df['p_ao'], to='mmHg'), name=latex('P_{ao}'), row=1, col=1)
    fig.add_scatter(x=df['t'], y=convert(df['p_pu'], to='mmHg'), name=latex('P_{pu}'), row=1, col=1)

    fig.add_scatter(x=df['t'], y=df['q_mt'], name=latex('Q_{mt}'), row=2, col=1)
    fig.add_scatter(x=df['t'], y=df['q_av'], name=latex('Q_{av}'), row=2, col=1)
    fig.add_scatter(x=df['t'], y=df['q_sys'], name=latex('Q_{sys}'), row=2, col=1)

    try:
        fig.add_scatter(x=df['t'], y=convert(df['p_aom'], to='mmHg'), name=latex('P_{ao,m}'), line_color='black', row=1, col=1)
        fig.add_scatter(x=df['t'], y=convert(df['p_aos'], to='mmHg'), name=latex('P_{ao,s}'), line_color='black', line_dash='dot', row=1, col=1)
        fig.add_scatter(x=df['t'], y=convert(df['p_aod'], to='mmHg'), name=latex('P_{ao,d}'), line_color='black', line_dash='dash', row=1, col=1)
    except KeyError:
        pass

    
    return fig

def plot_rv_pressures(df):

    # df = df.iloc[200:]

    fig = make_subplots(2, 1, shared_xaxes='all')
    fig.update_xaxes(range=X_RANGE)
    fig.update_layout(hovermode='x')
    fig.update_yaxes(row=1, col=1, title_text='Pressure (mmHg)')
    fig.update_yaxes(row=2, col=1, title_text='Flow rates (l/s)')
    fig.update_xaxes(row=2, col=1, title_text='Time (s)')

    fig.add_scatter(x=df['t'], y=convert(df['p_rv'], to='mmHg'), name=latex('P_{rv}'), row=1, col=1)
    fig.add_scatter(x=df['t'], y=convert(df['p_pa'], to='mmHg'), name=latex('P_{pa}'), row=1, col=1)
    fig.add_scatter(x=df['t'], y=convert(df['p_vc'], to='mmHg'), name=latex('P_{vc}'), row=1, col=1)

    fig.add_scatter(x=df['t'], y=df['q_tc'], name=latex('Q_{tc}'), row=2, col=1)
    fig.add_scatter(x=df['t'], y=df['q_pv'], name=latex('Q_{pv}'), row=2, col=1)
    fig.add_scatter(x=df['t'], y=df['q_pul'], name=latex('Q_{pul}'), row=2, col=1)

    try:
        fig.add_scatter(x=df['t'], y=convert(df['p_pam'], to='mmHg'), name=latex('P_{pa,m}'), line_color='black', row=1, col=1)
        fig.add_scatter(x=df['t'], y=convert(df['p_pas'], to='mmHg'), name=latex('P_{pa,s}'), line_color='black', line_dash='dot', row=1, col=1)
        fig.add_scatter(x=df['t'], y=convert(df['p_pad'], to='mmHg'), name=latex('P_{pa,d}'), line_color='black', line_dash='dash', row=1, col=1)
        fig.add_scatter(x=df['t'], y=convert(df['p_vcm'], to='mmHg'), name=latex('P_{vc,m}'), line_color='black', line_dash='dashdot', row=1, col=1)
    except KeyError:
        pass

    return fig

def plot_vent_interaction(df, df2=None):
    
    specs = [
        [{}, {'rowspan': 3}],
        [{}, None],
        [{}, None],
    ]

    fig = make_subplots(3, 2, shared_xaxes='columns', specs=specs)

    fig.update_xaxes(range=X_RANGE, col=1)
    fig.update_xaxes(row=3, col=1, title_text='Time (s)')
    fig.update_yaxes(row=1, col=1, title_text='Ventricle volume (ml)')
    fig.add_scatter(x=df['t'], y=convert(df['v_lv'], to='ml'), name='Left', row=1, col=1, showlegend=False, line_color=colors.DEFAULT_PLOTLY_COLORS[0])
    fig.add_scatter(x=df['t'], y=convert(df['v_rv'], to='ml'), name='Right', row=1, col=1, showlegend=False, line_color=colors.DEFAULT_PLOTLY_COLORS[1])
    if df2 is not None:
        fig.add_scatter(x=df2['t'], y=convert(df2['v_lv'], to='ml'), name='Left (linear)', row=1, col=1, showlegend=False, line_color=colors.DEFAULT_PLOTLY_COLORS[0], line_dash='dot')
        fig.add_scatter(x=df2['t'], y=convert(df2['v_rv'], to='ml'), name='Right (linear)', row=1, col=1, showlegend=False, line_color=colors.DEFAULT_PLOTLY_COLORS[1], line_dash='dot')

    fig.update_yaxes(row=2, col=1, title_text='Cardiac driver')
    fig.add_scatter(x=df['t'], y=df['e_t'], name=latex('e(t)'), row=2, col=1, showlegend=False, line_color='black')

    fig.update_yaxes(row=3, col=1, title_text='Septum volume (ml)')
    fig.add_scatter(x=df['t'], y=convert(df['v_spt'], to='ml'), name=latex('V_{spt}'), row=3, col=1, showlegend=False, line_color=colors.DEFAULT_PLOTLY_COLORS[2])
    fig.add_scatter(x=df['t'], y=convert(df['v_spt'], to='ml'), name=latex('V_{spt}'), row=3, col=1, showlegend=False, line_color='black')
    if df2 is not None: 
        fig.add_scatter(x=df2['t'], y=convert(df2['v_spt'], to='ml'), name=latex('V_{spt}'), row=3, col=1, showlegend=False, line_color='black', line_dash='dot')
    
    fig.update_yaxes(col=2, title_text='Ventricle pressure (mmHg)')
    fig.update_xaxes(col=2, title_text='Ventricle volume (ml)')

    df2 = df.iloc[200:]
    fig.add_scatter(x=convert(df2['v_lv'], to='ml'), y=convert(df2['p_lv'], to='mmHg'), name='Left', row=1, col=2, line_color=colors.DEFAULT_PLOTLY_COLORS[0])
    fig.add_scatter(x=convert(df2['v_rv'], to='ml'), y=convert(df2['p_rv'], to='mmHg'), name='Right', row=1, col=2, line_color=colors.DEFAULT_PLOTLY_COLORS[1])
    if df2 is not None:
        fig.add_scatter(x=convert(df2['v_lv'], to='ml'), y=convert(df2['p_lv'], to='mmHg'), name='Left (linear)', row=1, col=2, line_color=colors.DEFAULT_PLOTLY_COLORS[0], line_dash='dot')
        fig.add_scatter(x=convert(df2['v_rv'], to='ml'), y=convert(df2['p_rv'], to='mmHg'), name='Right (linear)', row=1, col=2, line_color=colors.DEFAULT_PLOTLY_COLORS[1], line_dash='dot')

    return fig


def latex(s):
    return fr"$\Large{{{s}}}$"

if __name__ == '__main__':

    # Choose a type of cardiovascular simulation to run from the three classes:

    cvs_class = SmithCardioVascularSystem
    # cvs_class = InertialSmithCVS
    # cvs_class = JallonHeartLungs

    # Choose a heartrate model. None uses the default static model. Defining
    # a function of time causes the variable heartrate model to be used. 

    f_hr = None
    # f_hr = lambda t: torch.full_like(t, fill_value=80.)
    # f_hr = lambda t: torch.full_like(t, fill_value=54.)
    # f_hr = lambda t: 80 + 20 * torch.tanh(0.3 * (t - 40))

    # Comment/uncomment to include blood pressure metric model. f_hr cannot be 
    # None if this is uncommented.
    # cvs_class = add_bp_metrics(cvs_class)

    # Instantiate cardiovascular simulation class
    cvs = cvs_class(f_hr=f_hr)

    # Uncomment this to use the Jallon v_spt linearisation
    # cvs._v_spt_method = 'jallon'

    t_final = 20  # seconds
    resolution = 50  # Hz

    with torch.no_grad():
        t_sol, sol = cvs.simulate(t_final, resolution)

    t, x, dx_dt, outputs = zip(*cvs.trajectory)
    t = torch.tensor(t)
    x = torch.stack(x)
    df = pd.DataFrame(outputs).apply(lambda s: [float(x) for x in s])
    df['t'] = t
    
    plot_states(cvs, df, t, x).write_html('states.html', auto_open=True, include_mathjax='cdn')
    plot_vent_interaction(df).write_html('vent.html', auto_open=True, include_mathjax='cdn')
    plot_lv_pressures(df).write_html('lv.html', auto_open=True, include_mathjax='cdn')
    plot_rv_pressures(df).write_html('rv.html', auto_open=True, include_mathjax='cdn')
