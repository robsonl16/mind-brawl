import dash
from dash import dcc
import time
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

from src.app import app
from src.data_stream import buffers, timestamps
from src.signal_processing import apply_filters, compute_band_powers, detect_blink
from src.layout import get_page_content, get_tab_content
from src.config import BUFFER_SECONDS, SAMPLING_RATE, EEG_CHANNELS, AXES, UPDATE_INTERVAL_MS

# === Main Layout Callback ===
@app.callback(Output('main-page-content', 'children'), Input('url', 'pathname'))
def display_page(_):
    return get_page_content()

# === Tab Content Callback ===
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab_content(tab):
    return get_tab_content(tab)

# === EEG Graph Callbacks ===
def create_eeg_callback(channel):
    @app.callback(
        Output(f'eeg-graph-{channel}', 'figure'),
        Input('interval', 'n_intervals'),
        State('filter-checklist', 'value')
    )
    def update_eeg_graph(_, selected_filters):
        t = np.array(timestamps['EEG'])
        y = np.array(buffers['EEG'][channel])
        fig = go.Figure()
        if len(t) > 1 and len(y) > 1:
            if len(t) > len(y): t = t[-len(y):]
            elif len(y) > len(t): y = y[-len(t):]
            t = t - t[0]

            if len(y) > 10 and selected_filters:
                y = apply_filters(y, SAMPLING_RATE, selected_filters)

            fig.add_trace(go.Scattergl(x=t, y=y, mode='lines', name=channel))
            fig.update_xaxes(range=[max(0, t[-1] - BUFFER_SECONDS), t[-1]])
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Time (s)", yaxis_title="µV",
            title=f"EEG - {channel}", template="plotly_white", showlegend=False
        )
        return fig
    return update_eeg_graph

for ch in EEG_CHANNELS:
    globals()[f'update_eeg_{ch}_callback'] = create_eeg_callback(ch)

# === Motion Sensor Graph Callback ===
@app.callback(
    [Output('accel-graph', 'figure'), Output('gyro-graph', 'figure')],
    Input('interval', 'n_intervals'),
    State('tabs', 'value')
)
def update_motion_graphs(_, active_tab):
    if active_tab != 'motion-tab':
        return dash.no_update, dash.no_update

    accel_fig = go.Figure()
    t_accel = np.array(timestamps['Accelerometer'])
    if len(t_accel):
        t_accel = t_accel - t_accel[0]
        for ax in AXES:
            y = np.array(buffers['Accelerometer'][ax])
            if len(t_accel) and len(y):
                accel_fig.add_trace(go.Scattergl(x=t_accel, y=y, mode='lines', name=ax))
    accel_fig.update_layout(
        xaxis=dict(range=[0, BUFFER_SECONDS]), margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (s)", yaxis_title="m/s²", title="Accelerometer", template="plotly_white"
    )

    gyro_fig = go.Figure()
    t_gyro = np.array(timestamps['Gyroscope'])
    if len(t_gyro):
        t_gyro = t_gyro - t_gyro[0]
        for ax in AXES:
            y = np.array(buffers['Gyroscope'][ax])
            if len(t_gyro) and len(y):
                gyro_fig.add_trace(go.Scattergl(x=t_gyro, y=y, mode='lines', name=ax))
    gyro_fig.update_layout(
        xaxis=dict(range=[0, BUFFER_SECONDS]), margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (s)", yaxis_title="°/s", title="Gyroscope", template="plotly_white"
    )
    return accel_fig, gyro_fig

# === Band Power Callback ===
@app.callback(
    Output('bandpower-graphs', 'children'),
    Input('interval', 'n_intervals'),
    State('tabs', 'value')
)
def update_bandpower_graphs(_, active_tab):
    if active_tab != 'band-tab':
        return dash.no_update
    
    graphs = []
    for ch in EEG_CHANNELS:
        y = np.array(buffers['EEG'][ch])
        if len(y) > 2 * SAMPLING_RATE:
            powers = compute_band_powers(y, SAMPLING_RATE)
            graphs.append(dcc.Graph(
                figure=go.Figure(
                    data=[go.Bar(x=list(powers.keys()), y=list(powers.values()), marker_color='indigo')],
                    layout=go.Layout(
                        title=f'Band Power - {ch}', yaxis_title="Power (µV²)",
                        template="plotly_white", margin=dict(l=20, r=20, t=40, b=20),
                        yaxis=dict(range=[0, 150])
                    )
                ),
                config={"displayModeBar": False}, style={'height': '250px'}
            ))
    return graphs

# === Game Callback ===
@app.callback(
    [Output('crosshair', 'style'),
     Output('target-outer', 'style'),
     Output('timer-bar', 'style'),
     Output('game-state-output', 'children'),
     Output('game-state-store', 'data')],
    [Input('interval', 'n_intervals'),
     Input('start-game-btn', 'n_clicks')],
    [State('game-state-store', 'data'),
     State('tabs', 'value')]
)
def update_zen_archer(_, n_clicks, game_state, active_tab):
    if active_tab != 'game-tab':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'start-game-btn.n_clicks':
        game_state = {
            'score': 0, 'target_pos': (175, 175), 'last_blink_time': time.time(), 'shot_timer': 10.0
        }

    game_state['shot_timer'] -= (UPDATE_INTERVAL_MS / 1000.0)
    if game_state['shot_timer'] <= 0:
        game_state['target_pos'] = (np.random.randint(25, 325), np.random.randint(25, 325))
        game_state['shot_timer'] = 10.0

    blinked, new_blink_time = detect_blink(buffers['EEG'], SAMPLING_RATE, last_blink_time=game_state['last_blink_time'])
    game_state['last_blink_time'] = new_blink_time

    focus_channels = ['AF7', 'AF8']
    alpha_power, beta_power, num_channels = 0, 0, 0
    for ch in focus_channels:
        y = np.array(buffers['EEG'][ch])
        if len(y) > 2 * SAMPLING_RATE:
            powers = compute_band_powers(y, SAMPLING_RATE)
            if 'Alpha' in powers and 'Beta' in powers:
                alpha_power += powers['Alpha']
                beta_power += powers['Beta']
                num_channels += 1

    if num_channels == 0:
        return dash.no_update, dash.no_update, dash.no_update, "Waiting for EEG data...", dash.no_update

    focus_metric = (beta_power / num_channels) / ((alpha_power / num_channels) + 1e-10)

    try:
        gyro_y, gyro_z = buffers['Gyroscope']['Y'][-1], buffers['Gyroscope']['Z'][-1]
    except IndexError:
        return dash.no_update, dash.no_update, dash.no_update, "Waiting for Gyro data...", dash.no_update

    sensitivity = 5
    crosshair_x = 235 + (gyro_z * sensitivity)
    crosshair_y = 235 + (gyro_y * sensitivity)
    shake_magnitude = np.clip(5 / (focus_metric + 1e-10), 1, 20)
    crosshair_x += np.random.uniform(-shake_magnitude, shake_magnitude)
    crosshair_y += np.random.uniform(-shake_magnitude, shake_magnitude)
    crosshair_x = np.clip(crosshair_x, 0, 470)
    crosshair_y = np.clip(crosshair_y, 0, 470)

    points = 0
    if blinked:
        target_center_x = game_state['target_pos'][0] + 75
        target_center_y = game_state['target_pos'][1] + 75
        dist_from_center = np.sqrt((crosshair_x - target_center_x)**2 + (crosshair_y - target_center_y)**2)

        if dist_from_center <= 25: points = 100
        elif dist_from_center <= 50: points = 50
        elif dist_from_center <= 75: points = 20
        
        if points > 0: game_state['score'] += points
        
        game_state['target_pos'] = (np.random.randint(25, 325), np.random.randint(25, 325))
        game_state['shot_timer'] = 10.0

    crosshair_style = {
        'position': 'absolute', 'width': '30px', 'height': '30px', 'border': '2px solid cyan',
        'top': f'{crosshair_y}px', 'left': f'{crosshair_x}px', 'transition': 'top 0.1s, left 0.1s'
    }
    target_style = {
        'position': 'absolute', 'width': '150px', 'height': '150px',
        'borderRadius': '50%', 'backgroundColor': 'lightblue',
        'left': f'{game_state["target_pos"][0]}px', 'top': f'{game_state["target_pos"][1]}px',
        'transition': 'top 0.5s, left 0.5s'
    }
    timer_style = {
        'position': 'absolute', 'bottom': '0', 'left': '0', 'height': '10px',
        'backgroundColor': 'orange', 'width': f"{game_state['shot_timer'] * 10}%"
    }
    state_text = f"Score: {game_state['score']} | Focus: {focus_metric:.2f} | Last Shot: {'HIT!' if blinked and points > 0 else ('MISS' if blinked else '--')}"

    return crosshair_style, target_style, timer_style, state_text, game_state
