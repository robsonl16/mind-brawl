import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import collections
import time
import threading
from scipy.signal import butter, filtfilt, iirnotch, welch

# === Settings ===
BUFFER_SECONDS = 5
UPDATE_INTERVAL_MS = 250
SAMPLING_RATE = 256  # Hz

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
AXES = ['X', 'Y', 'Z']

# === Global Buffers ===
buffers = {
    'EEG': {ch: collections.deque(maxlen=BUFFER_SECONDS * SAMPLING_RATE) for ch in EEG_CHANNELS},
    'Accelerometer': {ax: collections.deque(maxlen=BUFFER_SECONDS * 52) for ax in AXES},
    'Gyroscope': {ax: collections.deque(maxlen=BUFFER_SECONDS * 52) for ax in AXES},
}
timestamps = {
    'EEG': collections.deque(maxlen=BUFFER_SECONDS * SAMPLING_RATE),
    'Accelerometer': collections.deque(maxlen=BUFFER_SECONDS * 52),
    'Gyroscope': collections.deque(maxlen=BUFFER_SECONDS * 52),
}

# === Stream Connections ===
def connect_stream(name):
    print(f"Connecting to {name} stream...")
    streams = resolve_byprop('type', name, timeout=10)
    if not streams:
        print(f"❌ Failed to find {name} stream.")
        return None
    return StreamInlet(streams[0])

inlets = {
    'EEG': connect_stream('EEG'),
    'Accelerometer': connect_stream('ACC'),
    'Gyroscope': connect_stream('GYRO'),
}

# === Background Thread to Poll Streams ===
def stream_loop():
    while True:
        for name, inlet in inlets.items():
            if inlet:
                sample, ts = inlet.pull_sample(timeout=0.0)
                if sample:
                    keys = list(buffers[name].keys())
                    for i, key in enumerate(keys):
                        buffers[name][key].append(sample[i])
                    timestamps[name].append(ts)
        time.sleep(0.001)

threading.Thread(target=stream_loop, daemon=True).start()

# === Filter Utils ===
def apply_filters(data, fs, selected_filters):
    nyq = 0.5 * fs

    if 'notch' in selected_filters:
        b, a = iirnotch(w0=60/nyq, Q=30)
        data = filtfilt(b, a, data)

    if 'highpass' in selected_filters:
        b, a = butter(4, 1/nyq, btype='high')
        data = filtfilt(b, a, data)

    if 'lowpass' in selected_filters:
        b, a = butter(4, 30/nyq, btype='low')
        data = filtfilt(b, a, data)

    bandpass_bands = {
        # 'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    for band, (low, high) in bandpass_bands.items():
        if band in selected_filters:
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            data = filtfilt(b, a, data)

    return data

def compute_band_powers(signal, fs=256):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    band_ranges = {
        # 'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
    }
    powers = {}
    for band, (low, high) in band_ranges.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        powers[band] = np.trapz(psd[idx], freqs[idx])
    return powers

def detect_blink(eeg_buffer, threshold=75, cooldown=0.5, last_blink_time=0):
    """Detects a blink artifact in the EEG signal."""
    # Use frontal channels for blink detection
    frontal_channels = ['AF7', 'AF8']
    current_time = time.time()

    if current_time - last_blink_time < cooldown:
        return False, last_blink_time

    for ch in frontal_channels:
        data = np.array(eeg_buffer[ch])
        if len(data) < 50:  # Need enough data to check
            continue
        
        # Check the most recent ~200ms of data for a sharp spike
        recent_data = data[-int(0.2 * SAMPLING_RATE):]
        if len(recent_data) > 0 and (np.max(recent_data) - np.min(recent_data)) > threshold:
            return True, current_time
            
    return False, last_blink_time


# === Dash App ===
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Muse 2 Dashboard"

def build_graph(id_, title, y_label):
    return dcc.Graph(id=id_, config={"displayModeBar": False}, style={'height': '250px'})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='main-page-content')
])

@app.callback(
    Output('main-page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(_):
    return html.Div([
        html.H2("Muse 2 EEG Dashboard"),
        dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),
        # Store for game state
        dcc.Store(id='game-state-store', data={
            'score': 0, 
            'target_pos': (175, 175), # Initial position for the multi-ring target
            'last_blink_time': 0,
            'shot_timer': 10.0
        }),

        dcc.Tabs(id="tabs", value='eeg-tab', children=[
            dcc.Tab(label='EEG Time Series', value='eeg-tab'),
            dcc.Tab(label='Mind Game', value='game-tab'),
            dcc.Tab(label='Motion Sensors', value='motion-tab'),
            dcc.Tab(label='Band Power', value='band-tab'),

        ]),
        
        html.Div(id='tab-content')
    ])

# === EEG Helper Function ===
def build_eeg_figure(ch, filters, window=BUFFER_SECONDS):
    t = np.array(timestamps['EEG'])
    y = np.array(buffers['EEG'][ch])
    fig = go.Figure()
    if len(t) and len(y):
        if len(t) > len(y):
            t = t[-len(y):]
        elif len(y) > len(t):
            y = y[-len(t):]
        t = t - t[0]

        if len(y) > 10 and filters:
            y = apply_filters(y, SAMPLING_RATE, filters)

        fig.add_trace(go.Scattergl(x=t, y=y, mode='lines', name=ch))
        fig.update_xaxes(range=[0, window])
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (s)",
        yaxis_title="µV",
        title=f"EEG - {ch}",
        template="plotly_white",
        showlegend=False
    )
    return fig

# === EEG Callbacks (individual per channel) ===
def make_eeg_callback(ch):
    @app.callback(
        Output(f"eeg-graph-{ch}", "figure"),
        Input("interval", "n_intervals"),
        Input("filter-checklist", "value"),  # <-- Corrected ID here
    )
    def update_eeg(n, selected_filters):
        return build_eeg_figure(ch, selected_filters)
    return update_eeg

for ch in EEG_CHANNELS:
    make_eeg_callback(ch)

# === Plot Update Callback for Accel/Gyro ===
@app.callback(
    [Output('accel-graph', 'figure'),
     Output('gyro-graph', 'figure')],
    [Input('interval', 'n_intervals')]
)
def update_others(_):
    window = BUFFER_SECONDS

    accel_fig = go.Figure()
    t = np.array(timestamps['Accelerometer'])
    if len(t):
        t = t - t[0]
    for ax in AXES:
        y = np.array(buffers['Accelerometer'][ax])
        if len(t) and len(y):
            accel_fig.add_trace(go.Scattergl(x=t, y=y, mode='lines', name=ax))
    accel_fig.update_layout(
        xaxis=dict(range=[0, window]),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (s)",
        yaxis_title="Accel (m/s²)",
        title="Accelerometer",
        template="plotly_white"
    )

    gyro_fig = go.Figure()
    t = np.array(timestamps['Gyroscope'])
    if len(t):
        t = t - t[0]
    for ax in AXES:
        y = np.array(buffers['Gyroscope'][ax])
        if len(t) and len(y):
            gyro_fig.add_trace(go.Scattergl(x=t, y=y, mode='lines', name=ax))
    gyro_fig.update_layout(
        xaxis=dict(range=[0, window]),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time (s)",
        yaxis_title="Gyro (°/s)",
        title="Gyroscope",
        template="plotly_white"
    )

    return [accel_fig, gyro_fig]

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'eeg-tab':
        return html.Div([
            html.H4("Filter Controls"),
            dcc.Checklist(
                id='filter-checklist',
                options=[
                    {'label': 'Power Line Notch (60Hz)', 'value': 'notch'},
                    {'label': 'High-Pass (>1Hz)', 'value': 'highpass'},
                    {'label': 'Low-Pass (<50Hz)', 'value': 'lowpass'},
                    {'label': 'Bandpass Alpha (8–12Hz)', 'value': 'alpha'},
                    {'label': 'Bandpass Beta (12–30Hz)', 'value': 'beta'},
                    {'label': 'Bandpass Theta (4–8Hz)', 'value': 'theta'},
                    # {'label': 'Bandpass Delta (1–4Hz)', 'value': 'delta'},
                ],
                value=[],
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.H4("EEG Channels"),
            *[
                html.Div([
                    html.Label(f"EEG - {ch}"),
                    build_graph(f'eeg-graph-{ch}', f"EEG - {ch}", "µV")
                ]) for ch in EEG_CHANNELS
            ]
        ])
    
    elif tab == 'motion-tab':
        return html.Div([
            html.Div([
                html.H4("Accelerometer"),
                html.Label("Accelerometer (X/Y/Z in m/s²)"),
                build_graph('accel-graph', 'Accelerometer', 'm/s²')
            ]),
            html.Div([
                html.H4("Gyroscope"),
                html.Label("Gyroscope (X/Y/Z in °/s)"),
                build_graph('gyro-graph', 'Gyroscope', '°/s')
            ])
        ])
    
    elif tab == 'band-tab':
        return html.Div([
            html.H4("EEG Band Power (Delta, Theta, Alpha, Beta)"),
            html.Div(id='bandpower-graphs')  # <-- this gets populated by callback
        ])
    
    elif tab == 'game-tab':
        return html.Div([
            html.H4("The Zen Archer"),
            html.P("Aim with your head, focus to steady, and blink to shoot!"),
            html.Button('Start / Reset Game', id='start-game-btn', n_clicks=0),
            # Game Area
            html.Div(
                id='game-area',
                style={
                    'width': '500px', 'height': '500px', 'border': '2px solid black',
                    'margin': 'auto', 'position': 'relative', 'overflow': 'hidden',
                    'background': '#f0f0f0'
                },
                children=[
                    # Multi-ring Target
                    html.Div(id='target-outer', style={
                        'position': 'absolute', 'width': '150px', 'height': '150px',
                        'borderRadius': '50%', 'backgroundColor': 'lightblue', 'left': '175px', 'top': '175px'
                    }, children=[
                        html.Div(id='target-middle', style={
                            'position': 'absolute', 'width': '100px', 'height': '100px',
                            'borderRadius': '50%', 'backgroundColor': 'lightgreen', 'left': '25px', 'top': '25px'
                        }, children=[
                            html.Div(id='target-inner', style={
                                'position': 'absolute', 'width': '50px', 'height': '50px',
                                'borderRadius': '50%', 'backgroundColor': 'red', 'left': '25px', 'top': '25px'
                            })
                        ])
                    ]),
                    # The Crosshair
                    html.Div(id='crosshair', style={
                        'width': '30px', 'height': '30px', 'border': '2px solid cyan',
                        'position': 'absolute', 'top': '235px', 'left': '235px'
                    }),
                    # Shot Timer Bar
                    html.Div(id='timer-bar', style={
                        'position': 'absolute', 'bottom': '0', 'left': '0', 'height': '10px',
                        'backgroundColor': 'orange', 'width': '100%'
                    })
                ]
            ),
            html.Div(id='game-state-output', style={'textAlign': 'center', 'marginTop': '20px'})
        ])
        
        # return html.Div([
        #     html.H4("Focus vs. Relax"),
        #     html.P("Try to relax to move the bar left, or focus to move it right."),
        #     html.Div(
        #         # This is the container for our game bar
        #         style={'width': '80%', 'height': '50px', 'border': '1px solid black', 'margin': 'auto', 'position': 'relative'},
        #         children=[
        #             # This is the bar that will move
        #             html.Div(id='game-bar', style={'width': '10px', 'height': '100%', 'backgroundColor': 'red', 'position': 'absolute', 'left': '50%'})
        #         ]
        #     ),
        #     html.Div(id='game-state-output', style={'textAlign': 'center', 'marginTop': '20px'})
        # ])

    return html.Div("Unknown Tab")

@app.callback(
    Output('bandpower-graphs', 'children'),
    Input('interval', 'n_intervals'),
)
def update_bandpower(_):
    graphs = []
    t = np.array(timestamps['EEG'])
    if not len(t):
        return html.Div("Waiting for EEG data...")
    
    for ch in EEG_CHANNELS:
        y = np.array(buffers['EEG'][ch])
        if len(y) > 2 * SAMPLING_RATE:
            powers = compute_band_powers(y)
            graphs.append(
                dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                x=list(powers.keys()),
                                y=list(powers.values()),
                                marker_color='indigo'
                            )
                        ],
                        layout=go.Layout(
                            title=f'Band Power - {ch}',
                            yaxis_title="Power (µV²)",
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20),
                            yaxis=dict(range=[0, 150])
                        )
                    ),
                    config={"displayModeBar": False},
                    style={'height': '250px'}
                )
            )
    return graphs

@app.callback(
    [Output('crosshair', 'style'),
     Output('target-outer', 'style'),
     Output('timer-bar', 'style'),
     Output('game-state-output', 'children'),
     Output('game-state-store', 'data')],
    [Input('interval', 'n_intervals'),
     Input('start-game-btn', 'n_clicks')],
    [dash.dependencies.State('game-state-store', 'data'),
     dash.dependencies.State('tabs', 'value')]  # <-- Add the active tab as State
)
def update_zen_archer(_, n_clicks, game_state, active_tab):
    
    if active_tab != 'game-tab':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'start-game-btn.n_clicks':
        game_state = {
            'score': 0, 'target_pos': (175, 175), 'last_blink_time': time.time(), 'shot_timer': 10.0
        }

    # --- 1. Update Timer ---
    game_state['shot_timer'] -= (UPDATE_INTERVAL_MS / 1000.0)
    if game_state['shot_timer'] <= 0:
        # Time ran out, reset target
        game_state['target_pos'] = (np.random.randint(25, 325), np.random.randint(25, 325))
        game_state['shot_timer'] = 10.0 # Reset timer

    # --- 2. Detect Blink ---
    blinked, new_blink_time = detect_blink(buffers['EEG'], last_blink_time=game_state['last_blink_time'])
    game_state['last_blink_time'] = new_blink_time

    # --- 3. Calculate Focus Metric ---
    focus_channels = ['AF7', 'AF8']
    alpha_power, beta_power, num_channels = 0, 0, 0
    for ch in focus_channels:
        y = np.array(buffers['EEG'][ch])
        if len(y) > 2 * SAMPLING_RATE:
            powers = compute_band_powers(y)
            if 'Alpha' in powers and 'Beta' in powers:
                alpha_power += powers['Alpha']
                beta_power += powers['Beta']
                num_channels += 1

    if num_channels == 0:
        return dash.no_update, dash.no_update, dash.no_update, "Waiting for EEG data...", dash.no_update

    focus_metric = (beta_power / num_channels) / ((alpha_power / num_channels) + 1e-10)

    # --- 4. Get Gyroscope Data ---
    try:
        gyro_y, gyro_z = buffers['Gyroscope']['Y'][-1], buffers['Gyroscope']['Z'][-1]
    except IndexError:
        return dash.no_update, dash.no_update, dash.no_update, "Waiting for Gyro data...", dash.no_update

    # --- 5. Update Crosshair Position ---
    sensitivity = 5
    crosshair_x = 235 + (gyro_z * sensitivity)
    crosshair_y = 235 + (gyro_y * sensitivity)
    shake_magnitude = np.clip(5 / (focus_metric + 1e-10), 1, 20)
    crosshair_x += np.random.uniform(-shake_magnitude, shake_magnitude)
    crosshair_y += np.random.uniform(-shake_magnitude, shake_magnitude)
    crosshair_x = np.clip(crosshair_x, 0, 470)
    crosshair_y = np.clip(crosshair_y, 0, 470)

    # --- 6. Scoring Logic on Blink ---
    if blinked:
        # Center of target is at (pos + 75, pos + 75)
        target_center_x = game_state['target_pos'][0] + 75
        target_center_y = game_state['target_pos'][1] + 75
        dist_from_center = np.sqrt((crosshair_x - target_center_x)**2 + (crosshair_y - target_center_y)**2)

        points = 0
        if dist_from_center <= 25: # Inner ring
            points = 100
        elif dist_from_center <= 50: # Middle ring
            points = 50
        elif dist_from_center <= 75: # Outer ring
            points = 20
        
        if points > 0:
            game_state['score'] += points
        
        # Reset for next shot
        game_state['target_pos'] = (np.random.randint(25, 325), np.random.randint(25, 325))
        game_state['shot_timer'] = 10.0

    # --- 7. Prepare Outputs ---
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

# # === Game Callback ===
# @app.callback(
#     [Output('game-bar', 'style'),
#      Output('game-state-output', 'children')],
#     Input('interval', 'n_intervals')
# )
# def update_game_state(_):
#     # Use AF7 and AF8 as they are frontal channels sensitive to cognitive activity
#     focus_channels = ['AF7', 'AF8']
#     alpha_power = 0
#     beta_power = 0
#     num_channels = 0

#     for ch in focus_channels:
#         y = np.array(buffers['EEG'][ch])
#         if len(y) > 2 * SAMPLING_RATE:
#             powers = compute_band_powers(y)
#             if 'Alpha' in powers and 'Beta' in powers:
#                 alpha_power += powers['Alpha']
#                 beta_power += powers['Beta']
#                 num_channels += 1

#     if num_channels == 0:
#         # Not enough data yet, return default state
#         return dash.no_update, "Waiting for EEG data..."

#     # Average the powers
#     alpha_power /= num_channels
#     beta_power /= num_channels

#     # Simple ratio - add a small epsilon to avoid division by zero
#     # A higher ratio means more Beta (focus)
#     focus_metric = beta_power / (alpha_power + 1e-10)

#     # Normalize the metric to a percentage (0% to 100%) for the bar position
#     # We'll clamp the ratio between 0.5 (very relaxed) and 2.5 (very focused) for this example
#     # You will need to tune these clamp values based on your own brain activity!
#     clamped_ratio = np.clip(focus_metric, 0.5, 2.5)
#     bar_position_percent = (clamped_ratio - 0.5) / (2.5 - 0.5) * 100

#     bar_style = {'width': '10px', 'height': '100%', 'backgroundColor': 'red', 'position': 'absolute', 'left': f'{bar_position_percent}%'}
    
#     state_text = f"Focus Metric (Beta/Alpha Ratio): {focus_metric:.2f}"

#     return bar_style, state_text

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
