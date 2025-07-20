import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import collections
import time
import threading
from scipy.signal import butter, filtfilt, iirnotch

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
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    for band, (low, high) in bandpass_bands.items():
        if band in selected_filters:
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            data = filtfilt(b, a, data)

    return data

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Muse 2 Dashboard"

def build_graph(id_, title, y_label):
    return dcc.Graph(id=id_, config={"displayModeBar": False}, style={'height': '250px'})

app.layout = html.Div([
    html.H2("Muse 2 EEG Dashboard"),
    dcc.Interval(id='interval', interval=UPDATE_INTERVAL_MS, n_intervals=0),

    html.Div([
        html.H4("Filter Options"),
        dcc.Checklist(
            id='filter-selector',
            options=[
                {'label': 'Notch (60 Hz)', 'value': 'notch'},
                {'label': 'High-pass (>1Hz)', 'value': 'highpass'},
                {'label': 'Low-pass (<30Hz)', 'value': 'lowpass'},
                {'label': 'Delta (0.5–4 Hz)', 'value': 'delta'},
                {'label': 'Theta (4–8 Hz)', 'value': 'theta'},
                {'label': 'Alpha (8–13 Hz)', 'value': 'alpha'},
                {'label': 'Beta (13–30 Hz)', 'value': 'beta'},
            ],
            value=[],
            inline=True
        ),
    ]),

    html.Div([
        html.H4("EEG Channels"),
        *[build_graph(f'eeg-graph-{ch}', f"EEG - {ch}", "µV") for ch in EEG_CHANNELS]
    ]),

    html.Div([
        html.H4("Accelerometer"),
        build_graph('accel-graph', 'Accelerometer', 'm/s²')
    ]),

    html.Div([
        html.H4("Gyroscope"),
        build_graph('gyro-graph', 'Gyroscope', '°/s')
    ]),
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
for ch in EEG_CHANNELS:
    @app.callback(
        Output(f"eeg-graph-{ch}", "figure"),
        Input("interval", "n_intervals"),
        Input("filter-selector", "value"),
    )
    def update_eeg(n, selected_filters, ch=ch):  # closure binding
        return build_eeg_figure(ch, selected_filters)

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

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
