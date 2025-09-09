import collections
import time
import threading
from pylsl import StreamInlet, resolve_byprop
from src.config import BUFFER_SECONDS, SAMPLING_RATE, EEG_CHANNELS, AXES

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
    """Connect to an LSL stream by its name."""
    print(f"Connecting to {name} stream...")
    try:
        streams = resolve_byprop('type', name, timeout=10)
        if not streams:
            raise ConnectionError(f"Failed to find {name} stream.")
        print(f"✅ Found {name} stream.")
        return StreamInlet(streams[0])
    except Exception as e:
        print(f"❌ Error connecting to {name} stream: {e}")
        return None

inlets = {
    'EEG': connect_stream('EEG'),
    'Accelerometer': connect_stream('ACC'),
    'Gyroscope': connect_stream('GYRO'),
}

# === Background Thread to Poll Streams ===
def stream_loop():
    """Continuously poll data from LSL streams in a background thread."""
    while True:
        for name, inlet in inlets.items():
            if inlet:
                sample, ts = inlet.pull_sample(timeout=0.0)
                if sample:
                    keys = list(buffers[name].keys())
                    for i, key in enumerate(keys):
                        buffers[name][key].append(sample[i])
                    timestamps[name].append(ts)
        time.sleep(0.001) # Small sleep to prevent busy-waiting

def start_stream_thread():
    """Starts the background thread for data streaming."""
    thread = threading.Thread(target=stream_loop, daemon=True)
    thread.start()
