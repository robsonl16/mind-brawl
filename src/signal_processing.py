import numpy as np
import time
from scipy.signal import butter, filtfilt, iirnotch, welch

def apply_filters(data, fs, selected_filters):
    """Apply selected filters to the EEG data."""
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
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
    }
    for band, (low, high) in bandpass_bands.items():
        if band in selected_filters:
            b, a = butter(4, [low/nyq, high/nyq], btype='band')
            data = filtfilt(b, a, data)

    return data

def compute_band_powers(signal, fs):
    """Compute band powers for a given signal."""
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    band_ranges = {
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
    }
    powers = {}
    for band, (low, high) in band_ranges.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        powers[band] = np.trapz(psd[idx], freqs[idx])
    return powers

def detect_blink(eeg_buffer, fs, threshold=65, cooldown=0.75, last_blink_time=0):
    """
    Detects a blink artifact in the EEG signal.
    Tuning:
    - If blinks aren't registering, LOWER the threshold.
    - If you get false positives (blinks register when you don't blink), INCREASE the threshold or cooldown.
    """
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
        recent_data = data[-int(0.2 * fs):]
        if len(recent_data) > 0 and (np.max(recent_data) - np.min(recent_data)) > threshold:
            return True, current_time
            
    return False, last_blink_time
