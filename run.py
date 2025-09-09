from src.app import app
from src.layout import get_layout
from src.data_stream import start_stream_thread
import src.callbacks  # This import is crucial to register the callbacks

# Set the layout of the app
app.layout = get_layout()

# Start the background thread for data streaming
start_stream_thread()

# Main entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)
