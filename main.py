import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path('/home/madbob10/Dash')))

from dash import Dash
from dash_app.components.layout import get_layout
from dash_app.callbacks import callbacks

app = Dash(__name__, suppress_callback_exceptions=True)

# Store the latest plot state
app.latest_plot = {'type': None, 'file': None, 'figure': None}

# Set layout
app.layout = get_layout()

# Register callbacks
callbacks.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True, port=8050)