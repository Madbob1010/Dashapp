#!/usr/bin/env python3
import sys
import os
# Use relative path to trading_bot/

from dash import Dash
from dash_app.components.layout import get_layout
from dash_app.components.callbacks import register_callbacks

# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Store the latest plot state
app.latest_plot = {'type': None, 'file': None, 'figure': None}

# Set layout
app.layout = get_layout()

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)