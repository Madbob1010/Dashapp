from dash import dcc, html
import datetime
import os
from trading_bot.config import settings
from pathlib import Path
from .data_load_tab import get_data_load_content
from .data_view_tab import get_data_view_content# Define color scheme for consistency
COLORS = {
    'background': '#000000',  # Black background
    'text': '#FFFFFF',        # White text
    'accent': '#333333',      # Gray accent for borders and elements
    'border': '1px solid #333333'
}

def get_layout():
    """Return the Dash app layout."""
    return html.Div([
        html.H1("Trading Strategy Backtesting Dashboard", style={'textAlign': 'center', 'color': COLORS['text']}),
        dcc.Tabs(id='tabs', value='data-view', children=[
            dcc.Tab(label='Data View', value='data-view', style={'color': COLORS['text'], 'backgroundColor': COLORS['accent']}),
            dcc.Tab(label='Data Load', value='data-load', style={'color': COLORS['text'], 'backgroundColor': COLORS['accent']}),
        ], style={'backgroundColor': COLORS['background'], 'border': COLORS['border']}),
        html.Div(id='tabs-content', style={'backgroundColor': COLORS['background'], 'padding': '20px', 'border': COLORS['border']})
    ], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

def get_tabs_content(value):
    """Return content based on selected tab."""
    if value == 'data-load':
        return get_data_load_content()
    elif value == 'data-view':
        return get_data_view_content()
    return html.P("Select a tab to view content.", style={'color': COLORS['text']})
    