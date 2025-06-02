from dash import dcc, html
import datetime
import os
from trading_bot.config import settings
from pathlib import Path
from .data_load_tab import get_data_load_content
from .data_view_tab import get_data_view_content

def get_layout():
    """Return the Dash app layout."""
    return html.Div([
        html.H1("Trading Strategy Backtesting Dashboard", style={'textAlign': 'center', 'color': '#000000'}),
        dcc.Tabs(id='tabs', value='data-load', children=[
            dcc.Tab(label='Data Load', value='data-load', style={'color': "#000000"}),
            dcc.Tab(label='Data View', value='data-view', style={'color': "#000000"}),
        ], style={'backgroundColor': '#FFFFFF', 'border': '1px solid #000000'}),
        html.Div(id='tabs-content', style={'backgroundColor': '#FFFFFF', 'padding': '20px', 'border': '1px solid #000000'})
    ])

def get_tabs_content(value):
    """Return content based on selected tab."""
    if value == 'data-load':
        return get_data_load_content()
    elif value == 'data-view':
        return get_data_view_content()
    return html.P("Select a tab to view content.", style={'color': '#000000'})