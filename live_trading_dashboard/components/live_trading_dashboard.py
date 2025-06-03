from dash import dcc, html
import datetime
import os
from trading_bot.config import settings
from pathlib import Path

# Define color scheme for consistency
COLORS = {
    'background': '#000000',  # Black background
    'text': '#FFFFFF',        # White text
    'accent': '#333333',      # Gray accent for borders and elements
    'border': '1px solid #333333'
}

def get_live_trading_content():
    """Return content for the Live Trading Dashboard tab."""
    dropdown_style = {
        'width': '50%',
        'border': COLORS['border'],
        'backgroundColor': COLORS['accent'],
        'color': '#000000'  # Black text for dropdown
    }

    return html.Div([
        html.H3("Live Trading Dashboard", style={'color': COLORS['text']}),
        html.Label("Select Data Source (CSV):", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='live-data-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            style=dropdown_style
        ),
        dcc.Graph(
            id='live-price-chart',
            style={
                'border': COLORS['border'],
                'backgroundColor': COLORS['accent'],
                'height': '400px',
                'margin-top': '20px'
            }
        ),
        html.Hr(),
        html.H3("Trading Controls", style={'color': COLORS['text']}),
        html.Label("Select Trading Pair:", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='trading-pair-dropdown',
            options=[
                {'label': 'BTC/USD', 'value': 'BTCUSD'},
                {'label': 'ETH/USD', 'value': 'ETHUSD'},
                {'label': 'LTC/USD', 'value': 'LTCUSD'}
            ],
            value='BTCUSD',
            style={**dropdown_style, 'margin-bottom': '10px'}
        ),
        html.Label("Trade Amount:", style={'color': COLORS['text']}),
        dcc.Input(
            id='trade-amount',
            type='number',
            placeholder='Enter amount (e.g., 0.01 BTC)',
            style={
                'width': '50%',
                'margin-bottom': '10px',
                'border': COLORS['border'],
                'backgroundColor': COLORS['accent'],
                'color': COLORS['text']
            }
        ),
        html.Div([
            html.Button(
                'Buy',
                id='buy-button',
                n_clicks=0,
                style={
                    'margin-right': '10px',
                    'border': COLORS['border'],
                    'backgroundColor': '#28a745',
                    'color': COLORS['text']
                }
            ),
            html.Button(
                'Sell',
                id='sell-button',
                n_clicks=0,
                style={
                    'border': COLORS['border'],
                    'backgroundColor': '#dc3545',
                    'color': COLORS['text']
                }
            ),
        ], style={'margin': '10px 0'}),
        html.Div(
            id='trade-status',
            style={
                'color': COLORS['text'],
                'margin': '10px 0',
                'padding': '10px',
                'border': COLORS['border'],
                'backgroundColor': COLORS['accent']
            }
        ),
        html.Hr(),
        html.H3("Live Trading Settings", style={'color': COLORS['text']}),
        dcc.Checklist(
            id='trading-options',
            options=[
                {'label': 'Enable Auto-Trading', 'value': 'auto_trading'},
                {'label': 'Show Real-Time Alerts', 'value': 'alerts'},
                {'label': 'Use Stop-Loss', 'value': 'stop_loss'},
            ],
            value=['alerts'],
            style={'margin': '10px', 'color': COLORS['text']}
        ),
        html.Label("Update Interval (seconds):", style={'color': COLORS['text']}),
        html.Div(
            dcc.Slider(
                id='update-interval',
                min=1,
                max=60,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 61, 5)},
                className='slider-container'
            ),
            className='slider-container'
        ),
        dcc.Interval(
            id='live-update',
            interval=5*1000,  # Default 5 seconds
            n_intervals=0
        ),
        html.Div(
            id='live-status',
            style={'color': COLORS['text'], 'margin': '10px 0'}
        )
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})

def get_layout():
    """Return the Dash app layout."""
    return html.Div([
        html.H1("Live Trading Dashboard", style={'textAlign': 'center', 'color': COLORS['text']}),
        dcc.Tabs(id='tabs', value='live-trading', children=[
            dcc.Tab(label='Live Trading', value='live-trading', style={'color': COLORS['text'], 'backgroundColor': COLORS['accent']}),
        ], style={'backgroundColor': COLORS['background'], 'border': COLORS['border']}),
        html.Div(id='tabs-content', style={'backgroundColor': COLORS['background'], 'padding': '20px', 'border': COLORS['border']})
    ], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

def get_tabs_content(value):
    """Return content based on selected tab."""
    if value == 'live-trading':
        return get_live_trading_content()
    return html.P("Select a tab to view content.", style={'color': COLORS['text']})