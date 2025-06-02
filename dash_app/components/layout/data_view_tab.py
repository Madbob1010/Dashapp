from dash import dcc, html
import os
from trading_bot.config import settings
from pathlib import Path

def get_data_view_content():
    """Return content for the Data View tab."""
    config_dir = Path('/home/madbob10/Dash/configs')
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]

    return html.Div([
        html.H3("View Cryptocurrency Data", style={'color': '#000000'}),
        html.Label("Select Cryptocurrency Data:", style={'color': '#000000'}),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            style={'width': '50%', 'border': '1px solid #000000'}
        ),
        dcc.Graph(id='price-chart', style={'border': '1px solid #000000'}),
        html.Hr(),
        html.H3("Configuration", style={'color': '#000000'}),
        html.Label("Select Config File:", style={'color': '#000000'}),
        dcc.Dropdown(
            id='config-dropdown',
            options=[{'label': f, 'value': f} for f in config_files],
            value='default.json',
            style={'width': '50%', 'margin-bottom': '10px', 'border': '1px solid #000000'}
        ),
        html.Div(id='config-editor', style={'margin': '20px 0', 'padding': '10px', 'border': '1px solid #000000'}),
        html.Label("Save Config As:", style={'color': '#000000'}),
        dcc.Input(
            id='config-save-name',
            type='text',
            placeholder='Enter config name (e.g., strategy1.json)',
            style={'width': '50%', 'margin-bottom': '10px', 'border': '1px solid #000000'}
        ),
        html.Button('Save Config', id='save-config-button', n_clicks=0, style={'margin-left': '10px', 'border': '1px solid #000000'}),
        html.Div(id='save-config-status', style={'color': '#000000', 'margin': '10px 0'}),
        html.Hr(),
        html.H3("Run Backtest", style={'color': '#000000'}),
        html.Label("Select CSV Files (or Entire File):", style={'color': '#000000'}),
        dcc.Dropdown(
            id='backtest-csv-dropdown',
            options=[{'label': 'Entire File', 'value': 'all'}] + [{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            multi=True,
            style={'width': '50%', 'border': '1px solid #000000'}
        ),
        html.Label("Backtest Options:", style={'color': '#000000'}),
        dcc.Checklist(
            id='backtest-options',
            options=[
                {'label': 'Generate Plot', 'value': 'plot'},
                {'label': 'Use LSTM', 'value': 'use_lstm'},
            ],
            value=['plot', 'use_lstm'],
            style={'margin': '10px', 'color': '#000000'}
        ),
        html.Button('Run Backtest', id='backtest-button', n_clicks=0, style={'border': '1px solid #000000'}),
        html.Br(),
        html.Div(id='backtest-status', style={'color': '#000000'})
    ])