from dash import dcc, html
import datetime
import os
from dash_app.config.settings import DATA_DIR

def get_layout():
    """Return the Dash app layout."""
    return html.Div([
        html.H1("Trading Strategy Backtesting Dashboard", style={'textAlign': 'center'}),
        dcc.Tabs(id='tabs', value='data-load', children=[
            dcc.Tab(label='Data Load', value='data-load'),
            dcc.Tab(label='Data View', value='data-view'),
        ]),
        html.Div(id='tabs-content')
    ])

def get_data_load_content():
    """Return content for the Data Load tab."""
    return html.Div([
        html.H3("Fetch Cryptocurrency Data"),
        html.Label("Symbol (e.g., BTCUSDT):"),
        dcc.Input(id='symbol-input', value='BTCUSDT', type='text', style={'width': '50%'}),
        html.Br(),
        html.Label("Start Date:"),
        dcc.DatePickerSingle(
            id='start-date-picker',
            date=(datetime.datetime.now() - datetime.timedelta(days=7)).date(),
            display_format='YYYY-MM-DD',
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("End Date:"),
        dcc.DatePickerSingle(
            id='end-date-picker',
            date=datetime.datetime.now().date(),
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%'}
        ),
        html.Br(),
        html.Button('Fetch Data', id='fetch-button', n_clicks=0),
        html.Br(),
        html.Div(id='fetch-status'),
        html.Hr(),
        html.H3("Reprocess CSV Data"),
        html.Label("Select CSV File:"),
        dcc.Dropdown(
            id='reprocess-csv-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(DATA_DIR) if s.endswith('.csv')],
            value=None,
            style={'width': '50%'}
        ),
        html.Label("Start Date:"),
        dcc.DatePickerSingle(
            id='reprocess-start-date-picker',
            display_format='YYYY-MM-DD',
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("End Date:"),
        dcc.DatePickerSingle(
            id='reprocess-end-date-picker',
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("Target Timeframe:"),
        dcc.Dropdown(
            id='reprocess-timeframe',
            options=[
                {'label': '1 Minute', 'value': '1min'},
                {'label': '5 Minutes', 'value': '5min'},
                {'label': '15 Minutes', 'value': '15min'},
                {'label': '1 Hour', 'value': '1H'},
            ],
            value='1min',
            style={'width': '50%'}
        ),
        html.Button('Reprocess Data', id='reprocess-button', n_clicks=0),
        html.Br(),
        html.Div(id='reprocess-status')
    ])

def get_data_view_content():
    """Return content for the Data View tab."""
    return html.Div([
        html.H3("View Cryptocurrency Data"),
        html.Label("Select Cryptocurrency Data:"),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(DATA_DIR) if s.endswith('.csv')],
            value=None,
            style={'width': '50%'}
        ),
        dcc.Graph(id='price-chart'),
        html.Hr(),
        html.H3("Run Backtest"),
        html.Label("Select CSV Files (or Entire File):"),
        dcc.Dropdown(
            id='backtest-csv-dropdown',
            options=[{'label': 'Entire File', 'value': 'all'}] + [{'label': s, 'value': s} for s in os.listdir(DATA_DIR) if s.endswith('.csv')],
            value=None,
            multi=True,
            style={'width': '50%'}
        ),
        html.Label("Backtest Options:"),
        dcc.Checklist(
            id='backtest-options',
            options=[
                {'label': 'Generate Plot', 'value': 'plot'},
                {'label': 'Use LSTM', 'value': 'use_lstm'},
            ],
            value=[],
            style={'margin': '10px'}
        ),
        html.Button('Run Backtest', id='backtest-button', n_clicks=0),
        html.Br(),
        html.Div(id='backtest-status')
    ])