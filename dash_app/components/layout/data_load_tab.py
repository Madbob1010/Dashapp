from dash import dcc, html
import datetime
import os
from trading_bot.config import settings

def get_data_load_content():
    """Return content for the Data Load tab."""
    return html.Div([
        html.H3("Fetch Cryptocurrency Data", style={'color': '#000000'}),
        html.Label("Symbol (e.g., BTCUSDT):", style={'color': '#000000'}),
        dcc.Input(id='symbol-input', value='BTCUSDT', type='text', style={'width': '50%', 'border': '1px solid #000000'}),
        html.Br(),
        html.Label("Start Date:", style={'color': '#000000'}),
        dcc.DatePickerSingle(
            id='start-date-picker',
            date=(datetime.datetime.now() - datetime.timedelta(days=7)).date(),
            display_format='YYYY-MM-DD',
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("End Date:", style={'color': '#000000'}),
        dcc.DatePickerSingle(
            id='end-date-picker',
            date=datetime.datetime.now().date(),
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%'}
        ),
        html.Br(),
        html.Button('Fetch Data', id='fetch-button', n_clicks=0, style={'border': '1px solid #000000'}),
        html.Br(),
        html.Div(id='fetch-status', style={'color': '#000000'}),
        html.Hr(),
        html.H3("Reprocess CSV Data", style={'color': '#000000'}),
        html.Label("Select CSV File:", style={'color': '#000000'}),
        dcc.Dropdown(
            id='reprocess-csv-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            style={'width': '50%', 'border': '1px solid #000000'}
        ),
        html.Label("Start Date:", style={'color': '#000000'}),
        dcc.DatePickerSingle(
            id='reprocess-start-date-picker',
            display_format='YYYY-MM-DD',
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("End Date:", style={'color': '#000000'}),
        dcc.DatePickerSingle(
            id='reprocess-end-date-picker',
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%'}
        ),
        html.Br(),
        html.Label("Target Timeframe:", style={'color': '#000000'}),
        dcc.Dropdown(
            id='reprocess-timeframe',
            options=[
                {'label': '1 Minute', 'value': '1min'},
                {'label': '5 Minutes', 'value': '5min'},
                {'label': '15 Minutes', 'value': '15min'},
                {'label': '1 Hour', 'value': '1H'},
            ],
            value='1min',
            style={'width': '50%', 'border': '1px solid #000000'}
        ),
        html.Button('Reprocess Data', id='reprocess-button', n_clicks=0, style={'border': '1px solid #000000'}),
        html.Br(),
        html.Div(id='reprocess-status', style={'color': '#000000'})
    ])