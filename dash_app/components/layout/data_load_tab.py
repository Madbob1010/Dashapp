from dash import dcc, html
import datetime
import os
from trading_bot.config import settings

# Define color scheme for consistency
COLORS = {
    'background': '#000000',  # Black background
    'text': '#FFFFFF',        # White text
    'accent': '#333333',      # Gray accent for borders and elements
    'border': '1px solid #333333'
}

def get_data_load_content():
    """Return content for the Data Load tab."""
    return html.Div([
        html.H3("Fetch Cryptocurrency Data", style={'color': COLORS['text']}),
        html.Label("Symbol (e.g., BTCUSDT):", style={'color': COLORS['text']}),
        dcc.Input(
            id='symbol-input',
            value='BTCUSDT',
            type='text',
            style={'width': '50%', 'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Br(),
        html.Label("Start Date:", style={'color': COLORS['text']}),
        dcc.DatePickerSingle(
            id='start-date-picker',
            date=(datetime.datetime.now() - datetime.timedelta(days=7)).date(),
            display_format='YYYY-MM-DD',
            style={'width': '50%', 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Br(),
        html.Label("End Date:", style={'color': COLORS['text']}),
        dcc.DatePickerSingle(
            id='end-date-picker',
            date=datetime.datetime.now().date(),
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%', 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Br(),
        html.Button('Fetch Data', id='fetch-button', n_clicks=0, style={'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}),
        html.Br(),
        html.Div(id='fetch-status', style={'color': COLORS['text']}),
        html.Hr(),
        html.H3("Reprocess CSV Data", style={'color': COLORS['text']}),
        html.Label("Select CSV File:", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='reprocess-csv-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            style={'width': '50%', 'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': '#000000'}
        ),
        html.Label("Start Date:", style={'color': COLORS['text']}),
        dcc.DatePickerSingle(
            id='reprocess-start-date-picker',
            display_format='YYYY-MM-DD',
            style={'width': '50%', 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Br(),
        html.Label("End Date:", style={'color': COLORS['text']}),
        dcc.DatePickerSingle(
            id='reprocess-end-date-picker',
            display_format='YYYY-MM-DD',
            max_date_allowed=datetime.datetime.now().date(),
            style={'width': '50%', 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Br(),
        html.Label("Target Timeframe:", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='reprocess-timeframe',
            options=[
                {'label': '1 Minute', 'value': '1min'},
                {'label': '5 Minutes', 'value': '5min'},
                {'label': '15 Minutes', 'value': '15min'},
                {'label': '1 Hour', 'value': '1H'},
            ],
            value='1min',
            style={'width': '50%', 'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': '#000000'}
        ),
        html.Button('Reprocess Data', id='reprocess-button', n_clicks=0, style={'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}),
        html.Br(),
        html.Div(id='reprocess-status', style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})