from dash import dcc, html
import os
from trading_bot.config import settings
from pathlib import Path

# Define color scheme for consistency
COLORS = {
    'background': "#000000",  # Black background
    'text': "#FFFFFF",        # White text
    'accent': "#000000",      # Gray accent for borders and elements
    'border': '1px solid #333333'
}

def get_data_view_content():
    """Return content for the Data View tab."""
    config_dir = Path('/home/madbob10/Dash/configs')
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]

    # Simplified dropdown style without custom CSS
    dropdown_style = {
        'width': '50%',
        'border': COLORS['border'],
        'backgroundColor': COLORS['accent'],  # Gray background for dropdown
        'color': COLORS['text']              # White text for dropdown
    }

    return html.Div([
        html.H3("View Cryptocurrency Data", style={'color': COLORS['text']}),
        html.Label("Select Cryptocurrency Data:", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            style=dropdown_style
        ),
        dcc.Graph(id='price-chart', style={'border': COLORS['border'], 'backgroundColor': COLORS['accent']}),
        html.Hr(),
        html.H3("Configuration", style={'color': COLORS['text']}),
        html.Label("Select Config File:", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='config-dropdown',
            options=[{'label': f, 'value': f} for f in config_files],
            value='default.json',
            style={**dropdown_style, 'margin-bottom': '10px'}
        ),
        html.Div(
            id='config-editor',
            style={
                'margin': '20px 0',
                'padding': '10px',
                'border': COLORS['border'],
                'backgroundColor': COLORS['accent'],
                'color': COLORS['text'],
                'fontFamily': 'Arial, sans-serif'
            }
        ),
        html.Label("Save Config As:", style={'color': COLORS['text']}),
        dcc.Input(
            id='config-save-name',
            type='text',
            placeholder='Enter config name (e.g., strategy1.json)',
            style={'width': '50%', 'margin-bottom': '10px', 'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}
        ),
        html.Button('Save Config', id='save-config-button', n_clicks=0, style={'margin-left': '10px', 'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}),
        html.Div(id='save-config-status', style={'color': COLORS['text'], 'margin': '10px 0'}),
        html.Hr(),
        html.H3("Run Backtest", style={'color': COLORS['text']}),
        html.Label("Select CSV Files (or Entire File):", style={'color': COLORS['text']}),
        dcc.Dropdown(
            id='backtest-csv-dropdown',
            options=[{'label': 'Entire File', 'value': 'all'}] + [{'label': s, 'value': s} for s in os.listdir(settings.DATA_DIR) if s.endswith('.csv')],
            value=None,
            multi=True,
            style=dropdown_style
        ),
        html.Label("Backtest Options:", style={'color': COLORS['text']}),
        dcc.Checklist(
            id='backtest-options',
            options=[
                {'label': 'Generate Plot', 'value': 'plot'},
                {'label': 'Use LSTM', 'value': 'use_lstm'},
            ],
            value=['plot', 'use_lstm'],
            style={'margin': '10px', 'color': COLORS['text']}
        ),
        html.Button('Run Backtest', id='backtest-button', n_clicks=0, style={'border': COLORS['border'], 'backgroundColor': COLORS['accent'], 'color': COLORS['text']}),
        html.Br(),
        html.Div(id='backtest-status', style={'color': COLORS['text']})
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})