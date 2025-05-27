#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import datetime
from datalinux2 import fetch_multiple_cryptos, update_crypto_csv, reprocess_csv
from backtestrocm3 import run_backtest

# Initialize Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define data directory
data_dir = "/home/madbob10/Dash/data/"

# Layout with tabs
app.layout = html.Div([
    html.H1("Trading Strategy Backtesting Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='data-load', children=[
        dcc.Tab(label='Data Load', value='data-load'),
        dcc.Tab(label='Data View', value='data-view'),
    ]),
    html.Div(id='tabs-content')
])

# Render tab content
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'data-load':
        return html.Div([
            html.H3("Fetch Cryptocurrency Data"),
            html.Label("Symbol (e.g., BTCUSDT):"),
            dcc.Input(id='symbol-input', value='BTCUSDT', type='text', style={'width': '50%'}),
            html.Br(),
            html.Label("Start Date (YYYY-MM-DD):"),
            dcc.Input(id='start-date-input', value=(datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d'), type='text', style={'width': '50%'}),
            html.Br(),
            html.Label("End Date (YYYY-MM-DD):"),
            dcc.Input(id='end-date-input', value=datetime.datetime.now().strftime('%Y-%m-%d'), type='text', style={'width': '50%'}),
            html.Br(),
            html.Button('Fetch Data', id='fetch-button', n_clicks=0),
            html.Br(),
            html.Div(id='fetch-status'),
            html.Hr(),
            html.H3("Reprocess CSV Data"),
            html.Label("Select CSV File:"),
            dcc.Dropdown(
                id='reprocess-csv-dropdown',
                options=[{'label': s, 'value': s} for s in os.listdir(data_dir) if s.endswith('.csv')],
                value=None,
                style={'width': '50%'}
            ),
            html.Label("Start Date (YYYY-MM-DD):"),
            dcc.Input(id='reprocess-start-date', value='', type='text', style={'width': '50%'}),
            html.Br(),
            html.Label("End Date (YYYY-MM-DD):"),
            dcc.Input(id='reprocess-end-date', value='', type='text', style={'width': '50%'}),
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
    elif tab == 'data-view':
        return html.Div([
            html.H3("View Cryptocurrency Data"),
            html.Label("Select Cryptocurrency Data:"),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': s, 'value': s} for s in os.listdir(data_dir) if s.endswith('.csv')],
                value=None,
                style={'width': '50%'}
            ),
            dcc.Graph(id='price-chart'),
            html.Hr(),
            html.H3("Run Backtest"),
            html.Label("Select CSV Files (or Entire File):"),
            dcc.Dropdown(
                id='backtest-csv-dropdown',
                options=[{'label': 'Entire File', 'value': 'all'}] + [{'label': s, 'value': s} for s in os.listdir(data_dir) if s.endswith('.csv')],
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

# Callback to fetch data
@app.callback(
    Output('fetch-status', 'children'),
    Input('fetch-button', 'n_clicks'),
    State('symbol-input', 'value'),
    State('start-date-input', 'value'),
    State('end-date-input', 'value')
)
def fetch_data(n_clicks, symbol, start_date, end_date):
    if n_clicks == 0:
        return "Enter details and click 'Fetch Data' to start."
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        if start_date >= end_date:
            return "Error: Start date must be before end date."
        
        csv_path = update_crypto_csv(symbol, start_date, end_date, data_dir)
        if csv_path:
            return f"Success: Data for {symbol} saved to {csv_path}"
        else:
            return f"Error: No data fetched for {symbol}."
    except Exception as e:
        return f"Error: {str(e)}"

# Callback to reprocess data
@app.callback(
    Output('reprocess-status', 'children'),
    Input('reprocess-button', 'n_clicks'),
    State('reprocess-csv-dropdown', 'value'),
    State('reprocess-start-date', 'value'),
    State('reprocess-end-date', 'value'),
    State('reprocess-timeframe', 'value')
)
def reprocess_data(n_clicks, csv_file, start_date, end_date, timeframe):
    if n_clicks == 0:
        return "Select options and click 'Reprocess Data' to start."
    
    if not csv_file:
        return "Error: Please select a CSV file."
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
        csv_path = os.path.join(data_dir, csv_file)
        new_csv_path = reprocess_csv(csv_path, start_date, end_date, timeframe, data_dir)
        if new_csv_path:
            return f"Success: Reprocessed data saved to {new_csv_path}"
        else:
            return "Error: No data reprocessed."
    except Exception as e:
        return f"Error: {str(e)}"

# Callback to update the price chart
@app.callback(
    Output('price-chart', 'figure'),
    Input('symbol-dropdown', 'value')
)
def update_chart(selected_file):
    if not selected_file:
        return px.line(title="Select a dataset to view")
    
    csv_path = os.path.join(data_dir, selected_file)
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        symbol = selected_file.split('_1m_')[0]
        fig = px.line(df, x='timestamp', y='close', title=f'Close Price for {symbol}')
        fig.update_layout(template='plotly_dark')
        return fig
    except Exception as e:
        return px.line(title=f"Error loading {selected_file}: {str(e)}")

# Callback to run backtest
@app.callback(
    Output('backtest-status', 'children'),
    Input('backtest-button', 'n_clicks'),
    State('backtest-csv-dropdown', 'value'),
    State('backtest-options', 'value')
)
def run_backtest_callback(n_clicks, csv_files, options):
    if n_clicks == 0:
        return "Select files and options, then click 'Run Backtest' to start."
    
    if not csv_files:
        return "Error: Please select at least one CSV file or 'Entire File'."
    
    try:
        plot = 'plot' in options
        use_lstm = 'use_lstm' in options
        if csv_files == ['all']:
            csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            csv_files = [os.path.join(data_dir, f) for f in csv_files]
        
        results = []
        for csv_file in csv_files:
            result = run_backtest(csv_file, plot=plot, use_lstm=use_lstm)
            if result[1] is not None:  # Check if backtest succeeded
                symbol = os.path.basename(csv_file).split('_1m_')[0]
                best_key, best_return = result[1], result[2]
                results.append(
                    f"{symbol}: Best Parameters: EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}, "
                    f"Return: {best_return*100:.2f}%"
                )
            else:
                symbol = os.path.basename(csv_file).split('_1m_')[0]
                results.append(f"{symbol}: Failed to process")
        
        return html.Ul([html.Li(r) for r in results])
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)