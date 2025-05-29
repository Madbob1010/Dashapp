#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
from trading_bot.datalinux2 import fetch_multiple_cryptos, update_crypto_csv, reprocess_csv
from backtestrocm3 import run_backtest

# Initialize Dash app with callback exception suppression
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define directories
DATA_DIR = "/home/madbob10/Dash/data/"
BACKTEST_RESULTS_DIR = "/home/madbob10/Dash/data/backtest_results/"
BACKTEST_PLOTS_DIR = "/home/madbob10/Dash/data/backtest_plots/"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
os.makedirs(BACKTEST_PLOTS_DIR, exist_ok=True)

# Store the latest plot state
app.latest_plot = {'type': None, 'file': None, 'figure': None}

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
    elif tab == 'data-view':
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

# Callback to fetch data
@app.callback(
    Output('fetch-status', 'children'),
    Input('fetch-button', 'n_clicks'),
    State('symbol-input', 'value'),
    State('start-date-picker', 'date'),
    State('end-date-picker', 'date')
)
def fetch_data(n_clicks, symbol, start_date, end_date):
    if n_clicks == 0:
        return "Enter details and click 'Fetch Data' to start."
    
    try:
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        if start_date >= end_date:
            return "Error: Start date must be before end date."
        
        csv_path = update_crypto_csv(symbol, start_date, end_date, DATA_DIR)
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
    State('reprocess-start-date-picker', 'date'),
    State('reprocess-end-date-picker', 'date'),
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
        csv_path = os.path.join(DATA_DIR, csv_file)
        new_csv_path = reprocess_csv(csv_path, start_date, end_date, timeframe, DATA_DIR)
        if new_csv_path:
            return f"Success: Reprocessed data saved to {new_csv_path}"
        else:
            return "Error: No data reprocessed."
    except Exception as e:
        return f"Error: {str(e)}"

# Callback to update the price chart
@app.callback(
    Output('price-chart', 'figure'),
    Input('symbol-dropdown', 'value'),
    Input('backtest-button', 'n_clicks')
)
def update_chart(selected_file, backtest_n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return px.line(title="Select a dataset or run a backtest to view")
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'symbol-dropdown' and selected_file:
        csv_path = os.path.join(DATA_DIR, selected_file)
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            symbol = selected_file.split('_')[0]
            fig = px.line(df, x='timestamp', y='close', title=f'Close Price for {symbol}')
            fig.update_layout(template='plotly_dark')
            app.latest_plot = {'type': 'csv', 'file': csv_path, 'figure': fig}
            return fig
        except Exception as e:
            return px.line(title=f"Error loading {selected_file}: {str(e)}")
    
    elif trigger_id == 'backtest-button' and app.latest_plot['type'] == 'backtest':
        return app.latest_plot['figure']
    
    return px.line(title="Select a dataset or run a backtest to view")

# Callback to run backtest
@app.callback(
    Output('backtest-status', 'children'),
    Output('price-chart', 'figure', allow_duplicate=True),
    Input('backtest-button', 'n_clicks'),
    State('backtest-csv-dropdown', 'value'),
    State('backtest-options', 'value'),
    prevent_initial_call=True
)
def run_backtest_callback(n_clicks, csv_files, options):
    if n_clicks == 0:
        return "Select files and options, then click 'Run Backtest' to start.", dash.no_update
    
    if not csv_files:
        return "Error: Please select at least one CSV file or 'Entire File'.", dash.no_update
    
    try:
        plot = 'plot' in options
        use_lstm = 'use_lstm' in options
        if csv_files == ['all']:
            csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        else:
            csv_files = [os.path.join(DATA_DIR, f) for f in csv_files]
        
        results = []
        latest_fig = None
        for csv_file in csv_files:
            result = run_backtest(csv_file, plot=plot, use_lstm=use_lstm)
            if result[1] is not None:
                symbol = os.path.basename(csv_file).split('_')[0]
                best_key, best_return = result[1], result[2]
                result_text = (
                    f"{symbol}: Best Parameters: EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}, "
                    f"Return: {best_return*100:.2f}%"
                )
                results.append(result_text)
                
                # Update latest plot if this is the last file and plot is enabled
                if plot and csv_file == csv_files[-1]:
                    df = pd.read_csv(csv_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    pf = result[3][best_key]
                    best_ema = pd.Series(np.nan, index=df.index)
                    best_rsi = pd.Series(np.nan, index=df.index)
                    best_entries = (df['close'] > best_ema) & (best_rsi < best_key[2])
                    best_exits = (df['close'] < best_ema) & (best_rsi > (100 - best_key[2]))
                    if use_lstm and result[3].get('lstm_signals') is not None:
                        best_entries = best_entries & result[3]['lstm_signals']
                        best_exits = best_exits & ~result[3]['lstm_signals']
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=pf.value(),
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='yellow'),
                            yaxis="y2"
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[best_entries],
                            y=df['close'][best_entries],
                            mode='markers',
                            name='Entries',
                            marker=dict(symbol='circle', color='green', size=8)
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[best_exits],
                            y=df['close'][best_exits],
                            mode='markers',
                            name='Exits',
                            marker=dict(symbol='circle', color='red', size=8)
                        )
                    )
                    fig.update_layout(
                        height=600,
                        width=900,
                        title=f"Backtest Results for {symbol} (LSTM: {use_lstm})",
                        showlegend=True,
                        template='plotly_dark',
                        yaxis=dict(title='Price (USD)'),
                        yaxis2=dict(
                            title='Portfolio Value (USD)',
                            overlaying='y',
                            side='right'
                        )
                    )
                    latest_fig = fig
                    app.latest_plot = {'type': 'backtest', 'file': csv_file, 'figure': fig}
            else:
                symbol = os.path.basename(csv_file).split('_')[0]
                results.append(f"{symbol}: Failed to process")
        
        return html.Ul([html.Li(r) for r in results]), latest_fig or dash.no_update
    except Exception as e:
        return f"Error: {str(e)}", dash.no_update

if __name__ == '__main__':
    app.run(debug=True)