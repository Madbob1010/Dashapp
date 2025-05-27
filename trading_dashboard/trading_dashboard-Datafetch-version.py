#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import datetime
from datalinux2 import fetch_multiple_cryptos, update_crypto_csv

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
            html.H3("Load Cryptocurrency Data"),
            html.Label("Symbol:"),
            dcc.Input(id='symbol-input', type='text', placeholder='e.g. BTCUSDT'),
            html.Br(),
            html.Label("Start Date (YYYY-MM-DD):"),
            dcc.Input(id='start-date-input', type='text', placeholder='YYYY-MM-DD'),
            html.Br(),
            html.Label("End Date (YYYY-MM-DD):"),
            dcc.Input(id='end-date-input', type='text', placeholder='YYYY-MM-DD'),
            html.Br(),
            html.Button('Fetch Data', id='fetch-button', n_clicks=0),
            html.Br(),
            html.Div(id='fetch-status')
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
            dcc.Graph(id='price-chart')
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
        
        # Fetch data using datalinux2.py
        csv_path = update_crypto_csv(symbol, start_date, end_date, data_dir)
        if csv_path:
            return f"Success: Data for {symbol} saved to {csv_path}"
        else:
            return f"Error: No data fetched for {symbol}."
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

if __name__ == '__main__':
    app.run(debug=True)