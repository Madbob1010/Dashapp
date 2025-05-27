#!/usr/bin/env python3
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Define data directory (same as backtestrocm3.py and datalinux2.py)
data_dir = "/home/madbob10/Dash/data/"

# Sample layout
app.layout = html.Div([
    html.H1("Trading Strategy Backtesting Dashboard", style={'textAlign': 'center'}),
    html.Label("Select Cryptocurrency:"),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': s.replace('.csv', ''), 'value': s} for s in os.listdir(data_dir) if s.endswith('.csv')],
        value='BTCUSDT.csv',
        style={'width': '50%'}
    ),
    dcc.Graph(id='price-chart'),
    html.Div(id='backtest-results')
])

# Callback to update the price chart
@app.callback(
    Output('price-chart', 'figure'),
    Input('symbol-dropdown', 'value')
)
def update_chart(selected_symbol):
    csv_path = os.path.join(data_dir, selected_symbol)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig = px.line(df, x='timestamp', y='close', title=f'Close Price for {selected_symbol.replace(".csv", "")}')
    fig.update_layout(template='plotly_dark')
    return fig

if __name__ == '__main__':
    app.run(debug=True)