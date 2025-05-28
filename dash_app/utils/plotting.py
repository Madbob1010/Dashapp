import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash_app.config.settings import PLOT_TEMPLATE, PLOT_HEIGHT, PLOT_WIDTH

def create_backtest_plot(df: pd.DataFrame, result: tuple, symbol: str, use_lstm: bool):
    """Create a Plotly figure for backtest results."""
    best_key, best_return, portfolio_dict = result[1], result[2], result[3]
    pf = portfolio_dict[best_key]
    
    best_ema = pd.Series(np.nan, index=df.index)
    best_rsi = pd.Series(np.nan, index=df.index)
    best_entries = (df['close'] > best_ema) & (best_rsi < best_key[2])
    best_exits = (df['close'] < best_ema) & (best_rsi > (100 - best_key[2]))
    
    if use_lstm and 'lstm_signals' in portfolio_dict:
        best_entries = best_entries & portfolio_dict['lstm_signals']
        best_exits = best_exits & ~portfolio_dict['lstm_signals']
    
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
        height=PLOT_HEIGHT,
        width=PLOT_WIDTH,
        title=f"Backtest Results for {symbol} (LSTM: {use_lstm})",
        showlegend=True,
        template=PLOT_TEMPLATE,
        yaxis=dict(title='Price (USD)'),
        yaxis2=dict(
            title='Portfolio Value (USD)',
            overlaying='y',
            side='right'
        )
    )
    return fig