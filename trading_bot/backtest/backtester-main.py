import vectorbt as vbt
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple
from trading_bot.config.settings import PARAM_RANGE, LSTM_LOOKBACK, RESULTS_DIR
from trading_bot.utils.helpers import calculate_ema_talib, calculate_rsi_talib
from trading_bot.models.lstm import train_lstm_model
import pickle
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_signals_for_params(args: Tuple[Tuple[int, int, int], pd.DataFrame, Optional[pd.Series]]) -> Tuple[Tuple[int, int, int], pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute signals for a single parameter combination."""
    key, df, lstm_signals = args
    ema_length, rsi_length, rsi_threshold = key
    logging.debug(f"Processing parameters: {key}")

    try:
        ema_line = calculate_ema_talib(df['close'].values, ema_length)
        rsi = calculate_rsi_talib(df['close'].values, rsi_length)
    except Exception as e:
        logging.error(f"ta-lib calculation failed for {key}: {e}")
        raise

    if len(ema_line) != len(df) or len(rsi) != len(df):
        raise ValueError(f"Length mismatch for {key}: EMA ({len(ema_line)}), RSI ({len(rsi)}), DataFrame ({len(df)})")

    ema_line = pd.Series(ema_line, index=df.index, name=f'ema_{ema_length}').ffill()
    rsi = pd.Series(rsi, index=df.index, name=f'rsi_{rsi_length}').ffill()

    entries = (df['close'] > ema_line) & (rsi < rsi_threshold)
    exits = (df['close'] < ema_line) & (rsi > (100 - rsi_threshold))

    if lstm_signals is not None:
        entries = entries & lstm_signals
        exits = exits & ~lstm_signals

    if entries.isnull().any() or exits.isnull().any():
        raise ValueError(f"Invalid signals for {key}: Contains NaN values")

    return key, entries, exits, ema_line, rsi

def create_backtest_plot(df: pd.DataFrame, entries: pd.Series, exits: pd.Series, ema: pd.Series, rsi: pd.Series, symbol: str) -> go.Figure:
    """Create a plotly figure showing price, buy/sell signals, exits, EMA, and RSI."""
    logging.debug(f"Creating backtest plot for {symbol}")

    # Create subplots: price + EMA on top, RSI on bottom
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol} Price and EMA", "RSI")
    )

    # Plot closing price
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Plot EMA
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=ema,
            mode='lines',
            name=f'EMA {ema.name.split("_")[1]}',
            line=dict(color='orange')
        ),
        row=1, col=1
    )

    # Plot buy signals
    buy_signals = df[entries]['timestamp']
    buy_prices = df[entries]['close']
    fig.add_trace(
        go.Scatter(
            x=buy_signals,
            y=buy_prices,
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ),
        row=1, col=1
    )

    # Plot sell (exit) signals
    sell_signals = df[exits]['timestamp']
    sell_prices = df[exits]['close']
    fig.add_trace(
        go.Scatter(
            x=sell_signals,
            y=sell_prices,
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ),
        row=1, col=1
    )

    # Plot RSI
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=rsi,
            mode='lines',
            name=f'RSI {rsi.name.split("_")[1]}',
            line=dict(color='red')
        ),
        row=2, col=1
    )

    # Add RSI thresholds (30 and 70)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    fig.update_layout(
        title=f'Backtest Results for {symbol}',
        template='plotly_dark',
        showlegend=True,
        xaxis2_title="Date",  # Bottom subplot x-axis
        yaxis_title="Price",
        yaxis2_title="RSI",
        height=800
    )

    # Update x-axes to share zoom/pan
    fig.update_xaxes(matches='x')

    return fig

def run_vectorbt_backtest(
    df: pd.DataFrame,
    use_lstm: bool = False,
    plot: bool = False,
    param_range: Optional[dict] = None,
    lookback: int = LSTM_LOOKBACK
) -> Tuple[Dict[tuple, vbt.Portfolio], tuple, float, Optional[go.Figure]]:
    """Run vectorbt backtest with ta-lib EMA/RSI and optional LSTM signals."""
    logging.debug("Starting vectorbt_backtest")
    if param_range is None:
        param_range = PARAM_RANGE

    required_columns = ['close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")

    if len(df) < max(param_range['ema_length']) + max(param_range['rsi_length']) + lookback:
        raise ValueError(f"CSV has insufficient data for EMA/RSI/LSTM calculations: {len(df)} rows")

    lstm_signals = None
    if use_lstm:
        logging.debug("Training LSTM model")
        try:
            predictions, start_index = train_lstm_model(df['close'].values, lookback=lookback)
            lstm_predictions = pd.Series(np.nan, index=df.index, name='lstm_predictions')
            if start_index + len(predictions) > len(df):
                raise ValueError(f"Predictions (len={len(predictions)}) with start_index={start_index} exceed DataFrame length ({len(df)})")
            lstm_predictions.iloc[start_index:start_index + len(predictions)] = predictions.flatten().tolist()
            lstm_signals = (lstm_predictions > df['close']).astype(bool).fillna(False)
            logging.info(f"LSTM Buy Signals: {lstm_signals.sum()}")
            logging.info(f"LSTM Predictions sample: {lstm_predictions.iloc[start_index:start_index+2].to_list()}")
        except Exception as e:
            logging.error(f"LSTM signal generation failed: {e}")
            raise

    param_combinations = list(product(
        param_range['ema_length'],
        param_range['rsi_length'],
        param_range['rsi_threshold']
    ))

    num_processes = min(cpu_count(), len(param_combinations))
    logging.debug(f"Using {num_processes} processes for {len(param_combinations)} parameter combinations")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            compute_signals_for_params,
            [(key, df, lstm_signals) for key in param_combinations]
        )

    entries = {}
    exits = {}
    ema_dict = {}
    rsi_dict = {}
    for key, entry, exit, ema_line, rsi in results:
        entries[key] = entry
        exits[key] = exit
        ema_dict[key] = ema_line
        rsi_dict[key] = rsi

    logging.debug("Creating portfolios")
    portfolio = {}
    try:
        portfolio = {
            key: vbt.Portfolio.from_signals(
                close=df['close'],
                entries=entries[key],
                exits=exits[key],
                init_cash=10000,
                freq='1d'
            )
            for key in entries
        }
    except Exception as e:
        logging.error(f"Portfolio creation failed: {type(e).__name__} - {e}")
        raise

    logging.debug("Selecting best portfolio")
    best_key = max(portfolio, key=lambda k: portfolio[k].total_return())
    best_portfolio = portfolio[best_key]
    best_entries = entries[best_key]
    best_exits = exits[best_key]
    best_ema = ema_dict[best_key]
    best_rsi = rsi_dict[best_key]
    best_return = best_portfolio.total_return()

    ema_start = max(param_range['ema_length'])
    rsi_start = max(param_range['rsi_length'])
    logging.info(f"Best portfolio for {best_key}: Entries={best_entries.sum()} signals, Exits={best_exits.sum()} signals")
    logging.info(f"Best EMA sample (first 5 valid from index {ema_start}): {best_ema.iloc[ema_start:ema_start+5].to_list()}")
    logging.info(f"Best RSI sample (first 5 valid from index {rsi_start}): {best_rsi.iloc[rsi_start:rsi_start+5].to_list()}")
    logging.info(f"Best RSI stats: Min={best_rsi.min():.2f}, Max={best_rsi.max():.2f}, Mean={best_rsi.mean():.2f}")

    plot_fig = None
    if plot:
        logging.debug("Generating backtest plot")
        try:
            plot_fig = create_backtest_plot(df, best_entries, best_exits, best_ema, best_rsi, df.index.name or "Asset")
        except Exception as e:
            logging.error(f"Plot generation failed: {e}")
            plot_fig = None

    return portfolio, best_key, best_return, plot_fig

def save_results(symbol: str, best_key: tuple, best_return: float, use_lstm: bool):
    """Save backtest results to a pickle file."""
    result_file = RESULTS_DIR / f"results_{symbol}.pkl"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(result_file, 'wb') as f:
        pickle.dump({'best_key': best_key, 'best_return': best_return, 'use_lstm': use_lstm}, f)
    logging.info(f"Results saved to {result_file}")