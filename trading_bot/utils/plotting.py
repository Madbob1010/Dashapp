import plotly.graph_objects as go
import pandas as pd
from config.settings import PLOT_HEIGHT, PLOT_WIDTH, PLOTS_DIR
import logging
from colorama import Fore, Style

def generate_plot(symbol: str, df: pd.DataFrame, portfolio, best_entries: pd.Series, best_exits: pd.Series, use_lstm: bool) -> str:
    """Generate and save a simplified Plotly HTML plot."""
    print(f"{Fore.CYAN}[INFO] Generating simplified plot for {symbol}{Style.RESET_ALL}")
    logging.debug("Generating simplified plot")
    
    if portfolio.trades.count() == 0:
        logging.warning("No trades generated for the best parameters.")
        print(f"{Fore.YELLOW}[WARN] No trades generated for {symbol}{Style.RESET_ALL}")
    
    # Create plot
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
            y=portfolio.value(),
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
        template='plotly_dark',
        yaxis=dict(title='Price (USD)'),
        yaxis2=dict(
            title='Portfolio Value (USD)',
            overlaying='y',
            side='right'
        )
    )
    
    # Save plot
    plot_filename = f"backtest_{symbol}.html"
    plot_filepath = PLOTS_DIR / plot_filename
    try:
        fig.write_html(str(plot_filepath))
        print(f"{Fore.GREEN}[INFO] Plot saved as {plot_filepath}{Style.RESET_ALL}")
        return str(plot_filepath)
    except Exception as e:
        logging.error(f"Failed to save plot at {plot_filepath}: {e}")
        print(f"{Fore.RED}[ERROR] Could not save plot: {e}{Style.RESET_ALL}")
        return None