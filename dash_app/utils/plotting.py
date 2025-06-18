import plotly.graph_objects as go
import pandas as pd
import json
import os
import numpy as np
import traceback
from ..config.settings import PLOT_HEIGHT, PLOT_WIDTH, PLOTS_DIR
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PLOTS_DIR, 'plotting.log')),
        logging.StreamHandler()
    ]
)
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify').setLevel(logging.WARNING)

def generate_plot(symbol: str, df: pd.DataFrame, portfolio, best_entries: pd.Series, best_exits: pd.Series, use_lstm: bool, lstm_predictions: pd.Series = None) -> str:
    """Generate and save a simplified Plotly HTML plot and JSON figure data, including LSTM predictions."""
    logging.info(f"Generating plot for {symbol}")
    
    try:
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            logging.error(f"Invalid df type: {type(df)}")
            return None
        if not isinstance(best_entries, pd.Series) or not isinstance(best_exits, pd.Series):
            logging.error(f"Invalid entries/exits type: entries={type(best_entries)}, exits={type(best_exits)}")
            return None
        
        # Ensure boolean entries/exits and align indices
        best_entries = best_entries.astype(bool).reindex(df.index, fill_value=False)
        best_exits = best_exits.astype(bool).reindex(df.index, fill_value=False)
        logging.info(f"Entries dtype: {best_entries.dtype}, Exits dtype: {best_exits.dtype}")
        
        # Validate portfolio
        if not hasattr(portfolio, 'value'):
            logging.error("Portfolio object lacks 'value' attribute")
            return None
        
        # Convert portfolio value
        portfolio_value = portfolio.value()
        if isinstance(portfolio_value, np.ndarray):
            portfolio_value = portfolio_value.tolist()
        elif isinstance(portfolio_value, pd.Series):
            portfolio_value = portfolio_value.reindex(df.index, fill_value=np.nan)
        else:
            portfolio_value = pd.Series(portfolio_value, index=df.index)
        
        # Handle LSTM predictions
        if use_lstm and lstm_predictions is not None:
            if isinstance(lstm_predictions, np.ndarray):
                lstm_predictions = pd.Series(lstm_predictions.flatten(), index=df.index[-len(lstm_predictions):])
            elif isinstance(lstm_predictions, pd.Series):
                lstm_predictions = lstm_predictions.reindex(df.index, fill_value=np.nan)
            else:
                logging.warning("Invalid LSTM predictions type, ignoring")
                lstm_predictions = None
        
        # Create plot
        fig = go.Figure()
        
        # Add close price
        fig.add_trace(
            go.Scatter(
                x=df.index.astype(str),
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            )
        )
        
        # Add portfolio value
        fig.add_trace(
            go.Scatter(
                x=df.index.astype(str),
                y=portfolio_value,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='yellow'),
                yaxis="y2"
            )
        )
        
        # Add entries
        entry_indices = best_entries[best_entries].index
        if len(entry_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=entry_indices.astype(str),
                    y=df.loc[entry_indices, 'close'],
                    mode='markers',
                    name='Entries',
                    marker=dict(symbol='circle', color='green', size=8)
                )
            )
        
        # Add exits
        exit_indices = best_exits[best_exits].index
        if len(exit_indices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=exit_indices.astype(str),
                    y=df.loc[exit_indices, 'close'],
                    mode='markers',
                    name='Exits',
                    marker=dict(symbol='circle', color='red', size=8)
                )
            )
        
        # Add LSTM predictions
        if use_lstm and lstm_predictions is not None and not lstm_predictions.isna().all():
            valid_predictions = lstm_predictions.dropna()
            if not valid_predictions.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_predictions.index.astype(str),
                        y=valid_predictions,
                        mode='lines',
                        name='LSTM Predictions',
                        line=dict(color='purple', dash='dash')
                    )
                )
        
        # Update layout
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
            ),
            xaxis=dict(title='Date')
        )
        
        # Save plot
        plot_filename = f"backtest_{symbol}.html"
        plot_filepath = Path(PLOTS_DIR) / plot_filename
        json_filename = f"backtest_{symbol}.json"
        json_filepath = Path(PLOTS_DIR) / json_filename
        
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_html(str(plot_filepath))
            logging.info(f"HTML plot saved: {plot_filepath}")
            
            # Convert figure data to JSON-serializable format
            fig_dict = fig.to_dict()
            for trace in fig_dict['data']:
                for key in ['x', 'y']:
                    if key in trace:
                        if isinstance(trace[key], np.ndarray):
                            trace[key] = trace[key].tolist()
                        elif isinstance(trace[key], pd.Series):
                            trace[key] = trace[key].tolist()
                        elif isinstance(trace[key], list):
                            trace[key] = [str(x) if isinstance(x, (datetime, pd.Timestamp)) else x for x in trace[key]]
            with open(json_filepath, 'w') as f:
                json.dump(fig_dict, f)
            logging.info(f"JSON plot saved: {json_filepath}")
            
            if not plot_filepath.exists():
                raise FileNotFoundError(f"HTML file {plot_filepath} was not created")
            if not os.access(plot_filepath, os.R_OK):
                raise PermissionError(f"HTML file {plot_filepath} is not readable")
            if not json_filepath.exists():
                raise FileNotFoundError(f"JSON file {json_filepath} was not created")
            if not os.access(json_filepath, os.R_OK):
                raise PermissionError(f"JSON file {json_filepath} is not readable")
            
            return str(plot_filepath)
        except Exception as e:
            logging.error(f"Failed to save plot at {plot_filepath} or {json_filepath}: {e}\n{traceback.format_exc()}")
            return None
        
    except Exception as e:
        logging.error(f"Plot generation failed for {symbol}: {e}\n{traceback.format_exc()}")
        return None