import os
from pathlib import Path

# Base directory
BASE_DIR = Path('/home/madbob10/Dash')

# Data directories
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = DATA_DIR / 'backtest_results'
PLOTS_DIR = DATA_DIR / 'backtest_plots'

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Backtest parameter ranges
PARAM_RANGE = {
    'ema_length': list(range(15, 45, 5)),
    'rsi_length': list(range(15, 25, 5)),
    'rsi_threshold': list(range(30, 60, 5))
}

# LSTM parameters
LSTM_LOOKBACK = 15
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 128
USE_LSTM = True
PLOT_ENABLED = True

# Plotting settings
PLOT_HEIGHT = 600
PLOT_WIDTH = 900
PLOT_TEMPLATE = 'plotly_dark'

# Vectorbt settings
VBT_SETTINGS = {
    'plotting': {
        'layout': {
            'template': PLOT_TEMPLATE
        }
    }
}