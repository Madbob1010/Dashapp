import os
from pathlib import Path

# Base directory for data
DATA_DIR = Path("/home/madbob10/Dash/data")
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
PLOTS_DIR = DATA_DIR / "backtest_plots"

# Ensure directories exist
for directory in [DATA_DIR, BACKTEST_RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Plot settings
PLOT_TEMPLATE = "plotly_dark"
PLOT_HEIGHT = 600
PLOT_WIDTH = 900