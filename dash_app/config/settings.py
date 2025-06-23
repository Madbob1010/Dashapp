import os
from pathlib import Path
from dash_app.config.paths import BASE_DIR, DATA_DIR, RESULTS_DIR, PLOTS_DIR

# Define BASE_DIR relative to settings.py (Dash/dash_app/config -> Dash)
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot settings
PLOT_TEMPLATE = "plotly_dark"
PLOT_HEIGHT = 600
PLOT_WIDTH = 900