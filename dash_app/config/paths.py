# dash_app/config/paths.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "backtest_results")
PLOTS_DIR = os.path.join(BASE_DIR, "backtest_plots")