from dash import Input, Output
import os
from dash_app.config.settings import DATA_DIR
import logging

def register_callback(app):
    """Register the update_dropdowns callback."""
    @app.callback(
        Output('symbol-dropdown', 'options'),
        Output('backtest-csv-dropdown', 'options'),
        Input('fetch-status', 'children'),
        Input('reprocess-status', 'children')
    )
    def update_dropdowns(fetch_status, reprocess_status):
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        logging.debug(f"Dropdown options: {csv_files}")
        options = [{'label': f, 'value': f} for f in csv_files]
        backtest_options = options + [{'label': 'Entire File', 'value': 'all'}]
        return options, backtest_options