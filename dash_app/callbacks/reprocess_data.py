from dash import Input, Output, State
import datetime
import os
from trading_bot.datalinux2 import reprocess_csv
from dash_app.config.settings import DATA_DIR
import logging

def register_callback(app):
    """Register the reprocess_data callback."""
    @app.callback(
        Output('reprocess-status', 'children'),
        Input('reprocess-button', 'n_clicks'),
        State('reprocess-csv-dropdown', 'value'),
        State('reprocess-start-date-picker', 'date'),
        State('reprocess-end-date-picker', 'date'),
        State('reprocess-timeframe', 'value')
    )
    def reprocess_data(n_clicks, csv_file, start_date, end_date, timeframe):
        if n_clicks == 0:
            return "Select options and click 'Reprocess Data' to start."
        
        if not csv_file:
            return "Error: Please select a CSV file."
        
        logging.debug(f"Reprocessing {csv_file}, {start_date}, {end_date}, {timeframe}")
        try:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
            csv_path = os.path.join(DATA_DIR, csv_file)
            new_csv_path = reprocess_csv(csv_path, start_date, end_date, timeframe, str(DATA_DIR))
            if new_csv_path:
                logging.debug(f"Reprocessed data saved to {new_csv_path}")
                return f"Success: Reprocessed data saved to {new_csv_path}"
            else:
                logging.error("No data reprocessed")
                return "Error: No data reprocessed."
        except Exception as e:
            logging.error(f"Reprocess error: {str(e)}")
            return f"Error: {str(e)}"