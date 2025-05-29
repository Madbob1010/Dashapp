from dash import Input, Output, callback_context
import plotly.express as px
import pandas as pd
import os
from dash_app.config.settings import DATA_DIR
import logging

def register_callback(app):
    """Register the update_chart callback."""
    @app.callback(
        Output('price-chart', 'figure'),
        Input('symbol-dropdown', 'value'),
        Input('backtest-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def update_chart(selected_file, backtest_n_clicks):
        ctx = callback_context
        logging.debug(f"Update chart triggered: file={selected_file}, clicks={backtest_n_clicks}, triggered={ctx.triggered}")
        
        if not ctx.triggered:
            logging.debug("No trigger, returning default chart")
            return px.line(title="Select a dataset or run a backtest to view")
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        logging.debug(f"Trigger ID: {trigger_id}")
        
        if trigger_id == 'symbol-dropdown' and selected_file:
            csv_path = os.path.join(DATA_DIR, selected_file)
            logging.debug(f"Loading CSV: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                logging.debug(f"CSV loaded: rows={len(df)}, columns={df.columns.tolist()}")
                
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    logging.error(f"Missing required columns in {csv_path}: {df.columns}")
                    return px.line(title=f"Error: CSV {selected_file} missing 'timestamp' or 'close' column")
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isna().all():
                    logging.error(f"Invalid timestamps in {csv_path}")
                    return px.line(title=f"Error: Invalid timestamps in {selected_file}")
                
                if df['close'].isna().all() or not pd.api.types.is_numeric_dtype(df['close']):
                    logging.error(f"Invalid close prices in {csv_path}")
                    return px.line(title=f"Error: Invalid close prices in {selected_file}")
                
                symbol = selected_file.split('_')[0]
                logging.debug(f"Creating line plot for {symbol}")
                fig = px.line(df, x='timestamp', y='close', title=f'Close Price for {symbol}')
                fig.update_layout(template='plotly_dark')
                app.latest_plot = {'type': 'csv', 'file': csv_path, 'figure': fig}
                logging.debug("Chart created successfully")
                return fig
            except Exception as e:
                logging.error(f"Error loading {csv_path}: {str(e)}")
                return px.line(title=f"Error loading {selected_file}: {str(e)}")
        
        elif trigger_id == 'backtest-button' and app.latest_plot.get('type') == 'backtest':
            logging.debug(f"Returning backtest plot: {app.latest_plot}")
            return app.latest_plot['figure']
        
        logging.debug("Returning default chart (no valid trigger)")
        return px.line(title="Select a dataset or run a backtest to view")