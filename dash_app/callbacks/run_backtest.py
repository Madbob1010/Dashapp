from dash import Input, Output, State, html, dash
import pandas as pd
import os
import json
import plotly.graph_objects as go
from trading_bot.backtest.backtester.backtest import run_vectorbt_backtest
from trading_bot.backtest.backtester.utils import save_results
from dash_app.config.settings import DATA_DIR, PLOTS_DIR, PLOT_TEMPLATE, PLOT_HEIGHT, PLOT_WIDTH
import logging
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'backtest.log')),
        logging.StreamHandler()
    ]
)
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify').setLevel(logging.WARNING)

def register_callback(app):
    """Register the run_backtest callback."""
    @app.callback(
        Output('backtest-status', 'children'),
        Output('price-chart', 'figure', allow_duplicate=True),
        Input('backtest-button', 'n_clicks'),
        State('backtest-csv-dropdown', 'value'),
        State('backtest-options', 'value'),
        prevent_initial_call=True
    )
    def run_backtest_callback(n_clicks, csv_files, options):
        """
        Execute backtest for selected CSV files and display results and plot.

        Args:
            n_clicks (int): Number of clicks on the backtest button.
            csv_files (list): List of selected CSV files from dropdown.
            options (list): List of selected options (e.g., 'use_lstm', 'plot').

        Returns:
            tuple: (HTML list of results, Plotly figure)
        """
        logging.info(f"Backtest triggered: files={csv_files}, options={options}")

        # Default return value to avoid SchemaTypeValidationError
        default_result = (
            html.P("Backtest failed unexpectedly", style={'color': 'red'}),
            go.Figure().update_layout(
                title="No plot generated",
                template=PLOT_TEMPLATE,
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH
            )
        )

        # Validate inputs
        if n_clicks == 0:
            return html.P("Select files and options, then click 'Run Backtest' to start."), dash.no_update

        if not csv_files:
            logging.error("No CSV files selected")
            return html.P("Error: Please select at least one CSV file or 'Entire File'.", style={'color': 'red'}), go.Figure().update_layout(
                title="No plot generated",
                template=PLOT_TEMPLATE,
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH
            )

        # Process options
        use_lstm = 'use_lstm' in (options or [])
        plot_enabled = 'plot' in (options or [])
        logging.info(f"Plot enabled: {plot_enabled}, Use LSTM: {use_lstm}")

        # Handle 'all' CSV files
        try:
            if csv_files == ['all']:
                csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            else:
                csv_files = [f for f in csv_files]
        except Exception as e:
            logging.error(f"Failed to list CSV files: {e}\n{traceback.format_exc()}")
            return html.P(f"Error listing CSV files: {e}", style={'color': 'red'}), go.Figure().update_layout(
                title="No plot generated",
                template=PLOT_TEMPLATE,
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH
            )

        results = []
        latest_fig = go.Figure().update_layout(
            title="No valid plot generated",
            template=PLOT_TEMPLATE,
            height=PLOT_HEIGHT,
            width=PLOT_WIDTH
        )

        for csv_file in csv_files:
            csv_path = os.path.join(DATA_DIR, csv_file)
            logging.info(f"Running backtest on {csv_path}")

            try:
                # Load and validate CSV
                df = pd.read_csv(csv_path)
                if 'timestamp' not in df.columns or 'close' not in df.columns:
                    logging.error(f"Invalid CSV {csv_file}: missing required columns")
                    results.append(f"{csv_file.split('_')[0]}: Invalid CSV")
                    continue

                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df.set_index('timestamp', inplace=True)
                if df.index.isna().all():
                    logging.error(f"Invalid timestamps in {csv_file}")
                    results.append(f"{csv_file.split('_')[0]}: Invalid timestamps")
                    continue

                min_rows = 75 if use_lstm else 60  # EMA(45) + RSI(15) + LSTM(15)
                if df['close'].isna().any() or len(df) < min_rows:
                    logging.error(f"Error: Insufficient rows or NaN values in {csv_file}, rows={len(df)}, required={min_rows}")
                    results.append(f"{csv_file.split('_')[0]}: Invalid data")
                    continue

                # Extract symbol
                symbol = csv_file.split('_')[0]

                # Run backtest in main process
                start_time = time.time()
                try:
                    logging.info("Starting backtest execution")
                    result = run_vectorbt_backtest(
                        df=df,
                        symbol=symbol,
                        use_lstm=use_lstm,
                        plot=plot_enabled
                    )
                    elapsed_time = time.time() - start_time
                    logging.info(f"Backtest completed for {symbol} in {elapsed_time:.2f} seconds")
                except Exception as e:
                    logging.error(f"Backtest execution failed for {symbol}: {e}\n{traceback.format_exc()}")
                    results.append(f"{symbol}: Error in backtest: {e}")
                    continue

                # Validate result
                if result is None or len(result) != 7:
                    logging.error(f"Backtest returned invalid result for {symbol}: {result}")
                    results.append(f"{symbol}: Backtest failed")
                    continue

                portfolio, best_key, best_return, entries, exits, lstm_predictions, plot_filepath = result

                if best_key is not None:
                    # Format result for display
                    result_text = (
                        f"{symbol}: Best Parameters: EMA={best_key[0]}, RSI={best_key[1]}, RSI_Threshold={best_key[2]}, "
                        f"Return: {best_return*100:.2f}%"
                    )
                    results.append(result_text)

                    # Save results
                    try:
                        save_results(symbol, best_key, best_return, use_lstm, plot_filepath)
                    except Exception as e:
                        logging.error(f"Failed to save results for {symbol}: {e}")
                        results.append(f"{symbol}: Error saving results: {e}")

                    # Load plot if enabled
                    if plot_enabled:
                        try:
                            logging.info(f"Loading plot for {symbol}")
                            if not plot_filepath or not os.path.exists(plot_filepath):
                                logging.error(f"Plot file not generated for {symbol}")
                                latest_fig = go.Figure().update_layout(
                                    title=f"No plot generated for {symbol}",
                                    template=PLOT_TEMPLATE,
                                    height=PLOT_HEIGHT,
                                    width=PLOT_WIDTH
                                )
                            else:
                                json_filepath = plot_filepath.replace('.html', '.json')
                                if os.path.exists(json_filepath):
                                    with open(json_filepath, 'r') as f:
                                        fig_dict = json.load(f)
                                    latest_fig = go.Figure(fig_dict)
                                    app.latest_plot = {'type': 'backtest', 'file': csv_path, 'figure': latest_fig}
                                    logging.info(f"Backtest plot loaded for {symbol}")
                                else:
                                    logging.error(f"JSON file {json_filepath} not found")
                                    latest_fig = go.Figure().update_layout(
                                        title=f"JSON file not found for {symbol}",
                                        template=PLOT_TEMPLATE,
                                        height=PLOT_HEIGHT,
                                        width=PLOT_WIDTH
                                    )
                        except Exception as e:
                            logging.error(f"Failed to load plot for {symbol}: {e}\n{traceback.format_exc()}")
                            latest_fig = go.Figure().update_layout(
                                title=f"Failed to load plot for {symbol}: {e}",
                                template=PLOT_TEMPLATE,
                                height=PLOT_HEIGHT,
                                width=PLOT_WIDTH
                            )
                    logging.info(f"LSTM predictions status: {lstm_predictions is not None}")
                else:
                    logging.warning(f"Backtest failed for {symbol}: No valid results")
                    results.append(f"{symbol}: Failed to process")
            except Exception as e:
                logging.error(f"Backtest failed for {csv_file}: {e}\n{traceback.format_exc()}")
                results.append(f"{symbol}: Error: {e}")

        logging.info(f"Backtest results: {results}")
        if not plot_enabled:
            logging.warning("Plot option not enabled; no plot generated")
            latest_fig = go.Figure().update_layout(
                title="Plotting not enabled",
                template=PLOT_TEMPLATE,
                height=PLOT_HEIGHT,
                width=PLOT_WIDTH
            )

        # Format results as a table
        if results:
            result_table = html.Table(
                # Header
                [html.Tr([
                    html.Th("Symbol"),
                    html.Th("EMA"),
                    html.Th("RSI"),
                    html.Th("RSI Threshold"),
                    html.Th("Return (%)")
                ], style={'background-color': '#2c3e50', 'color': 'white'})] +
                # Rows
                [
                    html.Tr([
                        html.Td(result.split(':')[0]),
                        html.Td(result.split('EMA=')[1].split(',')[0]),
                        html.Td(result.split('RSI=')[1].split(',')[0]),
                        html.Td(result.split('RSI_Threshold=')[1].split(',')[0]),
                        html.Td(result.split('Return: ')[1].strip('%'))
                    ], style={'background-color': '#34495e' if i % 2 else '#2c3e50', 'color': 'white'})
                    for i, result in enumerate(results) if 'Error' not in result
                ],
                style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'margin': '10px 0',
                    'font-size': '14px',
                    'text-align': 'center'
                }
            )
            # Add error messages below the table
            error_messages = [
                html.P(result, style={'color': 'red', 'margin': '5px 0'})
                for result in results if 'Error' in result
            ]
            result_html = html.Div([result_table] + error_messages)
        else:
            result_html = html.P("No backtest results to display", style={'color': 'white'})

        return result_html, latest_fig