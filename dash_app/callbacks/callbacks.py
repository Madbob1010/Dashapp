from dash import Dash
import logging
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('watchdog.observers.inotify').setLevel(logging.WARNING)

def register_callbacks(app: Dash):
    """Register all Dash callbacks by dynamically loading callback modules."""
    # Define the list of callback modules to load
    callback_modules = [
        'render_content',
        'update_dropdowns',
        'fetch_data',
        'reprocess_data',
        'update_chart',
        'run_backtest',
        'config_editor'
    ]

    # Get the directory of the current file
    callback_dir = Path(__file__).parent

    # Dynamically import and register each callback module
    for module_name in callback_modules:
        try:
            # Import the module
            module = importlib.import_module(f'dash_app.callbacks.{module_name}')
            # Check if the module has a register_callback function
            if hasattr(module, 'register_callback'):
                module.register_callback(app)
                logging.info(f"Successfully registered callbacks from {module_name}")
            else:
                logging.warning(f"Module {module_name} does not have a register_callback function")
        except Exception as e:
            logging.error(f"Failed to load callback module {module_name}: {str(e)}")
            raise