import sys
from pathlib import Path
from dash import Dash

# Add project root to PYTHONPATH (use relative path for portability)
sys.path.append(str(Path(__file__).parent))

try:
    from dash_app.components.layout.global_vars import get_layout as get_default_layout
    from live_trading_dashboard.components.live_trading_dashboard import get_layout as get_trading_layout
    from dash_app.callbacks import callbacks
except ImportError as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

def select_layout():
    """Prompt user to select a layout once and return the corresponding layout."""
    print("Select a layout:")
    print("1. Trading Layout")
    print("2. Default Layout")
    choice = input("Enter 1 or 2 (default is 1): ").strip()
    
    if choice == '2':
        print("Selected Default Layout")
        return get_default_layout()
    print("Selected Trading Layout")
    return get_trading_layout()

# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Store the latest plot state
app.latest_plot = {'type': None, 'file': None, 'figure': None}

# Set layout based on user selection
try:
    app.layout = select_layout()
except Exception as e:
    print(f"Error setting layout: {e}")
    sys.exit(1)

# Register callbacks
try:
    callbacks.register_callbacks(app)
except Exception as e:
    print(f"Error registering callbacks: {e}")
    sys.exit(1)

if __name__ == '__main__':
    try:
        app.run(debug=False, port=8050)  # Debug disabled to prevent auto-reloading
    except Exception as e:
        print(f"Error running Dash app: {e}")
        sys.exit(1)