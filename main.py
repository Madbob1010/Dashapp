import sys
import os
import subprocess
import platform
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

def kill_process_on_port(port):
    """Kill any process using the specified port."""
    system = platform.system()
    try:
        if system in ["Linux", "Darwin"]:  # Linux or macOS
            # Find and kill process on port
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}"],
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-9", pid])
                    print(f"Killed process {pid} on port {port}")
        elif system == "Windows":
            # Find process using port
            result = subprocess.run(
                ["netstat", "-aon", "|", "findstr", f":{port}"],
                shell=True,
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line:
                    pid = line.split()[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"])
                    print(f"Killed process {pid} on port {port}")
        else:
            print(f"Unsupported OS: {system}")
    except subprocess.CalledProcessError as e:
        print(f"No process found on port {port} or error killing process: {e}")
    except Exception as e:
        print(f"Unexpected error while killing process on port {port}: {e}")

def select_layout():
    """Prompt user to select a layout once and return the layout with its port."""
    print("Select a layout:")
    print("1. Trading Layout (port 8050)")
    print("2. Default Layout (port 8051)")
    choice = input("Enter 1 or 2 (default is 1): ").strip()
    
    if choice == '2':
        print("Selected Default Layout (port 8051)")
        return get_default_layout(), 8051
    print("Selected Trading Layout (port 8050)")
    return get_trading_layout(), 8050

# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# Store the latest plot state
app.latest_plot = {'type': None, 'file': None, 'figure': None}

# Set layout and port based on user selection
try:
    layout, port = select_layout()
    # Kill any process on the selected port
    kill_process_on_port(port)
    app.layout = layout
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
        app.run(debug=False, port=port)
    except Exception as e:
        print(f"Error running Dash app on port {port}: {e}")
        sys.exit(1)