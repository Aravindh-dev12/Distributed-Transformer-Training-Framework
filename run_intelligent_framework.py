import os
import sys
import subprocess
import time
import webbrowser

def print_banner():
    banner = """
======================================================================
     █████  ███    ██ ████████ ██  ██████  ██████   █████  ██    ██ ██ 
    ██   ██ ████   ██    ██    ██ ██       ██   ██ ██   ██  ██  ██  ██ 
    ███████ ██ ██  ██    ██    ██ ██   ███ ██████  ███████   ████   ██ 
    ██   ██ ██  ██ ██    ██    ██ ██    ██ ██   ██ ██   ██    ██       
    ██   ██ ██   ████    ██    ██  ██████  ██   ██ ██   ██    ██    ██ 
======================================================================
         INTELLIGENT LLM HUB - DISTRIBUTED TRANSFORMER FRAMEWORK
======================================================================
  * Loaded with: Hugging Face Weights Bridge (Bi-directional)
  * Feature suite: Autoregressive Streaming Inference & Live Sampler
  * Training suite: Fine-tuning console + Dynamic Loss Chart
  * Destination Hub: Host & upload directly to profile 'Aravindhan11'
======================================================================
"""
    print(banner)

def main():
    print_banner()
    
    # Check if inside directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(current_dir, "server.py")
    
    if not os.path.exists(server_path):
        print(f"Error: Could not locate server.py in {current_dir}. Please run from the root directory.")
        sys.exit(1)
        
    print("Checking dependencies...")
    try:
        import torch
        import transformers
        import datasets
        print("-> Core dependencies (PyTorch, Transformers, Datasets) verified successfully.")
    except ImportError as e:
        print(f"Error: Missing core dependency. Please run 'uv sync' or install manually:\n  pip install torch transformers datasets rich")
        sys.exit(1)
        
    port = 8000
    print(f"\nStarting API Server on port {port}...")
    
    # Start server.py in a subprocess
    # Using sys.executable to ensure we use the same Python interpreter
    try:
        process = subprocess.Popen(
            [sys.executable, server_path, str(port)],
            cwd=current_dir
        )
        
        # Give the server a second to boot up
        time.sleep(2)
        
        # Open web browser
        url = f"http://localhost:{port}"
        print(f"\n-> Launching your Web Dashboard at: {url}")
        print("-> Press Ctrl+C in this terminal to stop the server and close the dashboard.")
        
        webbrowser.open(url)
        
        # Wait for the subprocess to finish
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nShutting down Intelligent Framework Hub...")
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            pass
        print("Server stopped. Goodbye!")
    except Exception as e:
        print(f"Error running the command center: {e}")

if __name__ == "__main__":
    main()
