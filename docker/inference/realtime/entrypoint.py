#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    """
    A simple entrypoint script that checks the passed command 
    and calls gunicorn (or another server) if "serve".
    """
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Start your inference server 
        # e.g., using gunicorn to serve app.py:app on port 8080
        subprocess.call([
            "gunicorn",
            "--bind", "0.0.0.0:8080",
            "app:app"
        ])
    # else:
    #     # Otherwise, just run whatever was passed 
    #     # (useful if you have other commands, e.g. "train")
    #     command = sys.argv[1:]
    #     return_code = subprocess.call(command)
    #     sys.exit(return_code)

if __name__ == "__main__":
    main()
