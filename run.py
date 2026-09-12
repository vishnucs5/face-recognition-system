#!/usr/bin/env python3
"""Main entry point for the Face Recognition Identification System."""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_ui():
    """Launch the Streamlit web dashboard."""
    app_path = PROJECT_ROOT / "app" / "ui" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"Starting Face Recognition Web Dashboard on http://localhost:8501 ...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping dashboard.")


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ["--ui", "ui", "dashboard"]:
        run_ui()
    elif sys.argv[1] == "enroll":
        from scripts import enroll
        sys.argv.pop(1)
        enroll.main()
    elif sys.argv[1] == "identify":
        from scripts import identify
        sys.argv.pop(1)
        identify.main()
    elif sys.argv[1] == "evaluate":
        from scripts import evaluate
        sys.argv.pop(1)
        evaluate.main()
    elif sys.argv[1] in ["prepare-data", "prepare_data", "sample-data"]:
        from scripts import prepare_sample_data
        prepare_sample_data.create_sample_dataset(enroll_identities=True)
    else:
        print("Usage:")
        print("  python run.py                (Launch Streamlit UI)")
        print("  python run.py enroll         --name <name> --image <path>")
        print("  python run.py identify       --image <path> [--threshold <val>]")
        print("  python run.py evaluate       [--test-dir <path>]")
        print("  python run.py prepare-data   (Generate benchmark sample data)")
        sys.exit(1)


if __name__ == "__main__":
    main()
