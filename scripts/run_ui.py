"""Launch the Streamlit chat UI."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "ui" / "streamlit_app.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        cwd=ROOT,
        check=True,
    )
