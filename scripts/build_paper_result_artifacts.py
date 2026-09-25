"""Compatibility entry point; maintained code lives in paper/code."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
for name in ['refresh-result-tables.py','draw-Fig2-E2-procedural-memory-model-sensitivity.py','draw-Fig3-effectiveness-evidence.py']:
    subprocess.run([sys.executable,str(root/'paper/code'/name)],check=True)
