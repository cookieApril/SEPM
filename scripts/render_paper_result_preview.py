"""Compatibility entry point for the actual LaTeX tail preview.
Pass --engine /path/to/tectonic when Tectonic is not on PATH.
"""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(root/'paper/code/build-results-preview.py'),*sys.argv[1:]],check=True)
