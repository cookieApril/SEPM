"""Print the project page to PDF with a headless Chromium-based browser.

Usage: python build.py [--browser PATH]

Writes SEPM_project_page_anonymous.pdf next to the HTML file.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "SEPM_project_page_anonymous.html"
BROWSERS = ("chromium", "chromium-browser", "google-chrome", "microsoft-edge", "msedge")


def find_browser() -> str:
    for name in BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("No Chromium-based browser found; pass --browser PATH.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default=None)
    args = parser.parse_args()

    out_pdf = PAGE.with_suffix(".pdf")
    subprocess.run(
        [args.browser or find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=8000", f"--print-to-pdf={out_pdf}", PAGE.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
