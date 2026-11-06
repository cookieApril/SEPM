"""Build the SEPM project page (public and anonymous) as HTML and PDF.

Usage: python build.py [--repo ../SEPM] [--browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"]

Outputs (next to this script):
  SEPM_project_page_public.html / .pdf
  SEPM_project_page_anonymous.html / .pdf
  README_anonymous.md   (anonymous README for the supplementary code archive)
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = {
    "public": {
        "VENUE": "Multi-Agent Systems · Procedural Memory",
        "AUTHORS": '<div class="authors">Yi Liu · Junwei Zhao · Li Sun · Jiansong Zhang · Sen Su</div>',
        "BUTTONS": (
            '<a class="btn b1" href="https://arxiv.org/abs/2610.14515">📄 Paper (arXiv)</a>'
            '<a class="btn b2" href="https://github.com/cookieApril/SEPM">💻 Code</a>'
            '<span class="btn b3">🧪 7 benchmarks</span>'
            '<span class="btn b4">🤝 AutoGen · DyLAN · CoELA</span>'
        ),
        "CLONE": "git clone https://github.com/cookieApril/SEPM.git &amp;&amp; cd SEPM",
        "CITATION": """<section class="keep">
  <h2><span class="dot" style="background:var(--sun)">5</span>Citation</h2>
<pre class="tree">@misc{liu2026sepm,
  title         = {SEPM: Self-Evolving Procedural Memory Grounded in Task Graphs
                   for Multi-Agent Collaboration},
  author        = {Yi Liu and Junwei Zhao and Li Sun and Jiansong Zhang and Sen Su},
  year          = {2026},
  eprint        = {2610.14515},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA}
}</pre>
</section>""",
        "FOOTER": "Contact: Yi Liu (louie@bupt.edu.cn) · Code released under the MIT License",
    },
    "anonymous": {
        "VENUE": "CVPR 2027 Submission · Supplementary Material",
        "AUTHORS": '<div class="authors">Anonymous CVPR submission</div>',
        "BUTTONS": (
            '<span class="btn b1">📄 Main paper</span>'
            '<span class="btn b2">💻 Code in supplementary</span>'
            '<span class="btn b3">🧪 7 benchmarks</span>'
            '<span class="btn b4">🤝 AutoGen · DyLAN · CoELA</span>'
        ),
        "CLONE": "cd SEPM                                         <span class=\"c\"># unpacked supplementary archive</span>",
        "CITATION": "",
        "FOOTER": "Anonymous CVPR 2027 submission · Code released under the MIT License",
    },
}


def anonymize_readme(text: str) -> str:
    """Remove every identifying element from the public README."""
    text = re.sub(r'  <a href="https://arxiv\.org[^\n]*\n', "", text)
    text = text.replace(
        "<p align=\"center\"><b>Team memory",
        "<p align=\"center\"><i>Anonymous CVPR 2027 submission · supplementary code</i></p>\n\n"
        "<p align=\"center\"><b>Team memory",
        1,
    )
    text = text.replace(
        "git clone https://github.com/cookieApril/SEPM.git && cd SEPM",
        "cd SEPM                                               # unpacked supplementary archive",
    )
    text = text.replace("- [📚 Citation](#-citation)\n", "")
    text = re.sub(r"\n## 📚 Citation.*?(?=\n## 📄 License)", "\n", text, flags=re.S)
    for marker in ("arxiv.org", "2610.14515", "cookieApril", "louie@", "Yi Liu", "bupt"):
        assert marker.lower() not in text.lower(), f"identifying text left in README: {marker}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=HERE.parent / "SEPM")
    parser.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    args = parser.parse_args()

    repo = args.repo.resolve()
    template = (HERE / "template.html").read_text(encoding="utf-8")
    images = (repo / "docs" / "images").as_uri()
    for name, values in VARIANTS.items():
        html = template.replace("{{IMG}}", images)
        for key, value in values.items():
            html = html.replace("{{" + key + "}}", value)
        assert "{{" not in html, f"unfilled placeholder in {name}"
        out_html = HERE / f"SEPM_project_page_{name}.html"
        out_pdf = HERE / f"SEPM_project_page_{name}.pdf"
        out_html.write_text(html, encoding="utf-8")
        subprocess.run(
            [args.browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             "--virtual-time-budget=8000", f"--print-to-pdf={out_pdf}", out_html.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
        print("wrote", out_pdf)

    readme = (repo / "README.md").read_text(encoding="utf-8")
    (HERE / "README_anonymous.md").write_text(anonymize_readme(readme), encoding="utf-8", newline="\n")
    print("wrote", HERE / "README_anonymous.md")
    (HERE / "README_public.md").write_text(readme, encoding="utf-8", newline="\n")
    print("wrote", HERE / "README_public.md")


if __name__ == "__main__":
    main()
