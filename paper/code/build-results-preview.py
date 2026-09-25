"""Compile the revised tail without changing the protected front of main.tex.

Usage: python paper/code/build-results-preview.py --engine /path/to/tectonic
Requires Tectonic and its LaTeX bundle. First use may download TeX resources.
The preview starts with section 5 and figure 2, matching the current manuscript.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

PAPER=Path(__file__).resolve().parents[1]
ROOT=PAPER.parent
parser=argparse.ArgumentParser()
parser.add_argument('--engine',default='tectonic')
parser.add_argument('--offline',action='store_true',help='Use only cached TeX resources')
args=parser.parse_args()
engine=shutil.which(args.engine) or str(Path(args.engine).resolve())
scratch=ROOT/'tmp/paper-revision/latex'; scratch.mkdir(parents=True,exist_ok=True)
for suffix in ['*.sty','*.bst','*.bib']:
    for source in PAPER.glob(suffix): shutil.copy2(source,scratch/source.name)
(scratch/'image').mkdir(exist_ok=True)
shutil.copy2(PAPER/'image/Fig2-E2-procedural-memory-model-sensitivity.pdf',scratch/'image/Fig2-E2-procedural-memory-model-sensitivity.pdf')
shutil.copy2(PAPER/'image/Fig3-effectiveness-evidence.pdf',scratch/'image/Fig3-effectiveness-evidence.pdf')
tail=PAPER.joinpath('main.tex').read_text(encoding='utf-8').split(r'\section{Experiments}',1)[1]
header=r'''\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{iclr2027_conference,times}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,url,caption}
\newcommand{\method}{\textsc{GEMS}}
\begin{document}
\setcounter{section}{4}
\setcounter{figure}{1}
\section{Experiments}
'''
# The isolated preview cannot resolve a label defined in the protected front.
# Its actual subsection number is 4.2; main.tex retains the real cross-reference.
tail=tail.replace(r'\ref{sec:blackboard}','4.2')
source=scratch/'Experiments-and-Appendix.tex'; source.write_text(header+tail,encoding='utf-8')
env=dict(os.environ)
env['TECTONIC_CACHE_DIR']=str(ROOT/'tmp/paper-revision/tectonic-cache')
command=[engine,'--untrusted','--keep-logs','--keep-intermediates']
if args.offline: command.append('--only-cached')
subprocess.run([*command,str(source)],cwd=scratch,env=env,check=True)
shutil.copy2(scratch/'Experiments-and-Appendix.pdf',PAPER/'Experiments-and-Appendix.pdf')
print(PAPER/'Experiments-and-Appendix.pdf')
