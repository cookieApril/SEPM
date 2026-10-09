# README figure assets

These PNGs are rasterized from the author's vector PDFs in `paper/image/`. The
`paper/` directory is ignored by Git, so the rendered copies live here for the
repository README. The figures and plotted values were not redrawn or changed.

| README asset | Source PDF | Manuscript figure |
| --- | --- | --- |
| `framework.png` | `Multi-Agents-Main-Fig.pdf` | Figure 2 |
| `case-study.png` | `Casestudy-Fig.pdf` | Figure 3 |
| `multiagentbench.png` | `small-Fig.pdf` | Figure 1 |
| `overall-performance.png` | `Table1-Fig.pdf` | Figure 4(a) |
| `token-cost.png` | `TokenCost.pdf` | Figure 4(b) |
| `graph-vs-text.png` | `graph_fig.pdf` | Figure 5(a) |

The first two were rendered with Poppler `pdftoppm -scale-to 2400`; the remaining
four with `-scale-to 1800`. All commands also used `-f 1 -l 1 -png -singlefile`.
