"""Draw Figure 2 from accepted E2 means using Python + matplotlib."""
from result_data import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
(PAPER / 'image').mkdir(exist_ok=True)
# Four aligned dot panels: native units, across-model mean guide, explicit spans.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.edgecolor': '#999999', 'axes.linewidth': .6})
fig, axes = plt.subplots(1, 4, figsize=(5.5, 4.2), sharey=True)
fig.subplots_adjust(left=.31, right=.985, top=.79, bottom=.18, wspace=.32)
colors, markers = ['#0072B2', '#009E73', '#D55E00'], ['o', 's', 'D']
model_style = {m:(colors[i],markers[i]) for i,(_,ms) in enumerate(groups) for m in ms}
for j, (ax, (key, name, n, scale)) in enumerate(zip(axes, benchmarks)):
    values = [by_model[m][key]*scale for m in models]
    ax.axvline(statistics.mean(values), color='#888888', ls='--', lw=.8, zorder=1)
    for y, (m,v) in enumerate(zip(models, values)):
        color, marker = model_style[m]
        ax.scatter(v, y, s=28, c=color, marker=marker, linewidth=.55, edgecolors='white', zorder=3)
    for sep in [5.5, 9.5]:
        ax.axhline(sep, color='#cccccc', lw=.6)
    ax.set_ylim(11.8, -.8)
    ax.set_yticks(range(12))
    ax.set_yticklabels(models if j == 0 else [])
    ax.tick_params(axis='y', length=0, labelsize=7.2, labelleft=(j == 0))
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', color='#e6e6e6', lw=.6)
    for spine in ['top','right','left']:
        ax.spines[spine].set_visible(False)
    panel_name = 'Research\n(MultiAgentBench)' if key == 'multiagentbench_research' else name
    ax.set_title(f'({chr(97+j)}) {panel_name}', fontsize=7.8, pad=9)
    if scale == 100:
        ax.set_xlim(0, 65); ax.set_xticks([0,20,40,60]); ax.set_xlabel('Task score (%)', fontsize=7.5)
    else:
        ax.set_xlim(4.10,4.42); ax.set_xticks([4.1,4.2,4.3,4.4]); ax.set_xlabel('Official rating', fontsize=7.5)
    span = stats[key]['span']
    ax.text(.5,-.18,f'Range: {span:.{3 if scale == 1 else 1}f}'+('\nrating points' if scale == 1 else ' pp'),
            transform=ax.transAxes, ha='center', va='top', fontsize=7.2)
# sharey shares label objects; set the first-axis model labels once at the end.
axes[0].set_yticklabels(models)
legend = [Line2D([0],[0],marker=markers[i],color='none',markerfacecolor=colors[i],markeredgecolor='white',
                 markersize=6,label=group) for i,(group,_) in enumerate(groups)]
legend += [Line2D([0],[0],color='#888888',ls='--',lw=.8,label='Across-model mean')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.59,.955),ncol=4,frameon=False,fontsize=7.0,
           handlelength=1.5, columnspacing=1.1)
for extension in ['pdf','svg','png']:
    fig.savefig(PAPER / f'image/Fig2-E2-procedural-memory-model-sensitivity.{extension}',dpi=300,facecolor='white')
plt.close(fig)

print(PAPER / 'image/Fig2-E2-procedural-memory-model-sensitivity.pdf')
