"""Draw Figure 3 from accepted E1 means and matched E3 comparisons."""
from result_data import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

(PAPER / 'image').mkdir(exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.edgecolor': '#888888', 'axes.linewidth': .6,
})

fig = plt.figure(figsize=(7.0, 4.4))
grid = fig.add_gridspec(2, 2, width_ratios=[.98, 1.38], height_ratios=[1, 1],
                        left=.22, right=.985, top=.82, bottom=.13, wspace=.78, hspace=.92)
ax_host = fig.add_subplot(grid[:, 0])
ax_pairs = fig.add_subplot(grid[0, 1])
ax_contrib = fig.add_subplot(grid[1, 1])

# (a) Every E1 benchmark-host block. Connected endpoints show the GEMS change;
# G-Memory appears only where that method has an evaluated result.
blocks = [
    ('ALFWorld / AutoGen', 'alfworld', 'autogen'),
    ('ALFWorld / DyLAN', 'alfworld', 'dylan'),
    ('WebArena / AutoGen', 'webarena', 'autogen'),
    ('OfficeBench / AutoGen', 'officebench', 'autogen'),
]
y = np.arange(len(blocks))[::-1]
for yi, (label, benchmark, host) in zip(y, blocks):
    baseline = 100 * lookup[(host, 'no-memory', benchmark)]['official_task_score_mean']
    gems = 100 * lookup[(host, 'team-memory', benchmark)]['official_task_score_mean']
    delta = gems - baseline
    edge = '#009E73' if delta > 1e-9 else '#8C8C8C'
    ax_host.annotate('', xy=(gems, yi + .065), xytext=(baseline, yi - .065),
                     arrowprops=dict(arrowstyle='-|>', color=edge, lw=1.6, shrinkA=5, shrinkB=5))
    ax_host.scatter(baseline, yi - .065, s=42, facecolor='white', edgecolor='#0072B2', lw=1.4, zorder=3)
    ax_host.scatter(gems, yi + .065, s=45, facecolor='#009E73', edgecolor='white', lw=.6, zorder=4)
    gmemory = lookup.get((host, 'gmemory', benchmark))
    if gmemory:
        gm = 100 * gmemory['official_task_score_mean']
        ax_host.scatter(gm, yi, s=48, marker='D', facecolor='#CC79A7', edgecolor='white', lw=.6, zorder=4)
        ax_host.text(gm, yi + .19, f'{gm:.1f}', color='#8E4C78', ha='center', va='bottom', fontsize=6.6)
    ax_host.text(baseline, yi - .20, f'{baseline:.1f}', color='#0072B2', ha='center', va='top', fontsize=6.6)
    ax_host.text(gems, yi + .20, f'{gems:.1f}', color='#007A5E', ha='center', va='bottom', fontsize=6.6)
    ax_host.text(92, yi, f'{delta:+.1f} pp', color=edge, ha='right', va='center', fontsize=6.8, fontweight='bold')
ax_host.set_yticks(y, [b[0] for b in blocks])
ax_host.set_xlim(0, 95)
ax_host.set_xticks([0, 20, 40, 60, 80])
ax_host.set_xlabel('Official task score (%)')
ax_host.set_title('(a) Host-conditioned E1 effects', loc='left', y=1.13, fontsize=8.5, fontweight='bold')
ax_host.grid(axis='x', color='#E7E7E7', lw=.6)
ax_host.tick_params(axis='y', length=0)
ax_host.spines[['top', 'right', 'left']].set_visible(False)
ax_host.legend(handles=[
    Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#0072B2', label='No added'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor='#CC79A7', markeredgecolor='white', label='G-Memory'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='#009E73', markeredgecolor='white', label='GEMS'),
], frameon=False, ncol=3, fontsize=6.5, loc='lower left', bbox_to_anchor=(-.03, 1.005),
   handletextpad=.25, columnspacing=.65)

# (b) Paired case transitions. Improvements and regressions diverge from zero;
# circle area and label encode tied cases.
benchmarks_display = ['ALFWorld', 'WebArena', 'MAB Research', 'OfficeBench']
benchmarks_raw = ['ALFWorld', 'WebArena', 'MultiAgentBench Research', 'OfficeBench']
counts = [e3['benchmarks'][name]['full_vs_no_extra'] for name in benchmarks_raw]
y2 = np.arange(4)[::-1]
for yi, count in zip(y2, counts):
    total = sum(count[k] for k in ['improved', 'tie', 'regressed'])
    imp = 100 * count['improved'] / total
    reg = 100 * count['regressed'] / total
    tie = 100 * count['tie'] / total
    ax_pairs.barh(yi, imp, left=0, height=.46, color='#009E73')
    ax_pairs.barh(yi, -reg, left=0, height=.46, color='#D55E00')
    ax_pairs.scatter(0, yi, s=28 + 1.5 * tie, color='#B8B8B8', edgecolor='white', lw=.7, zorder=4)
    ax_pairs.text(0, yi, str(count['tie']), ha='center', va='center', fontsize=6.4, fontweight='bold', zorder=5)
    if count['improved']:
        ax_pairs.text(imp + 1.3, yi, str(count['improved']), ha='left', va='center', color='#007A5E', fontweight='bold')
    if count['regressed']:
        ax_pairs.text(-reg - 1.3, yi, str(count['regressed']), ha='right', va='center', color='#A54400', fontweight='bold')
ax_pairs.axvline(0, color='#777777', lw=.7)
ax_pairs.set_yticks(y2, benchmarks_display)
ax_pairs.set_xlim(-20, 48)
ax_pairs.set_xticks([-20, 0, 20, 40])
ax_pairs.set_xticklabels(['20', '0', '20', '40'])
ax_pairs.set_xlabel('Share of matched cases (%)')
ax_pairs.set_title('(b) Paired E3 outcome balance', loc='left', fontsize=8.5, fontweight='bold')
ax_pairs.tick_params(axis='y', length=0)
ax_pairs.grid(axis='x', color='#ECECEC', lw=.6)
ax_pairs.spines[['top', 'right', 'left']].set_visible(False)
ax_pairs.text(.99, .02, 'circle = tied cases', transform=ax_pairs.transAxes, ha='right', va='bottom', color='#666666', fontsize=6.4)

# (c) Contribution contrasts. Color magnitude is normalized within each
# benchmark because Research uses rating points and the others use percentage points.
contrast_keys = ['joint', 'c1', 'c3']
row_labels = ['Joint system', '+ Procedural memory', '+ Alignment']
column_labels = ['ALFWorld', 'WebArena', 'MAB\nResearch', 'OfficeBench']
raw = np.array([[contrasts[key][name] for name in benchmarks_raw] for key in contrast_keys], dtype=float)
denom = np.maximum(np.max(np.abs(raw), axis=0), 1e-12)
normalized = raw / denom
cmap = LinearSegmentedColormap.from_list('effect', ['#D55E00', '#FAFAFA', '#009E73'])
ax_contrib.imshow(normalized, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        value = raw[i, j]
        label = f'{value:+.3f}' if j == 2 else f'{value:+.1f}'
        ax_contrib.text(j, i, label, ha='center', va='center', fontsize=7,
                        color='white' if abs(normalized[i, j]) > .68 else '#333333', fontweight='bold')
ax_contrib.set_xticks(range(4), column_labels)
ax_contrib.set_yticks(range(3), row_labels)
ax_contrib.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
ax_contrib.set_title('(c) Contribution contrast map', loc='left', fontsize=8.5, fontweight='bold', pad=8)
for spine in ax_contrib.spines.values():
    spine.set_visible(False)
ax_contrib.text(.5, -.19, 'annotations: pp; MAB Research: rating points', transform=ax_contrib.transAxes,
                ha='center', va='top', fontsize=6.4, color='#555555')

for extension in ['pdf', 'svg', 'png']:
    fig.savefig(PAPER / f'image/Fig3-effectiveness-evidence.{extension}', dpi=300, facecolor='white')
plt.close(fig)
print(PAPER / 'image/Fig3-effectiveness-evidence.pdf')
