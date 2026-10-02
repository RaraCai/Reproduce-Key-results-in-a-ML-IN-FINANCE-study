# src/plotting.py
"""Plotting configuration and helpers."""
import matplotlib.pyplot as plt
from matplotlib import rcParams

# font
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# fontsize
rcParams['font.size'] = 13
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['legend.fontsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

# lines and ticks
rcParams['axes.linewidth'] = 1.0
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 5
rcParams['lines.markeredgewidth'] = 0.6
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4

# grid
rcParams['grid.linestyle'] = ':'
rcParams['grid.alpha'] = 0.5
rcParams['grid.linewidth'] = 0.6

# legend
rcParams['legend.frameon'] = True
rcParams['legend.framealpha'] = 0.95
rcParams['legend.edgecolor'] = '0.8'
rcParams['legend.columnspacing'] = 0.8
rcParams['legend.handlelength'] = 1.5
rcParams['legend.labelspacing'] = 0.35
rcParams['legend.borderpad'] = 0.45

# output
rcParams['figure.dpi'] = 120
rcParams['savefig.dpi'] = 300

FIG_FORMAT = 'pdf'
FIGSIZE = (8, 6)

# colors and markers
COLORS = ['#8074C8','#7895C1','#A8CBDF',
          '#D6EFF4','#F2FAFC','#992224',
          '#B54764','#E3625D','#EF8B67',
          '#F0C284','#F5EBAE','#F7FBC9']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

REF_LINE = dict(color='#888888', linestyle='--', linewidth=1.0, alpha=0.8)


def _get_style(j, markevery=1):
    return dict(
        color=COLORS[j % len(COLORS)],
        marker=MARKERS[j % len(MARKERS)],
        markeredgecolor='white',     
        markevery=markevery,
    )


def plot_metric_grid(ax, cq_grid, results, z_grid, metric_name,
                     plot_z_indices=None, ridgeless_label=False,
                     markevery=1):
    if plot_z_indices is None:
        plot_z_indices = range(len(z_grid))
    for j in plot_z_indices:
        z_val = z_grid[j]
        label = f"$z = {z_val}$"
        if ridgeless_label and z_val == 0:
            label += " (Ridgeless)"
        ax.plot(cq_grid, results[metric_name][:, j], label=label,
                **_get_style(j, markevery))


def add_interpolation_marker(ax, label=r"$cq = 1$"):
    ax.axvline(1.0, label=label, **REF_LINE)


def finalize_and_save(fig, ax, *, savepath, title, xlabel, ylabel,
                      ylim=None, yscale=None, legend=True,
                      loc='upper right', ncol=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, which='both' if yscale == 'log' else 'major')
    if legend:
        if ncol is None:
            ncol = 1 if len(ax.get_lines()) <= 3 else 2
        ax.legend(ncol=ncol, loc=loc)
    fig.tight_layout()
    fig.savefig(savepath)