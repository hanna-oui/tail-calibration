import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Data ──────────────────────────────────────────────────────────────────────
months = [
    'Jan 07','Feb 07','Mar 07','Apr 07','May 07','Jun 07',
    'Jul 07','Aug 07','Sep 07','Oct 07','Nov 07','Dec 07',
    'Jan 08','Feb 08','Mar 08','Apr 08','May 08','Jun 08',
    'Jul 08','Aug 08','Sep 08','Oct 08','Nov 08','Dec 08'
]
t = np.arange(len(months))

realized = np.array([
    1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
    1.9, 2.1, 2.4, 2.7, 3.0, 3.3,
    3.6, 3.9, 4.2, 4.6, 5.0, 5.5,
    6.1, 6.8, 10.2, 12.8, 14.1, 13.5
])

crisis_idx   = [20, 21, 22, 23]
fhat_miss    = [0, 3, 5, 8, 11]   # routine months F-hat misses
ftilde_miss  = crisis_idx          # crisis months F-tilde misses

# ── F-hat bands (blue) ────────────────────────────────────────────────────────
fh_lo = np.where(
    np.isin(t, crisis_idx), realized - 2.8,
    np.where(np.isin(t, fhat_miss), realized + 0.4, realized - 0.65)
)
fh_hi = np.where(
    np.isin(t, crisis_idx), realized + 2.8,
    np.where(np.isin(t, fhat_miss), realized + 1.6, realized + 0.65)
)
fh_med = np.where(np.isin(t, crisis_idx), realized, realized * 0.95)

# ── F-tilde bands (red) ───────────────────────────────────────────────────────
ft_med = realized.copy().astype(float)
for offset, val in enumerate([7.2, 7.8, 8.2, 8.5]):
    ft_med[20 + offset] = val

ft_lo = np.where(np.isin(t, crisis_idx), ft_med - 0.7, ft_med - 0.45)
ft_hi = np.where(np.isin(t, crisis_idx), ft_med + 0.7, ft_med + 0.45)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5), facecolor='white')
ax.set_facecolor('white')

BLUE   = '#185FA5'
RED    = '#993C1D'
BLUE_A = (*plt.matplotlib.colors.to_rgb(BLUE), 0.18) #type:ignore
RED_A  = (*plt.matplotlib.colors.to_rgb(RED),  0.18) #type:ignore

# Crisis shading
ax.axvspan(19.5, 23.5, color='#DC3232', alpha=0.10, zorder=0)
for x in [19.5, 23.5]:
    ax.axvline(x, color='#DC3232', alpha=0.45, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(21.5, 17.7, 'Crisis', color='#DC3232', fontsize=12, ha='center', va='top', fontweight='500')

# F-hat confidence band
ax.fill_between(t, fh_lo, fh_hi, color=BLUE, alpha=0.18, zorder=2)
ax.plot(t, fh_lo, color=BLUE, alpha=0.5, linewidth=0.8, linestyle=(0, (3, 3)), zorder=3)
ax.plot(t, fh_hi, color=BLUE, alpha=0.5, linewidth=0.8, linestyle=(0, (3, 3)), zorder=3)
ax.plot(t, fh_med, color=BLUE, linewidth=1.8, linestyle=(0, (6, 3)), zorder=4)

# F-tilde confidence band
ax.fill_between(t, ft_lo, ft_hi, color=RED, alpha=0.18, zorder=2)
ax.plot(t, ft_lo, color=RED, alpha=0.5, linewidth=0.8, linestyle=(0, (3, 3)), zorder=3)
ax.plot(t, ft_hi, color=RED, alpha=0.5, linewidth=0.8, linestyle=(0, (3, 3)), zorder=3)
ax.plot(t, ft_med, color=RED, linewidth=1.8, linestyle=(0, (6, 3)), zorder=4)

# Realized line
point_colors = ['#DC3232' if i in crisis_idx else '#111111' for i in t]
point_sizes  = [7 if i in crisis_idx else 4 for i in t]
ax.plot(t, realized, color='#111111', linewidth=2.5, zorder=6)
ax.scatter(t, realized, c=point_colors, s=[s**2 for s in point_sizes],
           zorder=7, edgecolors='none')

# Miss crosses
def draw_cross(ax, i, color, size=0.35):
    x, y = t[i], realized[i]
    ax.plot([x - size, x + size], [y - size * 0.8, y + size * 0.8],
            color=color, linewidth=2, zorder=8)
    ax.plot([x + size, x - size], [y - size * 0.8, y + size * 0.8],
            color=color, linewidth=2, zorder=8)

for i in fhat_miss:
    draw_cross(ax, i, BLUE)
for i in ftilde_miss:
    draw_cross(ax, i, RED)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-0.5, 23.5)
ax.set_ylim(0, 18)
ax.set_xticks(t[::3])
ax.set_xticklabels([months[i] for i in t[::3]], rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(0, 19, 2))
ax.set_yticklabels([f'{v}%' for v in range(0, 19, 2)], fontsize=12)
ax.set_ylabel('Mortgage default rate (%)', fontsize=13, color='#555')
ax.tick_params(colors='#555')
ax.grid(color='#111111', alpha=0.08, linewidth=0.6)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=BLUE, alpha=0.5, label='$F_{\\text{Cover}}$ 80% interval (covers crisis)'),
    mpatches.Patch(facecolor=RED,  alpha=0.5, label='$F_{\\text{Miss}}$ 80% interval (misses crisis)'),
    Line2D([0], [0], color='#111111', linewidth=2.5, label='Realized $Y_t$'),
    mpatches.Patch(facecolor='#DC3232', alpha=0.15,
                   edgecolor='#DC3232', label='Crisis months'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          frameon=False, ncol=4, bbox_to_anchor=(0.0, 1.02),
          columnspacing=1.2, handlelength=1.2)

# ── Footer annotation ─────────────────────────────────────────────────────────
fig.text(0.25, -0.04,
         '$F_{\\text{Cover}}$ coverage: $\\mathbf{19/24 \\approx 0.79}$ (misses 5 routine months)',
         ha='center', fontsize=12, color='#555')
fig.text(0.70, -0.04,
         '$F_{\\text{Miss}}$ coverage: $\\mathbf{20/24 \\approx 0.83}$ (misses all 4 crisis months)',
         ha='center', fontsize=12, color='#555')

plt.tight_layout()
plt.savefig('./images/descriptive/mortgage_default_coverage.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()



