"""Regenerate paper figures from the measured per-factor tables.

Reads the two CSVs written by `make bias` in the project root, so the figure
cannot drift from the numbers quoted in the text.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "project" / "reports"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family":"serif","font.serif":["CMU Serif","DejaVu Serif"],
    "font.size":8,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.0})

FUND = {'book_to_market','earnings_yield','cash_flow_yield','sales_to_price','gross_profitability',
        'roe','operating_margin','asset_growth','accruals','net_share_issuance','turnover_1m'}

d = pd.read_csv(R/'pit_vs_naive.csv').set_index('factor')
s = pd.read_csv(R/'survivorship.csv').set_index('factor')
f = pd.DataFrame({"dat": d.naive_t-d.pit_t, "sur": s.survivor_t-s.pit_t}).dropna().sort_values("dat")
y, h = np.arange(len(f)), 0.38

fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.barh(y+h/2, f.dat, height=h, color="#2E4E8F", label="Period-end join", zorder=3)
ax.barh(y-h/2, f.sur, height=h, color="#A3352C", label="Current membership", zorder=3)
zero = f.dat.abs() < 1e-12
ax.scatter(np.zeros(zero.sum()), y[zero.values]+h/2, s=13, facecolors="white",
           edgecolors="#2E4E8F", linewidths=0.9, zorder=5, label="exactly zero")
ax.axvline(0, color="black", lw=0.6, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(f.index, fontsize=6.6, family="monospace")
for lab, name in zip(ax.get_yticklabels(), f.index):
    lab.set_color("#222222" if name in FUND else "#9A9A9A")
ax.set_xlabel("shift in $t$-statistic from point-in-time baseline", fontsize=8)
ax.set_xlim(-2.4, 4.2); ax.set_ylim(-0.8, len(f)-0.2)
ax.spines[["top","right","left"]].set_visible(False)
ax.grid(axis="x", lw=0.35, color="#CCCCCC", zorder=0)
ax.legend(frameon=False, fontsize=7.2, loc="lower right", handlelength=1.2, scatterpoints=1)
ax.tick_params(axis="y", length=0, pad=2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT/"fingerprints.pdf", bbox_inches="tight")
print(f"wrote {OUT/'fingerprints.pdf'} ({int(zero.sum())} exact zeros marked)")
