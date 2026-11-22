"""Regenerate Figure 2: Per-class Delta_bare with 95% bootstrap CI (B=1000).

M-3 remediation: the paper now uses B=1000 resamples (experiments/bootstrap_b1000.py,
results/BOOTSTRAP_B1000_8CLASS.json). This script regenerates fig2 from that file so
the figure, table, and methodology text all quote the same B.

Layout matches the original fig2: 3 panels (GLIP / GDino / OWLv2), error bars,
green / red / blue, significance markers. Only CI endpoints change (point
estimates are identical at B=200 and B=1000).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

B1000 = "results/BOOTSTRAP_B1000_8CLASS.json"

with open(B1000) as f:
    res = json.load(f)["models"]

classes = ["person", "car", "dog", "cat", "chair", "cup", "bottle", "book"]
panels = [
    ("GLIP", "GLIP-T", "tab:green"),
    ("GDino", "GDino", "tab:red"),
    ("OWLv2", "OWLv2", "tab:blue"),
]

fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.9), sharey=True)

for ax, (key, label, color) in zip(axes, panels):
    xs = range(len(classes))
    for i, c in enumerate(classes):
        r = res[key][c]
        pt, lo, hi = r["delta_bare"], r["ci_low"], r["ci_high"]
        sig = not r["ci_contains_zero"]
        marker = "o" if sig else "x"
        ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt=marker,
                    color=color, ecolor=color, elinewidth=1.4, capsize=3,
                    markersize=6, alpha=1.0 if sig else 0.55)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_title(label, fontsize=11)
    ax.set_xlim(-0.6, len(classes) - 0.4)

axes[0].set_ylabel(r"$\Delta_{\mathrm{bare}}$ (AP50)", fontsize=10)
fig.suptitle(r"Per-Class $\Delta_{\mathrm{bare}}$ with 95% Bootstrap CI (B=1000)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = "figures/fig2_delta_bare_ci.pdf"
with PdfPages(out) as pdf:
    pdf.savefig(fig)
print(f"saved -> {out}")
