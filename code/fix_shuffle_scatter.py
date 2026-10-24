# -*- coding: utf-8 -*-
"""Regenerate fig_shuffle_scatter.pdf with collision-free short labels.

Fixes visual-review defects: truncated 15-char names ("D16 USRANOVA id"),
label-on-label overprint in the dense low-VUS cluster, label running past
the right plot border. Same data, same axes, same diagonal and mean box.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "..", "results")
FIG = os.path.join(BASE, "..", "paper", "figures")

pairs = []
for f in glob.glob(os.path.join(RES, "c1_shuffled", "A-PM-shuf_*.json")):
    d = json.load(open(f, encoding="utf-8"))
    fid = d["flow"]
    m5f = os.path.join(RES, "m5", fid + ".json")
    if os.path.exists(m5f):
        m5 = json.load(open(m5f, encoding="utf-8"))
        real = m5["A-PM"]["mean"]
        pairs.append((real, d["mean"], fid))
pairs.sort()
assert len(pairs) == 20, "expected 20 flows, got %d" % len(pairs)

fig, ax = plt.subplots(figsize=(4.5, 4.0))
lim = max(max(x for x, y, f in pairs), max(y for x, y, f in pairs)) * 1.1 + 0.02
ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)

# markers first, register pixel positions
for real, shuf, fid in pairs:
    ax.scatter(real, shuf, s=40, c="#2ca02c", edgecolors="k", linewidths=0.5, zorder=3)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
axbb = ax.get_window_extent(renderer=renderer)
occupied = []
pt_pix = [ax.transData.transform((r, s)) for r, s, _ in pairs]


def pad(bb, p=1.5):
    return (bb.x0 - p, bb.y0 - p, bb.x1 + p, bb.y1 + p)


def hit(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def candidates():
    c = [(5, 4, "left"), (-5, 4, "right"), (5, -8, "left"), (-5, -8, "right"),
         (6, -13, "left"), (-6, -13, "right"), (0, 10, "center"), (0, -14, "center")]
    import math
    for r in range(14, 42, 4):
        for a in range(0, 360, 30):
            dx = round(r * math.cos(math.radians(a)))
            dy = round(r * math.sin(math.radians(a)))
            c.append((dx, dy, "left" if dx >= 0 else "right"))
    return c


fails = []
# label low-x first: dense cluster labels get placed before sparse ones
for real, shuf, fid in sorted(pairs, key=lambda t: (t[0] + t[1])):
    short = "_".join(fid.split("_")[:2]).replace("_", " ")   # e.g. "D16 NAB"
    xy = (real, shuf)
    pix = ax.transData.transform(xy)
    placed = None
    for dx, dy, ha in candidates():
        t = ax.annotate(short, xy, fontsize=6, textcoords="offset points",
                        xytext=(dx, dy), ha=ha, zorder=3)
        bb = pad(t.get_window_extent(renderer=renderer))
        inside = axbb.x0 <= bb[0] and bb[2] <= axbb.x1 and axbb.y0 <= bb[1] and bb[3] <= axbb.y1
        near_pt = any(bb[0] - 3 <= q[0] <= bb[2] + 3 and bb[1] - 3 <= q[1] <= bb[3] + 3
                      for q in pt_pix)
        if inside and not near_pt and not any(hit(bb, o) for o in occupied):
            occupied.append(bb)
            placed = t
            break
        t.remove()
    if placed is None:
        t = ax.annotate(short, xy, fontsize=6, textcoords="offset points",
                        xytext=(5, 4), ha="left", zorder=3)
        occupied.append(pad(t.get_window_extent(renderer=renderer)))
        fails.append(fid)

d_mean = np.mean([s - r for r, s, _ in pairs])
ax.text(0.05, 0.93, f"Mean shuffled - real = {d_mean:+.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
ax.set_xlabel("A-PM with real teacher scores (VUS-PR)")
ax.set_ylabel("A-PM with shuffled scores (VUS-PR)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shuffle_scatter.pdf"))
print("shuffle scatter: %d points, d_mean=%+.4f, fallback labels: %s"
      % (len(pairs), d_mean, fails or "none"))
