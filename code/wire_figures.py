# -*- coding: utf-8 -*-
"""把 Fig 5-9 接入 main.tex + 生成 Fig 9 scatter PDF。一次执行全部。"""
import os, json, glob

# 路径
PAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIG = os.path.join(PAP, "figures")
os.makedirs(FIG, exist_ok=True)

BS = chr(92)
NL = chr(10)

# ===== Step 1: 生成 Fig 9 scatter PDF =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

fig, ax = plt.subplots(figsize=(4.5, 4.0))
lim = max(max(x for x, y, f in pairs), max(y for x, y, f in pairs)) * 1.1 + 0.02
ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
for real, shuf, fid in pairs:
    ax.scatter(real, shuf, s=40, c="#2ca02c", edgecolors="k", linewidths=0.5, zorder=3)
    short = fid.replace("_", " ")[:15]
    ax.annotate(short, (real, shuf), fontsize=6, textcoords="offset points", xytext=(4, 4))
d_mean = np.mean([s - r for r, s, _ in pairs])
ax.text(0.05, 0.93, f"Mean shuffled - real = {d_mean:+.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
ax.set_xlabel("A-PM with real teacher scores (VUS-PR)")
ax.set_ylabel("A-PM with shuffled scores (VUS-PR)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_shuffle_scatter.pdf"))
plt.close(fig)
print(f"Fig 9 scatter: {len(pairs)} points, d_mean={d_mean:+.4f}")

# ===== Step 2: 接入 main.tex =====
main_path = os.path.join(PAP, "main.tex")
src = open(main_path, encoding="utf-8").read()

# Fig 5: R8 bimodal → R8 段后
a5 = "which is the cost the distillation was meant to avoid."
f5 = (a5 + NL + NL
      + BS + "begin{figure}[t]" + NL
      + BS + "centering" + NL
      + BS + "includegraphics[width=0.8" + BS + "textwidth]{figures/fig_r8_bimodal.pdf}" + NL
      + BS + "caption{R8 teacher-on-anomaly control: VUS-PR by flow for"
      + " A-PM (score distillation, original vs.\\ augmented) and B1-PM"
      + " (self-supervised, original vs.\\ augmented). The bimodal pattern"
      + " (Exathlon $+0.59$, occupancy $-0.48$) shows the score interface"
      + " inherits the teacher's condition-dependent inconsistency.}"
      + NL + BS + "label{fig:r8}" + NL
      + BS + "end{figure}")
assert a5 in src, "fig5"
src = src.replace(a5, f5)
print("Fig 5 wired")

# Fig 6: exclusion sensitivity → §6.2 敏感性句后
a6 = "so neither shared-lineage caveat drives the directional pattern."
f6 = (a6 + NL + NL
      + BS + "begin{figure}[t]" + NL
      + BS + "centering" + NL
      + BS + "includegraphics[width=" + BS + "textwidth]{figures/fig_exclusion_sensitivity.pdf}" + NL
      + BS + "caption{Exclusion sensitivity: confirmation macro VUS-PR with"
      + " the full 9-source set vs.\\ dropping the WADI or NAB stream. The"
      + " arm ordering is unchanged across all three exclusions.}"
      + NL + BS + "label{fig:sensitivity}" + NL
      + BS + "end{figure}")
assert a6 in src, "fig6"
src = src.replace(a6, f6)
print("Fig 6 wired")

# Fig 7: deployment → §3.1 末尾
a7 = "though full information-set" + NL + "equalization across all arms" + NL + "remains future work."
f7 = (a7 + NL + NL
      + BS + "begin{figure}[t]" + NL
      + BS + "centering" + NL
      + BS + "includegraphics[width=" + BS + "textwidth]{figures/fig_deployment.pdf}" + NL
      + BS + "caption{Cloud--edge deployment architecture. Left: the frozen"
      + " teacher (205M parameters) scores targets and students are trained"
      + " offline. Right: only the exported student (ONNX, optional INT8) is"
      + " deployed on the edge device.}"
      + NL + BS + "label{fig:deployment}" + NL
      + BS + "end{figure}")
assert a7 in src, "fig7"
src = src.replace(a7, f7)
print("Fig 7 wired")

# Fig 8: signal pathways → §3.3 Arms 末尾
a8 = "Full arm definitions, the" + NL + "comparison table, and implementation details are in" + NL + "Section~" + BS + "ref{sec:setup-arms}."
f8 = (a8 + NL + NL
      + BS + "begin{figure}[t]" + NL
      + BS + "centering" + NL
      + BS + "includegraphics[width=" + BS + "textwidth]{figures/fig_pipeline.pdf}" + NL
      + BS + "caption{Three training-signal pathways compared in this study."
      + " Top: score distillation (arms A/C). Middle: forecast distillation"
      + " (arm B2). Bottom: self-supervised training (arm B1). The linear"
      + " floor and PCA serve as non-neural references.}"
      + NL + BS + "label{fig:signals}" + NL
      + BS + "end{figure}")
assert a8 in src, "fig8"
src = src.replace(a8, f8)
print("Fig 8 wired")

# Fig 9: shuffle scatter → §7.1 洗牌段后
a9 = "though the interval still allows a positive effect of up to $+0.05$."
f9 = (a9 + NL + NL
      + BS + "begin{figure}[t]" + NL
      + BS + "centering" + NL
      + BS + "includegraphics[width=0.7" + BS + "textwidth]{figures/fig_shuffle_scatter.pdf}" + NL
      + BS + "caption{Shuffled-teacher control: per-flow VUS-PR for A-PM"
      + " with real teacher scores (x-axis) vs.\\ time-shuffled teacher"
      + " scores (y-axis). The diagonal is $y{=}x$.}"
      + NL + BS + "label{fig:shuffle}" + NL
      + BS + "end{figure}")
assert a9 in src, "fig9"
src = src.replace(a9, f9)
print("Fig 9 wired")

open(main_path, "w", encoding="utf-8").write(src)
print("All 5 figures wired into main.tex")
