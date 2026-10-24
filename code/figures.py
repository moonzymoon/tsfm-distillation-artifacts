# -*- coding: utf-8 -*-
"""四张图（PDF 矢量）→ paper/figures/。数据全部程序生成自 results/ JSON。
v2: F1 重排版防标注碰撞；F2 纳入辅助臂540 run并分家族着色；F3 热力图补 T0/PCA 列。
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

ARMS = ["C-PM", "C-TCN", "A-PM", "A-TCN", "B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "AB-PM", "AB-TCN", "LINEAR"]
NO_T = ["B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "LINEAR"]
GROUP = {"C-PM": "score-distilled", "C-TCN": "score-distilled", "A-PM": "score-distilled",
         "A-TCN": "score-distilled", "AB-PM": "score+forecast", "AB-TCN": "score+forecast",
         "B2-PM": "forecast-distilled", "B2-TCN": "forecast-distilled",
         "B1-PM": "self-supervised", "B1-TCN": "self-supervised", "LINEAR": "linear floor"}
COLOR = {"score-distilled": "#d62728", "score+forecast": "#9467bd",
         "forecast-distilled": "#ff7f0e", "self-supervised": "#2ca02c", "linear floor": "#7f7f7f"}

rows = {}
for f in glob.glob(os.path.join(RES, "m5", "D*.json")):
    d = json.load(open(f, encoding="utf-8"))
    rows[d["flow"]] = d
b = {}
for f in glob.glob(os.path.join(RES, "m5b", "*.json")):
    d = json.load(open(f, encoding="utf-8"))
    b[(d["arm"], d["flow"])] = d["mean"]


def g(arm, fl):
    if arm == "T0":
        return rows[fl]["teacher"]["vus_pr"]
    if arm == "PCA":
        return rows[fl]["pca"]["vus_pr"]
    if (arm, fl) in b:
        return b[(arm, fl)]
    return rows[fl][arm]["mean"]


flows = sorted(rows)

# ============ F1: pipeline schematic（v2 重排版：宽画布、标签全部脱离图形元素） ============
fig, ax = plt.subplots(figsize=(9.6, 3.6))
ax.axis("off")


def box(x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="k", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, ls=style, color="k"))


# 顶链：教师路径
box(1, 66, 15, 16, "Multivariate stream\n(normal-only\ntraining segment)", "#eef")
box(22, 66, 19, 16, "Frozen TSFM teacher\n(Chronos-Bolt-base)\n+ locked pipeline", "#f4cccc")
box(47, 66, 19, 16, "Teacher signals:\n$s_T$ scores / $\\hat{x}_T$ forecasts", "#fff2cc")
# 中链：学生训练
box(22, 30, 19, 16, "Student training\n(TCN-lite / PatchMLP-lite,\n$\\approx$0.27M params)", "#d9ead3")
box(47, 30, 19, 16, "Trained student\n(ONNX, INT8 opt.)", "#d9ead3")
# 右链：部署
box(72, 38, 24, 16, "Edge deployment\n0.08\u20130.52 ms/window\n(E1\u2013E2 budgets)", "#d9ead3")
# 底链：B1
box(47, 6, 19, 14, "B1 self-supervised\n(no teacher signal)", "#eeeeee")

arrow(16, 74, 22, 74)
arrow(41, 74, 47, 74)
# 教师信号 → 学生训练（A/C 用分数，B2 用预测，AB 用两者）
arrow(50, 66, 34, 46, style="--")
ax.text(52.5, 58.5, "score target (A/C, AB)\nforecast target (B2, AB)", fontsize=8, va="center", ha="left")
# 原始流 → 学生训练（标注在箭头上方远端，避开所有线）
arrow(8.5, 66, 24, 46)
ax.text(13.5, 63.5, "raw series (all arms)", fontsize=8, ha="left")
# 原始流 → B1（沿底部绕行，避开学生训练框）
ax.plot([3.5, 3.5], [64, 13], ls="--", lw=1.2, color="k")
arrow(3.5, 13, 47, 13, style="-")
ax.text(24, 9.5, "no teacher (arm B1)", fontsize=8, ha="center", color="#555")
# 训练 → 导出 → 部署
arrow(41, 38, 47, 38)
arrow(66, 38, 72, 44)
ax.text(69, 47, "deploy", fontsize=8, ha="center")
arrow(56.5, 20, 56.5, 30, style="--")
# 教师退役：叉（右上角，远离信号标注）
ax.plot([60, 64], [63, 66], color="#999", lw=1.4)
ax.plot([60, 64], [66, 63], color="#999", lw=1.4)
ax.text(65.5, 64.5, "retired after training", fontsize=7.5, color="#777", ha="left")
ax.set_xlim(0, 98)
ax.set_ylim(0, 86)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_pipeline.pdf"))
plt.close(fig)
print("F1 ok")

# ============ F2: fidelity-quality scatter（v2：全 540 run，含辅助臂） ============
fig, ax = plt.subplots(figsize=(5.4, 3.8))
seen = set()
n_pts = 0
for f_, r in rows.items():
    for k, run in r["runs"].items():
        arm = k.rsplit("-s", 1)[0]
        grp = GROUP.get(arm)
        if grp is None or not np.isfinite(run["spearman"]):
            continue
        lbl = grp if grp not in seen else None
        seen.add(grp)
        ax.scatter(run["spearman"], run["vus_pr"], s=13, alpha=0.6,
                   color=COLOR[grp], label=lbl)
        n_pts += 1
for f_ in NO_T:
    pass
# 辅助臂（m5b，3种子）——全家族绘制，图例五项齐
allsp = [x["spearman"] for f_, r in rows.items() for x in r["runs"].values()
         if np.isfinite(x["spearman"])]
for arm in ["B1-TCN", "B1-PM", "B2-TCN", "B2-PM", "AB-PM", "AB-TCN", "LINEAR"]:
    grp = GROUP[arm]
    lbl = grp if grp not in seen else None
    seen.add(grp)
    for fp in glob.glob(os.path.join(RES, "m5b", f"{arm}_D*.json")):
        z = json.load(open(fp, encoding="utf-8"))
        for run in z["runs"].values():
            if np.isfinite(run["spearman"]):
                allsp.append(run["spearman"])
                ax.scatter(run["spearman"], run["vus_pr"], s=13, alpha=0.6,
                           color=COLOR[grp], label=lbl)
                lbl = None
                n_pts += 1
med = float(np.median(allsp))
ax.axvline(med, ls="--", lw=1, color="k")
ax.text(med + 0.012, 0.985, f"median $\\rho$={med:.3f} ({len(allsp)} runs)",
        fontsize=8, transform=ax.get_xaxis_transform(), va="top")
ax.set_xlabel("Fidelity to teacher (per-run Spearman $\\rho$)")
ax.set_ylabel("Quality (VUS-PR)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_fidelity_quality.pdf"))
plt.close(fig)
print(f"F2 ok ({n_pts} points)")

# ============ F3: dev heatmap（v2：补 T0/PCA 列，去图内标题） ============
ALL = ARMS + ["T0", "PCA"]
fig, ax = plt.subplots(figsize=(8.0, 6.0))
M = np.array([[g(a, fl) for a in ALL] for fl in flows])
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.6, M.max()))
ax.set_xticks(range(len(ALL)))
ax.set_xticklabels(ALL, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(flows)))
ax.set_yticklabels([f.replace("_", " ") for f in flows], fontsize=7)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="white" if M[i, j] < M.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.025, label="VUS-PR")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_dev_heatmap.pdf"))
plt.close(fig)
print("F3 ok (13 cols incl T0/PCA)")

# ============ F3b: cost-quality（v2：训练后计时矩阵 + 防重叠标注 + 口径注） ============
TRAINED = os.path.join(RES, "cost", "timing_matrix_TRAINED.json")
if os.path.exists(TRAINED):
    tmrows = json.load(open(TRAINED, encoding="utf-8"))["rows"]
    lat = {}
    for r in tmrows:
        if r["dim"] != 66:
            continue
        p95 = r["agg"]["p95_ms"]["median"]
        if r["model"] == "TCN" and r["prec"] == "FP32":
            lat["TCN"] = p95
        if r["model"] == "PatchMLP" and r["prec"] == "INT8":
            lat["PatchMLP-INT8"] = p95
        if r["model"] == "PatchMLP" and r["prec"] == "FP32":
            lat["PatchMLP-FP32"] = p95
        if r["model"] == "LINEAR" and r["prec"] == "FP32":
            lat["LINEAR"] = p95
else:
    tm = json.load(open(os.path.join(RES, "cost", "timing_matrix_FINAL.json"), encoding="utf-8"))
    lat = {}
    for r in tm:
        if r["model"] == "TCN" and r["dim"] == 66 and r["prec"] == "FP32":
            lat["TCN"] = r["p95_ms"]
        if r["model"] == "PatchMLP" and r["dim"] == 66 and r["prec"] == "INT8":
            lat["PatchMLP-INT8"] = r["p95_ms"]
        if r["model"] == "PatchMLP" and r["dim"] == 66 and r["prec"] == "FP32":
            lat["PatchMLP-FP32"] = r["p95_ms"]
        if r["model"] == "LINEAR" and r["dim"] == 66:
            lat["LINEAR"] = r["p95_ms"]
m6 = {}
for arm in ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR"]:
    vals = [json.load(open(f, encoding="utf-8"))["mean"]
            for f in glob.glob(os.path.join(RES, "m6", f"{arm}_C*.json"))]
    m6[arm] = float(np.mean(vals))
base = json.load(open(os.path.join(RES, "m6", "baselines.json"), encoding="utf-8"))
pca_q = float(np.mean([v["pca"] for v in base.values() if v.get("pca") is not None]))
t_q = float(np.mean([v["teacher"] for v in base.values() if v.get("teacher") is not None]))
tc = json.load(open(os.path.join(RES, "cost", "teacher_cpu_timing.json"), encoding="utf-8"))
tc_b1 = json.load(open(os.path.join(RES, "cost", "teacher_cpu_timing_batch1.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(RES, "cost", "teacher_cpu_timing_batch1.json")) else tc
t_lat = tc_b1.get("teacher_cpu_ms_per_point_batch1", tc["teacher_cpu_ms_per_point"])
fig, ax = plt.subplots(figsize=(5.6, 4.2))
# 三个 PM 臂同 x 密集；B1-PM 上移、教师标签下移，给 (RPi) 序号点位让位（视觉验收修复）
pts = [  # (label, x, y, color, offset, ha)
    ("Linear floor", lat["LINEAR"], m6["LINEAR"], "#7f7f7f", (8, 15), "left"),
    ("C-PM (PM INT8)", lat["PatchMLP-INT8"], m6["C-PM"], "#d62728", (-8, -3), "right"),
    ("B2-PM (PM INT8)", lat["PatchMLP-INT8"], m6["B2-PM"], "#ff7f0e", (-8, -9), "right"),
    ("B1-PM (PM FP32)", lat.get("PatchMLP-FP32", 0.1177), m6["B1-PM"], "#8c564b", (12, 22), "left"),
    ("B1-TCN (TCN FP32)", lat["TCN"], m6["B1-TCN"], "#2ca02c", (10, 12), "left"),
    ("PCA (fixed)", 0.005, pca_q, "#17becf", (7, -14), "left"),
    ("Frozen teacher (T0)", t_lat, t_q, "k", (-6, -16), "right"),
]
for name, x, y, c, off, ha in pts:
    ax.scatter(x, y, s=46, color=c, zorder=3)
    ax.annotate(name, (x, y), textcoords="offset points", xytext=off, fontsize=8, ha=ha)
# R14：零调参经典基线点位（灰实心方块，序号 4/5；延迟实测 classical_timing.json）
CLS = os.path.join(RES, "cost", "classical_timing.json")
CLS_PTS = []
if os.path.exists(CLS):
    cz = json.load(open(CLS, encoding="utf-8"))
    CLS_PTS = [("Robust-Z", cz["Robust-Z"]["rounds_median"]["p95_ms"], 0.2392, (4, -12)),
               ("MA-residual", cz["MA-residual"]["rounds_median"]["p95_ms"], 0.3248, (8, 0))]
for i, (nm, x_, y_, off) in enumerate(CLS_PTS, 4):
    ax.scatter(x_, y_, s=60, marker="s", color="#5c6b73", zorder=4)
    ax.annotate(str(i), (x_, y_), textcoords="offset points", xytext=off,
                fontsize=8, color="#5c6b73")
# 树莓派 4B on-device 点位：粉色空心方块 + 序号（键在图注，防标签叠印）
RPI = os.path.join(RES, "cost", "rpi_timing.json")
if os.path.exists(RPI):
    rz = json.load(open(RPI, encoding="utf-8"))
    for i, e in enumerate(rz.get("entries", []), 1):
        ax.scatter(e["p95_ms"], e["quality"], s=70, facecolors="none",
                   edgecolors="#e377c2", linewidths=1.8, zorder=4)
        ax.annotate(str(i), (e["p95_ms"], e["quality"]), textcoords="offset points",
                    xytext=e.get("offset", (5, 5)), fontsize=8, color="#e377c2")
ax.axvspan(0, 1, alpha=0.06, color="green")
ax.axvspan(1, 5, alpha=0.04, color="orange")
ax.text(0.32, 0.02, "E1", transform=ax.get_xaxis_transform(), fontsize=8, color="green")
ax.text(1.6, 0.02, "E2", transform=ax.get_xaxis_transform(), fontsize=8, color="darkorange")
ax.set_xscale("log")
ax.set_xlabel("p95 per-window latency at the 66-channel configuration\n(ms, log scale; single-thread ORT, batch 1)")
ax.set_ylabel("Quality (VUS-PR, confirm macro)")
ax.set_ylim(0, 0.36)
ax.axhline(0, lw=0.5, color="0.85")
ax.set_xlim(0.001, 300)
ax.set_title("Quality: mean over 10 seeds on 9 held-out sources;\nlatency: trained artifacts, teacher end-to-end at batch 1", fontsize=7.5, color="0.35")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_cost_quality.pdf"))
plt.close(fig)
print("F3b ok (teacher at batch-1 cost)")

# ============================================================
# F5: R8 bimodal bar chart
# ============================================================
FLOWS_R8 = ["D1_SWaT_id1", "D3_SMD_id4", "D11_SVDB_id2", "D16_NAB_occ6005", "D18_Exathlon_id1"]
FLOW_LABELS_R8 = ["SWaT", "SMD", "SVDB", "Occupancy", "Exathlon"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(FLOWS_R8))
width = 0.2
a_orig, a_aug, b_orig, b_aug = [], [], [], []
for fid in FLOWS_R8:
    pa = json.load(open(os.path.join(RES, "r8_anomaly", f"A-PM-aug_{fid}.json"), encoding="utf-8"))["mean"]
    pb = json.load(open(os.path.join(RES, "r8_anomaly", f"B1-PM-aug_{fid}.json"), encoding="utf-8"))["mean"]
    po = g("A-PM", fid)
    bo = json.load(open(os.path.join(RES, "m5b", f"B1-PM_{fid}.json"), encoding="utf-8"))["mean"]
    a_orig.append(po); a_aug.append(pa); b_orig.append(bo); b_aug.append(pb)
a_orig, a_aug, b_orig, b_aug = np.array(a_orig), np.array(a_aug), np.array(b_orig), np.array(b_aug)
ax.bar(x - 0.3, b_orig, width, label="B1-PM original", color="#999999", alpha=0.7)
ax.bar(x - 0.1, b_aug, width, label="B1-PM augmented", color="#2ca02c", alpha=0.8)
ax.bar(x + 0.1, a_orig, width, label="A-PM original", color="#e6a0a0", alpha=0.8)
ax.bar(x + 0.3, a_aug, width, label="A-PM augmented", color="#d62728", alpha=0.9)
for i in range(len(x)):
    d = a_aug[i] - a_orig[i]                     # 同臂前后差（正文口径）
    if abs(d) > 0.05:
        sign = "+" if d > 0 else ""
        ax.annotate(f"{sign}{d:.2f}", (x[i] + 0.2, max(b_orig[i], b_aug[i], a_orig[i], a_aug[i]) + 0.02),
                    ha="center", fontsize=8, fontweight="bold",
                    color="#2ca02c" if d > 0 else "#d62728")
ax.set_xticks(x)
ax.set_xticklabels(FLOW_LABELS_R8, fontsize=9)
ax.set_ylabel("VUS-PR (3 seeds)")
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.set_ylim(0, None)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_r8_bimodal.pdf"))
plt.close(fig)
print("F5 ok (R8 bimodal)")

# ============================================================
# F6: Exclusion sensitivity bar chart
# ============================================================
FLOWS_C = ["C1_WADI", "C2_Genesis_id1", "C3_Daphnet_id1", "C4_MGAB", "C5_YAHOO",
           "C6_IOPS", "C7_MITDB_id5", "C8_NEweather", "C9_NAB_adexchange"]
CONF_LABELS = ["WADI", "Genesis", "Daphnet", "MGAB", "Yahoo", "IOPS", "MITDB", "NEweather", "AdExch"]
m6_all = {}
for arm in ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR", "A-PM"]:
    m6_all[arm] = {}
    for f in glob.glob(os.path.join(RES, "m6", f"{arm}_C*.json")):
        fid = os.path.basename(f).replace(f"{arm}_", "").replace(".json", "")
        m6_all[arm][fid] = json.load(open(f, encoding="utf-8"))["mean"]
conf_arms = ["C-PM", "B1-TCN", "B1-PM", "B2-PM", "LINEAR", "A-PM"]
conf_colors = ["#d62728", "#2ca02c", "#8c564b", "#ff7f0e", "#7f7f7f", "#c49c94"]
fig, ax = plt.subplots(figsize=(6.8, 3.6))
x = np.arange(len(CONF_LABELS))
width = 0.13
for k, arm in enumerate(conf_arms):
    vals = [next((v for kk, v in m6_all[arm].items() if fl.lower() in kk.lower()), 0.0)
            for fl in CONF_LABELS]
    assert max(vals) > 0, f"F6: arm {arm} matched no flow data"
    ax.bar(x + (k - 2.5) * width, vals, width, label=arm, color=conf_colors[k], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CONF_LABELS, fontsize=8, rotation=15, ha="right")
ax.set_ylabel("VUS-PR (10 seeds)")
ax.legend(frameon=False, fontsize=7, ncol=6, loc="upper left", columnspacing=0.9,
          handlelength=1.1)
ax.set_ylim(0, None)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_exclusion_sensitivity.pdf"))
plt.close(fig)
print("F6 ok (exclusion sensitivity)")

# ============================================================
# F7: Cloud-edge deployment architecture
# ============================================================
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.axis("off")
BS = chr(92)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.plot([50, 50], [2, 58], ls="--", lw=1.2, color="#999")
ax.text(25, 58, "Cloud (offline)", fontsize=10, ha="center", fontweight="bold", color="#4472c4")
ax.text(75, 58, "Edge (online)", fontsize=10, ha="center", fontweight="bold", color="#548235")
def rbox(x, y, w, h, text, fc, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec="k", lw=1.1))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)
def harrow(x1, y1, x2, y2, label="", ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="k"))
    if label: ax.text((x1+x2)/2, (y1+y2)/2 + ly, label, fontsize=7.5, ha="center")
# Cloud side
rbox(2, 48, 18, 7, "Multivariate streams\n(29 flows,\n22 collections)", "#eef", fs=8)
rbox(26, 48, 18, 7, "Frozen Chronos-Bolt\n(205M params, GPU)", "#f4cccc")
rbox(26, 34, 18, 7, "Signal caches\n$s_T$ scores / $\\hat{x}_T$ forecasts", "#fff2cc")
rbox(26, 20, 18, 7, "Student training\n(A/C/B1/B2/AB; B1 uses\nraw streams only)", "#d9ead3")
rbox(2, 20, 18, 7, "ONNX export\n(INT8 quantization)", "#d9ead3")
harrow(20, 51.5, 26, 51.5, "signals", ly=1.5)
harrow(35, 48, 35, 41, "cache")
harrow(26, 23.5, 20, 23.5, "export", ly=1.5)
# Edge side
rbox(56, 47, 18, 9, "Student ONNX/INT8\n($\\approx$0.27M params,\n$\\leq$5MB)", "#d9ead3")
rbox(80, 47, 18, 9, "Real-time inference\n(0.08\u20130.52 ms/window)", "#d9ead3")
harrow(74, 51.5, 80, 51.5, "deploy", ly=1.5)
rbox(56, 20, 18, 7, "Latency p50/p95\nRSS, file size\n(measured)", "#e2f0d9")
rbox(80, 20, 18, 7, "Energy, throughput\n(future work,\non-device)", "#f5f5f5")
harrow(65, 47, 65, 27, "")
harrow(89, 47, 89, 27, "")
ax.text(66.2, 37.5, "measure", fontsize=7.5, ha="left")
# Teacher retired
ax.text(75, 36, "Teacher retired:\nnot deployed on edge", fontsize=8, ha="center",
        style="italic", color="#999",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ccc", ls="--"))
ax.plot([69, 73], [37, 34], color="#999", lw=1.2)
ax.plot([69, 73], [34, 37], color="#999", lw=1.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_deployment.pdf"))
plt.close(fig)
print("F7 ok (deployment architecture)")
