# -*- coding: utf-8 -*-
"""Build results/cost/rpi_timing.json (figures.py hook schema) from raw Pi output."""
import io
import json

BASE = r"D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏"
raw = json.load(io.open(BASE + r"\results\cost\rpi_timing_raw.json", encoding="utf-8"))

# hook entries: (model, prec) -> label ; quality = m6 confirm macro (same as desktop points)
# offsets: in-axes index digit placement (key lives in the figure caption)
ENTRIES = [
    ("LINEAR", "FP32", "LINEAR", 0.3229, {"offset": (4, -11)}),
    ("PatchMLP", "INT8", "C-PM (PM INT8)", 0.2501, {"offset": (5, 5)}),
    ("TCN", "FP32", "B1-TCN (TCN FP32)", 0.2778, {"offset": (5, 5)}),
]
rows = {(r["model"], r["prec"]): r for r in raw["rows"]}
out = {"protocol": raw["meta"]["protocol"], "entries": [], "detail": []}
for model, prec, label, q, place in ENTRIES:
    r = rows[(model, prec)]
    p95 = r["agg"]["p95_ms"]["median"]
    out["entries"].append({"label": label, "p95_ms": p95, "quality": q,
                           "p50_ms": r["agg"]["p50_ms"]["median"],
                           "rss_mb": r["rss_mb"],
                           "throughput_win_per_s": round(1000.0 / r["agg"]["p50_ms"]["median"], 1),
                           **place})
for r in raw["rows"]:
    out["detail"].append({"model": r["model"], "prec": r["prec"],
                          "p50_ms": r["agg"]["p50_ms"], "p95_ms": r["agg"]["p95_ms"],
                          "file_MB": r["file_MB"], "rss_mb": r["rss_mb"]})
out["meta"] = {k: v for k, v in raw["meta"].items() if k != "protocol"}

with io.open(BASE + r"\results\cost\rpi_timing.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
for e in out["entries"]:
    print("%-22s p95 %.4f ms  p50 %.4f ms  rss %.1f MB  %.1f win/s  q %.4f"
          % (e["label"], e["p95_ms"], e["p50_ms"], e["rss_mb"],
             e["throughput_win_per_s"], e["quality"]))
print("meta:", out["meta"])
