# -*- coding: utf-8 -*-
"""生成附录成本表（训练后计时矩阵 → LaTeX），写入 paper/tables/cost_table.tex。
用法: python gen_cost_appendix.py  （需 results/cost/timing_matrix_TRAINED.json 已存在）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "cost", "timing_matrix_TRAINED.json")
DST = os.path.join(HERE, "..", "paper", "tables", "cost_table.tex")

BS = chr(92)

def L(s):
    return s.replace("@", BS)

d = json.load(open(SRC, encoding="utf-8"))
rows, drift = d["rows"], {(x["dim"], x["model"]): x for x in d["drift"]}

out = ["% AUTO-GENERATED from results/cost/timing_matrix_TRAINED.json — do not edit",
       L("@begin{table}[t]"),
       L("@centering"),
       L("@caption{Deployment-cost matrix on trained artifacts (case-C students, seed 0, "
         "M6 training protocol; idle desktop CPU, single-thread ONNX Runtime, batch 1, "
         "window $L{=}128$; median over 5 rounds of 50 warm-up + 500 timed windows, "
         "with round-to-round min--max of p95 in parentheses. The final column gives "
         "the INT8-minus-FP32 VUS-PR drift of the exported artifacts on each flow's test split.}"),
       L("@label{tab:costmatrix}"),
       L("@scriptsize"),
       L("@setlength{@tabcolsep}{2.4pt}"),
       L("@begin{tabular}{llrrrrrr}"),
       L("@toprule"),
       L("Ch & Model & Prec & p50 (ms) & p95 (ms) & p95 range & MB & $@Delta$VUS @\\"),
       L("@midrule")]

for r in sorted(rows, key=lambda x: (x["dim"], x["model"], x["prec"] != "FP32")):
    if r["prec"] != "FP32":
        continue
    q = next(x for x in rows if x["dim"] == r["dim"] and x["model"] == r["model"]
             and x["prec"] == "INT8")
    dr = drift.get((r["dim"], r["model"]), {})
    dv = (dr.get("vus_int8", 0) - dr.get("vus_fp32", 0)) if dr.get("vus_fp32") is not None else None
    for tag, x in (("fp", r), ("q8", q)):
        a = x["agg"]
        prec_name = "FP32" if tag == "fp" else "INT8"
        dvcell = f"{dv:+.3f}" if (tag == "q8" and dv is not None) else ("--" if tag == "q8" else "ref")
        out.append(f"{r['dim']} & {r['model']} & {prec_name} & {a['p50_ms']['median']:.3f} & "
                   f"{a['p95_ms']['median']:.3f} & {a['p95_ms']['min']:.3f}--{a['p95_ms']['max']:.3f} & "
                   f"{x['file_MB']:.2f} & {dvcell} {BS}{BS}")
    out.append(L("@addlinespace[2pt]"))

out += [L("@bottomrule"), L("@end{tabular}"), L("@end{table}")]
open(DST, "w", encoding="utf-8").write("\n".join(out))
print(f"written {DST}")
