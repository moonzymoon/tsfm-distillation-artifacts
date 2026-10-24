# -*- coding: utf-8 -*-
"""幂等重放：main.tex 的 main_edits + 两份 submission 的 all_edits（section 文件已修好跳过）。"""
import io

BS = chr(92)
NL = chr(10)
TX = BS + "times"

p_old = ("Causal Matrix Profile and LOF were designated as" + NL +
         "auxiliary classical baselines but ultimately not executed (deviation" + NL +
         "disclosed in R5): the fixed-configuration PCA together with the linear" + NL +
         "floor already spans the classical comparison family, and a faithful" + NL +
         "causal-MP implementation with the full tuning budget was judged" + NL +
         "disproportionate to its marginal value.")
p_new = ("Causal Matrix Profile and LOF were designated as auxiliary classical" + NL +
         "baselines. Causal Matrix Profile was ultimately not executed (deviation" + NL +
         "disclosed in R5): a faithful causal implementation with the full tuning" + NL +
         "budget was judged disproportionate to its marginal value. LOF was later" + NL +
         "added in a window-feature form under revision R11, fit on the full" + NL +
         "unlabeled test segment (transductive, non-streaming), alongside two" + NL +
         "zero-tuning streaming scorers.")
s_old = "matched-pairs rank-biserial correlation and raw differences with"
s_new = ("matched-pairs rank-biserial correlation (on ranks of absolute paired" + NL +
         "differences, with sign counts alongside) and raw differences with")

confirm_edits = [
    ("matched-pairs rank-biserial $r{=}-0.33$", "matched-pairs rank-biserial $r{=}-0.02$"),
    ("$p{=}0.73$ [Holm 1.0], $r{=}-0.11$", "$p{=}0.73$ [Holm 1.0], $r{=}+0.16$"),
    ("$p{=}0.20$ [Holm 0.60], $r{=}-0.22$", "$p{=}0.20$ [Holm 0.60], $r{=}-0.49$"),
    ("(1{,}485 total including all post-hoc controls and additions)",
     "(1{,}777 total including all disclosed controls and additions)"),
    ("In the five teacher-positive flows score distillation adds only" + NL +
     "$+0.019$ mean VUS-PR over the best self-supervised arm, while the" + NL +
     "fixed PCA baseline still dominates that subgroup (0.453).",
     "In the five sources where the teacher matches or beats the best" + NL +
     "self-supervised arm per flow (Genesis, Daphnet, MITDB, NEweather," + NL +
     "NAB-adexchange), score distillation adds only $+0.018$ mean VUS-PR over" + NL +
     "that arm, while the fixed PCA baseline still dominates that subgroup" + NL +
     "(0.453)."),
    ("The neural arms retain one clear niche." + NL +
     "On the 66-channel SWaT source the best self-supervised student reaches" + NL +
     "0.568 against 0.401 for the strongest classical baseline, so" + NL +
     "high-channel industrial streams remain the case for trained students.",
     "The neural arms retain one exploratory niche." + NL +
     "On the 66-channel SWaT development source the best self-supervised" + NL +
     "student (0.568) exceeds the strongest classical baseline there (PCA at" + NL +
     "0.446), an observation we do not extrapolate beyond this source."),
    ("baseline. Neural arms are separated by year of design only in the" + NL +
     "artifact package.}", "baseline.}"),
    ("The three classical baselines are zero-tuning methods added under R11;" + NL +
     "all other rows are preregistered arms, the teacher, or the fixed PCA" + NL +
     "baseline.}",
     "The three classical baselines are zero-tuning methods added under R11;" + NL +
     "LOF-window is fit on the full unlabeled test segment (transductive," + NL +
     "non-streaming) and is excluded from streaming-deployment claims;" + NL +
     "all other rows are preregistered arms, the teacher, or the fixed PCA" + NL +
     "baseline.}"),
    ("All cost figures below are measured on " + BS + "emph{trained} artifacts",
     "Two budget tiers anchor the cost axis." + NL +
     "Tier E1 requires a per-window p95 latency below 1" + BS + ",ms and an" + NL +
     "artifact of at most 1" + BS + ",MB; tier E2 extends both to 5" + BS + ",ms and" + NL +
     "5" + BS + ",MB." + NL +
     "Thresholds are defined on desktop single-thread measurements; the" + NL +
     "Raspberry~Pi~4B figures are device-side validation, not tier" + NL +
     "definitions." + NL +
     "All cost figures below are measured on " + BS + "emph{trained} artifacts"),
]

main_edits = [
    ("Macro score distillation reaches 0.256 versus 0.346 self-supervised at" + NL +
     "0.27M, 0.291 versus 0.344 at 1.6M, and 0.385 versus 0.399 at 5.8M," + NL +
     "where the gap $-0.014$ falls inside the preregistered 0.05 margin." + NL +
     "The closing size, however, prices itself out of the edge: a 5.8M" + NL +
     "student implies a FP32 artifact above 23" + BS + ",MB, an order of magnitude" + NL +
     "beyond the E1/E2 deployment budgets this study targets.",
     "Macro score distillation reaches 0.256 versus 0.346 self-supervised at" + NL +
     "0.27M, 0.262 versus 0.342 at 0.6M, 0.291 versus 0.344 at 1.6M, and" + NL +
     "0.332 versus 0.343 at 5.8M, where the gap $-0.011$ falls inside the" + NL +
     "preregistered 0.05 margin (paired mean difference $-0.011$, bootstrap" + NL +
     "95" + BS + "% CI $[-0.054, +0.030]$, five flows)." + NL +
     "The closing size, however, prices itself out of the edge: the exported" + NL +
     "5.8M student measures 23.2" + BS + ",MB FP32 and 5.8" + BS + ",MB quantized," + NL +
     "roughly five times the E2 storage ceiling and far outside E1."),
    ("Score distillation closes its deficit to self-supervision only near" + NL +
     "six million parameters, whose artifact exceeds 23" + BS + ",MB, so within edge" + NL +
     "budgets the interface remains the binding constraint.",
     "Score distillation closes its deficit to self-supervision only near" + NL +
     "six million parameters, whose exported artifact measures 23.2" + BS + ",MB" + NL +
     "(5.8" + BS + ",MB quantized), so within edge budgets the interface remains" + NL +
     "the binding constraint."),
    ("Teacher strength modulates the ceiling." + NL + "The interface gates the transfer.",
     "Teacher strength modulates the ceiling." + NL +
     "Within the tested boundary, the interface is the binding constraint."),
    ("The mechanism picture diverges in one instructive way.",
     "The mechanism picture diverges in one instructive way, read here as an" + NL +
     "exploratory case study rather than a controlled strength experiment."),
    ("second-teacher-family controls jointly trace it to the score interface" + NL +
     "itself, not to teacher capacity, context asymmetry, or the normal-only" + NL +
     "training signal.",
     "second-teacher-family controls jointly point to the score interface as" + NL +
     "the most consistent explanation within the tested teachers, capacities," + NL +
     "and training conditions, without excluding those factors in general."),
    ("PatchMLP students fit sub-millisecond, sub-megabyte edge budgets at" + NL +
     "every tested channel count.",
     "PatchMLP students fit sub-millisecond, sub-megabyte desktop edge budgets" + NL +
     "at every tested channel count."),
    ("Students run on device on a Raspberry~Pi~4B at the reference dimension," + NL +
     "8--278$" + TX + "$ below the teacher's desktop cost.",
     "On a Raspberry~Pi~4B the same artifacts measure 0.44 to 15.47" + BS + ",ms" + NL +
     "p95, within tier E2 for the quantized PatchMLP student and 8--278$" + TX + "$" + NL +
     "below the teacher's desktop cost."),
    ("Shuffled-teacher and anomaly-exposure controls trace the failure to the" + NL +
     "score interface itself, not to teacher capacity, context asymmetry, or" + NL +
     "the training constraint.",
     "Shuffled-teacher and anomaly-exposure controls point to the score" + NL +
     "interface rather than to teacher capacity, context asymmetry, or the" + NL +
     "training constraint within the tested conditions."),
    ("itself, while trained students keep a clear niche on the" + NL +
     "66-channel industrial source.",
     "itself, while trained students keep an exploratory edge on the" + NL +
     "66-channel development source (0.568 versus 0.446 for the strongest" + NL +
     "classical baseline)."),
    ("B1 (self-supervised) PatchMLP-lite INT8 & High-channel industrial niche (SWaT 0.568 vs." + BS + " 0.401 best classical); occupancy edge over C-PM is matched by zero-tuning robust-z " + BS + BS,
     "B1 (self-supervised) PatchMLP-lite INT8 & Exploratory dev-source niche (SWaT 0.568 vs." + BS + " 0.446 PCA; single-source observation); occupancy edge over C-PM is matched by zero-tuning robust-z " + BS + BS),
    ("First, within the five teacher-positive sources, where the teacher" + NL +
     "matches or beats the best self-supervised arm, score distillation still" + NL +
     "adds only $+0.019$ mean VUS-PR.",
     "First, within the five sources where the teacher matches or beats the" + NL +
     "best self-supervised arm per flow, score distillation still adds only" + NL +
     "$+0.018$ mean VUS-PR (group means, exploratory stratification)."),
    ("R13 & 2026-10-17 10:04 & 8284d2bac26f3625 & capacity sweep extension to 5.8M parameters (pre-run) " + BS + BS,
     "R13 & 2026-10-17 10:04 & 8284d2bac26f3625 & capacity sweep extension to 5.8M parameters (pre-run) " + BS + BS + NL +
     "R14 & 2026-10-17 22:58 & 0f3fd5a2b282210a & classical-baseline latency measurement and cost-plane update (pre-run) " + BS + BS),
    ("the preregistration revisions R1--R10 and their timing are part of",
     "the preregistration revisions R1--R14 and their timing are part of"),
    ("append-only revisions R1--R13, released", "append-only revisions R1--R14, released"),
    ("The preregistration text, this hash chain, and all per-run" + NL +
     "results are part of the artifact package.",
     "The preregistration text, this hash chain, and all per-run" + NL +
     "results are part of the artifact package." + NL +
     "The chain evidences disclosure transparency." + NL +
     "Full-length hashes and file timestamps are in the artifact package, and" + NL +
     "no external timestamping authority is claimed."),
]

all_edits = [(p_old, p_new), (s_old, s_new)] + confirm_edits + main_edits

for f, edits in [(r'paper' + BS + 'main.tex', main_edits)]:
    s = io.open(f, encoding='utf-8').read()
    for i, (old, new) in enumerate(edits):
        n = s.count(old)
        assert n == 1, '%s edit%d: %d (%s)' % (f, i, n, old[:44].replace(NL, '|'))
        s = s.replace(old, new)
    io.open(f, 'w', encoding='utf-8', newline=chr(10)).write(s)
    print('main.tex (%d)' % len(edits))

for f in [r'paper' + BS + 'main_submission.tex', r'submission_package' + BS + 'main_submission.tex']:
    s = io.open(f, encoding='utf-8').read()
    for i, (old, new) in enumerate(all_edits):
        n = s.count(old)
        assert n == 1, '%s edit%d: %d (%s)' % (f, i, n, old[:44].replace(NL, '|'))
        s = s.replace(old, new)
    io.open(f, 'w', encoding='utf-8', newline=chr(10)).write(s)
    print('%s (%d)' % (f.split(BS)[-1], len(all_edits)))
print('ALL OK')
