# -*- coding: utf-8 -*-
"""R13 入文：容量段四档重写 + 混淆段第三点改 + 结论联动句 + 链行 + ledger/摘要数字 + AI 声明 R13。"""
import io

BS = chr(92)
NL = chr(10)

# 1) Discussion 容量段重写（三档→四档 + 闭合叙事 + 预算联动）
c_old = ("A capacity sweep over PatchMLP students at 0.27M, 0.6M, and 1.6M" + NL +
         "parameters (five development flows, three seeds) shows the" + NL +
         "score-interface deficit narrowing with capacity while persisting at the" + NL +
         "largest tested size (macro 0.256 versus 0.346 self-supervised at 0.27M," + NL +
         "0.291 versus 0.344 at 1.6M).")
c_new = ("A capacity sweep over PatchMLP students at 0.27M, 0.6M, 1.6M, and 5.8M" + NL +
         "parameters (five development flows, three seeds; the largest size under" + NL +
         "revision R13) shows the score-interface deficit narrowing monotonically" + NL +
         "with capacity and closing only near six million parameters." + NL +
         "Macro score distillation reaches 0.256 versus 0.346 self-supervised at" + NL +
         "0.27M, 0.291 versus 0.344 at 1.6M, and 0.385 versus 0.399 at 5.8M," + NL +
         "where the gap $-0.014$ falls inside the preregistered 0.05 margin." + NL +
         "The closing size, however, prices itself out of the edge: a 5.8M" + NL +
         "student implies a FP32 artifact above 23\\,MB, an order of magnitude" + NL +
         "beyond the E1/E2 deployment budgets this study targets." + NL +
         "Within edge-sized students, the interface, not capacity, is the" + NL +
         "binding constraint.")

# 2) 混淆段第三点同步改
f_old = ("Third, the capacity sweep shows the deficit shrinking smoothly with" + NL +
         "student size rather than vanishing once the student could absorb a" + NL +
         "stronger teacher.")
f_new = ("Third, the capacity sweep shows the deficit shrinking smoothly with" + NL +
         "student size and closing only near six million parameters, far beyond" + NL +
         "the edge budgets at stake.")

# 3) 结论容量联动句（部署点后）
TX = chr(92) + "times"
k_old = ("Students are 200--1500$" + TX + "$ cheaper per point than the teacher on" + NL +
         "trained artifacts, and quantized PatchMLP students meet" + NL +
         "sub-millisecond, sub-megabyte budgets at every tested channel count.")
k_new = (k_old + NL +
         "The capacity sweep adds the flip side." + NL +
         "Score distillation closes its deficit to self-supervision only near" + NL +
         "six million parameters, whose artifact exceeds 23\\,MB, so within edge" + NL +
         "budgets the interface remains the binding constraint.")

# 4) R13 链行 + R11 时间戳勘正（与冻结记录一致）
r11_fix_old = "R11 & 2026-10-15 03:12 & 74143d38e48c8184"
r11_fix_new = "R11 & 2026-10-17 09:21 & 74143d38e48c8184"
r12_tail = "erratum: R11 timing-evidence wording corrected " + BS + BS
r13_row = (r12_tail + NL +
           "R13 & 2026-10-17 10:04 & 8284d2bac26f3625 & capacity sweep extension to 5.8M parameters (pre-run) " + BS + BS)

# 5) ledger + 摘要数字（+45）
l_old = ("Run ledger & 1{,}732 total: 1{,}270 preregistered (400 main grid + 420 auxiliary + 450 five-arm confirmation) + 462 post-hoc/controls and additions (90 A-PM confirmation addition + 40 shuffled-teacher R6 + 25 TimesFM subset + 30 interface ablation R7 + 30 anomaly-exposure R8 + 25 TimesFM full-flow R10 + 135 capacity-sweep R10 + 87 classical-baseline scoring R11).")
l_new = ("Run ledger & 1{,}777 total: 1{,}270 preregistered (400 main grid + 420 auxiliary + 450 five-arm confirmation) + 507 post-hoc/controls and additions (90 A-PM confirmation addition + 40 shuffled-teacher R6 + 25 TimesFM subset + 30 interface ablation R7 + 30 anomaly-exposure R8 + 25 TimesFM full-flow R10 + 135 capacity-sweep R10 + 87 classical-baseline scoring R11 + 45 capacity-sweep R13).")
a_old = ("1{,}270 preregistered runs plus 462" + NL + "disclosed additions")
a_new = ("1{,}270 preregistered runs plus 507" + NL + "disclosed additions")

# 6) AI 声明 R1--R12 → R1--R13
ai_old = "append-only revisions R1--R12, released"
ai_new = "append-only revisions R1--R13, released"

edits = [('cap', c_old, c_new), ('conf', f_old, f_new), ('concl', k_old, k_new),
         ('chain', r12_tail, r13_row), ('r11ts', r11_fix_old, r11_fix_new),
         ('ledger', l_old, l_new),
         ('abs', a_old, a_new), ('ai', ai_old, ai_new)]

for f in [r'paper' + BS + 'main.tex', r'paper' + BS + 'main_submission.tex',
          r'submission_package' + BS + 'main_submission.tex']:
    s = io.open(f, encoding='utf-8').read()
    for tag, old, new in edits:
        n = s.count(old)
        assert n == 1, '%s %s: %d' % (f, tag, n)
        s = s.replace(old, new)
    io.open(f, 'w', encoding='utf-8', newline=chr(10)).write(s)
    print('patched', f.split(BS)[-1])
print('ALL OK')
