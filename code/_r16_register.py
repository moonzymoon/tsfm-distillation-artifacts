# -*- coding: utf-8 -*-
"""R16 注册：泄漏披露+敏感性、Moirai 执行细节、确认集前瞻扩展三层设计、止步线。"""
import datetime
import hashlib
import io

BS = chr(92)
BASE = r'D:' + BS + '0科研' + BS + '工作1' + BS + '第20篇SCI_A3TSFM蒸馏'
p = BASE + BS + '预注册_v1.0_20261003.md'
sec = """

---

## 修订节 R16（2026-10-18，投稿前加固第三批，append-only；Moirai/扩源均为 pre-run 注册，敏感性为事后只读）

**触发**：投稿前最坏审稿人立场加固第二轮（外审终审两家意见融合裁决后）。

1. **训练段泄漏披露与敏感性（事后只读重分析）**：审计发现 _tsbadm 切分器的回退分支（首异常 <20% 时切 60%）在 16 个 _tsbadm 流中仅 D20_GECCO_id1 触发：全文件首异常位于 11.7%，回退至 60% 切分，训练段含 1,475 个异常点（1.77%）。其余 15 流训练段严格无异常；非 _tsbadm 流使用官方 train/test 文件，切分不依赖标签。论文中 "normal-only training segments" 的表述据此精确化。敏感性：排除 D20 后 11 个臂的 dev 宏均全部上升 0.006--0.013，臂间排序不变，score 系臂升幅最小——泄漏未偏袒任何结论方向（results/train_leak_audit.md、results/train_leak_sensitivity.md）。D20 不重切分不重跑（改历史协议风险大于收益），以披露+敏感性为处置。
2. **Moirai 第三教师执行（兑现 R9，pre-run 注册执行细节）**：模型 `Salesforce/moirai-1.1-R-base`（uni2ts 官方包；R9 注册时的 Moirai-Base 91M 的现行最接近实现，实际参数量与 revision 记录入缓存 meta 并在论文披露）。管线与 Chronos/TimesFM 同构：逐通道一步预测 → |x−x̂| → 跨通道 max；上下文 1024（patch 32 对齐），模型默认归一化；权重经 hf-mirror 下载并记录 revision。流与配置：R9 原文 5 流（D1/D3/D11/D16/D18）× 3 种子 × C-PM（epochs30/bs256/lr1e-3/stride2）= 15 训练 run。不可评分流按既有 SKIPPED 机制。**定位：探索性第三家族对照**——若结果方向与 Chronos/TimesFM 不一致，论文表述收窄为"教师家族与接口共同决定迁移"，不隐藏。
3. **确认集前瞻扩展（prospective extension，pre-run）**：受原 9 源功效限制（MDE 0.10--0.19）启发的前瞻性扩展批次。**报告层级**：primary = 原 9 源（冻结不变）→ prospective extension = 新独立源独立报告 → pooled secondary = 预先指定的层级汇总（source-cluster bootstrap，流嵌套于源）。**来源独立性**：仅接受开发集 20 流所属集合之外的全新集合；同集合新流不进入独立来源计数。**候选与备选顺序（锁定）**：Dodgers → Citrines → Bengaluru → GMU → IA4OSS；纳入标准：公开可得、逐点二值标签、测试段含异常、测试段长度 ≥ 2000；排除原因逐条记录（盲态标签核验，不接触模型指标）。**臂**：六臂（C-PM/B1-TCN/B1-PM/B2-PM/A-PM/LINEAR）× 10 种子 × 合格新源 + T0/PCA/三经典基线评分。**统计**：4 源内精确 Wilcoxon 最小 p=0.125（结构性不可确认显著），新批次报告方向与效应区间；pooled secondary 用 source-cluster bootstrap；MDE 以实测配对 SD 重算，不预设。
4. **显著化止步线（预写死）**：若 C-PM 对 B1-PM 的 pooled 优势达 Holm 校正显著且效应 >0.05 VUS-PR margin 且非单源驱动且新源方向一致 → 停机上报用户（需重定位论文主线）；其余情形如实披露为主线不变或异质性信号。任何情形禁止只报原 9 源。
5. **时点证据**：本节写入时，泄漏审计与敏感性重算已完成（results/train_leak_audit.md、train_leak_sensitivity.md），Moirai 无任何缓存/结果文件，新源数据未下载。
"""
with io.open(p, "a", encoding="utf-8", newline=chr(10)) as f:
    f.write(sec)
h = hashlib.sha256(io.open(p, "rb").read()).hexdigest()
p2 = BASE + BS + '预注册_v1.0_冻结记录.md'
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with io.open(p2, "a", encoding="utf-8", newline=chr(10)) as f:
    f.write("- R16 修订后 SHA256：%s（%s，append-only；泄漏披露+Moirai 细节+前瞻扩展 pre-run）\n" % (h, now))
print("R16:", h[:16], now)
