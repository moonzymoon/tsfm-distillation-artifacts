# Suggested Reviewers（JoS 投稿用，2026-10-10 拟）

选择原则：本文参考文献作者（熟悉领域）、无合作/同机构关系、国际分布、评测+轻量化+蒸馏三视角均衡。
邮箱请投稿时从各单位官网核实后填入系统（此处不预填，避免错误）。

| # | 姓名 | 机构 | 视角匹配 | 理由 |
|---|---|---|---|---|
| 1 | Kai-Chen Yang | （ASOC 论文通讯，官网核实） | **最对口** | 冻结 TSFM 异常检测的无泄漏评测（本文引文 [CLRA]），评测方法论与本文三轴设计同域 |
| 2 | John Paparrizos | The Ohio State University | 评测基准 | TSB-AD / VUS 作者；本文的流池、主指标（VUS-PR）与基准口径均出自其系列 |
| 3 | Vijay Ekambaram | IBM Research | 轻量 TSFM | Tiny Time Mixers（TTM）第一作者；TSFM 压缩与零/少样本迁移视角与本文 B2 接口问题直接相关 |
| 4 | Martin Uray | FH Salzburg (University of Applied Sciences) | 零样本审计 | 零样本 TSFM 多变量异常检测系统评测（本文引文），对教师条件依赖性结论有第一手经验 |
| 5 | Shengsheng Lin | 华南理工大学 SCUT（勘误：原误写电子科大） | 极轻量模型 | SparseTSF（1k 参数）第一作者；对 sub-MB 边缘预算与学生架构设计有直接发言权 |

邮箱核实（2026-10-17，官网/官方仓库来源）：

| # | 姓名 | 邮箱 | 来源 |
|---|---|---|---|
| 2 | John Paparrizos | paparrizos.1@osu.edu | OSU 工学院教职工目录 |
| 3 | Vijay Ekambaram | vijaye12@in.ibm.com | TTM NeurIPS 2024 论文 |
| 4 | Martin Uray | martin.uray@fh-salzburg.ac.at | FH Salzburg 教职工页 |
| 5 | Shengsheng Lin | cslinshengsheng@mail.scut.edu.cn | SparseTSF 官方 GitHub |
| 1 | Kai-Chen Yang | （提交时从 ScienceDirect 文章页作者面板核实，DOI 见参考文献 CLRA 条目） | |

回避说明：未列 IA-AD（JoS 2026 在刊邻居）作者——同刊过近；未列 Chronos 团队——教师模型出品方，存在立场关联。

# Title Page（投稿系统录入用）

**Title**: Distilling Frozen Time-Series Foundation Model Teachers into Lightweight Edge Anomaly Detectors: A Preregistered Quality–Fidelity–Cost Study

**Authors**: Yao Zhang¹*, Yanling Li¹, Yichen Wang²
**Affiliations**: ¹Department of Artificial Intelligence, Baoding University, Baoding, China; ²Department of Education, Baoding University, Baoding, China
**Corresponding author**: Yao Zhang, email: zhangyao@bdu.edu.cn
**ORCID**: Yao Zhang 0009-0008-7636-8158; Yanling Li 0000-0001-7576-2497; Yichen Wang 0009-0007-3682-7463

**Declarations**:
- Funding: The authors received no specific funding for this work.
- Conflict of interest: The authors declare no conflicts of interest.
- Data availability: All benchmark data are public; code, preregistration
  with hash chain, and per-run results are released at
  https://github.com/moonzymoon/tsfm-distillation-artifacts and archived
  with the Zenodo DOI https://doi.org/10.5281/zenodo.23623799.
- Author contributions (CRediT): Yao Zhang: Conceptualization,
  Methodology, Software, Investigation, Data curation, Writing - original
  draft, Visualization. Yanling Li: Supervision, Resources, Writing -
  review & editing. Yichen Wang: Data curation, Writing - review & editing.
- Generative AI disclosure: as in manuscript (ZCode/GLM assisted code,
  experiments, and drafting under a preregistered protocol; author
  verified all results).
