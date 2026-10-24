JoS 投稿包（2026-10-10 组装，终版）
================================

文件清单
  main_submission.tex    单文件稿源（sn-jnl 要求：无 \input，全部内联；边界刻画口径）
  refs.bib               参考文献库（26 条，全部审计核实，18 份 PDF 存档于 literature/refs_pdf/）
  sn-jnl.cls / sn-basic.bst  Springer Nature 模板 v3.1 (2024-12)
  figures/               7 张图（PDF 矢量）
  cover_letter_JoS.md    给编辑的信（边界刻画口径；[Author Name] 等占位待填）
  suggested_reviewers.md 5 人建议审稿人名单 + Title Page 录入稿（邮箱需官网核实后填）
  manuscript_preview.pdf 编译预览（31 页；与正式版逐字一致）

投稿步骤（全部就绪后）
  1. 把 main_submission.tex 开头作者块和 cover_letter_JoS.md、suggested_reviewers.md
     中的占位替换为真实信息（姓名/单位/邮箱/ORCID/基金）；
  2. 登录 https://www.editorialmanager.com/susc/ → Submit → 按系统录入
     （题名/摘要 250 词/6 关键词可从 tex 首页直接复制；分类与编辑按系统列表选）；
  3. 上传 main_submission.tex + refs.bib + figures/ + 模板两文件（或按系统提示合并上传
     manuscript_preview.pdf 作为编译版）；
  4. suggested reviewers 从 suggested_reviewers.md 录入（邮箱官网核实）；
  5. 提交后在"已投稿论文汇总表.xlsx"手动加行（台账规约见该目录 README；
     同时把第二十篇期刊列从"候选 Machine Learning"更新为 The Journal of Supercomputing）。

编译复现：pdflatex main_submission && bibtex main_submission && pdflatex x2
（MiKTeX/TeX Live 均可；figures/ 需与 tex 同目录）

Artifact 仓库：release/ 目录（48MB）已按数据声明承诺清单配齐
（代码/预注册+R1-R15 哈希链/全部逐 run 结果含 R6-R8 与训练后计时工件/缓存元数据），
建 GitHub/Zenodo 后取链接填入数据声明占位。
