JoS 投稿包（2026-10-25 终版，Snapp 平台）
========================================

平台：Springer Snapp（JoS 已从 Editorial Manager 迁移；投稿入口见期刊官网
"Submit manuscript" 按钮，或 https://www.editorialmanager.com/susc 重定向）。

上传用文件（本目录与 submission_flat/ 均为最终版，哈希 d81c3613）
  ../em_latex_source.zip  12 个源文件平铺 zip（main_submission.tex + refs.bib
                          + sn-jnl.cls + sn-basic.bst + 8 张图 PDF）——
                          Snapp 的 "Upload manuscript" 直接传它，平台自行编译
  manuscript_preview.pdf  终稿预览（33 页；含 GitHub 链接与 Zenodo DOI；
                          平台自行编译时无需上传，仅备核对）
  cover_letter_plain.txt  Cover Letter 全文（含 GitHub/Zenodo 链接，直接粘贴）
  cover_letter_JoS.md     同上的 markdown 版
  suggested_reviewers.md  5 人审稿人名单（4 人邮箱已核实；Kai-Chen Yang 的邮箱
                          提交时从 ScienceDirect 文章页作者面板核实）+
                          Title Page 录入稿（含全部声明文本）

系统录入速查
  Title: Distilling Frozen Time-Series Foundation Model Teachers into
  Lightweight Edge Anomaly Detectors: A Preregistered Quality-Fidelity-Cost Study
  Abstract: 148 词（main_submission.tex 的 bstract{...}，限 100-150）
  Keywords: time series anomaly detection; knowledge distillation;
  time-series foundation models; edge deployment; ONNX quantization;
  preregistered evaluation
  Funding: The authors received no specific funding for this work.
  Competing interests: The authors declare no competing interests.
  Data availability: 见 cover letter / Title Page 中的同一句（含两个链接）
  通讯作者电话：表格必填，提前备好

已发布的工件（正文与信中均已引用）
  GitHub: https://github.com/moonzymoon/tsfm-distillation-artifacts (tag v1.0)
  Zenodo: https://doi.org/10.5281/zenodo.23623799

编译复现：pdflatex main_submission && bibtex main_submission && pdflatex x2
（干净目录 + em_latex_source.zip 实测通过：33 页、零未解析引用，2026-10-25）
