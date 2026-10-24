# Artifact Release — Paper 20 (A3: TSFM Teacher Distillation Study)

预注册实证研究的完整工件包（Distilling Frozen Time-Series Foundation Model
Teachers into Lightweight Edge Anomaly Detectors: A Preregistered
Quality-Fidelity-Cost Study, The Journal of Supercomputing 投稿件）。

## 内容清单
- code/            全部实验代码（教师打分/学生训练/五臂/评测/统计/计时/图/树莓派实测方案）
- preregistration/ 预注册 v1.0 全文（含 R1-R17 append-only 修订）+ 冻结记录（SHA-256 哈希链）
- results/         全部逐 run JSON：
  - m5/m5b/m6      主网格（400）/辅助臂（420）/确认集（540，含 90 A-PM 增补）
  - m6ext          第十源扩展（CATSv2，R16：60 训练 + 4 评估）
  - c1_shuffled    洗牌教师对照 R6（40）
  - interface_ablation  接口消融 R7（30）
  - r8_anomaly     异常增广对照 R8（30，配 teacher_cache_aug 缓存）
  - timesfm_subset 第二教师子集（25）+ teacher_cache_timesfm_meta
  - moirai_subset  第三教师 Moirai R16（15，配 teacher_cache_moirai）
  - capacity_sweep 容量扫描 R10/R13（135+45）
  - classical_baselines 零调参经典基线 R11（87）
  - cost/          计时矩阵（随机权重旧版 + 训练后工件 v2：timing_matrix_TRAINED.{json,md}，
                   5轮x(50warm+500timed)）+ e2e v2（student_e2e_timing_v2.json）+
                   trained_artifacts/（36 个训练后 ONNX/INT8 工件，即论文附录 Table 11
                   与 INT8 质量漂移列的来源工件）
  - teacher_cache_meta / teacher_cache_aug   教师分数缓存元数据与 R8 增广缓存
- paper/           LaTeX 源与编译产物（sn-jnl v3.1，图 1 机制图 TikZ 源含 figurestyle.sty）
- submission_flat / submission_package  投稿件（平铺单文件 tex + 图 + 模板 + cover letter
                   + suggested reviewers）

## 运行账本（与论文 Reproduction details 表一致）
1,856 total = 1,270 预注册（400 主网格 + 420 辅助 + 450 五臂确认）
+ 586 披露增补（90 A-PM 确认增补 + 40 R6 洗牌 + 25 TimesFM 子集 + 30 R7 接口
+ 30 R8 异常增广 + 25 TimesFM 全流 R10 + 135 容量扫描 R10 + 87 经典基线 R11
+ 45 容量-1024 R13 + 60+4 扩展 R16 + 15 Moirai R16），逐 run JSON 可一一相加核对。
教师原始分数缓存（teacher_cache 165M / teacher_cache_timesfm 26M）未入包，
可用 build_cache.py / run_moirai_cache.py 复现。

## 复现路线
1. 环境：Python 3.11, torch 2.13, chronos-forecasting 2.3.2, timesfm 3.0.2, onnxruntime 1.28
2. 教师缓存：python build_cache.py（GPU, Chronos-Bolt-base）
3. 主网格：python run_m5.py；辅助臂：python run_m5b.py
4. 确认集：CUDA_VISIBLE_DEVICES=-1 python run_m6.py；A-PM 增补：python run_m6_apm.py
5. 负对照：python run_c1_shuffled.py；R7：python run_interface_ablation.py；R8：python run_r8_anomaly.py
6. 第二/第三教师：python run_timesfm_subset.py；python run_moirai_cache.py + run_moirai_train.py
7. 扩展源：python setup_extension_data.py + run_extension.py（CATSv2，结果在 results/m6ext）
8. 容量/基线：capacity 相关脚本；python run_classical_baselines.py
9. 统计与体检：python health_check.py
10. 计时（论文口径）：python timing_matrix_trained.py（训练后工件，空载机，单线程）
11. 附录表重生成：python gen_cost_appendix.py；全部图：python wire_figures.py

## 数据
全部公共基准（TSB-AD-M/TSB-UAD/NAB/SMD/PSM/MSL/SMAP/SWaT/HAI/MetroPT3/WADI/J-D/EMG/GHL/GECCO/Exathlon/Opportunity/Genesis/Daphnet/MITDB/SVDB/NEweather/CATSv2），
NAB 标签用官方 combined_labels.json。

## 完整性
预注册冻结与 R1-R17 修订的 SHA-256 哈希链见 paper/ 附录 Table 9（hash chain）
与 preregistration/冻结记录。
