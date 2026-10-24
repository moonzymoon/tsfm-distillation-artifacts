# P0 汇总：TimesFM 第二教师（C-PM 配置，5 种子）

覆盖 10 流；跳过 3 流（结构短于 ctx=1024，见 SKIPPED 标记）
宏均（覆盖流）：C-PM(TF)=0.2392 | C-PM(Chronos)=0.2741 | 最强 teacher-free=0.3722 | TimesFM T0=0.2784
median Spearman rho（C-PM-TF vs TimesFM 教师）= 0.7047（50 runs）
C-PM(TF) 逐流胜过最强 teacher-free 臂的流数：1 / 10

| 流 | C-PM(TF) | C-PM(Chronos) | TimesFM T0 | 最强free | 来自 |
|---|---|---|---|---|---|
| C3_Daphnet_id1 | 0.2166 | 0.3166 | 0.2409 | 0.2524 | LINEAR |
| C4_MGAB | 0.0088 | 0.0083 | 0.0093 | 0.1245 | B1-TCN |
| C6_IOPS | 0.0725 | 0.2168 | 0.2799 | 0.5738 | LINEAR |
| C7_MITDB_id5 | 0.0131 | 0.0078 | 0.0077 | 0.0230 | B1-TCN |
| C8_NEweather | 0.9361 | 0.9333 | 0.9399 | 0.9519 | PCA |
| D11_SVDB_id2 | 0.0686 | 0.0615 | 0.0724 | 0.0923 | B1-PM |
| D12_EMG | 0.1196 | 0.1188 | 0.1183 | 0.1190 | B1-PM |
| D18_Exathlon_id1 | 0.4897 | 0.4813 | 0.3924 | 0.9474 | LINEAR |
| D1_SWaT_id1 | 0.4364 | 0.5421 | 0.3413 | 0.5680 | B1-PM |
| D3_SMD_id4 | 0.0302 | 0.0547 | 0.3817 | 0.0701 | AB-PM |

## R10 判定

- J1（宏均无优势）: gap = -0.1330 → **不成立**
- J2（保真同量级）: median rho = 0.7047 → **不成立**
- 判定：**停机线触发——报告用户裁决**