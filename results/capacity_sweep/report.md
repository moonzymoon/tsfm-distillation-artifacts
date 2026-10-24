# 容量×接口扫描（R10，pre-run 注册；Chronos 教师，5 dev 流 × 3 种子）

## D1_SWaT_id1 (C=66, train [9522, 66])

| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |
|---|---|---|---|---|
| score-distill | PatchMLPLite | 265,617 | 0.5320 | -0.0935 |
| forecast-distill | PatchMLPLite | 265,617 | 0.5673 | -0.0846 |
| self-supervised | PatchMLPLite | 265,617 | 0.5680 | -0.0937 |
| score-distill | PatchMLP256 | 661,905 | 0.5638 | -0.1253 |
| forecast-distill | PatchMLP256 | 661,905 | 0.5563 | -0.091 |
| self-supervised | PatchMLP256 | 661,905 | 0.5522 | -0.0843 |
| score-distill | PatchMLP512 | 1,847,697 | 0.5523 | -0.1164 |
| forecast-distill | PatchMLP512 | 1,847,697 | 0.5543 | -0.0938 |
| self-supervised | PatchMLP512 | 1,847,697 | 0.5481 | -0.1259 |

## D3_SMD_id4 (C=38, train [10609, 38])

| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |
|---|---|---|---|---|
| score-distill | PatchMLPLite | 236,945 | 0.0519 | 0.7125 |
| forecast-distill | PatchMLPLite | 236,945 | 0.0281 | 0.0064 |
| self-supervised | PatchMLPLite | 236,945 | 0.0282 | 0.0065 |
| score-distill | PatchMLP256 | 604,561 | 0.0635 | 0.6622 |
| forecast-distill | PatchMLP256 | 604,561 | 0.0323 | 0.0231 |
| self-supervised | PatchMLP256 | 604,561 | 0.0319 | 0.0134 |
| score-distill | PatchMLP512 | 1,733,009 | 0.0719 | 0.442 |
| forecast-distill | PatchMLP512 | 1,733,009 | 0.0316 | -0.0367 |
| self-supervised | PatchMLP512 | 1,733,009 | 0.0303 | -0.0501 |

## D11_SVDB_id2 (C=2, train [54982, 2])

| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |
|---|---|---|---|---|
| score-distill | PatchMLPLite | 200,081 | 0.0612 | 0.8384 |
| forecast-distill | PatchMLPLite | 200,081 | 0.0906 | 0.3717 |
| self-supervised | PatchMLPLite | 200,081 | 0.0923 | 0.3671 |
| score-distill | PatchMLP256 | 530,833 | 0.0619 | 0.8403 |
| forecast-distill | PatchMLP256 | 530,833 | 0.0923 | 0.3661 |
| self-supervised | PatchMLP256 | 530,833 | 0.0902 | 0.3656 |
| score-distill | PatchMLP512 | 1,585,553 | 0.0635 | 0.8349 |
| forecast-distill | PatchMLP512 | 1,585,553 | 0.0919 | 0.3704 |
| self-supervised | PatchMLP512 | 1,585,553 | 0.0892 | 0.3361 |

## D18_Exathlon_id1 (C=18, train [12590, 18])

| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |
|---|---|---|---|---|
| score-distill | PatchMLPLite | 216,465 | 0.5147 | 0.0477 |
| forecast-distill | PatchMLPLite | 216,465 | 0.9218 | 0.1653 |
| self-supervised | PatchMLPLite | 216,465 | 0.9252 | 0.2887 |
| score-distill | PatchMLP256 | 563,601 | 0.5048 | 0.0255 |
| forecast-distill | PatchMLP256 | 563,601 | 0.9323 | 0.3635 |
| self-supervised | PatchMLP256 | 563,601 | 0.9137 | 0.1611 |
| score-distill | PatchMLP512 | 1,651,089 | 0.6475 | 0.2213 |
| forecast-distill | PatchMLP512 | 1,651,089 | 0.9413 | 0.3068 |
| self-supervised | PatchMLP512 | 1,651,089 | 0.9346 | 0.3566 |

## D12_EMG (C=8, train [243520, 8])

| 臂 | 骨干 | 参数量 | mean VUS-PR | median rho |
|---|---|---|---|---|
| score-distill | PatchMLPLite | 206,225 | 0.1190 | 0.9901 |
| forecast-distill | PatchMLPLite | 206,225 | 0.1190 | 0.983 |
| self-supervised | PatchMLPLite | 206,225 | 0.1190 | 0.9623 |
| score-distill | PatchMLP256 | 543,121 | 0.1181 | 0.9899 |
| forecast-distill | PatchMLP256 | 543,121 | 0.1189 | 0.9829 |
| self-supervised | PatchMLP256 | 543,121 | 0.1199 | 0.9565 |
| score-distill | PatchMLP512 | 1,610,129 | 0.1191 | 0.9898 |
| forecast-distill | PatchMLP512 | 1,610,129 | 0.1187 | 0.9829 |
| self-supervised | PatchMLP512 | 1,610,129 | 0.1190 | 0.9672 |
