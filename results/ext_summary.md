# n=10 三层分析（primary 9 冻结 / extension E2 独立 / pooled 层级 secondary）

## Primary（原 9 源，冻结）

| 臂 | 宏均 |
|---|---|
| C-PM | 0.2501 |
| A-PM | 0.2116 |
| B1-TCN | 0.2778 |
| B1-PM | 0.3039 |
| B2-PM | 0.2893 |
| LINEAR | 0.3229 |

## Extension（E2_CATSv2，独立新源）

| 臂 | 宏均 |
|---|---|
| C-PM | 0.0772 |
| A-PM | 0.0492 |
| B1-TCN | 0.0360 |
| B1-PM | 0.0360 |
| B2-PM | 0.0360 |
| LINEAR | 0.0360 |
| robustz (E2) | 0.0401 |
| mares (E2) | 0.0528 |
| lofwin (E2) | 0.2463 |
| T0 (E2) | 0.0538 |

## Pooled secondary（层级 bootstrap，流嵌套于源）

| C-PM pooled(10源) | 0.2328 |
| A-PM pooled(10源) | 0.1954 |
| B1-TCN pooled(10源) | 0.2536 |
| B1-PM pooled(10源) | 0.2771 |
| B2-PM pooled(10源) | 0.2640 |
| LINEAR pooled(10源) | 0.2942 |
- C-PM−B1-PM 差：原9源 -0.0538；E2 +0.0412；pooled -0.0443

## MDE

- C-PM vs B1-PM：paired SD 0.1415 → n=9 MDE ≈ 0.139；n=13（同 SD 假设）≈ 0.086