# -*- coding: utf-8 -*-
"""数据加载模块：预注册 v1.0 锁定的 29 条流（dev 20 + confirm 9）。
所有路径按公共库 D:\\0科研\\数据\\TSAD_数据集\\；输出统一为 (train_x, test_x, test_label)。
train 段无标签（蒸馏用），test 段带二值标签（评测用）。
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = r"D:\0科研\数据\TSAD_数据集"
TSBADM = os.path.join(ROOT, "TSB-AD-MV", "TSB-AD-M")
NAB_LABELS = os.path.join(ROOT, "NAB_labels_combined.json")


def _tsbadm(idx_name: str):
    """TSB-AD-M csv：特征列 + Label 列；文件内前 tr_ 比例为训练段的官方切片协议。
    文件名 tr_T_1st_S 中 T/S 非行数（2026-10-03 勘误），因此按 Label 首次出现点切分：
    train = 全文件，test = 尾段与官方 1st 长度近似——简化实现：train/test 按 50/50
    以首个异常点对齐（train 取首个异常前的完整前段，保证无标签泄漏；若前段不足
    20% 则取文件前 60% 并在缓存元数据记录该例外）。"""
    path = glob_one(idx_name)
    df = pd.read_csv(path)
    label = df["Label"].to_numpy().astype(np.int8)
    X = df.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    first_anom = int(np.argmax(label > 0)) if label.any() else len(label)
    cut = first_anom if first_anom >= 0.2 * len(label) else int(0.6 * len(label))
    return X[:cut], X[cut:], label[cut:]


def glob_one(idx_name: str) -> str:
    hits = [f for f in os.listdir(TSBADM) if f.endswith(".csv") and f.endswith(idx_name + ".csv")]
    if len(hits) != 1:
        raise FileNotFoundError(f"TSB-AD-M 唯一匹配失败: {idx_name} -> {hits}")
    return os.path.join(TSBADM, hits[0])


def _hai():
    d = os.path.join(ROOT, "HAI", "hai-22.04")
    tr = pd.read_csv(os.path.join(d, "train1.csv"))
    te = pd.read_csv(os.path.join(d, "test1.csv"))
    y = te["Attack"].to_numpy().astype(np.int8)
    cols = [c for c in tr.columns if c not in ("timestamp", "Attack")]
    # 仅取数值列（HAI 全为传感器/执行器数值），按列对齐
    Xtr, Xte = tr[cols].to_numpy(np.float32), te[cols].to_numpy(np.float32)
    return Xtr, Xte, y


def _emg():
    d = os.path.join(ROOT, "EMG")
    return (np.load(os.path.join(d, "train_data.npy")).astype(np.float32),
            np.load(os.path.join(d, "test_data.npy")).astype(np.float32),
            np.load(os.path.join(d, "test_label.npy")).astype(np.int8))


def _jd1_service0():
    """J-D1 service0：实测为 38 列多变量 KPI（train 23704×38 / test 23705×38，标签仅覆盖 test 前缀）。
    保留全部列作多变量流；test 截到标签长度。"""
    d = os.path.join(ROOT, "J-D", "Dataset1")
    tr = pd.read_csv(os.path.join(d, "train", "service0.csv"))
    te = pd.read_csv(os.path.join(d, "test", "service0.csv"))
    y = pd.read_csv(os.path.join(d, "test_label", "service0.csv")).iloc[:, -1].to_numpy().astype(np.int8)
    n = min(len(te), len(y))
    te = te.iloc[:n]
    return tr.to_numpy(np.float32), te.to_numpy(np.float32), y[:n]


def _nab(rel: str):
    lab = json.load(open(NAB_LABELS, encoding="utf-8"))[rel]
    df = pd.read_csv(os.path.join(ROOT, "NAB", rel))
    v = df["value"].to_numpy(np.float32).reshape(-1, 1)
    ts = pd.to_datetime(df["timestamp"])
    y = np.zeros(len(v), dtype=np.int8)
    # NAB 标签格式： anomaly 点列表（ISO 时间）→ 区间内全部记 1（区间=相邻标签点之间视为连续异常段）
    pts = sorted(pd.to_datetime(lab))
    for t in pts:
        i = int(np.searchsorted(ts.values, np.datetime64(t)))
        if 0 <= i < len(y):
            y[i] = 1
    # 官方 NAB 口径：标签点标记异常窗口起点；此处按点扩展到相邻标签之间（保守连续段）
    if len(pts) >= 2:
        y = np.zeros(len(v), dtype=np.int8)
        idxs = [int(np.searchsorted(ts.values, np.datetime64(t))) for t in pts]
        for a, b in zip(idxs[0::2], idxs[1::2] + [idxs[-1]]):
            y[a:b + 1] = 1
    cut = int(0.5 * len(v))  # NAB 无官方 train/test 切分 → 前 50% 蒸馏（预注册 §4 规则）
    return v[:cut], v[cut:], y[cut:]


def _tsbuad(folder: str, fname: str):
    df = pd.read_csv(os.path.join(ROOT, "TSB-UAD", folder, fname), header=None)
    v = df.iloc[:, 0].to_numpy(np.float32).reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy().astype(np.int8)
    cut = int(0.5 * len(v))
    return v[:cut], v[cut:], y[cut:]


def _neweather():
    d = os.path.join(ROOT, "NEweather")
    X = pd.read_csv(os.path.join(d, "NEweather_data.csv")).to_numpy(np.float32)
    y = pd.read_csv(os.path.join(d, "NEweather_class.csv")).iloc[:, 0].to_numpy().astype(np.int8)
    cut = int(0.5 * len(X))
    return X[:cut], X[cut:], y[cut:]


def _wadi():
    """C1：A1 14days（正常）蒸馏 → A2 attack（评测）。实测格式：
    A1 前3行元信息+表头（1.2M行）；A2 第1行数字伪表头+第2行真表头（~17.2万行），标签列
    'Attack LABLE (1:No Attack, -1:Attack)'——攻击=-1（反转约定）。
    成本控制（预注册R1规则②）：A1 训练段取前200k行 stride=8 抽样（25k点），A2 全量评测。
    传感器列按名称交集对齐，元数据记录。"""
    import re as _re
    A1 = os.path.join(ROOT, "WADI", "WADI.A1_9 Oct 2017", "WADI_14days.csv")
    A2 = os.path.join(ROOT, "WADI", "WADI.A2_19 Nov 2019", "WADI_attackdataLABLE.csv")
    h1 = pd.read_csv(A1, skiprows=4, nrows=2)                # 第5行为表头（带 \\UNC\ 前缀）
    norm1 = {}
    for c in h1.columns:
        base = str(c).split("\\")[-1].strip()
        if base and base not in norm1:
            norm1[base] = c
    te_full = pd.read_csv(A2, skiprows=1)                     # 真表头生效
    labcol = [c for c in te_full.columns if "LABLE" in c.upper() or "ATTACK" in c.upper()][-1]
    y = (te_full[labcol] == -1).astype(np.int8).to_numpy()    # 攻击=-1
    drop = {c for c in te_full.columns if not _re.match(r"^\d+_", str(c))}
    sens = [c for c in te_full.columns if c not in drop and c != labcol]
    common_a2 = [c for c in sens if str(c).strip() in norm1]  # A2 干净名序
    common_a1 = [norm1[str(c).strip()] for c in common_a2]    # A1 原始名序（同序）
    te = te_full[common_a2].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    tr_head = pd.read_csv(A1, skiprows=4, usecols=common_a1, nrows=200_000, engine="c")
    tr = tr_head.iloc[::8].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
    return tr, te, y[: len(te)]


FLOWS = {
    # ---- dev (20) ----
    "D1_SWaT_id1":      lambda: _tsbadm("id_1_Sensor_tr_3749_1st_9522"),  # 小切片（1.5万行×66维），弃 id_2 重流
    "D2_HAI":           _hai,
    "D3_SMD_id4":       lambda: _tsbadm("id_4_Facility_tr_7176_1st_10609"),
    "D4_SMD_id20":      lambda: _tsbadm("id_20_Facility_tr_5925_1st_17580"),
    "D5_PSM":           lambda: _tsbadm("id_1_Facility_tr_50000_1st_129872"),
    "D6_MSL_id10":      lambda: _tsbadm("id_10_Sensor_tr_1525_1st_4590"),
    "D7_MSL_id5":       lambda: _tsbadm("id_5_Sensor_tr_1150_1st_1250"),
    "D8_SMAP_id4":      lambda: _tsbadm("id_4_Sensor_tr_2160_1st_6449"),
    "D9_SMAP_id13":     lambda: _tsbadm("id_13_Sensor_tr_1173_1st_2750"),
    "D10_GHL_id2":      lambda: _tsbadm("id_2_Sensor_tr_50000_1st_51001"),
    "D11_SVDB_id2":     lambda: _tsbadm("id_2_Medical_tr_50000_1st_54982"),
    "D12_EMG":          _emg,
    "D13_JD1_svc0":     _jd1_service0,
    "D18_Exathlon_id1": lambda: _tsbadm("id_1_Facility_tr_10766_1st_12590"),
    "D19_OPP_id3":      lambda: _tsbadm("id_3_HumanActivity_tr_7016_1st_26691"),
    "D20_GECCO_id1":    lambda: _tsbadm("id_1_Sensor_tr_16165_1st_16265"),
    # ---- confirm (9) ----
    "C1_WADI":          _wadi,
    "C2_Genesis_id1":   lambda: _tsbadm("id_1_Sensor_tr_4055_1st_15538"),
    "C3_Daphnet_id1":   lambda: _tsbadm("id_1_HumanActivity_tr_9693_1st_20732"),
    "C7_MITDB_id5":     lambda: _tsbadm("id_5_Medical_tr_25000_1st_36913"),
    "C8_NEweather":     _neweather,
}
NAB_FLOWS = {
    "D14_NAB_cpu24ae8d": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "D15_NAB_cpu53ea38": "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",  # 原257a54后半段零异常→预注册备选
    "D16_NAB_occ6005":   "realTraffic/occupancy_6005.csv",
    "D17_NAB_machtemp":  "realKnownCause/machine_temperature_system_failure.csv",
    "C9_NAB_adexchange": "realAdExchange/exchange-2_cpm_results.csv",  # R4：原cpc流标签落在前半段(20%)被50%切分切尽，换同为文件序靠前且后半段含标签(37%/60%)的cpm流
}
TSBUAD_FLOWS = {
    "C4_MGAB":   ("MGAB", "1.test.out"),
    "C5_YAHOO":  ("YAHOO", "YahooA3Benchmark-TS1_data.out"),
    "C6_IOPS":   ("IOPS", "KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa.test.out"),
}


def load_flow(flow_id: str):
    """返回 (train_x [T,C] float32, test_x [T,C] float32, test_label [T] int8)。"""
    if flow_id in NAB_FLOWS:
        return _nab(NAB_FLOWS[flow_id])
    if flow_id in TSBUAD_FLOWS:
        return _tsbuad(*TSBUAD_FLOWS[flow_id])
    if flow_id in FLOWS:
        return FLOWS[flow_id]()
    raise KeyError(f"未知流: {flow_id}")


def all_flow_ids():
    return sorted(list(FLOWS) + list(NAB_FLOWS) + list(TSBUAD_FLOWS))
