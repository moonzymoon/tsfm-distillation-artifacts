# -*- coding: utf-8 -*-
"""扩展数据接入：解压选定 csv → data.py 注册 EXT 流（尾部 FLOWS.update，零改动既有条目）。"""
import io
import json
import os
import zipfile

BS = chr(92)
NL = chr(10)
HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(HERE, "..", "TSB-AD-M.zip")
EXTDIR = r"D:" + BS + "0科研" + BS + "数据" + BS + "TSAD_数据集" + BS + "TSB-AD-M-ext"
EXT_TSBADM = EXTDIR
FLOWS_JS = os.path.join(HERE, "..", "results", "extension_flows.json")
DATA_PY = os.path.join(HERE, "data.py")

FLOW_IDS = {"CreditCard": "E1_CreditCard", "CATSv2": "E2_CATSv2",
            "TAO": "E3_TAO", "LTDB": "E4_LTDB"}


def main():
    os.makedirs(EXTDIR, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    picked = json.load(open(os.path.join(HERE, "..", "results", "extension_flows.json"),
                            encoding="utf-8"))
    entries = []
    for src, info in picked.items():
        fname = os.path.basename(info["file"])
        with zf.open(info["file"]) as fsrc, open(os.path.join(EXTDIR, fname), "wb") as fdst:
            fdst.write(fsrc.read())
        fid = FLOW_IDS[src]
        entries.append('    "%s": lambda: _tsbadm_ext("%s"),' % (fid, fname))
        print("extracted", fname, "->", fid)
    # data.py 尾部追加 loader + 注册（append-only，不动既有行）
    s = io.open(DATA_PY, encoding='utf-8').read()
    if "_tsbadm_ext" in s:
        print("data.py already extended - skip")
        return
    add = ('''

# ----- R16-B 扩展流（TSB-AD-M 新集合，2026-10 追加；首异常{{PCT}}纯净训练段） -----
EXT_TSBADM = r"{{DIR}}"


def _tsbadm_ext(fname: str):
    import glob as _g
    hits = [f for f in os.listdir(EXT_TSBADM) if f == fname]
    assert len(hits) == 1, (fname, hits)
    df = pd.read_csv(os.path.join(EXT_TSBADM, hits[0]))
    label = df["Label"].to_numpy().astype(np.int8)
    X = df.drop(columns=["Label"]).to_numpy(dtype=np.float32)
    first_anom = int(np.argmax(label > 0)) if label.any() else len(label)
    cut = first_anom
    return X[:cut], X[cut:], label[cut:]


FLOWS.update({
{{ENTRIES}}
})
'''.replace("{{DIR}}", EXT_TSBADM).replace("{{PCT}}", "20%").replace("{{ENTRIES}}", NL.join(entries)))
    with io.open(DATA_PY, 'a', encoding='utf-8', newline=chr(10)) as f:
        f.write(add)
    print('data.py extended')


if __name__ == "__main__":
    main()
