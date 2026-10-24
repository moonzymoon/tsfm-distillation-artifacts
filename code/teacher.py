# -*- coding: utf-8 -*-
"""教师打分模块：冻结 Chronos-Bolt（预注册 §2）。
s_T(t) = max_c | x_t,c - xhat_median_c(t) |，xhat 由模型对上下文做一步概率预测取中位数分位。
归一化 = 模型内置 per-context scaling（即因果归一化，写入缓存元数据）。
前 ctx_len 个点无分定义 → 记 NaN，由调用方以流内最小分填充（VUS 容忍区间语义下无影响）。
"""
import os
import numpy as np
import torch

DEFAULT_TEACHER = "amazon/chronos-bolt-mini"   # 冒烟/试跑用 mini；主缓存用 amazon/chronos-bolt-base
CTX_LEN = 512                                  # 预注册：库推荐默认上下文，构建时写入缓存元数据
QUANTILE_MEDIAN_IDX = 4                        # bolt 默认分位 [0.1..0.9]，中位数=第4个
_LOCAL_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def resolve_model(name):
    """HF 直连被墙：本地 models/ 目录有完整权重则优先本地路径（经 curl 分段下载）。"""
    local = os.path.join(_LOCAL_MODELS, name.split("/")[-1])
    if os.path.exists(os.path.join(local, "model.safetensors")):
        return local
    return name


class BoltScorer:
    def __init__(self, model_name=DEFAULT_TEACHER, device=None):
        from chronos import ChronosBoltPipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = ChronosBoltPipeline.from_pretrained(resolve_model(model_name), device_map=self.device)
        self.model_name = model_name
        self.ctx = min(CTX_LEN, getattr(self.pipe, "config", {}).get("context_length", CTX_LEN)
                       if hasattr(self.pipe, "config") else CTX_LEN)

    @torch.no_grad()
    def forecast_median(self, series: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """series [T] 单通道 → xhat_median [T]（前 ctx 位 NaN）。"""
        T = len(series)
        out = np.full(T, np.nan, dtype=np.float32)
        if T <= self.ctx:
            return out
        starts = np.arange(self.ctx, T)           # t：预测 x[t]，上下文 series[t-ctx:t]
        ctxs = np.stack([series[t - self.ctx:t] for t in starts])
        med = np.empty(len(starts), dtype=np.float32)
        for i in range(0, len(starts), batch_size):
            cb = torch.from_numpy(ctxs[i:i + batch_size])
            q = self.pipe.predict(inputs=cb, prediction_length=1)    # [b, nq, pl=1]
            med[i:i + batch_size] = q[:, QUANTILE_MEDIAN_IDX, 0].cpu().numpy()
        out[starts] = med
        return out

    def score_flow(self, X: np.ndarray, batch_size: int = 128):
        """X [T,C] → (s_T [T], Xhat [T,C])；s_T=max_c |x-xhat|，NaN 位置记 NaN。"""
        T, C = X.shape
        Xhat = np.full((T, C), np.nan, dtype=np.float32)
        for c in range(C):
            Xhat[:, c] = self.forecast_median(np.ascontiguousarray(X[:, c]), batch_size)
        with np.errstate(invalid="ignore"):
            resid = np.abs(X - Xhat)
        s = np.nanmax(resid, axis=1) if C > 1 else resid[:, 0]
        return s.astype(np.float32), Xhat
