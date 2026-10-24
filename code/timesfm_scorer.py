# -*- coding: utf-8 -*-
"""TimesFM 第二教师评分器（次要臂）：与 teacher.py 同一锁定管线，
点预测=TimesFM 均值点输出（其自然点预测形态；Chronos 用中位数分位——
差异属各模型自然点预测口径，缓存元数据记录）。上下文=包默认值，运行时记录。
权重本地加载（models/timesfm-2.5-200m/，curl 分段下载）。
"""
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(HERE, "..", "models", "timesfm-2.5-200m")
CTX_LEN = 1024  # 锁定：max_context=1024（patch 32 的整倍数）；元数据记录完整 ForecastConfig


class TimesFMScorer:
    def __init__(self, device=None):
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wrapper = TimesFM_2p5_200M_torch(torch_compile=False)
        if os.path.exists(os.path.join(LOCAL_DIR, "model.safetensors")):
            wrapper.from_pretrained(LOCAL_DIR)
        else:
            wrapper.from_pretrained("google/timesfm-2.5-200m-pytorch")
        wrapper.model.to(self.device).eval()  # 包装类内层才是 nn.Module
        fc = ForecastConfig(max_context=CTX_LEN, max_horizon=128,
                            per_core_batch_size=64, normalize_inputs=True)
        wrapper.compile(forecast_config=fc)   # compile 在包装类上（内层的是 torch.compile）
        self.model = wrapper                  # forecast 走包装类
        self.ctx = CTX_LEN

    @torch.no_grad()
    def _forecast_points(self, contexts: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """contexts [N, ctx] → 均值点预测 [N]。分批防显存。"""
        outs = []
        for i in range(0, len(contexts), batch_size):
            batch = [contexts[i + j] for j in range(min(batch_size, len(contexts) - i))]
            mean_preds, _ = self.model.forecast(horizon=1, inputs=batch)
            outs.append(np.asarray(mean_preds)[:, 0])
        return np.concatenate(outs).astype(np.float32)

    def forecast_mean(self, series: np.ndarray, batch_size: int = 64) -> np.ndarray:
        T = len(series)
        out = np.full(T, np.nan, dtype=np.float32)
        if T <= self.ctx:
            return out
        starts = np.arange(self.ctx, T)
        ctxs = np.stack([series[t - self.ctx:t] for t in starts])
        out[starts] = self._forecast_points(ctxs, batch_size)
        return out

    def score_flow(self, X: np.ndarray, batch_size: int = 64):
        T, C = X.shape
        Xhat = np.full((T, C), np.nan, dtype=np.float32)
        for c in range(C):
            Xhat[:, c] = self.forecast_mean(np.ascontiguousarray(X[:, c]), batch_size)
        with np.errstate(invalid="ignore"):
            resid = np.abs(X - Xhat)
        s = np.nanmax(resid, axis=1) if C > 1 else resid[:, 0]
        return s.astype(np.float32), Xhat
