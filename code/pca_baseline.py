# -*- coding: utf-8 -*-
"""B3 预锁定强基线：PCA 重构（预注册 v1.0 §3）。因果纪律：仅 train 段拟合（不用标签）。
部署存储 = mu/sd + PCA 成分（计入资源表）。"""
import numpy as np
from sklearn.decomposition import PCA


class PCABaseline:
    def __init__(self, var=0.95, random_state=0):
        self.var, self.random_state = var, random_state

    def fit(self, Xtr):
        self.mu = Xtr.mean(0)
        self.sd = Xtr.std(0) + 1e-8
        self.pca = PCA(n_components=self.var, random_state=self.random_state).fit((Xtr - self.mu) / self.sd)
        self.storage_mb = (Xtr.shape[1] * 2 + self.pca.components_.size + self.pca.mean_.size) * 4 / 1e6
        return self

    def score(self, Xte):
        z = (Xte - self.mu) / self.sd
        rec = self.pca.inverse_transform(self.pca.transform(z))
        err = np.abs(z - rec)
        return (err.max(1) if err.shape[1] > 1 else err[:, 0]).astype(np.float32)
