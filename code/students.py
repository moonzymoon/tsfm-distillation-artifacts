# -*- coding: utf-8 -*-
"""学生架构：TCN-lite / PatchMLP-lite（预注册 §3；参数量目标 ≤1M，硬约束=导出大小+p95 延迟）。
输入窗口 x_{t-L+1..t}（含当前观测），输出标量分数。"""
import numpy as np
import torch
import torch.nn as nn


class CausalConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        if self.pad:
            x = x[:, :, :-self.pad]          # 因果：裁掉右侧看得到的未来
        return self.act(x)


class TCNLite(nn.Module):
    """空洞因果卷积 ×4 + 池化 + 头。通道 96/112/128/128，L=128 时参数约 0.18M。"""
    def __init__(self, c_in, L=128, ch=(96, 112, 128, 128), k=5):
        super().__init__()
        blocks, prev = [], c_in
        for i, c in enumerate(ch):
            blocks.append(CausalConvBlock(prev, c, k, dilation=2 ** i))
            prev = c
        self.body = nn.Sequential(*blocks)
        self.feat_dim = ch[-1] * 4
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(4), nn.Flatten(),
                                  nn.Linear(prev * 4, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, x):                    # x [B,C,L]
        return self.head(self.body(x)).squeeze(-1)


class PatchMLPLite(nn.Module):
    """patch 线性嵌入 + 2 层 mixer（token-mix/chan-mix）+ 头；谱系引 PatchTST/MLP-Mixer。"""
    def __init__(self, c_in, L=128, patch=8, dim=128):
        super().__init__()
        assert L % patch == 0
        self.patch, self.n = patch, L // patch
        self.embed = nn.Linear(c_in * patch, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.tmix = nn.Linear(self.n, self.n)
        self.norm2 = nn.LayerNorm(dim)
        self.cmix = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.feat_dim = dim * self.n
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(dim * self.n, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, x):                    # x [B,C,L]
        B, C, L = x.shape
        p = x.reshape(B, C, self.n, self.patch).permute(0, 2, 1, 3).reshape(B, self.n, C * self.patch)
        h = self.embed(p)
        h = h + self.tmix(self.norm1(h).transpose(1, 2)).transpose(1, 2)
        h = h + self.cmix(self.norm2(h))
        return self.head(h).squeeze(-1)


class NextStepHead(nn.Module):
    """B1/B2 臂：同骨干 + 下一步值回归头（预测窗口末点），分数=|残差|的 max 聚合。"""
    def __init__(self, backbone_cls, c_in, L, **kw):
        super().__init__()
        self.backbone = backbone_cls(c_in, L, **kw)
        self.reg = nn.Linear(self.backbone.feat_dim, 1)

    def forward(self, x):                    # 训练走 _feat+reg 分支
        raise NotImplementedError("使用 score_fn")


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_windows(X: np.ndarray, L: int, stride: int = 1):
    """X [T,C] → 窗口 [N,C,L]（窗口末点即打分时点 t）。"""
    idx = np.arange(0, len(X) - L + 1, stride)
    out = np.stack([X[i:i + L] for i in idx])
    return out.transpose(0, 2, 1).astype(np.float32), idx + L - 1   # [N,C,L], 打分时点
