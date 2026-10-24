# -*- coding: utf-8 -*-
"""蒸馏与对照臂训练（预注册 §3）。
案A 分数蒸馏 / 案C 排序蒸馏（O(MK) 采样）/ B1 无教师自监督 / B2 预测蒸馏→残差 / 线性下界。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from students import TCNLite, PatchMLPLite, NextStepHead, make_windows


class RankLoss(nn.Module):
    """案C：每批 M 窗口，每锚点采 K 个配对（近距+远距混合），滤教师近并列；O(MK)。"""
    def __init__(self, K=8, eps=1e-8):
        super().__init__()
        self.K, self.eps = K, eps

    def forward(self, z, sT):                     # z [M] 学生分, sT [M] 教师分
        M = z.shape[0]
        if M < 2:
            return z.sum() * 0.0
        idx = torch.randint(0, M, (M, self.K), device=z.device)
        ar = torch.arange(M, device=z.device).unsqueeze(1).expand(M, self.K)
        idxc = idx.clamp(0, M - 1)
        zi = z.unsqueeze(1).expand(M, self.K)
        zj = z[idxc]
        si = sT.unsqueeze(1).expand(M, self.K)
        sj = sT[idxc]
        keep = (idx != ar) & ((si - sj).abs() > self.eps)
        if keep.sum() == 0:
            return z.sum() * 0.0
        pair = F.softplus(-torch.sign(si - sj).detach() * (zi - zj))
        return pair[keep].mean()


def loss_case_A(z, sT, use_log1p: bool):
    t = torch.log1p(sT) if use_log1p else sT
    return F.huber_loss(z, t)


def train_case(model_name, X_train, sT_train, L=128, use_log1p=False, case="A",
               epochs=30, bs=256, lr=1e-3, seed=0, device="cpu", stride=1,
               teacher_Xhat=None):
    """统一训练入口。case ∈ {A, C, B1, B2, LINEAR}。
    返回 (score_fn(X)->[T] np.float32, info dict)。"""
    torch.manual_seed(seed); np.random.seed(seed)
    C = X_train.shape[1]
    dev = torch.device(device)
    if case != "LINEAR":
        if model_name == "TCNLite":
            backbone = TCNLite
        elif model_name == "PatchMLPLite":
            backbone = PatchMLPLite
        else:
            raise ValueError(model_name)

    if case in ("A", "C"):
        net = backbone(C, L).to(dev)
        W, t_idx = make_windows(X_train, L, stride)          # [N,C,L], 打分时点
        s_t = sT_train[t_idx]
        keep = ~np.isnan(s_t)
        Wt = torch.from_numpy(W[keep]); st = torch.from_numpy(s_t[keep].astype(np.float32))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        rank = RankLoss()
        n = len(Wt)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = Wt[perm[i:i + bs]].to(dev); bt = st[perm[i:i + bs]].to(dev)
                z = net(b)
                loss = loss_case_A(z, bt, use_log1p) if case == "A" else rank(z, bt)
                opt.zero_grad(); loss.backward(); opt.step()
        def score_fn(X, _net=net, _L=L, _dev=dev):
            _net.eval()
            Wq, tidx = make_windows(X, _L, 1)
            outs = []
            with torch.no_grad():
                for i in range(0, len(Wq), 512):
                    outs.append(_net(torch.from_numpy(Wq[i:i + 512]).to(_dev)).cpu().numpy())
            s = np.full(len(X), np.nan, dtype=np.float32)
            s[tidx] = np.concatenate(outs)
            return s
        return score_fn, {"params": sum(p.numel() for p in net.parameters())}

    if case in ("B1", "B2"):
        # 骨干特征 → 回归下一步值；分数 = max_c |x_t − xhat_student_c|
        # （B1 目标=真值 x_t；B2 目标=教师预测 xhat_T,t —— DistilTS 式对照）
        # 教师预测缓存前 ctx_len 点为 NaN：含 NaN 目标/输入的窗口一律丢弃（防 NaN 中毒）
        reg = NextStepHead(backbone, C, L).to(dev)
        Win, t_in = make_windows(X_train, L + 1, stride)     # [N,C,L+1]
        Xin, Xtg = Win[:, :, :-1], Win[:, :, -1]
        if case == "B2":
            assert teacher_Xhat is not None, "B2 需要教师预测缓存 Xhat"
            Xtg = teacher_Xhat[t_in].astype(np.float32)
        keep = (~np.isnan(Xin).any(axis=(1, 2))) & (~np.isnan(Xtg).any(axis=1))
        Xin, Xtg, t_in = Xin[keep], Xtg[keep], t_in[keep]
        Xi = torch.from_numpy(Xin); Xt = torch.from_numpy(Xtg)
        opt = torch.optim.Adam(reg.parameters(), lr=lr)
        n = len(Xi)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = Xi[perm[i:i + bs]].to(dev); bt = Xt[perm[i:i + bs]].to(dev)
                feat = _feat(reg.backbone, b)
                pred = reg.reg(feat)                          # [B,1] → 广播到 C 通道
                loss = F.huber_loss(pred.expand(-1, bt.shape[1]), bt)
                opt.zero_grad(); loss.backward(); opt.step()
        def score_fn(X, _reg=reg, _L=L, _dev=dev):
            _reg.eval()
            Wq, tidx = make_windows(X, _L, 1)
            Xnext = X[tidx]                                   # [N,C]
            outs = []
            with torch.no_grad():
                for i in range(0, len(Wq), 512):
                    b = torch.from_numpy(Wq[i:i + 512]).to(_dev)
                    feat = _feat(_reg.backbone, b)
                    outs.append(_reg.reg(feat).expand(-1, X.shape[1]).cpu().numpy())
            pred = np.concatenate(outs)
            s = np.full(len(X), np.nan, dtype=np.float32)
            with np.errstate(invalid="ignore"):
                s[tidx] = np.nanmax(np.abs(Xnext - pred), axis=1)
            return s
        return score_fn, {"params": sum(p.numel() for p in reg.parameters())}

    if case == "LINEAR":
        # 线性预测 + 绝对残差（预注册：非纯线性标量头）；同样丢弃 NaN 窗口
        W, t_idx = make_windows(X_train, L, stride)
        Xin, Xtg = W[:, :, :-1], X_train[t_idx].astype(np.float32)
        keep = (~np.isnan(Xin).any(axis=(1, 2))) & (~np.isnan(Xtg).any(axis=1))
        Xin, Xtg = Xin[keep], Xtg[keep]
        Xi = torch.from_numpy(Xin); Xt = torch.from_numpy(Xtg)
        lin = nn.Linear(C * (L - 1), 1).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=lr)
        n = len(Xi)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = Xi[perm[i:i + bs]].to(dev); bt = Xt[perm[i:i + bs]].to(dev)
                pred = lin(b.flatten(1))
                loss = F.huber_loss(pred.expand(-1, bt.shape[1]), bt)
                opt.zero_grad(); loss.backward(); opt.step()
        def score_fn(X, _lin=lin, _L=L, _dev=dev):
            _lin.eval()
            Wq, tidx = make_windows(X, _L, 1)
            Wq = Wq[:, :, :-1]                            # 与训练一致：去掉末点，预测末点
            Xnext = X[tidx]
            outs = []
            with torch.no_grad():
                for i in range(0, len(Wq), 1024):
                    b = torch.from_numpy(Wq[i:i + 1024]).to(_dev)
                    outs.append(_lin(b.flatten(1)).expand(-1, X.shape[1]).cpu().numpy())
            pred = np.concatenate(outs)
            s = np.full(len(X), np.nan, dtype=np.float32)
            with np.errstate(invalid="ignore"):
                s[tidx] = np.nanmax(np.abs(Xnext - pred), axis=1)
            return s
        return score_fn, {"params": C * (L - 1) + 1}

    raise ValueError(case)


def train_AB(model_name, X_train, s_tr, xhat_tr, L=128, lam=0.1, epochs=30, bs=256,
             lr=1e-3, seed=0, device="cpu", stride=2):
    """B消融臂（5.3裁定3备案实现）：A分数蒸馏 + λ·预测对齐（学生对齐教师缓存中位数预测）。
    共享骨干+双头；λ=0.1 固定（R3记录：管道API不暴露隐藏态，特征对齐降级为预测对齐）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    C = X_train.shape[1]; dev = torch.device(device)
    net = {"TCNLite": TCNLite, "PatchMLPLite": PatchMLPLite}[model_name](C, L).to(dev)
    reg = nn.Linear(net.feat_dim, 1).to(dev)
    W, t_idx = make_windows(X_train, L, stride)
    s_t = s_tr[t_idx]
    keep = ~np.isnan(s_t) & ~np.isnan(xhat_tr[t_idx]).any(axis=1)
    Wt = torch.from_numpy(W[keep]); st = torch.from_numpy(s_t[keep].astype(np.float32))
    xt = torch.from_numpy(xhat_tr[t_idx][keep].astype(np.float32))
    opt = torch.optim.Adam(list(net.parameters()) + list(reg.parameters()), lr=lr)
    n = len(Wt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = Wt[perm[i:i + bs]].to(dev); bt = st[perm[i:i + bs]].to(dev)
            bx = xt[perm[i:i + bs]].to(dev)
            z = net(b)
            pred = reg(_feat(net, b))
            loss = F.huber_loss(z, bt) + lam * F.huber_loss(pred.expand(-1, bx.shape[1]), bx)
            opt.zero_grad(); loss.backward(); opt.step()
    def score_fn(X, _net=net, _L=L, _dev=dev):
        _net.eval()
        Wq, tidx = make_windows(X, _L, 1)
        outs = []
        with torch.no_grad():
            for i in range(0, len(Wq), 512):
                outs.append(_net(torch.from_numpy(Wq[i:i + 512]).to(_dev)).cpu().numpy())
        s = np.full(len(X), np.nan, dtype=np.float32)
        s[tidx] = np.concatenate(outs)
        return s
    return score_fn, {"params": sum(p.numel() for p in net.parameters()) + reg.parameters().__next__().numel()}


def _feat(backbone, x):
    """骨干池化特征（与各 head 的池化方式严格一致，供回归头）。"""
    import torch.nn.functional as F
    if hasattr(backbone, "body"):                     # TCN：与 head 同样 pool4+flatten
        h = backbone.body(x)
        return F.adaptive_avg_pool1d(h, 4).flatten(1)
    B = x.shape[0]                                    # PatchMLP：embed+mixer → flatten
    h = backbone.embed(x.reshape(B, backbone.n, -1))
    h = h + backbone.tmix(backbone.norm1(h).transpose(1, 2)).transpose(1, 2)
    h = h + backbone.cmix(backbone.norm2(h))
    return h.flatten(1)
