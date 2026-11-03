import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam(X, iters, a0, a1, lr0, lr1, n0, n1, rng, fix=None, balance=0.0):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a0 != a1 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        if n0 <= 0:
            nm = 0.0
        else:
            if n1 <= 0:
                nm = n0 * (1.0 - f)
            else:
                nm = n0 * ((n1 / n0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = H - WG[:, None] * X
        if balance > 0:
            nn = G.max(axis=1)
            med = float(np.median(nn))
            scale = 1.0 + balance * np.tanh((nn - med) / 0.005)
            Grad = Grad * scale[:, None]
        m = b1 * m + (1.0 - b1) * Grad
        v = b2 * v + (1.0 - b2) * (Grad * Grad)
        bc1 = 1.0 - b1 ** (t + 1)
        bc2 = 1.0 - b2 ** (t + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        rad = (step * X).sum(axis=1, keepdims=True)
        step = step - rad * X
        if fix is not None:
            step[fix] = 0.0
        X = X - step
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            if fix is not None:
                N[fix] = 0.0
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-cfull[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            p = len(I)
        cij = (X @ X.T)[I, J]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cij, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best - 1e-15:
            X, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _best_hole(X_rem, rng, n_cand=12000):
    d = X_rem.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_rem.T
    mm = dots.max(axis=1)
    top = np.argpartition(mm, 4)[:4]
    best_v = C[top[0]]
    best_s = 1e9
    for k in top:
        vv = C[k].copy()
        for it in range(220):
            dts = X_rem @ vv
            cur = float(dts.max())
            act = dts > cur - 0.02
            if int(act.sum()) == 0:
                break
            g = X_rem[act].sum(axis=0)
            gt = g - float(g @ vv) * vv
            nn = float(np.linalg.norm(gt))
            if nn < 1e-12:
                break
            gt /= nn
            step = 0.08 * (0.15 ** (it / 220.0))
            vv = vv - step * gt
            vv = vv / max(1e-12, float(np.linalg.norm(vv)))
        s = float((X_rem @ vv).max())
        if s < best_s:
            best_s = s
            best_v = vv
    return best_v

def _gen_e8():
    pts = []
    sq = np.sqrt(2.0)
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si / sq
                    v[j] = sj / sq
                    pts.append(v)
    for mask in range(256):
        v = np.empty(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                v[k] = 0.5 / (sq/2.0) / 2.0 if False else 0.5
                # will divide by sq below
                nm += 1
            else:
                v[k] = -0.5
        # count minus: entries -0.5 correspond to bit 0
        nminus = 8 - bin(mask).count("1")
        # Actually above: bit1 => +0.5, bit0 => -0.5, so nminus = 8-popcount
        if nminus % 2 == 0:
            pts.append(v / sq)
    return np.array(pts)  # 240x8

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(20261103)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (162, 8):
                vv = _norm(vv)
                s = _max_cos(vv)
                if s < sb:
                    sb = s
                    Xb = vv.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy()
    s_best = float(sb)
    n, d = X_best.shape
    try:
        E8 = _gen_e8()
        E8 = _norm(E8)
    except Exception:
        E8 = None
    cands = []
    trial = 0
    # diverse + basin loop
    while time.time() < t0 + 108 and trial < 400:
        if time.time() > deadline - 50:
            break
        trial += 1
        Xt = None
        if E8 is not None and trial <= 4:
            idx = rng.choice(E8.shape[0], n, replace=False)
            Xt = E8[idx].copy()
            Xt = _norm(Xt + rng.randn(n, d) * (0.03 + 0.02 * trial))
            # two-stage: pinned? no, full anneal with balance
            Xt = _adam(Xt, 900, 25, 450, 0.005, 0.0012, 0.003, 0.0, rng, balance=0.8)
            Xt = _adam(Xt, 450, 150, 1400, 0.0015, 0.0004, 0.0, 0.0, rng, balance=0.4)
        elif trial <= 8:
            Xt = _norm(rng.randn(n, d))
            Xt = _adam(Xt, 1000, 20, 400, 0.005, 0.0012, 0.003, 0.0, rng, balance=0.8)
            Xt = _adam(Xt, 450, 150, 1400, 0.0015, 0.0004, 0.0, 0.0, rng, balance=0.4)
        else:
            mode = (trial % 6)
            Xt = X_best.copy()
            if mode == 0:
                Xt = _norm(Xt + rng.randn(n, d) * 0.010)
                Xt = _adam(Xt, 750, 25, 550, 0.004, 0.001, 0.0018, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
            elif mode == 1:
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                nn = G.max(axis=1)
                gm = float(nn.max())
                cnt = (G > gm - 2e-4).sum(axis=1)
                # victims: most overloaded + most isolated (rattler-like)
                order_high = np.argsort(-cnt)
                order_low = np.argsort(nn)
                k = 3 + (trial % 3)  # 3..5
                victims = order_high[:k].tolist()
                # add a loose point
                for q in order_low:
                    if q not in victims:
                        victims.append(int(q))
                        break
                victims = victims[:k+1]
                fix = np.ones(n, dtype=bool)
                fix[victims] = False
                for vv in victims:
                    mask = np.ones(n, dtype=bool); mask[vv] = False
                    # hole w.r.t. current Xt without vv (use X_best structure for rest)
                    hole = _best_hole(Xt[mask], rng, n_cand=12000)
                    Xt[vv] = hole
                Xt = _norm(Xt)
                # stage1: move victims only
                fix2 = np.ones(n, dtype=bool); fix2[victims] = False
                # fix=True means frozen, so freeze complement
                frozen = np.ones(n, dtype=bool); frozen[victims] = False
                Xt = _adam(Xt, 400, 30, 400, 0.004, 0.001, 0.001, 0.0, rng, fix=frozen, balance=0.5)
                Xt = _adam(Xt, 800, 25, 600, 0.004, 0.0008, 0.0012, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
            elif mode == 2:
                # cap re-optimisation: random direction, take ~18 nearest points
                u = rng.randn(d); u /= np.linalg.norm(u)
                proj = Xt @ u
                sel = np.argsort(-proj)[:18]
                frozen = np.ones(n, dtype=bool); frozen[sel] = False
                # randomise cap partially toward holes
                for vv in sel:
                    if rng.rand() < 0.5:
                        mask = np.ones(n, dtype=bool); mask[vv] = False
                        Xt[vv] = _best_hole(Xt[mask], rng, n_cand=6000)
                    else:
                        Xt[vv] = Xt[vv] + rng.randn(d) * 0.18
                Xt = _norm(Xt)
                Xt = _adam(Xt, 400, 30, 400, 0.004, 0.001, 0.001, 0.0, rng, fix=frozen, balance=0.5)
                Xt = _adam(Xt, 750, 25, 550, 0.004, 0.0008, 0.0012, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
            elif mode == 3:
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(n, 1)
                cf = G[iu]
                k = 8
                top = np.argpartition(-cf, k)[:k]
                for q in top:
                    i = int(iu[0][q]); j = int(iu[1][q])
                    di = Xt[i] - Xt[j]
                    amt = 0.03 + 0.02 * rng.rand()
                    Xt[i] = Xt[i] + di * amt
                    Xt[j] = Xt[j] - di * amt
                Xt = _norm(Xt)
                Xt = _adam(Xt, 750, 30, 550, 0.004, 0.001, 0.0018, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
            elif mode == 4:
                k = 14
                idx = rng.choice(n, k, replace=False)
                # replace half by holes, half by noise
                for vv in idx[:k//2]:
                    mask = np.ones(n, dtype=bool); mask[vv] = False
                    Xt[vv] = _best_hole(Xt[mask], rng, n_cand=6000)
                Xt[idx[k//2:]] = _norm(Xt[idx[k//2:]] + rng.randn(k - k//2, d) * 0.22)
                frozen = np.ones(n, dtype=bool); frozen[idx] = False
                Xt = _norm(Xt)
                Xt = _adam(Xt, 450, 25, 400, 0.0045, 0.001, 0.0015, 0.0, rng, fix=frozen, balance=0.5)
                Xt = _adam(Xt, 800, 25, 550, 0.004, 0.0008, 0.0012, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
            else:
                Xt = _norm(Xt + rng.randn(n, d) * 0.028)
                Xt = _adam(Xt, 800, 20, 480, 0.0045, 0.001, 0.0025, 0.0, rng, balance=0.7)
                Xt = _adam(Xt, 350, 150, 1300, 0.0014, 0.0004, 0.0, 0.0, rng, balance=0.3)
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best:
            budget = min(6.0, max(2.0, (deadline - time.time()) / 9.0))
            if time.time() + budget < deadline - 6:
                Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 6)
                s2 = _max_cos(Xt2)
                if s2 < s_best:
                    X_best, s_best = Xt2, s2
                    cands.append((s2, Xt2.copy()))
                elif s_try < s_best:
                    X_best, s_best = Xt, s_try
                    cands.append((s_try, Xt.copy()))
                else:
                    cands.append((s2, Xt2.copy()))
            else:
                X_best, s_best = Xt, s_try
        else:
            if s_try < s_best + 0.0025:
                cands.append((s_try, Xt.copy()))
    cands = sorted(cands, key=lambda z: z[0])[:6]
    for _, Xc in cands:
        if time.time() > deadline - 13:
            break
        remain = deadline - time.time() - 4
        if remain < 3:
            break
        budget = min(9.0, remain / max(1, len(cands)))
        budget = max(budget, 2.5)
        Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        budget = min(13.0, deadline - time.time() - 2)
        if budget > 2:
            Xp = _lp_polish(X_best, budget, t_deadline=deadline - 1)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
