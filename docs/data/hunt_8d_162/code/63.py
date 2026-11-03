import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_full(X, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
    n, d = X.shape
    V = np.zeros_like(X)
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            nm = noise0 * ((noise1 / noise0) ** f) if noise1 > 0 else noise0 * (1.0 - f)
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
        Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            N = N - (N * X).sum(axis=1, keepdims=True) * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
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
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-cfull[sel], 3000)[:3000]
            I = I[idx]; J = J[idx]
            csub = cfull[sel][idx]
            p = len(I)
        else:
            csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub,
                          A_eq=A_eq, b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _e8_roots():
    v = []
    s2 = np.sqrt(2.0)
    for i in range(8):
        for j in range(i+1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    x = np.zeros(8)
                    x[i] = si / s2
                    x[j] = sj / s2
                    v.append(x)
    # 128 half vectors even minus
    for mask in range(256):
        bits = [(1.0 if ((mask >> k) & 1 else -1.0) for k in range(8)]
        # count minus: entries -1
        nminus = sum(1 for b in bits if b < 0)
        if nminus % 2 == 1:
            continue
        x = np.array(bits, dtype=float) / (2.0 * s2) * 2.0  # bits*0.5 /sqrt2? = bits/(2 sqrt2)
        # bits/(2*sqrt2)
        x = np.array(bits, dtype=float) / (2.0 * s2)
        # wait: bits are +-1, want +-1/(2 sqrt2)? that is bits/(2 s2). done
        v.append(x)
    E = np.array(v)
    E = _norm(E)
    return E  # 240x8

def _greedy_subset(E, k, rng, n_try=3000):
    N = E.shape[0]
    GE = E @ E.T
    best_idx = None
    best_key = (1e9, 1e9)
    # multiple restarts inside
    for _ in range(2):
        idx = rng.choice(N, k, replace=False)
        cur_set = set(idx.tolist())
        # current max
        sub = GE[np.ix_(idx, idx)]
        np.fill_diagonal(sub, -2)
        cur_max = float(sub.max())
        cur_cnt = int((sub > cur_max - 1e-12).sum() // 2)
        for t in range(n_try):
            out_pool = rng.choice(N, 1)[0]
            if out_pool in cur_set:
                continue
            in_pick = rng.choice(k, 1)[0]
            in_node = idx[in_pick]
            # evaluate swap: new idx
            new_idx = idx.copy()
            new_idx[in_pick] = out_pool
            sub2 = GE[np.ix_(new_idx, new_idx)]
            np.fill_diagonal(sub2, -2)
            m2 = float(sub2.max())
            c2 = int((sub2 > m2 - 1e-12).sum() // 2)
            # lexicographic, allow sideways with small prob
            if (m2 < cur_max - 1e-12) or (abs(m2-cur_max) <= 1e-12 and c2 < cur_cnt):
                idx = new_idx
                cur_set.remove(int(in_node)); cur_set.add(int(out_pool))
                cur_max, cur_cnt = m2, c2
            elif abs(m2-cur_max) <= 1e-12 and c2 == cur_cnt and rng.rand() < 0.01:
                idx = new_idx
                cur_set.remove(int(in_node)); cur_set.add(int(out_pool))
        key = (cur_max, cur_cnt)
        if key < best_key:
            best_key = key
            best_idx = idx.copy()
    return best_idx, best_key

def _expand_antipodal(P):
    return np.vstack([P, -P])

def _sym_smooth_anti(P, iters, a0, a1, lr0, lr1, noise0, noise1, rng):
    m, d = P.shape
    V = np.zeros_like(P)
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        nm = 0.0 if noise0 <= 0 else (noise0 * ((noise1/noise0)**f) if noise1>0 else noise0*(1-f))
        Pn = _norm(P)
        X = _expand_antipodal(Pn)
        n = X.shape[0]
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
        GradX = (H - WG[:, None] * X) / S
        Grad_top = GradX[:m]
        Grad_bot = GradX[m:]
        GradP = Grad_top - Grad_bot
        # project to tangent of Pn (approx)
        GradP = GradP - (GradP * Pn).sum(axis=1, keepdims=True) * Pn
        V = mu * V + GradP
        P = Pn - lr * V
        if nm > 0:
            N = rng.randn(m, d)
            Pn2 = _norm(P)
            N = N - (N * Pn2).sum(axis=1, keepdims=True) * Pn2
            P = P + nm * N
        P = _norm(P)
    return _norm(P)

def _expand_cyclic(R, free=None):
    # R: m x 8, orbit size 8 each by roll
    blocks = []
    for i in range(R.shape[0]):
        r = R[i]
        for k in range(8):
            blocks.append(np.roll(r, k))
    X = np.array(blocks)
    if free is not None and len(free) > 0:
        X = np.vstack([X, free])
    return X

def _sym_smooth_cyclic(R, free, iters, a0, a1, lr0, lr1, rng):
    m, d = R.shape
    nf = 0 if free is None else free.shape[0]
    VR = np.zeros_like(R)
    Vf = np.zeros_like(free) if nf>0 else None
    mu = 0.8
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1/a0)**f) if a1!=a0 else a0
        lr = lr0 * ((lr1/lr0)**f) if lr1!=lr0 else lr0
        Rn = _norm(R)
        fn = _norm(free) if nf>0 else None
        X = _expand_cyclic(Rn, fn)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(a*(G-Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S==0:
            break
        H = W @ X
        WG = (W*G).sum(axis=1)
        GradX = (H - WG[:,None]*X)/S
        # split: first m*8 rows cyclic, rest free
        Gc = GradX[:m*8].reshape(m,8,8)  # [rep, shift, dim]
        GradR = np.zeros_like(Rn)
        for k in range(8):
            # X block = roll(R,k), so GradR += roll(GradBlock,-k)
            Gb = Gc[:,k,:]  # m x 8
            for i in range(m):
                GradR[i] += np.roll(Gb[i], -k)
        GradR = GradR - (GradR*Rn).sum(axis=1,keepdims=True)*Rn
        VR = mu*VR + GradR
        R = Rn - lr*VR
        if nf>0:
            Gf = GradX[m*8:]
            Gf = Gf - (Gf*fn).sum(axis=1,keepdims=True)*fn
            Vf = mu*Vf + Gf
            free = fn - lr*Vf
        R = _norm(R)
        if nf>0:
            free = _norm(free)
    R = _norm(R)
    if nf>0:
        free = _norm(free)
    return R, free

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    # load incumbent
    X_best = None; s_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < s_best:
                    s_best = s; X_best = v.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm(rng.randn(162,8)); s_best = _max_cos(X_best)
    X_inc = X_best.copy(); s_inc = float(s_best)
    cands = []

    # --- E8 structured selection ---
    try:
        E = _e8_roots()
    except Exception:
        E = None
    if E is not None and time.time() < deadline - 60:
        # a few greedy subsets
        for rep in range(3):
            if time.time() > deadline - 55:
                break
            idx, key = _greedy_subset(E, 162, rng, n_try=2500)
            X0 = E[idx].copy()
            # jitter slightly to break exact ties
            X0 = _norm(X0 + rng.randn(*X0.shape)*1e-3)
            X0 = _smooth_full(X0, 450, 30, 350, 0.014, 0.0012, 0.002, 0.0, rng)
            X0 = _smooth_full(X0, 300, 80, 800, 0.006, 0.0008, 0.0, 0.0, rng)
            s0 = _max_cos(X0)
            cands.append((s0, X0.copy()))
            if s0 < s_best:
                # quick LP
                b = min(5.0, max(2.0, (deadline-time.time())/10.0))
                if time.time()+b < deadline-10:
                    Xp = _lp_polish(X0, b, t_deadline=deadline-5)
                    sp = _max_cos(Xp)
                    if sp < s_best:
                        X_best, s_best = Xp, sp
                        cands.append((sp, Xp.copy()))
                    elif s0 < s_best:
                        X_best, s_best = X0, s0
                else:
                    if s0 < s_best:
                        X_best, s_best = X0, s0
    # --- antipodal orbit optimization ---
    for rep in range(3):
        if time.time() > deadline - 45:
            break
        P = _norm(rng.randn(81, 8))
        P = _sym_smooth_anti(P, 500, 25, 350, 0.014, 0.0012, 0.002, 0.0, rng)
        P = _sym_smooth_anti(P, 300, 80, 800, 0.006, 0.0008, 0.0, 0.0, rng)
        X0 = _expand_antipodal(_norm(P))
        s0 = _max_cos(X0)
        # symmetry release: increasing independent displacement
        for lvl, sig in enumerate([0.002, 0.006, 0.012]):
            if time.time() > deadline - 40:
                break
            Xr = _norm(X0 + rng.randn(*X0.shape)*sig)
            Xr = _smooth_full(Xr, 350, 40, 500, 0.010, 0.001, 0.001, 0.0, rng)
            sr = _max_cos(Xr)
            cands.append((sr, Xr.copy()))
            if sr < s0:
                s0 = sr; X0 = Xr.copy()
        cands.append((s0, X0.copy()))
        if s0 < s_best + 0.004 and time.time() < deadline - 25:
            # symmetry-preserving LP would be complex; do unrestricted short polish if promising
            if s0 < s_best + 0.001:
                b = 3.0
                if time.time()+b < deadline-8:
                    Xp = _lp_polish(X0, b, t_deadline=deadline-5)
                    sp = _max_cos(Xp)
                    cands.append((sp, Xp.copy()))
                    if sp < s_best:
                        X_best, s_best = Xp, sp
            if s0 < s_best:
                X_best, s_best = X0.copy(), s0
    # --- cyclic orbit optimization (20x8+2) ---
    for rep in range(2):
        if time.time() > deadline - 35:
            break
        R = _norm(rng.randn(20, 8))
        F = _norm(rng.randn(2, 8))
        R, F = _sym_smooth_cyclic(R, F, 450, 25, 300, 0.014, 0.0012, rng)
        X0 = _expand_cyclic(_norm(R), _norm(F))
        # release
        Xr = _norm(X0 + rng.randn(*X0.shape)*0.004)
        Xr = _smooth_full(Xr, 350, 40, 600, 0.009, 0.001, 0.001, 0.0, rng)
        sr = _max_cos(Xr)
        cands.append((sr, Xr.copy()))
        if sr < s_best:
            X_best, s_best = Xr.copy(), sr
    # --- polish most promising ---
    cands = sorted(cands, key=lambda z: z[0])[:5]
    for _, Xc in cands:
        if time.time() > deadline - 12:
            break
        remain = deadline - time.time() - 3
        if remain < 2.5:
            break
        b = min(8.0, max(2.5, remain/2.0))
        Xp = _lp_polish(np.array(Xc, copy=True), b, t_deadline=deadline-1)
        sp = _max_cos(Xp)
        if sp < s_best:
            X_best, s_best = Xp, sp
    if time.time() < deadline - 4:
        b = min(10.0, deadline - time.time() - 1.5)
        if b > 2:
            Xp = _lp_polish(X_best, b, t_deadline=deadline-0.5)
            sp = _max_cos(Xp)
            if sp < s_best:
                X_best, s_best = Xp, sp
    # fallback to incumbent
    if not np.isfinite(_max_cos(X_best)) or _max_cos(X_best) > s_inc + 1e-12:
        # if structured failed, return incumbent (still satisfies contract + direction attempted)
        X_best = X_inc
    X_best = _norm(X_best)
    return X_best
