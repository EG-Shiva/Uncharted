import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    rm = g.max(axis=1)
    iu = np.triu_indices(x.shape[0], 1)
    vals = (x @ x.T)[iu]
    k = int(np.argmax(vals))
    return rm, float(vals.max()), int(iu[0][k]), int(iu[1][k]), vals

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.0e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csel = cuv[sel]
        p = len(I)
        if p > 1500:
            order = np.argsort(-csel)[:1500]
            I = I[order]; J = J[order]; csel = csel[order]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _quench_movable(X_init, mov, steps, T0, T1, lr0, lr1, alpha=0.6):
    X = _norm_rows(X_init.copy())
    n, d = X.shape
    mov = np.asarray(mov, dtype=int)
    freeze_mask = np.ones(n, dtype=bool)
    freeze_mask[mov] = False
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad_g = (E @ X) / S
        rm = G.max(axis=1)
        Q = (G - rm[:, None]) / T
        np.clip(Q, -50, 0, out=Q)
        Er = np.exp(Q)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = Er / rs
        Ws = 0.5 * (W + W.T)
        Grad_r = Ws @ X
        Grad = alpha * Grad_g + (1.0 - alpha) * Grad_r
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gt[freeze_mask] = 0.0
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        maxn = 10.0 * np.sqrt(len(mov) * d)
        if sn > maxn and sn > 0:
            step *= (maxn / sn)
        X[mov] = X[mov] - lr * step[mov]
        X[mov] = _norm_rows(X[mov])
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X, best_m

def _hill_tangent(X_init, mov, iters, sigma, rng):
    X = X_init.copy()
    mov = np.asarray(mov, dtype=int)
    best = _max_cos(X)
    best_X = X.copy()
    for _ in range(iters):
        D = rng.standard_normal(size=(len(mov), 12)) * sigma
        Xm = X[mov]
        D = D - np.sum(D * Xm, axis=1, keepdims=True) * Xm
        Y = X.copy()
        Ym = Xm + D
        Ym = Ym / (np.linalg.norm(Ym, axis=1, keepdims=True) + 1e-12)
        Y[mov] = Ym
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        v = float(G.max())
        if v < best:
            best = v
            best_X = Y.copy()
            X = Y
    return best_X, best

def _refine_single(s0, R, steps=200, T=0.004, lr=0.12, mu=0.85):
    s = s0 / (np.linalg.norm(s0) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=5000):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    k = min(3, n_sample)
    idx = np.argpartition(scores, k)[:k]
    best_s = None; best_v = 1e100
    for kk in idx:
        s1 = _refine_single(Z[kk], R, steps=180, T=0.005, lr=0.13)
        s1 = _refine_single(s1, R, steps=180, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _pair_split(X, a, b, delta):
    Y = X.copy()
    xa = X[a]; xb = X[b]
    # tangent directions to separate
    ta = (xa - xb) - float((xa - xb) @ xa) * xa
    tb = (xb - xa) - float((xb - xa) @ xb) * xb
    na = float(np.linalg.norm(ta)); nb = float(np.linalg.norm(tb))
    if na < 1e-12 or nb < 1e-12:
        return Y
    ta /= na; tb /= nb
    Y[a] = _norm_rows((xa + delta * ta)[None, :])[0]
    Y[b] = _norm_rows((xb + delta * tb)[None, :])[0]
    return Y

def _fingerprint(X):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    v = (X @ X.T)[iu]
    top = np.sort(v)[-25:]
    return tuple(np.round(top * 20000).astype(int).tolist())

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 26.0
    rng = np.random.default_rng(12345)
    n, d = 71, 12
    best_X = None
    best_score = 1e100
    for f in ("best.npy", "record.npy"):
        try:
            Xc = _norm_rows(np.asarray(np.load(f), dtype=float))
            if Xc.shape == (71, 12) and np.all(np.isfinite(Xc)):
                s = _max_cos(Xc)
                if s < best_score:
                    best_score = s; best_X = Xc.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(best_X)
    best_X = _norm_rows(best_X)
    best_score = _max_cos(best_X)
    cur_X = best_X.copy()
    cur_score = best_score
    # archive / tabu
    seen_fp = set()
    try:
        seen_fp.add(_fingerprint(best_X))
    except Exception:
        pass
    tabu_sets = []
    def is_tabu(mov):
        fs = frozenset(int(v) for v in mov)
        return fs in tabu_sets
    def push_tabu(mov):
        tabu_sets.append(frozenset(int(v) for v in mov))
        if len(tabu_sets) > 120:
            tabu_sets.pop(0)
    push_tabu(list(range(8)))
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        try:
            rm, gm, ia, ib, _ = _row_max(cur_X)
        except Exception:
            rm = np.zeros(n); ia, ib = 0, 1; gm = cur_score
        # choose movable size 8..20 (freeze 51..63)
        k = int(rng.integers(8, 21))
        order_hi = np.argsort(-rm + rng.random(n) * 1e-4)
        # build movable: max pair + (k-2) with row-max closest to global max
        mov = [int(ia), int(ib)]
        for idx in order_hi:
            if len(mov) >= k:
                break
            ii = int(idx)
            if ii not in mov:
                mov.append(ii)
        # occasionally inject a low-load point for diversity
        if rng.random() < 0.25:
            order_lo = np.argsort(rm + rng.random(n) * 1e-4)
            cand = int(order_lo[0])
            if cand not in mov:
                mov[-1] = cand
        mov = np.array(mov, dtype=int)
        # tabu: mutate if recently visited
        tries = 0
        while is_tabu(mov) and tries < 6:
            # swap one element
            pos = int(rng.integers(0, len(mov)))
            # pick replacement not in mov
            pool = [i for i in range(n) if i not in set(mov.tolist())]
            mov[pos] = int(rng.choice(pool))
            tries += 1
        # choose surgical move
        u = rng.random()
        X0 = cur_X.copy()
        if u < 0.45:
            # plain coordinated perturbation of movable
            sig = float(10 ** rng.uniform(-2.6, -1.3))
            D = rng.standard_normal((len(mov), d)) * sig
            Xm = X0[mov]
            D = D - np.sum(D * Xm, axis=1, keepdims=True) * Xm
            X0[mov] = _norm_rows(Xm + D)
        elif u < 0.65:
            # relocate a chosen rattler to farthest hole of config with itself removed
            # pick rattler in mov with largest row-max (or random if tie)
            sub = sorted(mov.tolist(), key=lambda i: -rm[i])
            r = int(sub[0] if len(sub) > 0 else int(rng.integers(0, n)))
            keep = [i for i in range(n) if i != r]
            R = X0[keep]
            try:
                h = _find_hole(R, rng, n_sample=5000)
            except Exception:
                h = _norm_rows(rng.standard_normal((1, d)))[0]
            X0[r] = h
            # small tangent noise on rest of movable
            rest = [v for v in mov.tolist() if v != r]
            if len(rest) > 0:
                sig = float(10 ** rng.uniform(-2.8, -1.8))
                D = rng.standard_normal((len(rest), d)) * sig
                Xr = X0[np.array(rest)]
                D = D - np.sum(D * Xr, axis=1, keepdims=True) * Xr
                X0[np.array(rest)] = _norm_rows(Xr + D)
        elif u < 0.82:
            # move close pair apart symmetrically along tangent bisector
            delta = float(rng.uniform(0.015, 0.07))
            X0 = _pair_split(X0, int(ia), int(ib), delta)
            # add noise to other movable
            rest = [v for v in mov.tolist() if v not in (int(ia), int(ib))]
            if len(rest) > 0:
                sig = float(10 ** rng.uniform(-2.8, -1.6))
                D = rng.standard_normal((len(rest), d)) * sig
                Xr = X0[np.array(rest)]
                D = D - np.sum(D * Xr, axis=1, keepdims=True) * Xr
                X0[np.array(rest)] = _norm_rows(Xr + D)
        else:
            # swap high-load and low-load point
            hi = int(np.argmax(rm + rng.random(n) * 1e-6))
            lo = int(np.argmin(rm + rng.random(n) * 1e-6))
            if hi != lo:
                tmp = X0[hi].copy()
                X0[hi] = X0[lo].copy()
                X0[lo] = tmp
            sig = float(10 ** rng.uniform(-2.8, -1.6))
            D = rng.standard_normal((len(mov), d)) * sig
            Xm = X0[mov]
            D = D - np.sum(D * Xm, axis=1, keepdims=True) * Xm
            X0[mov] = _norm_rows(Xm + D)
        # multi-start Adam on movable subset with true-max tracking
        nstart = 2
        if tleft < 8:
            nstart = 1
        cands = []
        for si in range(nstart):
            T0 = float(10 ** rng.uniform(-2.6, -1.8))
            T1 = float(10 ** rng.uniform(-4.3, -3.3))
            lr0 = float(10 ** rng.uniform(-2.9, -2.2))
            lr1 = lr0 * float(10 ** rng.uniform(-1.2, -0.5))
            steps = 1100
            if tleft > 30 and rng.random() < 0.25:
                steps = 1700
            if tleft < 8:
                steps = 700
            Xstart = X0.copy()
            if si > 0:
                sig2 = float(10 ** rng.uniform(-2.5, -1.5))
                D2 = rng.standard_normal((len(mov), d)) * sig2
                Xm2 = Xstart[mov]
                D2 = D2 - np.sum(D2 * Xm2, axis=1, keepdims=True) * Xm2
                Xstart[mov] = _norm_rows(Xm2 + D2)
            try:
                Xq, _ = _quench_movable(Xstart, mov, steps, T0, T1, lr0, lr1, alpha=0.6)
            except Exception:
                continue
            # CMA-style random tangent proposals with true max
            try:
                sig_h = float(10 ** rng.uniform(-2.7, -1.9))
                nit = 250 if tleft > 10 else 120
                Xh, _ = _hill_tangent(Xq, mov, nit, sig_h, rng)
            except Exception:
                Xh = Xq
            cands.append(Xh)
        if len(cands) == 0:
            continue
        # pick best candidate by exact score
        scored = []
        for c in cands:
            try:
                scored.append((float(_max_cos(c)), c))
            except Exception:
                pass
        if len(scored) == 0:
            continue
        scored.sort(key=lambda t: t[0])
        Xc = scored[0][1]
        # short wide-band LP acceptance
        lp_secs = 1.0
        if tleft < 3:
            lp_secs = 0.5
        try:
            Xp = _lp_polish(Xc, seconds=lp_secs, deadline=HOP_END)
            sp = float(_max_cos(Xp))
        except Exception:
            sp = float(_max_cos(Xc)); Xp = Xc
        # fingerprint tabu: force novelty
        try:
            fp = _fingerprint(Xp)
        except Exception:
            fp = None
        is_new_fp = (fp not in seen_fp) if fp is not None else True
        push_tabu(mov)
        if fp is not None:
            seen_fp.add(fp)
            if len(seen_fp) > 5000:
                seen_fp.clear()
                seen_fp.add(fp)
        # acceptance: global best always; wandering current accepts small regress or novel topology
        if np.isfinite(sp) and sp < best_score - 1e-12:
            best_X = _norm_rows(Xp.copy()); best_score = sp
            cur_X = best_X.copy(); cur_score = best_score
        elif np.isfinite(sp) and (sp < cur_score + 0.0008) and (is_new_fp or sp < cur_score):
            cur_X = _norm_rows(Xp.copy()); cur_score = sp
        else:
            # occasional random restart of current from best with shake to avoid stagnation
            if it % 37 == 0 and rng.random() < 0.4:
                sig = float(10 ** rng.uniform(-2.2, -1.2))
                cur_X = _norm_rows(best_X + rng.standard_normal((n, d)) * sig)
                cur_score = float(_max_cos(cur_X))
    # final intensive squeeze of best
    try:
        trem = HARD_END - time.time() - 0.6
        if trem > 2.0:
            qs = 4500 if trem > 16 else 2200
            try:
                Xf, _ = _quench_movable(best_X, np.arange(n), qs, 0.0006, 0.00008, 0.0025, 0.00025, alpha=0.5)
                sf = float(_max_cos(Xf))
                if sf < best_score:
                    best_X, best_score = _norm_rows(Xf), sf
            except Exception:
                pass
            trem2 = HARD_END - time.time() - 0.5
            if trem2 > 1.0:
                try:
                    Xp2 = _lp_polish(best_X, seconds=min(trem2, 30.0), deadline=HARD_END - 0.4)
                    sp2 = float(_max_cos(Xp2))
                    if sp2 < best_score:
                        best_X, best_score = _norm_rows(Xp2), sp2
                except Exception:
                    pass
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    for i in range(n):
        if not np.all(np.isfinite(best_X[i])) or float(np.linalg.norm(best_X[i])) < 1e-12:
            v = rng.standard_normal(d)
            v = v / (np.linalg.norm(v) + 1e-12)
            best_X[i] = v
    best_X = _norm_rows(best_X)
    return best_X
