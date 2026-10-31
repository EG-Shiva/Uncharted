import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _smooth(X, steps, T0, T1, lr0, lr1, mu=0.9, ms0=0.04, ms1=0.004):
    X = X.copy()
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / float(max(1, steps))
        T = T0 * (T1 / T0) ** frac
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (lr1 / lr0) ** frac
        ms = ms0 * (ms1 / ms0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp((G - gmax) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        P = W / S
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        raw = lr * V
        rn = np.sqrt(np.sum(raw * raw, axis=1, keepdims=True)) + 1e-12
        factor = np.minimum(1.0, ms / rn)
        X = X - raw * factor
        X = _norm(X)
    return X

def _lp(x, deadline, radius0=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            from scipy.sparse import coo_matrix as _coo
            A_ub = _coo((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = _coo((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _refine_single(s0, Xref, steps=120, beta=90.0, lr=0.06, rng=None):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    for _ in range(steps):
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w /= (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - np.dot(g, s) * s
        s = s - lr * gt
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            if rng is None:
                s = np.array([1.0, 0, 0, 0])
            else:
                s = rng.normal(size=4)
            ns = np.linalg.norm(s)
        s = s / ns
    return s

def _deepest_hole(cur, rng, n_sample=15000):
    S = rng.normal(size=(n_sample, 4))
    S = _norm(S)
    D = S @ cur.T
    md = D.max(axis=1)
    bi = int(np.argmin(md))
    # refine top candidates
    idx = np.argpartition(md, 4)[:4]
    best_s = None; best_v = 1e9
    for k in idx:
        s = _refine_single(S[k], cur, steps=120, beta=90.0, lr=0.06, rng=rng)
        v = float(np.max(cur @ s))
        if v < best_v:
            best_v = v; best_s = s
    # also refine global best with longer
    s0 = _refine_single(S[bi], cur, steps=160, beta=140.0, lr=0.04, rng=rng)
    v0 = float(np.max(cur @ s0))
    if v0 < best_v:
        return s0
    return best_s

def _loads(X, m=None, tol=1e-3, base=0.80):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    deg = np.sum(G > m - tol, axis=1).astype(float)
    exc = np.maximum(G - base, 0.0)
    # exclude diagonal (already -2)
    ssum = exc.sum(axis=1)
    return deg + 8.0 * ssum, deg, m, G

def solve():
    t0 = time.time()
    LIMIT = 172.0
    SEARCH_END = t0 + 136.0
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (168, 4):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands = [_norm(rng.normal(size=(168, 4)))]
    X_best = None; best = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best:
            best = v; X_best = c.copy()
    n = 168
    cycle = 0
    # quick squeeze of incumbent to have tight baseline (short LP)
    # do not spend much here; parent already squeezed
    while time.time() < SEARCH_END:
        cycle += 1
        if time.time() > SEARCH_END - 2.0:
            break
        if cycle % 2 == 1:
            # ---- overshoot branch ----
            K = int([1, 2, 3][cycle % 3]) if cycle < 6 else int(rng.integers(1, 4))
            # start from best descendant
            base = X_best.copy()
            # tiny jitter before growing to diversify holes
            if rng.random() < 0.5:
                base = base + rng.normal(0, 0.002, size=base.shape)
                base = _norm(base)
            ext = base.copy()
            for _ in range(K):
                h = _deepest_hole(ext, rng, n_sample=12000)
                ext = np.vstack([ext, h[None, :]])
            # refine enlarged code hard (worse max expected)
            steps = 700 if ext.shape[0] < 171 else 500
            T0 = float(10.0 ** rng.uniform(-2.4, -1.8))
            ext_opt = _smooth(ext, steps, T0, T0 * 0.1, 0.28, 0.022)
            # optionally one more short anneal
            if rng.random() < 0.5:
                ext_opt = _smooth(ext_opt, 250, 0.003, 0.0006, 0.12, 0.012)
            load, deg, m, G = _loads(ext_opt, tol=1e-3, base=0.80)
            # delete point with largest load; try top-2 and keep better after short refine
            order = np.argsort(-load, kind="stable")
            tried = []
            for di in order[:2]:
                mask = np.ones(ext_opt.shape[0], dtype=bool); mask[di] = False
                cand168 = ext_opt[mask]
                # short joint refinement
                cand168 = _smooth(cand168, 350, 0.006, 0.0007, 0.22, 0.018)
                v = _max_cos(cand168)
                tried.append((v, cand168))
                if time.time() > SEARCH_END:
                    break
            tried.sort(key=lambda t: t[0])
            v0, c0 = tried[0]
            if v0 < best + 0.004:
                # opportunistic LP squeeze
                dl = min(4.0, max(1.0, SEARCH_END - time.time() - 1.0))
                if dl > 1.0 and time.time() + dl < t0 + LIMIT - 8:
                    try:
                        c1, v1 = _lp(c0, time.time() + dl)
                    except Exception:
                        c1, v1 = c0, v0
                else:
                    c1, v1 = c0, v0
                if v1 < best:
                    best = v1; X_best = c1.copy()
            # also allow walking: if close, keep as base for hole diversity? no, keep best only
        else:
            # ---- undershoot branch ----
            D = int(rng.integers(8, 19))  # 8-18
            load, deg, m, G = _loads(X_best, tol=1e-3, base=0.80)
            order = np.argsort(-load, kind="stable")
            # add geographic spread: prefer high-load but avoid deleting tight pairs together? simple top-D with small random swap
            chosen = order[:D].copy()
            # randomize 1-2 of them
            if rng.random() < 0.5:
                chosen[-1] = int(rng.integers(n))
                chosen[-2] = int(rng.integers(n))
            chosen = np.unique(chosen)
            mask = np.ones(n, dtype=bool); mask[chosen] = False
            small = X_best[mask].copy()
            # polish small code to its own optimum
            T0 = float(10.0 ** rng.uniform(-2.4, -1.8))
            small = _smooth(small, 650, T0, T0 * 0.1, 0.28, 0.02)
            cur = small
            # grow back by 2-4 insertions + short joint refinements
            while cur.shape[0] < 168:
                if time.time() > SEARCH_END:
                    break
                need = 168 - cur.shape[0]
                b = int(rng.integers(2, 5))
                b = min(b, need)
                for _ in range(b):
                    h = _deepest_hole(cur, rng, n_sample=12000)
                    cur = np.vstack([cur, h[None, :]])
                # short joint refinement after every batch
                cur = _smooth(cur, 220, 0.005, 0.0008, 0.20, 0.018)
            if cur.shape[0] != 168:
                continue
            # final short refine
            cur = _smooth(cur, 350, 0.004, 0.0006, 0.18, 0.015)
            v = _max_cos(cur)
            if v < best + 0.004:
                dl = min(4.0, max(1.0, SEARCH_END - time.time() - 1.0))
                if dl > 1.0 and time.time() + dl < t0 + LIMIT - 8:
                    try:
                        c1, v1 = _lp(cur, time.time() + dl)
                    except Exception:
                        c1, v1 = cur, v
                else:
                    c1, v1 = cur, v
                if v1 < best:
                    best = v1; X_best = c1.copy()
    # final polish with remaining time
    remaining = (t0 + LIMIT - 2.0) - time.time()
    if remaining > 6.0 and X_best is not None:
        try:
            X_best, best = _lp(X_best, time.time() + min(remaining - 1.0, 30.0))
        except Exception:
            pass
        if time.time() < t0 + LIMIT - 5.0:
            try:
                X2 = _smooth(X_best, 300, 0.002, 0.0005, 0.10, 0.01)
                if _max_cos(X2) < best:
                    X_best = X2; best = _max_cos(X_best)
                    # one more tiny LP if time
                    if time.time() < t0 + LIMIT - 8.0:
                        try:
                            X3, v3 = _lp(X_best, time.time() + 4.0)
                            if v3 < best:
                                X_best, best = X3, v3
                        except Exception:
                            pass
            except Exception:
                pass
    X_best = _norm(np.where(np.isfinite(X_best), X_best, 0.0))
    return X_best
