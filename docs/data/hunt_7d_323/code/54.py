import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _random_orth(d, rng):
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    q = q * s[None, :]
    return q

def _anneal(X, steps, T0, T1, e0, e1, deadline=None, t0=None):
    n, d = X.shape
    # ensure normalized
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        if deadline is not None and t0 is not None and (s % 50 == 0):
            if time.time() - t0 > deadline:
                break
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if steps > 1 else T0
        eta = e0 * (e1 / e0) ** frac if steps > 1 else e0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights
        # (G-m)<=0
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-30
        A = (W @ X) / rs
        # tangential component of average neighbour
        dot = (A * X).sum(axis=1, keepdims=True)
        At = A - dot * X
        # limit max displacement to eta
        nr = np.linalg.norm(At, axis=1, keepdims=True) + 1e-30
        # scale so that mean norm maps to eta? use max-norm scaling
        # move opposite to At (away from neighbours)
        step = At / nr * np.minimum(nr, 1.0)  # clip huge
        X = X - eta * step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds, t0_global, deadline_global, init_radius=4e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - t0_global > deadline_global:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cols.reshape(p, 2 * d + 1)[:, :d] = I[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, 2 * d] = n * d
        vals.reshape(p, 2 * d + 1)[:, :d] = x[J]
        vals.reshape(p, 2 * d + 1)[:, d:2 * d] = x[I]
        vals.reshape(p, 2 * d + 1)[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _build_shells():
    d = 7
    # D7 roots: +-e_i +-e_j
    D7 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d); v[i] = si; v[j] = sj
                    D7.append(v)
    D7 = np.array(D7) / np.sqrt(2.0)  # 84
    Axes = np.zeros((14, d))
    for i in range(d):
        Axes[2 * i, i] = 1.0; Axes[2 * i + 1, i] = -1.0
    # E7 via E8 orthogonal to e1+e2
    E8a = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8); v[i] = si; v[j] = sj
                    E8a.append(v)
    E8a = np.array(E8a)
    E8b = []
    for mask in range(256):
        v = np.full(8, 0.5)
        bits = [(mask >> k) & 1 for k in range(8)]
        for k, b in enumerate(bits):
            if b:
                v[k] = -0.5
        if int(sum(bits)) % 2 == 0:
            E8b.append(v)
    E8b = np.array(E8b)  # 128
    E8 = np.vstack([E8a, E8b])
    v0 = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    dots = E8 @ v0
    E7_8 = E8[np.abs(dots) < 1e-9]  # expect 126
    # basis of v0 perp: (e1-e2)/sqrt2, e3..e8
    B = np.zeros((8, 7))
    B[0, 0] = 1 / np.sqrt(2.0); B[1, 0] = -1 / np.sqrt(2.0)
    for k in range(6):
        B[2 + k, 1 + k] = 1.0
    E7 = E7_8 @ B
    E7 = E7 / np.linalg.norm(E7, axis=1, keepdims=True)
    # W4 shell: 4 x +-1
    W4 = []
    for comb in itertools.combinations(range(d), 4):
        for mask in range(16):
            v = np.zeros(d)
            for t, pos in enumerate(comb):
                v[pos] = 1.0 if ((mask >> t) & 1) == 0 else -1.0
            W4.append(v)
    W4 = np.array(W4) / 2.0  # norm 2 -> unit, 560
    return D7, Axes, E7, W4

def _cyclic_seed(n, d, k, rng):
    # random block rotation of order k
    U = _random_orth(d, rng)
    # 3 blocks 2x2 + 1x1
    R = np.zeros((d, d))
    # choose p's coprime at least one
    while True:
        ps = rng.integers(1, k, size=3)
        import math
        if any(math.gcd(int(p), k) == 1 for p in ps):
            break
        if k < 3:
            break
    for b, p in enumerate(ps):
        th = 2 * np.pi * float(p) / float(k)
        c, s = np.cos(th), np.sin(th)
        R[2 * b, 2 * b] = c; R[2 * b, 2 * b + 1] = -s
        R[2 * b + 1, 2 * b] = s; R[2 * b + 1, 2 * b + 1] = c
    if k % 2 == 1:
        R[6, 6] = 1.0
    else:
        R[6, 6] = 1.0 if rng.random() < 0.5 else -1.0
    Q = U @ R @ U.T
    # powers
    Qp = [np.eye(d)]
    for _ in range(1, k):
        Qp.append(Qp[-1] @ Q)
    Qp = np.array(Qp)  # k,d,d
    pts = []
    nb = (n + k - 1) // k + 1
    for _ in range(nb):
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        orb = Qp @ v  # k,d
        # normalize (orthogonal preserves)
        pts.append(orb)
        if len(pts) * k >= n + k:
            break
    pts = np.vstack(pts)
    # trim/pad
    if pts.shape[0] >= n:
        idx = rng.choice(pts.shape[0], size=n, replace=False)
        pts = pts[idx]
    else:
        extra = rng.standard_normal((n - pts.shape[0], d))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([pts, extra])
    pts = pts[:n]
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def solve():
    t0 = time.time()
    DEAD = 170.0  # return by ~170s
    LP_DEAD = 165.0
    rng = np.random.default_rng(20261101)
    n, d = 323, 7
    # load fallback
    best_X = None; best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = rng.standard_normal((n, d))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_v = _max_cos(best_X)
    fallback_X = best_X.copy(); fallback_v = float(best_v)

    try:
        D7, Axes, E7, W4 = _build_shells()
    except Exception:
        return fallback_X / np.linalg.norm(fallback_X, axis=1, keepdims=True)

    # seed builders
    def make_seed(si):
        r = rng  # shared deterministic stream
        typ = si % 6
        Q1 = _random_orth(d, r); Q2 = _random_orth(d, r); Q3 = _random_orth(d, r)
        if typ == 0:
            a = E7  # 126
            b = D7 @ Q1.T  # 84
            need = n - a.shape[0] - b.shape[0]
            c = r.standard_normal((need, d)); c /= np.linalg.norm(c, axis=1, keepdims=True)
            X = np.vstack([a, b, c])
        elif typ == 1:
            a = D7 @ Q1.T; b = D7 @ Q2.T; c = D7 @ Q3.T  # 252
            need = n - 252
            e = r.standard_normal((need, d)); e /= np.linalg.norm(e, axis=1, keepdims=True)
            X = np.vstack([a, b, c, e])
        elif typ == 2:
            k = [17, 23, 29, 31][si % 4]
            X = _cyclic_seed(n, d, k, r)
        elif typ == 3:
            k = [11, 13, 19][si % 3]
            X = _cyclic_seed(n, d, k, r)
            # mix a rotated E7 shell partially: replace 60 pts with E7 sample rotated
            idx = r.choice(n, size=60, replace=False)
            samp = E7[r.choice(E7.shape[0], size=60, replace=False)] @ Q1.T
            X[idx] = samp
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif typ == 4:
            # W4 sample + axes + random
            sel = r.choice(W4.shape[0], size=180, replace=False)
            a = W4[sel] @ Q1.T
            b = Axes @ Q2.T  # 14
            need = n - 180 - 14
            c = r.standard_normal((need, d)); c /= np.linalg.norm(c, axis=1, keepdims=True)
            X = np.vstack([a, b, c])
        else:
            X = r.standard_normal((n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
        # tiny jitter to break exact degeneracy
        X = X + r.standard_normal(X.shape) * 1e-4
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    elites = []  # (score, X, degvar)
    nseeds = 14
    for si in range(nseeds):
        if time.time() - t0 > 110:
            break
        try:
            Xs = make_seed(si)
        except Exception:
            continue
        # cheap anneal
        try:
            Xa = _anneal(Xs, steps=450, T0=0.05, T1=0.008, e0=0.08, e1=0.008,
                         deadline=110, t0=t0)
        except Exception:
            continue
        try:
            v = _max_cos(Xa)
        except Exception:
            continue
        # degree distribution diversity: contacts within 1e-3 of max
        try:
            G = Xa @ Xa.T; np.fill_diagonal(G, -2.0)
            cc = (G > v - 1e-3).sum(axis=1)
            degvar = float(cc.var()) + float(cc.max())
        except Exception:
            degvar = 0.0
        elites.append((v, Xa, degvar))
        # track global best from seeds (not yet fallback)
        if v < best_v - 1e-12:
            best_v = v; best_X = Xa.copy()

    # keep diverse elite pool: sort by score, keep best few distinct
    elites_sorted = sorted(elites, key=lambda t: t[0])
    kept = []
    for v, X, dv in elites_sorted:
        dup = False
        for _, Y, _ in kept:
            # distinct if max cos differs or degree var differs enough
            try:
                if abs(v - _max_cos(Y)) < 2e-4 and abs(dv - float(((Y@Y.T> v-1e-3).sum(axis=1)).var())) < 2.0:
                    # check correlation? treat as dup if very close
                    pass
            except Exception:
                pass
        kept.append((v, X, dv))
        if len(kept) >= 3:
            break

    # refine kept with longer anneal
    for ki, (v0, X0, dv) in enumerate(list(kept)):
        if time.time() - t0 > 140:
            break
        try:
            Xr = _anneal(X0.copy(), steps=1200, T0=0.015, T1=0.0035, e0=0.04, e1=0.004,
                         deadline=140, t0=t0)
            vr = _max_cos(Xr)
            if vr < best_v - 1e-12:
                best_v = vr; best_X = Xr.copy()
            # update kept
            kept[ki] = (vr, Xr, dv)
        except Exception:
            continue

    # LP-polish only a seed whose smooth score approaches 0.665
    # choose best refined
    if kept:
        kept_sorted = sorted(kept, key=lambda t: t[0])
        cand_v, cand_X, _ = kept_sorted[0]
        # threshold: within 0.01 of 0.665 (i.e. <0.675) to be worth polishing
        if cand_v < 0.675 and (time.time() - t0) < LP_DEAD - 10:
            remain = LP_DEAD - (time.time() - t0)
            budget = min(20.0, max(2.0, remain - 2.0))
            try:
                Xp = _lp_polish(cand_X, seconds=budget, t0_global=t0, deadline_global=LP_DEAD, init_radius=3e-3)
                vp = _max_cos(Xp)
                if vp < best_v:
                    best_v = vp; best_X = Xp
            except Exception:
                pass

    # fallback: always retain loaded record
    if fallback_v < best_v:
        best_X = fallback_X; best_v = fallback_v

    # final safety: short LP squeeze on best if time remains and best is fallback-like?
    # avoid heavy timeout: only if best is already <0.666 and time remains
    if (time.time() - t0) < 160 and best_v < 0.6655:
        remain = 168 - (time.time() - t0)
        if remain > 3:
            try:
                best_X = _lp_polish(best_X, seconds=min(8.0, remain - 2), t0_global=t0, deadline_global=168, init_radius=1e-3)
            except Exception:
                pass

    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # sanitize
    best_X[~np.isfinite(best_X)] = 0.1
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
