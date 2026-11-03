import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_sym(R):
    K = R.shape[0]
    N = K * 18 + 1
    X = np.empty((N, 9))
    idx = 0
    for k in range(K):
        v = R[k]
        for s in range(9):
            # cyclic coordinate shift
            X[idx] = np.roll(v, s)
            idx += 1
        for s in range(9):
            X[idx] = -np.roll(v, s)
            idx += 1
    X[idx] = np.ones(9, dtype=float) / 3.0
    return X

def _anneal_sym(R, steps, T0=0.02, T1=0.002, lr0=0.08, lr1=0.02, mu=0.6, t_end=None):
    K, d = R.shape
    V = np.zeros_like(R)
    lT0, lT1 = np.log(T0), np.log(T1)
    for s in range(steps):
        if t_end is not None and s % 20 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        X = _build_sym(R)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ X
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        DirX = WY * inv[:, None] - (r * inv)[:, None] * X
        DirX[~mk] = 0.0
        DirR = np.zeros_like(R)
        for k in range(K):
            acc = np.zeros(d)
            base = k * 18
            for ss in range(9):
                acc += np.roll(DirX[base + ss], -ss)
                acc += -np.roll(DirX[base + 9 + ss], -ss)
            DirR[k] = acc
        # tangent projection
        DirR = DirR - np.sum(DirR * R, axis=1, keepdims=True) * R
        V *= mu
        V += DirR
        sc = float(np.mean(np.linalg.norm(V, axis=1))) + 1e-12
        # DirR is ~18x scale; normalise like parent
        R = R - (lr / max(1.0, sc * 5.0)) * V
        R = _norm(R)
    return R

def _anneal_full(X, steps, T0=0.015, T1=0.002, lr0=0.08, lr1=0.02, mu=0.6, t_end=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    for s in range(steps):
        if t_end is not None and s % 20 == 0 and time.time() > t_end:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ X
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        Dir = WY * inv[:, None] - (r * inv)[:, None] * X
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        sc = float(np.mean(np.linalg.norm(V, axis=1))) + 1e-12
        X = X - (lr / max(1.0, sc * 5.0)) * V
        X = _norm(X)
    return X

def _lp_polish(X, budget):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.3 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c, -2500)[-2500]
            sel = c >= kth
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
    return x

def solve():
    t0 = time.time()
    T_LIM = 155.0
    def remain():
        return T_LIM - (time.time() - t0)
    n, d = 181, 9
    rng = np.random.default_rng(20261103)
    try:
        fb = np.load("best.npy")
        fb = np.asarray(fb, dtype=float)
        if fb.shape == (n, d):
            fb = _norm(fb)
        else:
            fb = None
    except Exception:
        fb = None
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (n, d):
            rec = _norm(rec)
        else:
            rec = None
    except Exception:
        rec = None
    fb_val = _max_cos(fb) if fb is not None else np.inf

    K = 10  # 10*18+1 = 181, orbit of cyclic shifts + sign flips
    best_sym = None
    best_sym_val = np.inf
    # symmetry-constrained search with several restarts
    n_restart = 5
    for ri in range(n_restart):
        if remain() < 60:
            break
        rgen = np.random.default_rng(9000 + ri)
        R = _norm(rgen.standard_normal((K, d)))
        # diversify: one restart from projected record/best
        if ri == 1 and fb is not None:
            # average fb over group to get reps: pick 10 spread points
            idx = rgen.choice(n, size=K, replace=False)
            R = fb[idx].copy() + rgen.standard_normal((K, d)) * 0.05
            R = _norm(R)
        elif ri == 2 and rec is not None:
            idx = rgen.choice(n, size=K, replace=False)
            R = rec[idx].copy() + rgen.standard_normal((K, d)) * 0.05
            R = _norm(R)
        R = _anneal_sym(R, steps=700, T0=0.025, T1=0.002, lr0=0.10, lr1=0.02, mu=0.6, t_end=t0 + T_LIM - 45)
        Xsym = _build_sym(_norm(R))
        v = _max_cos(Xsym)
        if v < best_sym_val:
            best_sym_val = v
            best_sym = Xsym.copy()
    if best_sym is None:
        best_sym = fb.copy() if fb is not None else _norm(rng.standard_normal((n, d)))
        best_sym_val = _max_cos(best_sym)

    # break symmetry: perturbations of increasing size + unconstrained anneal
    cands = []
    cands.append(best_sym.copy())
    scales = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    for si, sc in enumerate(scales):
        if remain() < 45:
            break
        rgen = np.random.default_rng(5000 + si)
        Xp = best_sym + rgen.standard_normal(best_sym.shape) * sc
        Xp = _norm(Xp)
        Xp = _anneal_full(Xp, steps=350, T0=0.012, T1=0.0015, lr0=0.07, lr1=0.015, mu=0.6, t_end=t0 + T_LIM - 30)
        cands.append(Xp.copy())
    # also one longer free anneal from best symmetric
    if remain() > 40 and len(cands) > 1:
        vals = [_max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        Xb = cands[bi].copy()
        Xb = _anneal_full(Xb, steps=400, T0=0.006, T1=0.0008, lr0=0.05, lr1=0.01, mu=0.65, t_end=t0 + T_LIM - 25)
        cands.append(Xb)

    vals = [_max_cos(c) for c in cands]
    order = np.argsort(vals)
    # LP-polish top copies with increasing perturbation history
    bestX = cands[int(order[0])].copy()
    bestV = float(vals[int(order[0])])
    nlp = 0
    for oi in order[:3]:
        if remain() < 8:
            break
        # small extra jitter before LP to escape exact symmetry
        rgen = np.random.default_rng(7777 + int(oi))
        Xq = cands[int(oi)].copy()
        # keep first copy exact, jitter others tiny
        if nlp > 0:
            Xq = _norm(Xq + rgen.standard_normal(Xq.shape) * 2e-4)
        bud = min(9.0, remain() - 2.0)
        if bud < 2.0:
            break
        Xp = _lp_polish(Xq, budget=bud)
        vp = _max_cos(Xp)
        if vp < bestV:
            bestV = vp
            bestX = Xp.copy()
        nlp += 1

    # fallback to lab best if we failed to improve
    if fb is not None and fb_val < bestV:
        return fb.copy()
    bestX = _norm(bestX)
    return bestX
