import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X, steps, beta0, beta1, s0, s1, mu=0.72):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        step = s0 + (s1 - s0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # softmax over pairs (upper weight to max)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ X
        proj = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - proj * X
        nrm = np.sqrt((gt ** 2).sum(axis=1, keepdims=True)) + 1e-15
        direc = gt / nrm
        vel = mu * vel + (1.0 - mu) * direc
        vn = np.sqrt((vel ** 2).sum(axis=1, keepdims=True)) + 1e-15
        vdir = vel / vn
        X = X - step * vdir
        X = _norm(X)
    return X

def _deep_hole(others, rng, m_try=2500, refine=40):
    cand = _norm(rng.standard_normal((m_try, 6)))
    dots = cand @ others.T
    worst = dots.max(axis=1)
    y = cand[int(np.argmin(worst))].copy()
    step = 0.05
    for _ in range(refine):
        d = others @ y
        j = int(np.argmax(d))
        g = others[j] - (others[j] @ y) * y
        n = np.linalg.norm(g) + 1e-15
        g /= n
        y = y - step * g
        ny = np.linalg.norm(y) + 1e-15
        y /= ny
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    best_x = X.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        idx = np.where(c_all > thresh)[0]
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c_all, dtype=bool)
        sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    def clean(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        return _norm(a)
    cands = []
    if xb is not None:
        c = clean(xb)
        if c is not None:
            cands.append(c)
    if xr is not None:
        c = clean(xr)
        if c is not None:
            cands.append(c)
    rng0 = np.random.default_rng(0)
    if not cands:
        cands.append(_norm(rng0.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    rng = np.random.default_rng(912345)
    iu0 = np.triu_indices(116, 1)
    explore_end = t0 + BUDGET - 30.0
    it = 0
    temp0, temp1 = 0.0030, 0.00010
    while time.time() < explore_end:
        it += 1
        frac = min(1.0, (time.time() - t0) / max(explore_end - t0, 1e-9))
        temp = temp0 + (temp1 - temp0) * frac
        trial = cur_x.copy()
        Gf = trial @ trial.T
        np.fill_diagonal(Gf, -2.0)
        pm = Gf.max(axis=1)
        mcur = float(pm.max())
        cnt = (Gf > mcur - 1e-7).sum(axis=1)
        order_worst = np.argsort(-pm)
        order_hub = np.argsort(-cnt)
        op = rng.random()
        if op < 0.18:
            sigma = float(rng.uniform(0.02, 0.12))
            trial = trial + sigma * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(220, 380)),
                            beta0=30, beta1=170, s0=0.025, s1=0.004)
        elif op < 0.38:
            k = int(rng.integers(4, 10))
            hubs = order_hub[:max(14, 2 * k)]
            idx = rng.choice(hubs, size=k, replace=False)
            sigma = float(rng.uniform(0.10, 0.40))
            trial[idx] += sigma * rng.standard_normal((k, 6))
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(280, 480)),
                            beta0=25, beta1=210, s0=0.03, s1=0.003)
        elif op < 0.56:
            flat = Gf[iu0]
            topk = np.argsort(-flat)[:4]
            moved = set()
            for tk in topk:
                if rng.random() < 0.8:
                    a = int(iu0[0][tk]); b = int(iu0[1][tk])
                    for vtx in (a, b):
                        if vtx in moved:
                            continue
                        moved.add(vtx)
                        other = b if vtx == a else a
                        dvec = trial[vtx] - trial[other]
                        dvec = dvec - (dvec @ trial[vtx]) * trial[vtx]
                        n = np.linalg.norm(dvec) + 1e-15
                        dvec /= n
                        amp = float(rng.uniform(0.06, 0.24))
                        trial[vtx] = trial[vtx] + amp * dvec + 0.018 * rng.standard_normal(6)
                    if len(moved) >= 6:
                        break
            trial = _norm(trial)
            trial = trial + 0.005 * rng.standard_normal(trial.shape)
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(280, 460)),
                            beta0=35, beta1=230, s0=0.022, s1=0.003)
        elif op < 0.76:
            k = int(rng.integers(2, 6))
            hubs = order_hub[:12].copy()
            rng.shuffle(hubs)
            idxs = hubs[:k]
            for wi in idxs:
                mask = np.ones(116, dtype=bool); mask[wi] = False
                others = trial[mask]
                y = _deep_hole(others, rng, m_try=2500, refine=40)
                y = y + 0.004 * rng.standard_normal(6)
                y /= (np.linalg.norm(y) + 1e-15)
                trial[wi] = y
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(280, 480)),
                            beta0=40, beta1=240, s0=0.02, s1=0.003)
        else:
            k = int(rng.integers(5, 14))
            # bias to worst points for subset reinit
            pool = order_worst[:40]
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            if len(idx) < k:
                extra = rng.choice(116, size=k - len(idx), replace=False)
                idx = np.concatenate([idx, extra])
            trial[idx] = _norm(rng.standard_normal((k, 6)))
            trial = _norm(trial)
            trial = _smooth(trial, steps=int(rng.integers(320, 560)),
                            beta0=20, beta1=190, s0=0.03, s1=0.0035)
        v = _maxcos(trial)
        # occasional LP squeeze for promising trials
        if v < best_val + 0.0015 and (it % 25 == 0) and (time.time() < explore_end - 3):
            tp = _lp_polish(trial, 0.7)
            v2 = _maxcos(tp)
            trial = tp
            v = v2
        if v < best_val - 1e-12:
            best_val = v
            x_best = trial.copy()
            cur_x = trial.copy(); cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-12)):
            if v < cur_val + 0.03:
                cur_x = trial.copy(); cur_val = v
        if it % 80 == 0 and time.time() < explore_end:
            cur_x = x_best.copy()
            cur_val = best_val
        if time.time() >= explore_end:
            break
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 4:
        x_best = _lp_polish(x_best, remain)
    else:
        x_best = _norm(x_best)
    return _norm(x_best)
