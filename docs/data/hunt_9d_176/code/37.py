import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_quench(X, steps, beta0, beta1, lr0, lr1, seed_off=0):
    n, d = X.shape
    x = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Gr = W @ x
        proj = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - proj * x
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        x = x - (lr / gn) * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x /= nr
    return x

def lp_polish(X, deadline, r_init=0.004, cap=2500):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2*d] = J[:, None] * d + ar
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    cand = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cand = a / nr
                    break
        except Exception:
            continue
    if cand is None:
        cand = rng.standard_normal((n, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    best_X = cand.copy()
    best_v = max_cos(best_X)
    # stage 1: short LP to lock in seed optimum
    d1 = min(t0 + 22.0, HARD - 60.0)
    if time.time() < d1:
        p = lp_polish(best_X, d1, r_init=0.003)
        v = max_cos(p)
        if v < best_v and np.all(np.isfinite(p)):
            best_X, best_v = p, v
    cur = best_X.copy()
    # stage 2: cheap jitter + quench screening (no LP), keep track
    # contact counts to target hubs occasionally
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    thr = best_v - 1e-7
    cc = np.sum(G0 > thr, axis=1)
    order_hub = np.argsort(-cc)
    trial = 0
    LOOP_END = min(t0 + 125.0, HARD - 35.0)
    best_trial_X = None
    while time.time() < LOOP_END and trial < 60:
        trial += 1
        y = cur.copy()
        if trial % 3 == 0:
            # hub-targeted: perturb top-contact points slightly more
            k = int(rng.integers(3, 9))
            idx = order_hub[rng.choice(n, size=k, replace=False) % n][:k]
            # actually pick among top 40 hubs
            pool = order_hub[:40]
            idx = pool[rng.choice(len(pool), size=k, replace=False)]
            amp = 0.008 + 0.008 * rng.random()
            y[idx] += amp * rng.standard_normal((k, d))
        elif trial % 3 == 1:
            k = int(rng.integers(4, 12))
            idx = rng.choice(n, size=k, replace=False)
            amp = 0.004 + 0.006 * rng.random()
            y[idx] += amp * rng.standard_normal((k, d))
        else:
            amp = 0.002 + 0.004 * rng.random()
            y += amp * rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        steps = 400 if trial < 30 else 600
        yq = smooth_quench(y, steps=steps, beta0=30.0, beta1=220.0, lr0=0.03, lr1=0.003)
        vq = max_cos(yq)
        # accept as new cur if close (basin hop) to diversify
        if vq < best_v + 0.004 and rng.random() < 0.3:
            cur = yq
            G0 = cur @ cur.T
            np.fill_diagonal(G0, -2.0)
            cc = np.sum(G0 > (vq - 1e-7), axis=1)
            order_hub = np.argsort(-cc)
        if vq < best_v:
            best_trial_X = yq.copy()
            # immediate short polish if promising and time allows
            if vq < best_v + 0.001:
                pass
        # keep the best quench even if worse than best, for final polish pool
        if best_trial_X is None or vq < max_cos(best_trial_X):
            if vq < best_v + 0.002:
                best_trial_X = yq.copy()
    # stage 3: LP-polish up to 2 most promising quenched candidates
    cands_lp = []
    if best_trial_X is not None:
        cands_lp.append(best_trial_X)
    # also add a fresh small global jitter of best
    if time.time() < HARD - 30.0:
        y = best_X + 0.003 * rng.standard_normal((n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        yq = smooth_quench(y, steps=500, beta0=30.0, beta1=250.0, lr0=0.025, lr1=0.002)
        cands_lp.append(yq)
    for c in cands_lp:
        if time.time() >= HARD - 12.0:
            break
        dl = min(time.time() + 12.0, HARD - 10.0)
        p = lp_polish(c, dl, r_init=0.004)
        v = max_cos(p)
        if v < best_v:
            best_X, best_v = p, v
            cur = p.copy()
    # stage 4: final polish of overall best
    if time.time() < HARD - 5.0:
        dl = HARD - 3.0
        p = lp_polish(best_X, dl, r_init=0.002)
        v = max_cos(p)
        if v < best_v:
            best_X, best_v = p, v
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
