import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, lr, T0, T1, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    # precompute decay for T
    if steps > 1:
        logr = np.log(T1 / T0) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        # softmax weights
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(42)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    it = 0
    # exploration budget: leave ~37s for final LP
    EXPL = 133.0
    while time.time() - t0 < EXPL:
        it += 1
        elapsed = time.time() - t0
        T_acc = 0.0015 * (1.0 - elapsed / (EXPL + 5)) + 0.00025
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        if r < 0.65:
            nk = int(rng.integers(1, 4))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            delta = 0.002 + rng.random() * 0.004
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            prop = cur.copy()
            for vv in victims:
                C = 6000
                cand = rng.normal(size=(C, d))
                cand = _norm(cand)
                Dots = cand @ prop.T
                Dots[:, vv] = -2.0
                md = Dots.max(axis=1)
                K = 8
                idx = np.argpartition(md, K)[:K]
                if rng.random() < 0.5:
                    pick = idx[rng.integers(K)]
                else:
                    pick = idx[np.argmin(md[idx])]
                prop[vv] = cand[pick]
            prop = _relax(prop, steps=700, lr=0.015, T0=0.012, T1=0.0015)
        elif r < 0.85:
            sigma = 0.03 + rng.random() * 0.07
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=750, lr=0.015, T0=0.012, T1=0.0012)
        else:
            sigma = 0.008 + rng.random() * 0.012
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=500, lr=0.010, T0=0.008, T1=0.0010)
        val = _max_cos(prop)
        improved_cur = False
        if val < cur_val - 1e-12:
            cur = prop
            cur_val = val
            improved_cur = True
        else:
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur = prop
                cur_val = val
        if val < best_val - 1e-12:
            # extra squeeze on new record
            prop2 = _relax(prop, steps=1100, lr=0.008, T0=0.005, T1=0.0008)
            val2 = _max_cos(prop2)
            if val2 < val:
                prop = prop2
                val = val2
                cur = prop
                cur_val = val
            best = prop.copy()
            best_val = val
        # safety time break
        if time.time() - t0 > EXPL + 2:
            break
    remain = 172.0 - (time.time() - t0)
    if remain > 5:
        # use most of remaining for LP, keep 3s margin
        lp_time = max(5.0, remain - 3.0)
        # cap to 40s to avoid overrun
        lp_time = min(lp_time, 40.0)
        try:
            best = _lp_polish(best, lp_time)
        except Exception:
            pass
    best = _norm(best)
    # shrink tiny margin not needed; ensure finite
    best[~np.isfinite(best)] = 0.0
    # renormalize zero-fixed rows if any
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(bad.sum()), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
