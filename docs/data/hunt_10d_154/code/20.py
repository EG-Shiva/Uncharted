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

def _relax(x, steps, lr, T0, T1):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
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

def _hole_insert(prop, victim, rng, C=12000, refine_steps=180):
    n, d = prop.shape
    cand = rng.standard_normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    K = 8
    if C >= K:
        idx = np.argpartition(md, K)[:K]
        if rng.random() < 0.5:
            bi = idx[np.argmin(md[idx])]
        else:
            bi = idx[rng.integers(K)]
    else:
        bi = int(np.argmin(md))
    y = cand[bi]
    # single-point subgradient refinement: minimise max_j <y, prop_j>
    lr0 = 0.08
    T = 0.004
    for it in range(refine_steps):
        dots = prop @ y
        dots[victim] = -2.0
        M = dots.max()
        # softmax over dots to get smooth gradient
        e = np.exp((dots - M) / T)
        e[victim] = 0.0
        S = e.sum()
        if S <= 0 or not np.isfinite(S):
            break
        g = (e @ prop) / S
        gt = g - np.dot(g, y) * y
        lr = lr0 * (1.0 - 0.7 * it / max(refine_steps, 1))
        y = y - lr * gt
        ny = np.linalg.norm(y)
        if ny == 0 or not np.isfinite(ny):
            y = cand[bi]
            break
        y = y / ny
    return y

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn > 60:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 3000:
            order = np.argsort(-c[sel])[:3000]
            I, J = I[order], J[order]
            p = 3000
            # recompute sel values: need c values for b_ub
            cvals = c[sel][order]
        else:
            cvals = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cvals, A_eq=A_eq, b_eq=np.zeros(n),
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
    DEAD = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(1)
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
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    it = 0
    while time.time() < HOP_END and time.time() < DEAD - 25:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(HOP_END - t0, 1.0)
        T_acc = 0.0016 * (1.0 - frac) + 0.00022
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        if r < 0.60:
            nk = int(rng.integers(1, 4))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.002 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.5
            s = probs.sum()
            if s <= 0:
                probs = np.full(n, 1.0 / n)
            else:
                probs /= s
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            prop = cur.copy()
            for vv in victims:
                prop[vv] = _hole_insert(prop, int(vv), rng, C=12000, refine_steps=160)
            prop = _relax(prop, steps=int(600 + rng.integers(250)), lr=0.015, T0=0.012, T1=0.0014)
        elif r < 0.85:
            sigma = 0.03 + rng.random() * 0.07
            prop = cur + sigma * rng.standard_normal(size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=int(650 + rng.integers(200)), lr=0.015, T0=0.012, T1=0.0012)
        else:
            sigma = 0.006 + rng.random() * 0.014
            prop = cur + sigma * rng.standard_normal(size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=480, lr=0.010, T0=0.008, T1=0.0010)
        val = _max_cos(prop)
        if val < cur_val - 1e-12:
            cur = prop
            cur_val = val
        else:
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur = prop
                cur_val = val
        if val < best_val - 1e-12:
            p2 = _relax(prop.copy(), steps=1000, lr=0.008, T0=0.005, T1=0.0008)
            v2 = _max_cos(p2)
            if v2 < val:
                prop = p2
                val = v2
                cur = prop.copy()
                cur_val = val
            best = prop.copy()
            best_val = val
        if time.time() > HOP_END:
            break
    remain = DEAD - time.time()
    if remain > 6:
        lp_time = min(max(5.0, remain - 3.0), 38.0)
        try:
            best = _lp_polish(best, lp_time, deadline=DEAD - 2.0)
        except Exception:
            pass
    best = _norm(best)
    nr = np.linalg.norm(best, axis=1)
    bad = ~(np.isfinite(nr) & (nr > 1e-12))
    if np.any(bad):
        rb = rng.standard_normal(size=(int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
