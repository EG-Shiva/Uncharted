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
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1,1e-12) / max(T0,1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        E = np.exp((G - mv) / max(T, 1e-12))
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

def _hole_best(prop, victim, rng, C=8000):
    d = prop.shape[1]
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ prop.T
    # exclude victim itself
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    # return few best, pick one
    K = 8
    idx = np.argpartition(md, K)[:K]
    # 50% best, 50% random among K
    if rng.random() < 0.5:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    return cand[pick]

def _lp_polish(x, seconds, t_deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > t_deadline:
            break
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
    DEAD = t0 + 165.0
    rng = np.random.default_rng(7)
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
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    # quick center rattlers once
    # exploration: leave ~32s for final polish
    EXPL = DEAD - 32.0
    it = 0
    while time.time() < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL - t0, 1.0)
        T_acc = 0.0016 * (1.0 - frac) + 0.00022
        r = rng.random()
        prop = cur.copy()
        if r < 0.55:
            # jammed-point kick to hole
            nk = int(rng.integers(1, 4))  # 1-3
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            delta = 0.003 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            # also consider total contact load for weighting
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_best(prop, vv, rng, C=6000)
            prop = _relax(prop, steps=int(500 + rng.integers(300)), lr=0.015, T0=0.012, T1=0.0014)
        elif r < 0.78:
            sigma = 0.025 + rng.random() * 0.07
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=650, lr=0.015, T0=0.011, T1=0.0012)
        else:
            # rattler / low-contact recentering
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            cnt = (G > mcur - 1e-9).sum(axis=1)
            # low contact indices
            order = np.argsort(cnt)
            nk = int(rng.integers(2, 6))
            victims = order[:nk]
            # sometimes add a random one
            if rng.random() < 0.4:
                victims = np.append(victims, rng.integers(n))
            for vv in victims:
                # larger candidate pool for single-point global opt
                prop[vv] = _hole_best(prop, int(vv), rng, C=12000)
            prop = _relax(prop, steps=550, lr=0.012, T0=0.008, T1=0.0010)
        v = _max_cos(prop)
        # accept?
        if v < cur_val or rng.random() < np.exp((cur_val - v) / max(T_acc, 1e-9)):
            cur = prop
            cur_val = v
        if v < best_val:
            # snap to local optimum with short LP
            if time.time() < EXPL - 3.0:
                try:
                    snapped = _lp_polish(prop, 2.0, t_deadline=EXPL)
                    vs = _max_cos(snapped)
                except Exception:
                    snapped, vs = prop, v
                if vs < best_val:
                    best, best_val = snapped.copy(), vs
                    cur, cur_val = snapped.copy(), vs
                elif v < best_val:
                    best, best_val = prop.copy(), v
            else:
                best, best_val = prop.copy(), v
        # occasional intensify around best
        if it % 25 == 0 and time.time() < EXPL - 5:
            # light anneal of best
            tmp = _relax(best.copy(), steps=300, lr=0.008, T0=0.004, T1=0.0008)
            vt = _max_cos(tmp)
            if vt < best_val:
                best, best_val = tmp, vt
                cur, cur_val = tmp.copy(), vt
    # final sharpen + polish with remaining time
    remain = max(DEAD - time.time(), 5.0)
    lp_time = min(30.0, max(remain - 4.0, 5.0))
    # sharpening relax
    try:
        best = _relax(best.copy(), steps=1200, lr=0.010, T0=0.005, T1=0.0005)
    except Exception:
        pass
    try:
        best = _lp_polish(best, lp_time, t_deadline=DEAD - 1.0)
    except Exception:
        pass
    best = _norm(best)
    return best
