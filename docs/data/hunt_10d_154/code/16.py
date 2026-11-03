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

def _contacts(x, delta=0.005):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = (G > m - delta).sum(axis=1)
    return m, cnt, G

def _hole_pick(prop, victim, rng, C=8000, K=8):
    d = prop.shape[1]
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    D = cand @ prop.T
    D[:, victim] = -2.0
    md = D.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    if rng.random() < 0.5:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    return cand[pick]

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
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
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
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
    DEAD = t0 + 168.0
    EXPL = DEAD - 30.0
    rng = np.random.default_rng(12345)
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
    it = 0
    stag = 0
    while time.time() < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL - t0, 1.0)
        T_acc = 0.0016 * (1.0 - frac) + 0.0002
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        prop = cur.copy()
        if stag > 60:
            # big escape
            prop = cur + rng.normal(scale=0.20, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=650, lr=0.015, T0=0.014, T1=0.0012)
            r_mode = "escape"
        elif r < 0.50:
            nk = int(rng.integers(1, 4))
            mcur, cnt, _ = _contacts(cur, delta=0.003 + rng.random() * 0.005)
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_pick(prop, vv, rng, C=7000)
            steps = int(500 + rng.integers(350))
            prop = _relax(prop, steps=steps, lr=0.015, T0=0.012, T1=0.0012 + rng.random() * 0.001)
            r_mode = "jam"
        elif r < 0.75:
            sigma = 0.02 + rng.random() * 0.10
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            steps = int(550 + rng.integers(300))
            prop = _relax(prop, steps=steps, lr=0.015, T0=0.012, T1=0.0012)
            r_mode = "shake"
        else:
            # rattler / loose refill
            mcur, cnt, G = _contacts(cur, delta=0.004)
            # loose = few contacts within delta
            loose = np.where(cnt <= 5)[0]
            if len(loose) == 0:
                loose = np.argsort(cnt)[:4]
            nk = min(len(loose), int(rng.integers(1, 5)))
            victims = rng.choice(loose, size=nk, replace=False) if len(loose) >= nk else loose
            for vv in victims:
                prop[vv] = _hole_pick(prop, vv, rng, C=7000)
            prop = _relax(prop, steps=int(500 + rng.integers(300)), lr=0.015, T0=0.011, T1=0.0010)
            r_mode = "rattler"
        prop_val = _max_cos(prop)
        # accept
        if prop_val < cur_val or rng.random() < np.exp((cur_val - prop_val) / T_acc):
            cur = prop
            cur_val = prop_val
            if prop_val < best_val:
                best = prop.copy()
                best_val = prop_val
                stag = 0
            else:
                stag += 1
        else:
            stag += 1
        # opportunistic LP snap when promising
        if prop_val < best_val + 0.003 and time.time() < EXPL - 2:
            # only snap occasionally to save time
            if (it % 4 == 0) or (prop_val < best_val + 0.0005):
                try:
                    snapped = _lp_polish(prop, seconds=1.5, deadline=EXPL)
                    sv = _max_cos(snapped)
                    if sv < best_val:
                        best = snapped.copy()
                        best_val = sv
                        cur = snapped.copy()
                        cur_val = sv
                        stag = 0
                    elif sv < cur_val and rng.random() < 0.7:
                        cur = snapped.copy()
                        cur_val = sv
                except Exception:
                    pass
        if best_val < 0.39:
            break
    # final polish
    remain = DEAD - time.time() - 1.5
    if remain > 4:
        try:
            best = _lp_polish(best, seconds=remain, deadline=DEAD - 1.0)
        except Exception:
            pass
    best = _norm(best)
    return best
