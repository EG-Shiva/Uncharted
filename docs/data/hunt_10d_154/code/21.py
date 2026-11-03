import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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

def _refine_hole(y, X, T=0.004, iters=160, lr=0.08):
    # minimize softmax max dot to fixed set X, y on sphere
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    for t in range(1, iters + 1):
        d = X @ y
        mv = d.max()
        e = np.exp((d - mv) / T)
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ X  # gradient of softmax
        gt = g - (g @ y) * y
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        st = mhat / (np.sqrt(vhat) + eps)
        # decay lr slightly
        y = y - lr * st
        n = np.linalg.norm(y)
        if n == 0 or not np.isfinite(n):
            break
        y = y / n
    return y

def _hole_replace(prop, victim, rng, C=7000):
    n, d = prop.shape
    # candidate sampling
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    K = 12
    idx = np.argpartition(md, K)[:K]
    # order by md
    o = idx[np.argsort(md[idx])]
    # refine top 3 and pick best
    X = np.delete(prop, victim, axis=0)
    best_y = None
    best_v = 1e9
    for k in range(3):
        y0 = o[k] if k < len(o) else o[0]
        y = cand[y0].copy()
        # two temperatures refinement
        y = _refine_hole(y, X, T=0.005, iters=120, lr=0.08)
        y = _refine_hole(y, X, T=0.0015, iters=80, lr=0.04)
        v = float((X @ y).max())
        if v < best_v:
            best_v = v
            best_y = y
    # occasional random among K to diversify
    if rng.random() < 0.25:
        y = cand[o[rng.integers(len(o))]].copy()
        y = _refine_hole(y, X, T=0.004, iters=100, lr=0.08)
        # keep if not much worse
        v = float((X @ y).max())
        if v < best_v + 0.005 and rng.random() < 0.5:
            best_y = y
    return best_y

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
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
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    cur = best.copy()
    cur_val = best_val
    EXPL = DEAD - 34.0
    it = 0
    while time.time() < EXPL:
        it += 1
        elapsed = time.time() - t0
        total = max(EXPL - t0, 1.0)
        frac = elapsed / total
        T_acc = 0.0014 * (1.0 - frac) + 0.0002
        r = rng.random()
        prop = cur.copy()
        if r < 0.62:
            nk = int(rng.integers(1, 3))  # 1-2
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            delta = 0.002 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.4
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_replace(prop, vv, rng, C=6000)
            prop = _relax(prop, steps=int(450 + rng.integers(300)), lr=0.014 + rng.random() * 0.006, T0=0.010 + rng.random() * 0.004, T1=0.0012 + rng.random() * 0.0008)
        elif r < 0.82:
            sigma = 0.02 + rng.random() * 0.06 * (1.0 - 0.5 * frac)
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=int(500 + rng.integers(300)), lr=0.014 + rng.random() * 0.006, T0=0.011 + rng.random() * 0.004, T1=0.0010 + rng.random() * 0.0008)
        else:
            nk = int(rng.integers(3, 6))  # 3-5 big kick
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            delta = 0.003 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.4
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_replace(prop, vv, rng, C=5000)
            prop = _relax(prop, steps=int(550 + rng.integers(350)), lr=0.015, T0=0.013, T1=0.0015)
        prop_val = _mc(prop)
        if prop_val < best_val:
            best_val = float(prop_val)
            best = prop.copy()
        # Metropolis on cur
        if prop_val < cur_val or rng.random() < np.exp((cur_val - prop_val) / max(T_acc, 1e-9)):
            cur = prop.copy()
            cur_val = float(prop_val)
        # occasional restart from best if stuck
        if it % 120 == 0 and rng.random() < 0.4:
            # jump back toward best with noise
            cur = best + rng.normal(scale=0.015, size=best.shape)
            cur = _norm(cur)
            cur_val = _mc(cur)
    # greedy single-point hole tuning on best before final LP
    try:
        Xb = best.copy()
        # a few passes over worst points
        for _ in range(2):
            if time.time() > DEAD - 30:
                break
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            # worst pairs
            order = np.argsort(G.max(axis=1))[-12:]
            for vv in order:
                if time.time() > DEAD - 30:
                    break
                y = _hole_replace(Xb, int(vv), rng, C=4000)
                trial = Xb.copy()
                trial[vv] = y
                v = _mc(trial)
                if v < _mc(Xb):
                    Xb = trial
        vxb = _mc(Xb)
        if vxb < best_val:
            best = Xb
            best_val = vxb
    except Exception:
        pass
    # final polish
    remain = DEAD - time.time() - 1.0
    if remain < 5:
        remain = 5
    if remain > 36:
        remain = 36
    try:
        best = _lp_polish(best, seconds=remain, deadline=DEAD - 0.5)
    except Exception:
        pass
    best = _norm(best)
    return best
