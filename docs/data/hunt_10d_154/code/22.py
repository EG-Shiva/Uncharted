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
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
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

def _quick_push(x, iters=60, eta0=0.015, thr_off=0.008):
    for it in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        W = (G > M - thr_off).astype(float)
        F = W @ x
        dot = (F * x).sum(axis=1, keepdims=True)
        Ft = F - dot * x
        eta = eta0 * (1.0 - 0.7 * it / max(iters, 1))
        x = _norm(x - eta * Ft)
    return x

def _hole_pick(prop, victim, rng, C=12000):
    n, d = prop.shape
    cand = _norm(rng.normal(size=(C, d)))
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    K = 10
    idx = np.argpartition(md, K)[:K]
    sub = idx[np.argsort(md[idx])]
    r = rng.random()
    if r < 0.55:
        pick = sub[0]
    elif r < 0.8:
        pick = sub[rng.integers(1, min(4, K))]
    else:
        pick = sub[rng.integers(K)]
    best_c = cand[pick].copy()
    best_v = float(md[pick])
    # local hole deepening: hill climb around pick
    sig = 0.06
    for _ in range(2):
        pert = _norm(best_c[None, :] + rng.normal(scale=sig, size=(2500, d)))
        D2 = pert @ prop.T
        D2[:, victim] = -2.0
        m2 = D2.max(axis=1)
        j = int(np.argmin(m2))
        if float(m2[j]) < best_v:
            best_v = float(m2[j])
            best_c = pert[j]
        sig *= 0.5
    return best_c

def _lp_polish(x, t_end):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.8 and radius > 1e-14:
        if time.time() + 1.0 > t_end:
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
            x, best, radius = y, v, min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_LIM = 172.0
    t_end = t0 + T_LIM
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
    t_expl_end = t_end - 34.0
    it = 0
    while time.time() < t_expl_end:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(t_expl_end - t0, 1e-9)
        T_acc = 0.0017 * (1.0 - frac) + 0.00018
        if T_acc < 0.00015:
            T_acc = 0.00015
        r = rng.random()
        prop = cur.copy()
        if r < 0.48:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.002 + rng.random() * 0.006
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.4
            probs /= probs.sum()
            nk = int(rng.integers(1, 4))
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            C = 14000 if rng.random() < 0.6 else 24000
            for vv in victims:
                prop[vv] = _hole_pick(prop, int(vv), rng, C=C)
            prop = _quick_push(prop, iters=40, eta0=0.012)
            steps = int(450 + rng.integers(400))
            lr = 0.012 + rng.random() * 0.010
            T0 = 0.008 + rng.random() * 0.008
            T1 = 0.0006 + rng.random() * 0.0012
            prop = _relax(prop, steps=steps, lr=lr, T0=T0, T1=T1)
        elif r < 0.70:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.003 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.4
            probs /= probs.sum()
            nk = int(rng.integers(4, 7))
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            for vv in victims:
                prop[vv] = _hole_pick(prop, int(vv), rng, C=9000)
            prop = _quick_push(prop, iters=30, eta0=0.012)
            prop = _relax(prop, steps=int(550 + rng.integers(350)), lr=0.013 + rng.random() * 0.008,
                          T0=0.010 + rng.random() * 0.006, T1=0.0006 + rng.random() * 0.001)
        elif r < 0.86:
            sigma = 0.02 + rng.random() * 0.10
            if rng.random() < 0.6:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                mcur = float(G.max())
                cnt = (G > mcur - 0.004).sum(axis=1)
                w = (cnt.astype(float) / max(float(cnt.max()), 1.0) * 0.7 + 0.3)
                noise = rng.normal(scale=sigma, size=cur.shape) * w[:, None]
                prop = _norm(cur + noise)
            else:
                prop = _norm(cur + rng.normal(scale=sigma, size=cur.shape))
            prop = _relax(prop, steps=int(450 + rng.integers(350)), lr=0.012 + rng.random() * 0.010,
                          T0=0.009 + rng.random() * 0.006, T1=0.0006 + rng.random() * 0.0012)
        else:
            u = _norm(rng.normal(size=(1, d)))[0]
            dots = cur @ u
            k = int(rng.integers(8, 32))
            thr = np.sort(dots)[n - k]
            cap = np.where(dots >= thr)[0]
            shift = rng.normal(size=(d,)) * (0.05 + rng.random() * 0.13)
            prop[cap] = prop[cap] + shift[None, :]
            prop[cap] = prop[cap] + rng.normal(scale=0.012, size=(len(cap), d))
            prop = _norm(prop)
            prop = _relax(prop, steps=int(550 + rng.integers(350)), lr=0.013 + rng.random() * 0.008,
                          T0=0.010 + rng.random() * 0.006, T1=0.0007 + rng.random() * 0.001)
        pv = _max_cos(prop)
        if pv < cur_val:
            cur, cur_val = prop, pv
            if pv < best_val:
                best, best_val = prop.copy(), pv
        else:
            if rng.random() < np.exp(-(pv - cur_val) / T_acc):
                cur, cur_val = prop, pv
        if it % 55 == 0 and cur_val > best_val + 0.004:
            cur, cur_val = best.copy(), best_val
        if it % 25 == 0 and time.time() > t_expl_end - 2:
            break
    try:
        # short anneal to tighten before LP
        best = _relax(best, steps=900, lr=0.010, T0=0.004, T1=0.00035)
        best = _lp_polish(best, t_end - 0.5)
    except Exception:
        pass
    best = _norm(best)
    return best
