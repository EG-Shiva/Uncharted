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

def _contact_counts(x, m, delta):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > m - delta).sum(axis=1), g

def _relax(x, steps, lr, T0, T1, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    if T0 <= 0: T0 = 1e-12
    if T1 <= 0: T1 = 1e-12
    if steps > 1:
        logr = np.log(T1 / T0) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
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

def _refine_single_point(prop, idx, steps=80, lr=0.02, T=0.005):
    # hill-climb single point to minimise its max dot to others
    for _ in range(steps):
        others = np.delete(prop, idx, axis=0)
        v = prop[idx:idx+1]
        d = (others @ v.T).ravel()
        mv = d.max()
        w = np.exp((d - mv) / max(T, 1e-12))
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        grad = (w @ others) / s  # direction to move away from
        # tangent component at v
        vv = v.ravel()
        g = grad - np.dot(grad, vv) * vv
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        # move opposite to grad (away from neighbours)
        vv = vv - lr * g / (ng + 1e-12) * min(1.0, ng * 5)
        # also small random tangent jitter to escape saddle
        vv = vv / np.linalg.norm(vv)
        prop[idx] = vv
        T *= 0.98
        if T < 0.0008:
            T = 0.0008
    return prop

def _hole_candidate(prop, victim, rng, C=8000, refine=True):
    d = prop.shape[1]
    cand = _norm(rng.normal(size=(C, d)))
    dots = cand @ prop.T
    dots[:, victim] = -2.0
    md = dots.max(axis=1)
    K = 10
    idx = np.argpartition(md, K)[:K]
    if rng.random() < 0.6:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    prop[victim] = cand[pick]
    if refine:
        _refine_single_point(prop, victim, steps=60, lr=0.03, T=0.006)
    return prop

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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
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
    elite = [(best_val, best.copy())]
    EXPL = t0 + 132.0
    it = 0
    while time.time() < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL - t0, 1.0)
        T_acc = 0.0018 * (1.0 - frac) + 0.0002
        r = rng.random()
        prop = cur.copy()
        if r < 0.50:
            # jammed kick to holes
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.002 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            # occasionally large kick
            if rng.random() < 0.18:
                nk = int(rng.integers(4, 7))
            elif rng.random() < 0.4:
                nk = int(rng.integers(2, 4))
            else:
                nk = 1
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            C = int(6000 + rng.integers(6000))
            for vv in victims:
                _hole_candidate(prop, vv, rng, C=C, refine=True)
            steps = int(550 + rng.integers(350))
            lr = 0.012 + rng.random() * 0.008
            T0 = 0.010 + rng.random() * 0.005
            T1 = 0.0010 + rng.random() * 0.0008
            prop = _relax(prop, steps=steps, lr=lr, T0=T0, T1=T1)
        elif r < 0.72:
            # rattler recycle + overloaded neighbourhood shake
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            # rattlers: max per point far below global max
            per_max = G.max(axis=1)
            ratt = np.where(per_max < mcur - 0.015)[0]
            cnt = (G > mcur - 0.004).sum(axis=1)
            prop = cur.copy()
            if len(ratt) > 0:
                nr = min(len(ratt), int(rng.integers(1, 3)))
                sel_r = rng.choice(ratt, size=nr, replace=False)
                for vv in sel_r:
                    _hole_candidate(prop, int(vv), rng, C=9000, refine=True)
            # shake neighbours of most loaded point
            hub = int(np.argmax(cnt))
            dots_hub = (cur @ cur[hub])
            dots_hub[hub] = -2
            nbrs = np.argsort(dots_hub)[-8:]
            sig = 0.02 + rng.random() * 0.03
            for nb in nbrs:
                if rng.random() < 0.6:
                    prop[nb] = prop[nb] + rng.normal(scale=sig, size=d)
            prop = _norm(prop)
            prop = _relax(prop, steps=int(550 + rng.integers(300)), lr=0.013, T0=0.011, T1=0.0011)
        elif r < 0.90:
            sigma = 0.025 + rng.random() * 0.075
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=int(600 + rng.integers(300)), lr=0.015, T0=0.012, T1=0.0012)
        else:
            # fine local shake
            sigma = 0.006 + rng.random() * 0.012
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=450, lr=0.009, T0=0.007, T1=0.0009)
        val = _max_cos(prop)
        # short LP snap for promising basins to get true value
        snapped = False
        if val < best_val + 0.002 and val < cur_val + 0.003 and (time.time() < EXPL - 8):
            # only snap if improvement potential and time allows; 1.2s snap
            try:
                ps = _lp_polish(prop, 1.2, deadline=EXPL)
                vs = _max_cos(ps)
                if vs < val:
                    prop, val = ps, vs
                    snapped = True
            except Exception:
                pass
        if val < cur_val - 1e-12:
            cur = prop.copy()
            cur_val = val
        else:
            if rng.random() < np.exp((cur_val - val) / max(T_acc, 1e-9)):
                cur = prop.copy()
                cur_val = val
        if val < best_val - 1e-12:
            # extra squeeze on record
            try:
                prop2 = _relax(prop, steps=900, lr=0.007, T0=0.004, T1=0.0007)
                val2 = _max_cos(prop2)
                if val2 < val:
                    prop, val = prop2, val2
                # quick snap
                if time.time() < EXPL - 5:
                    ps2 = _lp_polish(prop, 2.0, deadline=EXPL)
                    vs2 = _max_cos(ps2)
                    if vs2 < val:
                        prop, val = ps2, vs2
            except Exception:
                pass
            best = prop.copy()
            best_val = val
            cur = prop.copy()
            cur_val = val
            elite.append((best_val, best.copy()))
            elite = sorted(elite, key=lambda e: e[0])[:4]
        if time.time() > EXPL:
            break
    # final polish elites with remaining time
    elite = sorted(elite, key=lambda e: e[0])
    remain = DEAD - time.time()
    # reserve 3s margin
    budget = max(5.0, remain - 3.0)
    budget = min(budget, 38.0)
    start_pol = time.time()
    # distribute budget: best gets 60%, others share rest
    cands = [e[1] for e in elite[:3]]
    if len(cands) == 0:
        cands = [best]
    # ensure current best included
    vals = []
    for i, c in enumerate(cands):
        if i == 0:
            tb = budget * 0.6
        else:
            tb = budget * 0.4 / max(len(cands) - 1, 1)
        tb = max(tb, 2.0)
        if time.time() + tb > DEAD - 2.0:
            tb = max(1.0, DEAD - 2.0 - time.time())
        try:
            cp = _lp_polish(_norm(c), tb, deadline=DEAD - 2.0)
        except Exception:
            cp = c
        vals.append((_max_cos(cp), cp))
        if time.time() > DEAD - 3.0:
            break
    vals = sorted(vals, key=lambda e: e[0])
    best = vals[0][1]
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
