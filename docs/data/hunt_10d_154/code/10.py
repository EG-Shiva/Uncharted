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

def _relax_sm(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    if steps > 1:
        logr = np.log(T1 / T0) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
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

def _relax_poly(x, steps, lr, pwr=4.0, margin=0.35):
    # minimize sum_{i<j} relu(G-margin)^pwr, tangent Adam
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        H = G - margin
        H[H < 0] = 0.0
        # power
        Hp = H ** (pwr - 1.0)
        # gradient: p * Hp @ x
        Grad = (Hp @ x)
        s = np.abs(Grad).max()
        if s == 0 or not np.isfinite(s):
            break
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # decay lr slightly
        x = x - lr * step
        x = _norm(x)
    return x

def _hole_replace(prop, victims, rng, C=6000, K=8):
    n, d = prop.shape
    for vv in victims:
        cand = _norm(rng.normal(size=(C, d)))
        Dots = cand @ prop.T
        Dots[:, vv] = -2.0
        md = Dots.max(axis=1)
        idx = np.argpartition(md, K)[:K]
        if rng.random() < 0.5:
            pick = idx[rng.integers(K)]
        else:
            # best with prob, else second tier
            pick = idx[np.argmin(md[idx])]
        prop[vv] = cand[pick]
    return prop

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
                    best = a
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    elite = [(best_val, best.copy())]
    EXPL = 138.0
    it = 0
    stall = 0
    while time.time() - t0 < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / EXPL
        T_acc = 0.0016 * (1.0 - frac) + 0.0002
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        prop = None
        # contact stats of cur
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mcur = Gc.max()
        delta = 0.002 + rng.random() * 0.005
        cnt = (Gc > mcur - delta).sum(axis=1)
        # rattlers under tight threshold
        tight = 1e-7
        cnt_tight = (Gc > mcur - tight).sum(axis=1)
        rattlers = np.where(cnt_tight == 0)[0]
        if r < 0.48:
            nk = int(rng.integers(1, 5))
            if rng.random() < 0.75:
                probs = cnt.astype(float) + 0.5
                probs /= probs.sum()
                victims = rng.choice(n, size=nk, replace=False, p=probs)
            else:
                # include rattlers/random
                victims = rng.choice(n, size=nk, replace=False)
                if len(rattlers) > 0 and rng.random() < 0.7:
                    victims[0] = rattlers[rng.integers(len(rattlers))]
            prop = cur.copy()
            C = 5000 + int(rng.integers(0, 4000))
            prop = _hole_replace(prop, victims, rng, C=C, K=8)
            if rng.random() < 0.2:
                prop = _relax_poly(prop, steps=400, lr=0.01, pwr=4.0, margin=mcur - 0.02)
                prop = _relax_sm(prop, steps=550, lr=0.015, T0=0.012, T1=0.0012)
            else:
                T0 = 0.008 + rng.random() * 0.008
                T1 = 0.0008 + rng.random() * 0.0012
                lr = 0.012 + rng.random() * 0.008
                steps = 600 + int(rng.integers(0, 300))
                prop = _relax_sm(prop, steps=steps, lr=lr, T0=T0, T1=T1)
        elif r < 0.68:
            # cluster shake around most jammed point
            seed = int(np.argmax(cnt + rng.random(n) * 2.0))
            dists = Gc[seed].copy()
            k = int(rng.integers(6, 13))
            neigh = np.argpartition(-dists, k)[:k]
            prop = cur.copy()
            sigma = 0.10 + rng.random() * 0.15
            prop[neigh] += rng.normal(scale=sigma, size=(k, d))
            prop = _norm(prop)
            # also teleport 1-2 rattlers to holes
            if len(rattlers) > 0 and rng.random() < 0.8:
                nr_ = 1 + int(rng.integers(0, 2))
                nr_ = min(nr_, len(rattlers))
                vic = rng.choice(rattlers, size=nr_, replace=False)
                prop = _hole_replace(prop, vic, rng, C=6000, K=8)
            T0 = 0.010 + rng.random() * 0.006
            prop = _relax_sm(prop, steps=650, lr=0.015, T0=T0, T1=0.0012)
        elif r < 0.85:
            sigma = 0.02 + rng.random() * 0.06
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            if rng.random() < 0.3:
                prop = _relax_poly(prop, steps=350, lr=0.012, pwr=3.0 + rng.random() * 3.0, margin=mcur - 0.03)
            T0 = 0.008 + rng.random() * 0.006
            prop = _relax_sm(prop, steps=600, lr=0.013, T0=T0, T1=0.0010)
        else:
            # pair-break: take a top max pair and separate them
            iu0 = np.triu_indices(n, 1)
            # find top few pairs quickly
            flat = Gc[iu0]
            k2 = 5
            topk = np.argpartition(-flat, k2)[:k2]
            pick = topk[rng.integers(k2)]
            # map back: need indices
            # brute: get I,J of pick-th? easier: find i,j via search
            # instead find via argmax sampling: pick random among pairs > mcur-1e-7
            selpairs = np.where(flat > mcur - 1e-7)[0]
            if len(selpairs) > 0:
                sp = selpairs[rng.integers(len(selpairs))]
                # decode sp to (i,j): use iu arrays (build once per iter is heavy but n small)
                Iall = iu0[0][sp]
                Jall = iu0[1][sp]
                prop = cur.copy()
                # push apart along tangent difference
                vi = prop[Iall].copy()
                vj = prop[Jall].copy()
                diff = vi - vj
                diff = diff - 0.5 * (diff @ (vi + vj)) * (vi + vj) / max(np.linalg.norm(vi + vj) ** 2, 1e-12)
                alpha = 0.02 + rng.random() * 0.04
                prop[Iall] = _norm((vi + alpha * diff)[None, :])[0]
                prop[Jall] = _norm((vj - alpha * diff)[None, :])[0]
                prop += rng.normal(scale=0.01, size=prop.shape)
                prop = _norm(prop)
            else:
                prop = _norm(cur + rng.normal(scale=0.02, size=cur.shape))
            prop = _relax_sm(prop, steps=550, lr=0.012, T0=0.010, T1=0.0010)
        val = _max_cos(prop)
        if val < cur_val - 1e-12:
            cur = prop
            cur_val = val
            stall = 0
        else:
            stall += 1
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur = prop
                cur_val = val
        if val < best_val - 1e-12:
            # squeeze record basin harder
            prop2 = _relax_sm(prop, steps=1000, lr=0.008, T0=0.005, T1=0.0007)
            val2 = _max_cos(prop2)
            if val2 < val:
                prop = prop2
                val = val2
                cur = prop
                cur_val = val
            best = prop.copy()
            best_val = val
            elite.append((best_val, best.copy()))
            if len(elite) > 4:
                elite.sort(key=lambda z: z[0])
                elite = elite[:4]
            stall = 0
        # restart from elite if stalled
        if stall > 35:
            # pick best or random elite with perturbation
            elite.sort(key=lambda z: z[0])
            if rng.random() < 0.8:
                cur = elite[0][1].copy()
                cur_val = elite[0][0]
                cur = _norm(cur + rng.normal(scale=0.03, size=cur.shape))
                cur = _relax_sm(cur, steps=400, lr=0.012, T0=0.010, T1=0.002)
                cur_val = _max_cos(cur)
            else:
                idx = rng.integers(len(elite))
                cur = elite[idx][1].copy()
                cur_val = elite[idx][0]
            stall = 0
        if time.time() - t0 > EXPL + 1:
            break
    # final intensive polish of elites
    elite.sort(key=lambda z: z[0])
    # quick extra relax for top 2 within time
    for k in range(min(2, len(elite))):
        if time.time() - t0 > 150:
            break
        v0, x0 = elite[k]
        try:
            x1 = _relax_sm(x0.copy(), steps=900, lr=0.007, T0=0.004, T1=0.0006)
            v1 = _max_cos(x1)
            if v1 < v0:
                elite[k] = (v1, x1)
        except Exception:
            pass
    elite.sort(key=lambda z: z[0])
    best = elite[0][1].copy()
    best_val = elite[0][0]
    remain = 172.0 - (time.time() - t0)
    if remain > 6:
        lp_time = min(max(5.0, remain - 3.0), 28.0)
        try:
            best = _lp_polish(best, lp_time)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(bad.sum()), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
