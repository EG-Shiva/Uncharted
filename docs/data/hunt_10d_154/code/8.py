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

def _relax_softmax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / max(T, 1e-12))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
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

def _tighten(x, steps, lr0):
    # hard-contact subgradient descent on near-max pairs
    n, d = x.shape
    lr = lr0
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # shrinking window focuses to true max
        frac = s / max(steps - 1, 1)
        thr = m - (0.004 * (1.0 - frac) + 0.0004)
        M = G > thr
        # grad = sum of contacting neighbours
        Grad = M @ x
        cnt = M.sum(axis=1, keepdims=True) + 1.0
        Grad = Grad / np.sqrt(cnt)
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip extreme
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # normalize to avoid blowup on overloaded points
        Gt = Gt / np.maximum(gn, 1.0)
        x = x - lr * Gt
        x = _norm(x)
        lr *= 0.9985
    return x

def _hole_fill(prop, victims, rng, C=8000, K=8):
    n, d = prop.shape
    for vv in victims:
        # chunk candidates to limit memory
        best_pick = None
        best_md = 1e9
        top_cand = None
        # draw in two chunks
        for _ in range(2):
            Cc = C // 2
            cand = _norm(rng.normal(size=(Cc, d)))
            Dots = cand @ prop.T
            Dots[:, vv] = -2.0
            md = Dots.max(axis=1)
            # keep K best of this chunk
            idx = np.argpartition(md, K)[:K]
            # update global best
            j = idx[np.argmin(md[idx])]
            if md[j] < best_md:
                best_md = float(md[j])
                best_pick = cand[j].copy()
            if top_cand is None:
                top_cand = cand[idx]
                top_md = md[idx]
            else:
                top_cand = np.vstack([top_cand, cand[idx]])
                top_md = np.concatenate([top_md, md[idx]])
        # stochastic pick among pooled top candidates
        if rng.random() < 0.5:
            # random among top 2K
            kk = top_cand.shape[0]
            pick = top_cand[rng.integers(kk)]
        else:
            pick = best_pick
        prop[vv] = pick
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
        m = float(c.max())
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
        except Exception:
            break
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
                    best = a
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    EXPL = 133.0
    it = 0
    while time.time() - t0 < EXPL:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / EXPL
        T_acc = 0.0016 * (1.0 - frac) + 0.00022
        if T_acc < 0.0002:
            T_acc = 0.0002
        r = rng.random()
        # contact counts for targeting
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(Gc.max())
        delta = 0.002 + rng.random() * 0.004
        cnt = (Gc > mcur - delta).sum(axis=1)
        if r < 0.60:
            nk = int(rng.integers(1, 4))  # 1-3
            probs = cnt.astype(float) + 0.5
            probs /= probs.sum()
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            prop = cur.copy()
            prop = _hole_fill(prop, victims, rng, C=8000, K=8)
            prop = _relax_softmax(prop, steps=480, lr=0.016, T0=0.013, T1=0.0012)
            prop = _tighten(prop, steps=140, lr0=0.006)
        elif r < 0.80:
            sigma = 0.02 + rng.random() * 0.07
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax_softmax(prop, steps=520, lr=0.015, T0=0.012, T1=0.0010)
            prop = _tighten(prop, steps=140, lr0=0.005)
        else:
            # cluster shake around most jammed point
            w = int(np.argmax(cnt))
            # nearest neighbours of w
            row = Gc[w].copy()
            row[w] = -2.0
            nn = np.argsort(-row)[:6]
            group = np.concatenate([[w], nn])
            # sometimes include random rattler
            ratt = np.where(cnt == 0)[0]
            prop = cur.copy()
            sigma = 0.015 + rng.random() * 0.03
            prop[group] = prop[group] + rng.normal(scale=sigma, size=(len(group), d))
            if len(ratt) > 0 and rng.random() < 0.5:
                v2 = rng.choice(ratt)
                # move rattler a bit too to explore hole
                prop[v2] = prop[v2] + rng.normal(scale=0.05, size=d)
            prop = _norm(prop)
            prop = _relax_softmax(prop, steps=420, lr=0.013, T0=0.009, T1=0.0009)
            prop = _tighten(prop, steps=160, lr0=0.005)
        val = _max_cos(prop)
        if val < cur_val - 1e-12:
            cur = prop
            cur_val = val
        else:
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur = prop
                cur_val = val
        if val < best_val - 1e-12:
            # extra squeeze on record
            try:
                prop2 = _relax_softmax(prop, steps=700, lr=0.008, T0=0.004, T1=0.0006)
                prop2 = _tighten(prop2, steps=220, lr0=0.003)
                val2 = _max_cos(prop2)
            except Exception:
                val2 = val
                prop2 = prop
            if val2 < val:
                prop = prop2
                val = val2
                cur = prop.copy()
                cur_val = val
            best = prop.copy()
            best_val = val
        if time.time() - t0 > EXPL + 2:
            break
    remain = 172.0 - (time.time() - t0)
    if remain > 6:
        lp_time = max(5.0, remain - 3.0)
        lp_time = min(lp_time, 40.0)
        try:
            best = _lp_polish(best, lp_time)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = _norm(rng.normal(size=(int(np.sum(bad)), d)))
        best[bad] = rb
    best = _norm(best)
    return best
