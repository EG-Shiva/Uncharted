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
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-05:
            T = 8e-05
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
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
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _greedy_hole(prop, victim, rng, C=5000, K=8):
    d = prop.shape[1]
    cand = _norm(rng.normal(size=(C, d)))
    Dots = cand @ prop.T
    Dots[:, victim] = -2.0
    md = Dots.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    if rng.random() < 0.5:
        pick = idx[np.argmin(md[idx])]
    else:
        pick = idx[rng.integers(K)]
    return cand[pick]

def _lp_polish(x, seconds, deadline=1e18, topk=3200):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-03
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-13:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        # keep plateau: at least 3e-7 band plus top-K
        thr = max(25.0 * radius, 4e-07)
        sel = c > m - thr
        # ensure topk included
        if np.sum(sel) < topk:
            order = np.argpartition(-c, min(topk, len(c) - 1))[:topk] if False else None)
            # simpler: argsort topk
            k = min(topk, len(c))
            tb = np.argpartition(-c, k - 1)[:k]
            sel2 = np.zeros_like(sel)
            sel2[tb] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > topk:
            # keep tightest topk
            cv = c[sel]
            od = np.argpartition(-cv, topk - 1)[:topk]
            I, J = I[od], J[od]
            cvals = cv[od]
        else:
            cvals = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        if v < best - 1e-12:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 3e-07:
                # try one more tiny step then exit tightening
                pass
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
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
    # 1) corrected LP on loaded best: unlock plateau
    try:
        if time.time() < DEAD - 30:
            nb = _lp_polish(best, 6.0, deadline=DEAD - 25)
            vb = _max_cos(nb)
            if vb < best_val:
                best, best_val = nb, vb
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    # 2) few fresh random restarts to change basin
    try:
        for rs in range(4):
            if time.time() > t0 + 45:
                break
            if time.time() > DEAD - 40:
                break
            x0 = _norm(rng.normal(size=(n, d)))
            x0 = _relax(x0, steps=700, lr=0.020, T0=0.016, T1=0.0012)
            x0 = _relax(x0, steps=500, lr=0.010, T0=0.003, T1=0.00025)
            v0 = _max_cos(x0)
            if v0 < cur_val + 0.015:
                x0p = _lp_polish(x0, 1.5, deadline=DEAD - 25)
                v0p = _max_cos(x0p)
                if v0p < v0:
                    x0, v0 = x0p, v0p
            if v0 < best_val:
                best, best_val = x0.copy(), v0
            if v0 < cur_val:
                cur, cur_val = x0.copy(), v0
    except Exception:
        pass
    # ensure cur is best for hopping
    cur = best.copy()
    cur_val = best_val
    EXPL_END = t0 + 135.0
    if EXPL_END > DEAD - 30:
        EXPL_END = DEAD - 30
    it = 0
    while time.time() < EXPL_END:
        it += 1
        elapsed = time.time() - t0
        frac = max(0.0, (elapsed) / max(EXPL_END - t0, 1.0))
        T_acc = 0.0022 * (1.0 - frac) + 0.0002
        r = rng.random()
        prop = cur.copy()
        if r < 0.50:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            rowmax = G.max(axis=1)
            # victims: mix of jammed and random rattlers
            nj = int(rng.integers(3, 9))
            # jammed = largest rowmax
            jam = np.argsort(-rowmax)[:max(nj, 10)]
            nk = int(rng.integers(3, 8))
            if rng.random() < 0.3:
                nk = int(rng.integers(8, 16))
            victims = rng.choice(jam, size=min(nk, len(jam)), replace=False)
            # add 1-2 pure random to diversify
            if rng.random() < 0.5:
                extra = rng.choice(n, size=1, replace=False)
                victims = np.unique(np.concatenate([victims, extra]))
            for vv in victims:
                prop[int(vv)] = _greedy_hole(prop, int(vv), rng, C=5000, K=10)
            prop = _relax(prop, steps=500, lr=0.016, T0=0.012, T1=0.0015)
            prop = _relax(prop, steps=400, lr=0.009, T0=0.0025, T1=0.0002)
        elif r < 0.75:
            sigma = 0.08 + rng.random() * 0.22
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=600, lr=0.016, T0=0.012, T1=0.0008)
            prop = _relax(prop, steps=300, lr=0.008, T0=0.002, T1=0.0002)
        else:
            # block reinit 8-25 points
            nb = int(rng.integers(8, 26))
            victims = rng.choice(n, size=nb, replace=False)
            prop[victims] = _norm(rng.normal(size=(nb, d)))
            prop = _relax(prop, steps=650, lr=0.018, T0=0.014, T1=0.0010)
            prop = _relax(prop, steps=350, lr=0.009, T0=0.0025, T1=0.0002)
        val = _max_cos(prop)
        if val < cur_val - 1e-12:
            cur, cur_val = prop.copy(), val
        else:
            if rng.random() < np.exp((cur_val - val) / T_acc):
                cur, cur_val = prop.copy(), val
        if val < best_val - 5e-06:
            # extra cold squeeze
            try:
                prop2 = _relax(prop, steps=500, lr=0.006, T0=0.0015, T1=0.00012)
                v2 = _max_cos(prop2)
                if v2 < val:
                    prop, val = prop2, v2
                    cur, cur_val = prop.copy(), val
            except Exception:
                pass
            # snap with corrected LP (cheap)
            try:
                if time.time() < EXPL_END - 4:
                    ps = _lp_polish(prop, 1.4, deadline=EXPL_END)
                    vs = _max_cos(ps)
                    if vs < val:
                        prop, val = ps, vs
                        cur, cur_val = prop.copy(), val
            except Exception:
                pass
            if val < best_val:
                best, best_val = prop.copy(), val
        if time.time() > EXPL_END:
            break
    # final polish
    remain = DEAD - time.time() - 1.5
    if remain > 5:
        try:
            lp_t = min(max(6.0, remain - 4.0), 28.0)
            best = _lp_polish(best, lp_t, deadline=DEAD - 1.0)
            best_val = _max_cos(best)
        except Exception:
            pass
        # ultra-cold adam to equalise then tiny LP
        try:
            if DEAD - time.time() > 6:
                best = _relax(best, steps=400, lr=0.004, T0=0.0008, T1=0.0001)
        except Exception:
            pass
        try:
            if DEAD - time.time() > 4:
                best = _lp_polish(best, min(6.0, DEAD - time.time() - 1.0), deadline=DEAD - 0.5)
        except Exception:
            pass
    best = _norm(best)
    bad = ~np.isfinite(best).all(axis=1)
    if np.any(bad):
        best[bad] = _norm(rng.normal(size=(int(np.sum(bad)), d)))
    best = _norm(best)
    return best
