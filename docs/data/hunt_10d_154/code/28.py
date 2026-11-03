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
        if T < 5e-5:
            T = 5e-5
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

def _holes(prop, victims, rng, C=5000, K=8):
    d = prop.shape[1]
    for vv in victims:
        cand = _norm(rng.normal(size=(C, d)))
        Dots = cand @ prop.T
        Dots[:, vv] = -2.0
        md = Dots.max(axis=1)
        idx = np.argpartition(md, K)[:K]
        if rng.random() < 0.45:
            pick = idx[np.argmin(md[idx])]
        else:
            pick = idx[rng.integers(K)]
        prop[vv] = cand[pick]
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
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest 4000 to bound LP size
            order = np.argsort(-c[sel])[:4000]
            I, J = I[order], J[order]
            p = 4000
            thr = float(c[sel][order[-1]])
            sel2 = np.zeros_like(sel)
            # rebuild b using threshold is complex; just use selected subset
            bsel = (x[I] * x[J]).sum(axis=1)
            b_ub = -bsel
        else:
            b_ub = -c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
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
    rng = np.random.default_rng(20261103)
    n, d = 154, 10
    DEAD = t0 + 170.0
    EXPL_END = t0 + 132.0
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
    # short initial snap to get true basin value
    try:
        if time.time() - t0 < 25:
            best = _lp_polish(best, 2.0, deadline=DEAD - 40)
            best_val = _max_cos(best)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    # diverse pool: 2 fast random restarts
    pool = [best.copy()]
    try:
        for _ in range(2):
            if time.time() > t0 + 30:
                break
            R = _norm(rng.normal(size=(n, d)))
            R = _relax(R, steps=500, lr=0.02, T0=0.02, T1=0.002)
            R = _relax(R, steps=300, lr=0.010, T0=0.004, T1=0.0008)
            pool.append(R.copy())
            v = _max_cos(R)
            if v < best_val - 1e-7:
                Rs = _lp_polish(R, 0.8, deadline=EXPL_END)
                vs = _max_cos(Rs)
                if vs < best_val:
                    best, best_val = Rs.copy(), vs
                    cur, cur_val = Rs.copy(), vs
                    pool[0] = Rs.copy()
    except Exception:
        pass
    it = 0
    while time.time() < EXPL_END:
        it += 1
        elapsed = time.time() - t0
        frac = max(0.0, min(1.0, (elapsed) / max(EXPL_END - t0, 1.0)))
        T_acc = 0.0022 * (1.0 - frac) + 0.00022
        r = rng.random()
        prop = cur.copy()
        if r < 0.34:
            # hub-to-hole, deliberately break core
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            delta = 0.002 + rng.random() * 0.006
            cnt = (G > mcur - delta).sum(axis=1)
            probs = cnt.astype(float) + 0.4
            probs /= probs.sum()
            nk = int(rng.integers(2, 5))
            if rng.random() < 0.15:
                nk = int(rng.integers(5, 8))
            victims = rng.choice(n, size=nk, replace=False, p=probs)
            prop = _holes(prop, victims, rng, C=5000, K=8)
            prop = _relax(prop, steps=550, lr=0.016, T0=0.014, T1=0.0022)
            prop = _relax(prop, steps=300, lr=0.009, T0=0.0035, T1=0.0008)
        elif r < 0.54:
            # dense-cap cluster excision: change point count locally
            u = _norm(rng.normal(size=(1, d)))[0]
            dots = cur @ u
            k = int(rng.integers(6, 13))
            victims = np.argsort(-dots)[:k]
            rng.shuffle(victims)
            prop = _holes(prop, victims, rng, C=4000, K=10)
            prop = _relax(prop, steps=600, lr=0.017, T0=0.015, T1=0.0020)
            prop = _relax(prop, steps=300, lr=0.009, T0=0.0035, T1=0.0008)
        elif r < 0.68:
            # rattler / low-contact refill + neighbour jitter
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            cnt = (G > mcur - 0.006).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:4]
            nk = min(len(ratt), int(rng.integers(1, 4)))
            victims = rng.choice(ratt, size=max(nk, 1), replace=False)
            prop = _holes(prop, victims, rng, C=7000, K=8)
            prop = _relax(prop, steps=550, lr=0.014, T0=0.010, T1=0.0010)
        elif r < 0.86:
            sigma = 0.025 + rng.random() * 0.06
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=600, lr=0.015, T0=0.012, T1=0.0011)
        else:
            # fresh diverse restart (escapes same basin)
            if rng.random() < 0.5 and len(pool) > 1:
                base = pool[rng.integers(len(pool))]
                sigma = 0.05 + rng.random() * 0.10
                prop = base + rng.normal(scale=sigma, size=base.shape)
                prop = _norm(prop)
            else:
                prop = _norm(rng.normal(size=(n, d)))
            prop = _relax(prop, steps=650, lr=0.020, T0=0.022, T1=0.0025)
            prop = _relax(prop, steps=300, lr=0.010, T0=0.004, T1=0.0008)
        val = _max_cos(prop)
        # fast path: relaxed already beats snapped best -> polish immediately
        if val < best_val - 2e-7:
            try:
                prop2 = _relax(prop, steps=700, lr=0.007, T0=0.0035, T1=0.0006)
                v2 = _max_cos(prop2)
                if v2 < val:
                    prop, val = prop2, v2
                if time.time() < EXPL_END - 2:
                    ps = _lp_polish(prop, 1.0, deadline=EXPL_END)
                    vs = _max_cos(ps)
                    if vs < val:
                        prop, val = ps, vs
            except Exception:
                pass
            if val < best_val:
                best, best_val = prop.copy(), val
            cur, cur_val = prop.copy(), val
            pool.append(prop.copy())
            if len(pool) > 6:
                pool.pop(1)
            continue
        # bounded LP snap judging: only snap if within gate
        gate = 0.010 * (1.0 - frac) + 0.003
        needs_snap = (val < cur_val + gate)
        # snap every ~4th promising proposal to save time
        if needs_snap and (val < cur_val or (it % 3 == 0)):
            if time.time() > EXPL_END - 3:
                # no time for snaps, relaxed SA only
                if val < cur_val or rng.random() < np.exp((cur_val - val) / max(T_acc, 1e-6)):
                    cur, cur_val = prop.copy(), val
                continue
            try:
                ps = _lp_polish(prop, 0.6, deadline=EXPL_END)
                vs = _max_cos(ps)
            except Exception:
                continue
            # acceptance on snapped values only (topology-changed basin value)
            if vs < cur_val - 1e-12:
                cur, cur_val = ps.copy(), vs
                if vs < best_val:
                    best, best_val = ps.copy(), vs
            else:
                if rng.random() < np.exp((cur_val - vs) / max(T_acc, 1e-6)):
                    cur, cur_val = ps.copy(), vs
            # track best relaxed that snapped well for diversity
            if vs < best_val + 0.004 and rng.random() < 0.10:
                pool.append(ps.copy())
                if len(pool) > 6:
                    pool.pop(1)
        else:
            # far worse: occasional relaxed SA to keep moving without LP cost
            if rng.random() < 0.02 * np.exp((cur_val - val) / max(T_acc, 1e-6)):
                cur, cur_val = prop.copy(), val
        if time.time() > EXPL_END:
            break
    remain = DEAD - time.time() - 2.0
    if remain > 4:
        try:
            best = _lp_polish(best, min(max(6.0, remain - 1), 32.0), deadline=DEAD - 1.0)
        except Exception:
            pass
    best = _norm(best)
    bad = ~np.isfinite(best).all(axis=1)
    if np.any(bad):
        best[bad] = _norm(rng.normal(size=(int(np.sum(bad)), d)))
    best = _norm(best)
    return best
