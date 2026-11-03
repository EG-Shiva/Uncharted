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
    b1p, b2p = 1.0, 1.0
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
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        if len(I) > 2500:
            # keep tightest only to bound LP size
            k = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[k], J[k]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:len(I)] if len(I)==np.sum(sel) else (x[I]*x[J]).sum(axis=1)*-1.0 + 0.0, A_eq=A_eq, b_eq=np.zeros(n),
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

def _lp_polish2(x, seconds, deadline=1e18):
    # correct b_ub handling (recompute for truncated set)
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    Gfull = None
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        c = cfull[sel]
        if len(I) > 2500:
            k = np.argpartition(-c, 2500)[:2500]
            I, J, c = I[k], J[k], c[k]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def _greedy_regrow(fixed, k_new, rng, C=6000):
    # fixed: (nf,d) frozen points; sequentially place k_new points to minimize max dot
    d = fixed.shape[1]
    placed = []
    for k in range(k_new):
        cand = _norm(rng.normal(size=(C, d)))
        if fixed.shape[0] > 0:
            md = cand @ fixed.T
            m0 = md.max(axis=1)
        else:
            m0 = np.full(C, -2.0)
        if len(placed) > 0:
            P = np.stack(placed, axis=0)
            md2 = cand @ P.T
            m1 = md2.max(axis=1)
            m = np.maximum(m0, m1)
        else:
            m = m0
        K = 8
        idx = np.argpartition(m, K)[:K]
        # random among top-K, bias to best
        if rng.random() < 0.5:
            pick = idx[np.argmin(m[idx])]
        else:
            pick = idx[rng.integers(K)]
        placed.append(cand[pick])
    if len(placed) == 0:
        return np.zeros((0, d))
    return np.stack(placed, axis=0)

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    EXPL_END = t0 + 138.0
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
    # initial polish to true basin
    try:
        if time.time() < t0 + 30:
            best = _lp_polish2(best, 3.0, deadline=DEAD)
            best_val = _max_cos(best)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    it = 0
    while time.time() < EXPL_END:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL_END - t0, 1e-9)
        T_acc = 0.0012 * (1.0 - frac) + 0.00008
        r = rng.random()
        prop = cur.copy()
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        if r < 0.45:
            # core breakout: delete top overloaded cluster
            delta = 0.003 + rng.random() * 0.005
            cnt = (G > mcur - delta).sum(axis=1)
            k = int(rng.integers(4, 10))  # 4-9
            victims = np.argsort(-cnt)[:k]
            # randomize: with prob replace one victim by random
            if rng.random() < 0.3:
                victims[rng.integers(k)] = rng.integers(n)
            victims = np.unique(victims)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            fixed = prop[keep_mask]
            regrown = _greedy_regrow(fixed, len(victims), rng, C=6000)
            prop[keep_mask] = fixed
            # place regrown
            vi = 0
            for vv in victims:
                prop[vv] = regrown[vi]
                vi += 1
            prop = _relax(prop, steps=420, lr=0.015, T0=0.012, T1=0.0022)
            prop = _relax(prop, steps=300, lr=0.009, T0=0.0035, T1=0.0007)
        elif r < 0.75:
            # cap reshuffle: pick random direction, take s nearest, regrow globally
            s = int(rng.integers(12, 26))
            center = _norm(rng.normal(size=(1, d)))[0]
            dots = cur @ center
            victims = np.argsort(-dots)[:s]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            fixed = prop[keep_mask]
            regrown = _greedy_regrow(fixed, s, rng, C=5000)
            vi = 0
            for vv in victims:
                prop[vv] = regrown[vi]
                vi += 1
            prop = _relax(prop, steps=450, lr=0.015, T0=0.013, T1=0.0020)
            prop = _relax(prop, steps=280, lr=0.008, T0=0.003, T1=0.0006)
        elif r < 0.90:
            # half-space rotation: split by random hyperplane, rotate one side
            u = _norm(rng.normal(size=(1, d)))[0]
            side = (cur @ u) > 0
            ns = int(side.sum())
            if ns < 10 or ns > n - 10:
                sigma = 0.03 + rng.random() * 0.05
                prop = cur + rng.normal(scale=sigma, size=cur.shape)
                prop = _norm(prop)
            else:
                # random small rotation in 10D
                A = rng.normal(size=(d, d))
                A = A - A.T
                ang = 0.05 + rng.random() * 0.25
                # exp approx via Cayley or matrix exp series: use QR of I+ang*A/2
                R = np.eye(d) + ang * A
                # orthogonalize via QR
                Q, _ = np.linalg.qr(R)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                prop[side] = (cur[side] @ Q.T)
                prop = _norm(prop)
                # jitter rotated side slightly
                prop[side] = _norm(prop[side] + rng.normal(scale=0.01, size=(ns, d)))
            prop = _relax(prop, steps=500, lr=0.014, T0=0.011, T1=0.0012)
        else:
            # global jitter + cold
            sigma = 0.015 + rng.random() * 0.04
            prop = cur + rng.normal(scale=sigma, size=cur.shape)
            prop = _norm(prop)
            prop = _relax(prop, steps=450, lr=0.013, T0=0.010, T1=0.0010)
        # bounded LP polish BEFORE judging (topology-changing evaluation)
        ptime = 0.7
        # leave headroom for final polish
        if time.time() > EXPL_END - 6:
            ptime = 0.35
        try:
            prop_p = _lp_polish2(prop, ptime, deadline=EXPL_END)
        except Exception:
            prop_p = prop
        val_p = _max_cos(prop_p)
        # acceptance on polished values
        if val_p < cur_val - 1e-12:
            cur, cur_val = prop_p, val_p
        else:
            if rng.random() < np.exp((cur_val - val_p) / max(T_acc, 1e-9)):
                cur, cur_val = prop_p, val_p
        if val_p < best_val - 1e-12:
            best, best_val = prop_p.copy(), val_p
        if time.time() > EXPL_END:
            break
    # final polish with remaining time
    remain = DEAD - time.time() - 1.5
    if remain > 4:
        try:
            best = _lp_polish2(best, min(max(6.0, remain - 1.0), 30.0), deadline=DEAD)
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
