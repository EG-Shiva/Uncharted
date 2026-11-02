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

def _contacts(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    np.fill_diagonal(G, -np.inf)
    pw = G.max(axis=1)
    return C, m, pw, G

def _lp_polish(x, seconds, r0=4e-3, cap=700):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 40:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        if c.size > cap:
            part = np.argpartition(c, -cap)[-cap:]
            sel_idx = part[c[part] > thresh]
            if sel_idx.size < 50:
                sel_idx = part[np.argsort(c[part])[-50:]]
        else:
            sel_idx = np.where(c > thresh)[0]
        if sel_idx.size == 0:
            break
        I = iuI[sel_idx]; J = iuJ[sel_idx]
        p = len(I)
        csel = c[sel_idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            remain = seconds - (time.time() - start)
            tl = min(1.5, max(0.3, remain))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
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
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def _refine_single(p0, Y, steps=100, T0=0.02, T1=0.002, lr0=0.2, lr1=0.02, mu=0.5):
    p = p0 / (np.linalg.norm(p0) + 1e-12)
    vel = np.zeros(3 if False else p.shape)
    # use tangent momentum
    v = np.zeros_like(p)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        dots = Y @ p
        m = float(dots.max())
        w = np.exp((dots - m) / max(T, 1e-12))
        w = w / (w.sum() + 1e-12)
        grad = w @ Y
        gt = grad - np.dot(grad, p) * p
        nrm = np.linalg.norm(gt) + 1e-12
        gt = gt / nrm
        v = mu * v + (1.0 - mu) * gt
        p = p - lr * v
        p = p / (np.linalg.norm(p) + 1e-12)
    return p

def _best_insertion(Y, rng, n_samples=20000, n_best=6):
    d = Y.shape[1]
    S = rng.standard_normal((n_samples, d))
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    dots = S @ Y.T
    md = dots.max(axis=1)
    idx = np.argpartition(md, n_best)[:n_best]
    best_p = None
    best_v = 2.0
    for j in idx:
        pr = _refine_single(S[j], Y, steps=90)
        v = float((Y @ pr).max())
        if v < best_v:
            best_v = v
            best_p = pr
    # also refine the raw hole (argmin) a bit longer for safety
    return best_p

def _adam_all(X, steps, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        mx = float(G.max())
        E = (G - mx) / max(T, 1e-12)
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (k + 1)
        c2 = 1 - b2 ** (k + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        # normalize per-row to keep stable
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn, 1.0)
        X = X - lr * step
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    HARD_STOP = 150.0
    DEADLINE = 172.0
    rng = np.random.default_rng(20261114)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((54, 21))
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21):
        try:
            xb = _norm(np.asarray(xb, dtype=float).reshape(54, 21))
        except Exception:
            xb = _norm(rng.standard_normal((54, 21)))
    try:
        xr = _norm(np.asarray(np.load("record.npy"), dtype=float).reshape(54, 21))
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    n, d = 54, 21
    best_val = _max_cos(xb)
    C0, m0, _, _ = _contacts(xb)
    def chash(C):
        return tuple(sorted(np.asarray(C).tolist()))
    pool = [(best_val, xb.copy(), chash(C0))]
    hashes = set([chash(C0)])
    pool_best = best_val
    pool_best_X = xb.copy()
    it = 0
    while time.time() - t0 < HARD_STOP:
        it += 1
        remain_hard = HARD_STOP - (time.time() - t0)
        if remain_hard < 3.0:
            break
        rb_start = time.time()
        # pick base: mostly pool best, every 5th random elite
        if len(pool) > 1 and (it % 5 == 0):
            bi = int(rng.integers(0, len(pool)))
            base = pool[bi][1]
        else:
            base = pool_best_X
        Cb, mb, pwb, Gb = _contacts(base)
        # seed choice rotating high / low / random
        rmode = it % 3
        if rmode == 1:
            seed = int(np.argmax(Cb))
        elif rmode == 2:
            seed = int(np.argmin(Cb))
        else:
            seed = int(rng.integers(0, n))
        K = 7 + (it % 7)  # 7..13
        # neighbours of seed by dot (contact graph)
        row = Gb[seed].copy()
        row[seed] = -np.inf
        order = np.argsort(-row)
        # order excludes seed; take top K (K includes seed? we delete seed+K neighbours -> total K+1?)
        # spec: delete seed together with 7-13 neighbours => total 8-14. Use K neighbours.
        nb = order[:K].tolist()
        del_idx = [seed] + nb
        del_idx = sorted(set(del_idx))
        # keep mask
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        kept = base[keep_mask].copy()
        # reinsert one at a time, kept frozen
        inserted = []
        cur = kept
        ok = True
        for vv in range(len(del_idx)):
            # time guard: never let one rebuild exceed ~8s
            if time.time() - rb_start > 8.0:
                ok = False
                break
            if len(inserted) == 0:
                Y = cur
            else:
                Y = np.vstack([cur] + inserted)
            # anneal not needed here; _best_insertion handles it
            ns = 20000
            # shrink sample if late to save time
            if remain_hard < 20:
                ns = 12000
            try:
                p_new = _best_insertion(Y, rng, n_samples=ns, n_best=6)
            except Exception:
                ok = False
                break
            inserted.append(p_new.reshape(1, -1))
            if time.time() - t0 >= HARD_STOP - 1.0:
                ok = False
                break
        if not ok:
            continue
        cand = np.vstack([cur] + inserted)
        # cand should be 54 x 21 (order changed, fine)
        # 500-1000 Adam steps on all points
        steps = 600 if (remain_hard > 30) else 450
        # vary temperature slightly per iteration for diversity
        if it % 2 == 0:
            cand = _adam_all(cand, steps=steps, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003)
        else:
            cand = _adam_all(cand, steps=steps, T0=0.015, T1=0.0015, lr0=0.025, lr1=0.0025)
        if time.time() - rb_start > 8.0:
            # skip LP, evaluate anyway
            v = _max_cos(cand)
            lp_done = False
        else:
            # short LP polish, budget to fit ~8s rebuild
            used = time.time() - rb_start
            budget = min(1.6, max(0.4, 7.5 - used))
            # also respect hard stop
            budget = min(budget, HARD_STOP - (time.time() - t0) - 0.5)
            if budget > 0.35:
                try:
                    cand = _lp_polish(cand, seconds=budget, r0=3e-3, cap=650)
                except Exception:
                    pass
            v = _max_cos(cand)
        # pool update
        try:
            Cc, _, _, _ = _contacts(cand)
            h = chash(Cc)
        except Exception:
            h = None
        if v < pool_best - 1e-12:
            pool_best = v
            pool_best_X = cand.copy()
            if h is not None and h not in hashes:
                hashes.add(h)
            pool.append((v, cand.copy(), h))
            pool = sorted(pool, key=lambda z: z[0])[:5]
            # rebuild hashes set
            hashes = set([z[2] for z in pool if z[2] is not None])
        elif v < pool_best + 1e-4 and h is not None and h not in hashes:
            # accept new topology within tolerance
            pool.append((v, cand.copy(), h))
            pool = sorted(pool, key=lambda z: z[0])[:5]
            hashes = set([z[2] for z in pool if z[2] is not None])
            # if pool truncated and best still there, keep pool_best
            pool_best = min(pool_best, min(z[0] for z in pool))
            # ensure pool_best_X corresponds
            for z in pool:
                if abs(z[0] - pool_best) < 1e-15:
                    pool_best_X = z[1].copy()
                    break
        # opportunistic: if pool worst is much worse, keep size 5
    # hard-stop: LP-polish pool best with remaining time
    remain = DEADLINE - (time.time() - t0)
    if remain > 2.0:
        try:
            pool_best_X = _lp_polish(pool_best_X, seconds=max(2.0, remain - 1.0), r0=2e-3, cap=800)
        except Exception:
            pass
    pool_best_X = _norm(pool_best_X)
    return pool_best_X
