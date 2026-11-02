import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _near_stats(X, thr):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    nnear = int(np.count_nonzero(c > thr))
    cnt = (G > thr).sum(axis=1)
    return nnear, cnt, float(c.max())

def _sparse_descent(X, cap, steps, alpha=35.0, tau=3e-4, wc=1.0, ws=50.0, lr0=0.02, lr1=0.003):
    n, d = X.shape
    thr = cap - 0.002
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        # smooth count gradient
        Z = (G - thr) / tau
        # stable sigmoid
        S = np.empty_like(G)
        # clip Z for exp
        Zc = np.clip(Z, -30, 30)
        S = 1.0 / (1.0 + np.exp(-Zc))
        dC = S * (1.0 - S) / tau
        np.fill_diagonal(dC, 0.0)
        # high-temp softmax gradient
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum()) + 1e-30
        dS = W / sw
        Wtot = wc * dC + ws * dS
        # symmetrize (already sym)
        Grad = Wtot @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt) / n) + 1e-12
        X = X - (lr / (gn + 1e-12) * 0.02 + lr * 0.0) * Gt
        # scale: normalize step by mean row norm
        # above: use global norm; simpler fixed scaling
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _low_stress_dir(X, idx, active_thr):
    n, d = X.shape
    xi = X[idx]
    G = X @ X.T
    # active neighbours
    nbrs = np.where(G[idx] > active_thr)[0]
    nbrs = nbrs[nbrs != idx]
    if len(nbrs) == 0:
        # random tangent
        v = np.random.standard_normal(d)
        v -= np.dot(v, xi) * xi
        nv = np.linalg.norm(v) + 1e-12
        return v / nv
    A = X[nbrs] - G[idx, nbrs][:, None] * xi  # (m,d) tangent rows
    # SVD, smallest right singular vector
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        v = Vt[-1].copy()
    except Exception:
        v = np.random.standard_normal(d)
    # project to tangent at xi (remove normal comp = xi direction which is null)
    v -= np.dot(v, xi) * xi
    nv = float(np.linalg.norm(v))
    if not np.finite(nv) or nv < 1e-12:
        v = np.random.standard_normal(d)
        v -= np.dot(v, xi) * xi
        nv = float(np.linalg.norm(v)) + 1e-12
    return v / nv

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-14:
        it += 1
        if it > 60:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.argpartition(-c, 700)[:700]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((58, 22))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)
    init_val = float(best_val)
    n, d = 58, 22
    cur = best.copy()
    cur_val = float(best_val)

    deltas = [1e-3, 3e-4, 1e-4, 0.0]
    # stage configs: steps, lr range
    for di, delta in enumerate(deltas):
        if time.time() > T_END - 35:
            break
        cap = init_val + delta
        thr = cap - 0.002
        # try a few restarts per delta
        ntry = 3 if delta != 0.0 else 4
        for t in range(ntry):
            if time.time() > T_END - 32:
                break
            # start: perturb cur/best
            base = cur.copy() if rng.random() < 0.7 else best.copy()
            # initial kick: global noise + stress-directed kicks
            sig = 0.012 if delta >= 3e-4 else 0.006
            if t == 0 and di == 0:
                sig = 0.015
            y = base + sig * rng.standard_normal(base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # identify heavily loaded contacts via degrees at current max
            try:
                G0 = y @ y.T
                np.fill_diagonal(G0, -2.0)
                m0 = float(G0.max())
                cnt0 = (G0 > m0 - 0.004).sum(axis=1)
                order = np.argsort(-cnt0, kind="stable")
                # perturb top 4 hubs along low-stress dirs
                nhub = 4 if delta > 0 else 3
                for hh in range(nhub):
                    idx = int(order[hh % n])
                    # active threshold
                    athr = float((y @ y.T)[idx].max()) - 0.004 if False else m0 - 0.004
                    vdir = _low_stress_dir(y, idx, m0 - 0.004)
                    step = (0.02 if delta >= 3e-4 else 0.01) * (1.0 if rng.random() < 0.5 else -1.0)
                    # random magnitude jitter
                    step *= 0.7 + 0.6 * rng.random()
                    y[idx] = y[idx] + step * vdir
                    y[idx] /= np.linalg.norm(y[idx]) + 1e-16
            except Exception:
                pass
            # sparsifying descent
            steps = 700 if delta != 0.0 else 900
            # reduce steps if late
            if time.time() > T_END - 60 and steps > 500:
                steps = 500
            try:
                y = _sparse_descent(y, cap, steps, alpha=35.0, tau=3e-4, wc=1.0, ws=60.0,
                                    lr0=0.025 if delta > 0 else 0.015, lr1=0.003)
            except Exception:
                continue
            # extra stress kicks + short re-tighten (creates rattlers then re-tightens)
            try:
                for _ in range(2):
                    G1 = y @ y.T
                    np.fill_diagonal(G1, -2.0)
                    m1 = float(G1.max())
                    # find most loaded pair
                    k = int(np.argmax(G1))
                    i = k // n; j = k % n
                    # perturb its endpoints along low-stress dirs
                    for idx in (i, j):
                        if rng.random() < 0.8:
                            vd = _low_stress_dir(y, idx, m1 - 0.004)
                            st = 0.008 + 0.008 * rng.random()
                            if rng.random() < 0.5:
                                st = -st
                            y[idx] = y[idx] + st * vd
                            y[idx] /= np.linalg.norm(y[idx]) + 1e-16
                    y = _sparse_descent(y, cap, 150, alpha=40.0, tau=2.5e-4, wc=1.0, ws=80.0,
                                        lr0=0.008, lr1=0.002)
            except Exception:
                pass
            try:
                v = _max_cos(y)
                nn, cnt, _ = _near_stats(y, thr)
                nlow = int(np.count_nonzero(cnt < 10))
            except Exception:
                continue
            # accept if sparsified enough or improved max (allow rise up to cap)
            # we keep pool: if near-count small, LP-polish it even if max slightly above best
            do_polish = False
            if v < cur_val:
                cur = y.copy(); cur_val = v
                do_polish = True
            elif nn < 150 and nlow >= 3 and v < init_val + delta + 1e-4:
                # promising sparse topology: polish it
                do_polish = True
            elif v < init_val + delta and nn < 220 and t == 0:
                do_polish = True
            if do_polish:
                budget = 4.0 if delta != 0.0 else 6.0
                budget = min(budget, max(1.0, T_END - 30 - time.time()))
                if budget >= 1.0:
                    try:
                        yp = _lp_polish(y, budget, T_END - 28)
                        vp = _max_cos(yp)
                        if vp < cur_val:
                            cur = yp.copy(); cur_val = vp
                        if vp < best_val:
                            best = yp.copy(); best_val = vp
                        # even if not better than cur, keep sparse polished for next stage
                        if nn < 150 and vp < init_val + 5e-4:
                            cur = yp.copy(); cur_val = vp
                    except Exception:
                        pass
            if cur_val < best_val:
                best = cur.copy(); best_val = cur_val
            # early exit if new record by margin
            if best_val < init_val - 2e-6:
                pass
        # end tries for this delta: tighten cur toward best if cur worse
        if cur_val > best_val + 0.001:
            cur = best.copy(); cur_val = float(best_val)
        # LP-polish stage best briefly before decreasing delta
        try:
            b = min(4.0, max(1.0, T_END - 30 - time.time()))
            if b >= 1.0 and time.time() < T_END - 32:
                curp = _lp_polish(cur, b, T_END - 28)
                vp = _max_cos(curp)
                if vp < cur_val:
                    cur = curp; cur_val = vp
                if vp < best_val:
                    best = curp.copy(); best_val = vp
        except Exception:
            pass
    # final polish with remaining budget
    try:
        budget = min(28.0, max(4.0, T_END - time.time() - 2.0))
        if budget > 1.0:
            best = _lp_polish(best, budget, T_END - 1.0)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    for i in range(best.shape[0]):
        nr = float(np.linalg.norm(best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            vv = rng.standard_normal(d)
            vv /= np.linalg.norm(vv)
            best[i] = vv
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
