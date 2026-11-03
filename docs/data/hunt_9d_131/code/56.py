import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 3e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]
        p = I.shape[0]
        if p == 0:
            break
        if p > 6000:
            idx = np.where(sel)[0]
            o = np.argsort(-c[sel])[:6000]
            I = iu0[idx[o]]; J = iu1[idx[o]]
            csel = c[idx[o]]
            p = 6000
        else:
            csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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

def _anneal(x, steps=600, T0=0.02, T1=0.003, lr=0.01, seed_off=0):
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best_x = x.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for t in range(1, steps + 1):
        frac = t / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        G = x @ x.T
        g = G[iu0, iu1]
        mg = float(g.max())
        if mg < best_val:
            best_val = mg
            best_x = x.copy()
        e = np.exp((g - mg) / T)
        s = float(e.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        W = np.zeros((n, n), dtype=x.dtype)
        W[iu0, iu1] = e / s
        W = W + W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m1 / bc1
        vhat = v1 / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - delta
        x = _norm(x)
    # return best on trajectory
    return best_x

def _find_hole(Xk, rng, n_samples=25000):
    d = Xk.shape[1]
    Z = rng.standard_normal(size=(n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx].copy()
    # local refinement: minimize softmax max
    T = 0.008
    lr = 0.04
    for _ in range(150):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-300)
        g = W @ Xk
        gg = float(np.dot(g, z))
        gt = g - gg * z
        z = z - lr * gt
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z

def solve():
    t00 = time.time()
    BUDGET = 168.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        rng0 = np.random.default_rng(0)
        xb = rng0.standard_normal(size=(131, 9))
    if xb.shape != (131, 9):
        try:
            xb = np.reshape(xb, (131, 9))
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal(size=(131, 9))
    xb = _norm(xb)
    n, d = (131, 9)
    # ensure local optimum start
    xb = _lp_polish(xb, seconds=3.0)
    best = _mc(xb)
    cur = xb.copy()
    cur_val = best
    rng = np.random.default_rng(1)
    it = 0
    # precompute triu for contact counts
    while time.time() - t00 < BUDGET - 13.0:
        it += 1
        elapsed = time.time() - t00
        prog = elapsed / BUDGET
        action = it % 5
        Xt = cur.copy()
        if action == 0:
            # single hub -> hole
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m - 5e-7, axis=1)
            order = np.argsort(-cnt)
            # pick among top 15 randomly to diversify
            pool = order[:15]
            k = int(rng.integers(0, len(pool)))
            idx_move = int(pool[k])
            keep_mask = np.ones(n, dtype=bool); keep_mask[idx_move] = False
            Xk = cur[keep_mask]
            z = _find_hole(Xk, rng)
            Xt[idx_move] = z
            Xt = _norm(Xt)
            Xt = _anneal(Xt, steps=550, T0=0.018, T1=0.003, lr=0.009)
        elif action == 1:
            # triple hub -> holes sequentially
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m - 5e-7, axis=1)
            order = np.argsort(-cnt)
            pool = order[:20]
            chosen = rng.choice(pool, size=3, replace=False)
            keep_mask = np.ones(n, dtype=bool); keep_mask[chosen] = False
            Xk = cur[keep_mask]
            for qq, im in enumerate(chosen):
                z = _find_hole(Xk, rng, n_samples=15000)
                Xt[int(im)] = z
                Xk = np.vstack([Xk, z[None, :]])
            Xt = _norm(Xt)
            Xt = _anneal(Xt, steps=700, T0=0.022, T1=0.0035, lr=0.010)
        elif action == 2:
            # cap re-optimisation: perturb points in random cap strongly
            u = rng.standard_normal(size=(d,)); u = u / np.linalg.norm(u)
            dots = cur @ u
            sel_idx = np.where(dots > 0.35)[0]
            if sel_idx.size < 5:
                sel_idx = np.argsort(-dots)[:12]
            if sel_idx.size > 45:
                sel_idx = np.argsort(-dots)[:45]
            Xt[sel_idx] = Xt[sel_idx] + 0.22 * rng.standard_normal(size=(sel_idx.size, d))
            Xt = _norm(Xt)
            Xt = _anneal(Xt, steps=600, T0=0.020, T1=0.003, lr=0.010)
        elif action == 3:
            # global shake medium
            sigma = 0.09 if (it % 10 < 5) else 0.16
            Xt = Xt + sigma * rng.standard_normal(size=Xt.shape)
            Xt = _norm(Xt)
            Xt = _anneal(Xt, steps=700, T0=0.025, T1=0.004, lr=0.009)
        else:
            # rattler-targeted: move rattlers + one hub to holes
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m - 1e-9, axis=1)
            ratt = np.where(cnt == 0)[0]
            G2 = cur @ cur.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(G2.max())
            cnt2 = np.sum(G2 > m2 - 5e-7, axis=1)
            order = np.argsort(-cnt2)
            hub = int(order[0])
            mv = [hub]
            if ratt.size > 0:
                # move up to 2 rattlers as well
                nr = min(2, ratt.size)
                pick = rng.choice(ratt, size=nr, replace=False)
                mv.extend([int(v) for v in pick])
            keep_mask = np.ones(n, dtype=bool); keep_mask[np.array(mv)] = False
            Xk = cur[keep_mask]
            for im in mv:
                z = _find_hole(Xk, rng, n_samples=15000)
                Xt[int(im)] = z
                Xk = np.vstack([Xk, z[None, :]])
            Xt = _norm(Xt)
            Xt = _anneal(Xt, steps=650, T0=0.020, T1=0.003, lr=0.010)
        v = _mc(Xt)
        # decide LP polish if promising
        do_lp = (v < best + 0.006) and (time.time() - t00 < BUDGET - 13.0)
        if do_lp:
            Xt = _lp_polish(Xt, seconds=1.1)
            v = _mc(Xt)
        # update best
        if v < best - 1e-12:
            xb = Xt.copy(); best = v
            cur = Xt.copy(); cur_val = v
        else:
            # annealing acceptance for walk
            Temp = 0.0015 * (1.0 - prog) + 0.00015
            if v < cur_val or rng.random() < np.exp((cur_val - v) / Temp):
                cur = Xt.copy(); cur_val = v
        # occasional restart to best if wandering too far
        if cur_val > best + 0.008 and (it % 12 == 0):
            cur = xb.copy(); cur_val = best
    rem = BUDGET - (time.time() - t00) - 0.8
    if rem > 1.0:
        rem = min(rem, 11.0)
        xb = _lp_polish(xb, seconds=float(rem))
    xb = _norm(xb)
    return xb
