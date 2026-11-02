import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return cnt, float(m)

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _anneal(x0, movable, t_target, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    mov = np.asarray(movable, dtype=bool)
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        return x
    lrb = np.log(lr1 / lr0) / (steps - 1)
    beb = np.log(beta1 / beta0) / (steps - 1)
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    inv = np.zeros(n, dtype=bool)
    for s in range(steps):
        beta = beta0 * np.exp(beb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        z = beta * (g - t_target)
        z = np.clip(z, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        # only pairs involving a moved point
        mask = mov[Iu] | mov[Ju]
        sig = sig * mask
        W = np.zeros((n, n))
        W[Iu, Ju] = sig
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        gt[~mov] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mh = m1 / c1; vh = v1 / c2
        step = mh / (np.sqrt(vh) + eps)
        x[mov] = x[mov] - lr * step[mov]
        nr = np.linalg.norm(x[mov], axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x[mov] /= nr
    return x

def _void_refill(existing_list, rng, n_try=20000):
    # existing: (m,4) array
    ref = existing_list
    # batch sampling to limit memory
    best_val = 2.0
    best_c = None
    B = 5000
    # generate in chunks and keep global best
    for a in range(0, n_try, B):
        b = min(B, n_try - a)
        cand = rng.normal(size=(b, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ ref.T
        mx = D.max(axis=1)
        j = int(np.argmin(mx))
        if mx[j] < best_val:
            best_val = float(mx[j])
            best_c = cand[j].copy()
    c = best_c
    # local refinement: hill climb on max-dot
    cur_v = best_val
    sig = 0.06
    for it in range(250):
        prop = c + rng.normal(0, sig, size=4)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        v = float((prop @ ref.T).max())
        if v < cur_v:
            c, cur_v = prop, v
            sig *= 1.02
        else:
            sig *= 0.985
        if sig < 2e-4:
            sig = 2e-4
    return c

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    patch_K = [7, 11, 15]
    deltas = [1e-4, 3e-4, 1e-3]
    trial = 0
    # time budget: leave ~12s for final polish
    while time.time() - t0 < 148.0:
        trial += 1
        rem = 160.0 - (time.time() - t0)
        if rem < 12:
            break
        cnt, mx = _contacts(x_best, tol=1e-7)
        ratt = np.where(cnt == 0)[0]
        order_hub = np.argsort(-cnt)
        if trial == 1 and len(ratt) > 0:
            hub = int(ratt[0])
        elif trial % 3 == 0 and len(ratt) > 0:
            hub = int(ratt[rng.integers(len(ratt))])
        elif trial % 3 == 1:
            top = order_hub[:10]
            hub = int(top[rng.integers(len(top))])
        else:
            hub = int(rng.integers(n))
        k = patch_K[(trial - 1) % len(patch_K)]
        delta = deltas[(trial - 1) % len(deltas)]
        t_target = min(0.849102, best_val) - delta
        dots_hub = x_best[hub] @ x_best.T
        order = np.argsort(-dots_hub)
        del_idx = order[:k + 1].copy()
        keep = np.ones(n, dtype=bool)
        keep[del_idx] = False
        frozen = x_best[keep].copy()
        # reinsert one at a time at largest voids
        new_pts = []
        for s in range(len(del_idx)):
            if new_pts:
                ref = np.vstack([frozen] + new_pts)
            else:
                ref = frozen
            c = _void_refill(ref, rng, n_try=20000)
            new_pts.append(c.reshape(1, 4))
        y = np.empty_like(x_best)
        y[keep] = frozen
        y[del_idx] = np.vstack(new_pts)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        mov = np.zeros(n, dtype=bool)
        mov[del_idx] = True
        # optimize only reinserted at target
        y = _anneal(y, mov, t_target, steps=350, beta0=60.0, beta1=600.0, lr0=0.008, lr1=0.0008)
        # short all-free anneal
        allmov = np.ones(n, dtype=bool)
        y = _anneal(y, allmov, t_target + delta * 0.25, steps=150, beta0=80.0, beta1=500.0, lr0=0.003, lr1=0.0004)
        lb = min(4.5, max(1.5, rem - 8.0))
        # cap per-trial LP to keep cycling
        lb = min(lb, 4.0)
        y, v = _lp_polish(y, lb)
        if v < best_val - 1e-12:
            x_best, best_val = y, v
            continue
        if v < best_val + 0.002:
            # extra all-free anneal since contact graph may have changed
            if time.time() - t0 > 152.0:
                continue
            y2 = _anneal(y, allmov, min(best_val - 5e-5, v - 1e-4), steps=250, beta0=100.0, beta1=800.0, lr0=0.002, lr1=0.0003)
            rem2 = 160.0 - (time.time() - t0)
            if rem2 < 3:
                continue
            y2, v2 = _lp_polish(y2, min(4.0, max(1.0, rem2 - 6.0)))
            if v2 < best_val - 1e-12:
                x_best, best_val = y2, v2
        # small chance: keep slightly worse diverse base to escape? no, keep incumbent
    # final polish of incumbent
    remf = 178.0 - (time.time() - t0)
    if remf > 2:
        x_best, best_val = _lp_polish(x_best, min(remf - 1.0, 10.0))
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
