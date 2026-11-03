import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_deadline(x, deadline, r_init=0.004, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    bnds = [(-radius, radius)] * (n * d) + [(None, None)]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-12:
                break
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2*d] = J[:, None] * d + np.arange(d)
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        # update bounds for current radius
        bnds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
            radius *= 0.5
            continue
        y = y / nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def adam_quench(x, steps=400, a0=40.0, a1=280.0, lr=0.003):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        if not np.all(np.isfinite(Gt)):
            break
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x = x / nr
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def hole_fill(cand, victims, rng, ntrials=2048):
    n, d = cand.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    cur = cand[keep_mask].copy()
    for j in victims:
        trials = rng.standard_normal((ntrials, d))
        trials /= np.linalg.norm(trials, axis=1, keepdims=True)
        dots = trials @ cur.T
        mm = dots.max(axis=1)
        pick = int(np.argmin(mm))
        # small local refinement of pick: few perturbations around best
        best_t = trials[pick]
        best_v = float(mm[pick])
        for _ in range(3):
            pert = best_t[None, :] + 0.05 * rng.standard_normal((256, d))
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
            dd = (pert @ cur.T).max(axis=1)
            bi = int(np.argmin(dd))
            if float(dd[bi]) < best_v:
                best_v = float(dd[bi]); best_t = pert[bi]
        cand[j] = best_t
        cur = np.vstack([cur, best_t[None, :]])
    return cand

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LOOP_END = t0 + 152.0
    FINAL_RES = 18.0
    rng = np.random.default_rng(42)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    # initial squeeze
    if time.time() < HARD_END - 5:
        dl = min(time.time() + 22.0, HARD_END - 3)
        best_X = lp_deadline(best_X, dl, r_init=0.004)
        best_v = max_cos(best_X)
    curr_X = best_X.copy()
    curr_v = best_v
    it = 0
    while time.time() < LOOP_END:
        it += 1
        if time.time() > HARD_END - FINAL_RES - 8:
            break
        # choose victims: mostly worst per-point max, sometimes random
        ppm, gm = per_point_max(curr_X)
        cycle = it % 5
        if cycle == 0:
            k = 1
        elif cycle in (1, 3):
            k = 2
        else:
            k = 3
        # 80% worst, 20% random among top-20
        if rng.random() < 0.8:
            victims = np.argpartition(-ppm, k)[:k]
        else:
            top = np.argpartition(-ppm, 20)[:20]
            victims = rng.choice(top, size=k, replace=False)
        cand = curr_X.copy()
        # hole reinsert
        cand = hole_fill(cand, victims, rng, ntrials=2048)
        # extra shake of neighbours slightly to break symmetry
        shake_idx = rng.choice(n, size=6, replace=False)
        cand[shake_idx] += 0.02 * rng.standard_normal((6, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # quench
        steps = 350 if k <= 2 else 450
        cand = adam_quench(cand, steps=steps)
        vq = max_cos(cand)
        # quick screen: only LP if promising (within 0.01 of best)
        if vq > best_v + 0.012:
            # too bad, perturb curr slightly and continue
            continue
        dl = min(time.time() + 5.5, LOOP_END + 2.0, HARD_END - FINAL_RES)
        if dl - time.time() < 1.0:
            break
        cand = lp_deadline(cand, dl, r_init=0.003, cap=2400)
        v = max_cos(cand)
        if v < curr_v - 1e-12:
            curr_X, curr_v = cand, v
        # accept sideways with small prob to move basin
        elif v < curr_v + 0.001 and rng.random() < 0.15:
            curr_X, curr_v = cand, v
        if v < best_v:
            best_X, best_v = cand.copy(), v
            curr_X, curr_v = cand.copy(), v
        # if stuck, restart from best with shake
        if it % 7 == 0 and best_v < curr_v - 1e-9:
            curr_X = best_X.copy(); curr_v = best_v
    # final squeeze on best
    remain = HARD_END - time.time()
    if remain > 3:
        best_X = lp_deadline(best_X, time.time() + min(remain - 1.0, 16.0), r_init=0.0025, cap=2800)
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
