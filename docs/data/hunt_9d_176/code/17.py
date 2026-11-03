import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_quench(x, steps, a0=50.0, a1=320.0, lr=0.0025):
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
        Gt = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        x = x - lr * (1.0 - 0.4 * frac) * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def lp_squeeze(X, deadline, cap=2000, r_init=0.003):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < 40:
        it += 1
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def hole_reinsert(cand, idx, rng, ntrials=1500):
    n, d = cand.shape
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    cur = cand[mask].copy()
    for j in idx:
        trials = rng.standard_normal((ntrials, d))
        trials /= np.linalg.norm(trials, axis=1, keepdims=True)
        dots = trials @ cur.T
        mm = dots.max(axis=1)
        pick = int(np.argmin(mm))
        cand[j] = trials[pick]
        cur = np.vstack([cur, cand[j][None, :]])
    return cand

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    FINAL_RESERVE = 13.0
    rng = np.random.default_rng(0)
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
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    curr_X = best_X.copy()
    curr_v = best_v
    it = 0
    # quick initial short squeeze already essentially done; keep best
    while time.time() < HARD_END - FINAL_RESERVE and it < 200:
        it += 1
        # choose kick size: mostly small, occasionally large
        if it % 7 == 0:
            k = int(rng.integers(7, 12))
            steps = 700
            sigma = 0.0  # pure hole reinsert for large kick
            ntrials = 2000
        elif it % 3 == 0:
            k = int(rng.integers(4, 7))
            steps = 550
            sigma = 0.35
            ntrials = 1500
        else:
            k = int(rng.integers(2, 5))
            steps = 450
            sigma = 0.25
            ntrials = 1200
        # stress = per-point worst cosine
        try:
            G = curr_X @ curr_X.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            # add small random to break ties deterministically via rng
            worst = worst + 1e-12 * rng.standard_normal(n)
            idx = np.argpartition(-worst, k)[:k]
        except Exception:
            idx = rng.choice(n, size=k, replace=False)
        cand = curr_X.copy()
        if sigma > 0 and (it % 2 == 0):
            # gaussian shake of worst points
            cand[idx] += sigma * rng.standard_normal((k, d))
            nr = np.linalg.norm(cand, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            cand /= nr
        else:
            try:
                cand = hole_reinsert(cand, idx, rng, ntrials=ntrials)
            except Exception:
                cand[idx] += 0.5 * rng.standard_normal((k, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # light global jitter to help escape
        if it % 5 == 0:
            cand += 0.01 * rng.standard_normal((n, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        cand = smooth_quench(cand, steps=steps)
        try:
            v = max_cos(cand)
        except Exception:
            continue
        # promising? squeeze briefly to get true basin value
        if v < best_v + 0.001 and time.time() < HARD_END - FINAL_RESERVE - 5:
            # only squeeze if very close to best to save time
            if v < best_v + 0.0004:
                cand_sq = lp_squeeze(cand, min(time.time() + 2.5, HARD_END - FINAL_RESERVE))
                try:
                    v2 = max_cos(cand_sq)
                except Exception:
                    v2 = v
                if v2 < v:
                    cand, v = cand_sq, v2
        if v < best_v:
            best_v = v
            best_X = cand.copy()
        # basin-hopping acceptance
        if v < curr_v or rng.random() < 0.08:
            curr_X, curr_v = cand.copy(), v
        # if stuck, restart from best with shake
        if it % 25 == 0 and time.time() < HARD_END - FINAL_RESERVE:
            # partial restart
            curr_X = best_X.copy()
            curr_X += 0.02 * rng.standard_normal((n, d))
            curr_X /= np.linalg.norm(curr_X, axis=1, keepdims=True)
            curr_X = smooth_quench(curr_X, steps=300)
            try:
                curr_v = max_cos(curr_X)
            except Exception:
                curr_v = best_v + 0.001
            if curr_v < best_v:
                best_v = curr_v
                best_X = curr_X.copy()
    # final squeeze of best
    try:
        best_X = lp_squeeze(best_X, HARD_END - 0.5, cap=2200, r_init=0.002)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
