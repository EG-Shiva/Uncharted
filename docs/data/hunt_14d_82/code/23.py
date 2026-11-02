import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, radius_init=1e-4, t_end=None, max_it=200, max_pairs=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while it < max_it:
        it += 1
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - max(25.0 * radius, 2e-7)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, max_pairs - 1)[:max_pairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 1.0})
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _smooth_quench(x, steps, eta, step0, decay=0.999):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights focused near max
        S = (G - m) / eta
        # cut off far tail for stability
        S[S < -8.0] = -8.0
        W = np.exp(S)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = float(np.sqrt((Gt * Gt).mean()) + 1e-18)
        if not np.isfinite(nrm) or nrm < 1e-18:
            break
        step = step0 * (decay ** s)
        x = x - (step / nrm) * Gt
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(1)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        xb = rng.standard_normal(size=(n, d))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    # initial squeeze with moderate radius
    if t_end - time.time() > 12.0:
        xb = _lp_polish(xb, min(4.0, t_end - time.time() - 8.0), radius_init=3e-4, t_end=t_end - 8.0)
    best_val = _max_cos(xb)
    global_best = xb.copy()
    # extra micro polish from small radius to exploit 1e-6 gap
    if t_end - time.time() > 12.0:
        tmp = _lp_polish(global_best, min(3.0, t_end - time.time() - 8.0), radius_init=8e-6, t_end=t_end - 8.0)
        v = _max_cos(tmp)
        if v < best_val:
            global_best = tmp
            best_val = v
    current = global_best.copy()
    cur_val = best_val
    sigmas = [0.004, 0.001, 0.0004, 0.00015, 0.008, 0.002, 0.0006]
    it = 0
    while time.time() < t_end - 6.0:
        it += 1
        if time.time() >= t_end - 6.0:
            break
        base = global_best if (it % 5 == 0 or rng.random() < 0.7) else current
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        # locate worst pair
        flat = int(np.argmax(G))
        wi, wj = flat // n, flat % n
        trial = base.copy()
        r = rng.random()
        sigma = sigmas[it % len(sigmas)]
        if r < 0.55:
            # kick worst pair + its near neighbours
            neigh_i = np.where(G[wi] > m0 - 0.01)[0]
            neigh_j = np.where(G[wj] > m0 - 0.01)[0]
            idx = set([int(wi), int(wj)])
            for k in neigh_i[:4]:
                idx.add(int(k))
            for k in neigh_j[:4]:
                idx.add(int(k))
            # add a few random rattlers to keep diversity
            for _ in range(2):
                idx.add(int(rng.integers(0, n)))
            idx = np.array(sorted(idx), dtype=int)
            trial[idx] = base[idx] + sigma * rng.standard_normal(size=(len(idx), d))
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial /= nr
        elif r < 0.8:
            trial = base + (sigma * 0.5) * rng.standard_normal(size=base.shape)
            nr = np.linalg.norm(trial, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            trial /= nr
        else:
            # single-point teleport of worst endpoint into a hole
            victim = int(wi) if rng.random() < 0.5 else int(wj)
            mask = np.ones(n, dtype=bool)
            mask[victim] = False
            Xk = base[mask]
            cand = rng.standard_normal(size=(1500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Xk.T
            worst = dots.max(axis=1)
            h = cand[int(np.argmin(worst))]
            for _ in range(15):
                g = Xk @ h
                mm = float(np.max(g))
                w = np.exp((g - mm) / 0.03)
                grad = w @ Xk
                grad = grad - (grad @ h) * h
                nn = float(np.linalg.norm(grad))
                if nn < 1e-12 or not np.isfinite(nn):
                    break
                h = h - (0.05 / (nn + 1e-12)) * grad
                h /= (np.linalg.norm(h) + 1e-18)
            trial[victim] = h
        # fast smooth quench, eta tied to gap scale
        eta = 3e-4 if (it % 2 == 0) else 8e-5
        trial = _smooth_quench(trial, steps=250, eta=eta, step0=min(0.02, sigma * 1.2 + 1e-4), decay=0.998)
        vq = _max_cos(trial)
        if vq > best_val + 0.008 and vq > cur_val + 0.004:
            if vq < cur_val:
                current = trial.copy()
                cur_val = vq
            continue
        # decide LP budget
        if vq < best_val + 0.0015:
            budget_lp = 1.0
        elif vq < best_val + 0.004:
            budget_lp = 0.5
        else:
            budget_lp = 0.3
        if time.time() + budget_lp + 5.5 > t_end:
            budget_lp = max(0.2, t_end - 5.5 - time.time())
        if budget_lp < 0.2:
            break
        # two-stage: moderate then micro radius
        trial2 = _lp_polish(trial, budget_lp * 0.6, radius_init=5e-5, t_end=t_end - 5.0)
        trial2 = _lp_polish(trial2, budget_lp * 0.4, radius_init=5e-6, t_end=t_end - 5.0)
        v = _max_cos(trial2)
        if v < best_val - 1e-15:
            global_best = trial2.copy()
            best_val = v
            current = trial2.copy()
            cur_val = v
        elif v < cur_val:
            current = trial2.copy()
            cur_val = v
        else:
            Temp = 0.0004
            if rng.random() < np.exp((cur_val - v) / Temp):
                current = trial2.copy()
                cur_val = v
        if it % 25 == 0:
            current = global_best.copy()
            cur_val = best_val
    if time.time() < t_end - 0.4:
        global_best = _lp_polish(global_best, max(0.6, t_end - time.time() - 0.3), radius_init=3e-6, t_end=t_end - 0.15)
        global_best = _lp_polish(global_best, max(0.4, t_end - time.time() - 0.1), radius_init=2e-4, t_end=t_end - 0.05)
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
