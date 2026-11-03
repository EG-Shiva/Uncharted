import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_opt(X, steps, alpha0, alpha1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mask = np.ones((n, n), dtype=float)
    np.fill_diagonal(mask, 0.0)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = alpha0 * (alpha1 / alpha0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = G.max()
        E = np.exp(alpha * (G - cmax)) * mask
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        Grad = W @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = (step * X).sum(axis=1, keepdims=True)
        step_t = step - sdot * X
        nm = np.linalg.norm(step_t, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / nm)
        step_t = step_t * clip
        X = X - step_t
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def hole_for_set(Xfixed, rng, n_try=4000, refine=180):
    d = Xfixed.shape[1]
    Z = rng.normal(size=(n_try, d))
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30)
    dm = (Z @ Xfixed.T).max(axis=1)
    y = Z[int(np.argmin(dm))].copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    mm = np.zeros(d); vvv = np.zeros(d)
    for t in range(1, refine + 1):
        frac = t / refine
        alpha = 30.0 * (500.0 / 30.0) ** frac
        lr = 0.01 * (0.002 / 0.01) ** frac
        s = Xfixed @ y
        smax = s.max()
        e = np.exp(alpha * (s - smax))
        w = e / (e.sum() + 1e-30)
        g = w @ Xfixed
        gt = g - (g @ y) * y
        mm = b1 * mm + (1 - b1) * gt
        vvv = b2 * vvv + (1 - b2) * (gt * gt)
        mh = mm / (1 - b1 ** t)
        vh = vvv / (1 - b2 ** t)
        st = lr * mh / (np.sqrt(vh) + eps)
        st = st - (st @ y) * y
        nn = np.linalg.norm(st) + 1e-12
        if nn > 0.05:
            st *= 0.05 / nn
        y = y - st
        y /= (np.linalg.norm(y) + 1e-30)
    return y

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 176.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None:
            try:
                z = np.asarray(z, dtype=float)
                if z.shape == (154, 10):
                    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-30)
                    if np.all(np.isfinite(z)):
                        cands.append(z)
            except Exception:
                pass
    if len(cands) == 0:
        rng0 = np.random.RandomState(0)
        z = rng0.normal(size=(154, 10))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(z) for z in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = best.copy()
    cur_val = best_val
    rng = np.random.RandomState(12345)
    n, d = best.shape
    it = 0
    while time.time() - t0 < BUDGET - 30.0:
        it += 1
        elapsed = time.time() - t0
        remain = BUDGET - elapsed
        if remain < 12:
            break
        frac = elapsed / max(BUDGET - 30.0, 1.0)
        frac = min(max(frac, 0.0), 1.0)
        T_acc = 0.0015 * (1.0 - frac) + 0.00012
        cand = cur.copy()
        r = rng.rand()
        if r < 0.50:
            sigma = rng.uniform(0.005, 0.020)
            cand = cand + rng.normal(scale=sigma, size=cand.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            G = cand @ cand.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            mcur = G.max()
            delta = 0.003 + rng.rand() * 0.004
            cnt = (G > mcur - delta).sum(axis=1)
            k = int(rng.randint(1, 4))
            score = (rm - rm.min() + 1e-6) + 0.02 * cnt.astype(float)
            ssum = score.sum()
            if not np.isfinite(ssum) or ssum <= 0:
                victims = rng.choice(n, size=k, replace=False)
            else:
                probs = score / ssum
                victims = rng.choice(n, size=k, replace=False, p=probs)
            for idx in victims:
                Xf = np.delete(cand, idx, axis=0)
                cand[idx] = hole_for_set(Xf, rng, n_try=4000, refine=180)
        elif r < 0.75:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mcur = G.max()
            cnt = (G > mcur - 0.005).sum(axis=1)
            center = int(np.argmax(cnt + rng.rand(n) * 5.0))
            sims = (cur @ cur[center])
            neigh = np.argsort(sims)[-12:]
            szs = int(rng.randint(6, 13))
            subset = rng.choice(neigh, size=szs, replace=False)
            sigma = rng.uniform(0.08, 0.25)
            cand[subset] += rng.normal(scale=sigma, size=(szs, d))
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            if rng.rand() < 0.4:
                G2 = cand @ cand.T
                np.fill_diagonal(G2, -2.0)
                rm2 = G2.max(axis=1)
                idx = int(np.argmax(rm2))
                Xf = np.delete(cand, idx, axis=0)
                cand[idx] = hole_for_set(Xf, rng, n_try=4000, refine=180)
        elif r < 0.90:
            sz = int(rng.randint(8, 26))
            subset = rng.choice(n, size=sz, replace=False)
            sigma = rng.uniform(0.07, 0.22)
            cand[subset] += rng.normal(scale=sigma, size=(sz, d))
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
        else:
            sigma = rng.uniform(0.03, 0.09)
            cand = cur + rng.normal(scale=sigma, size=cur.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
        steps = 600
        if remain < 70:
            steps = 450
        a0 = 40.0
        a1 = rng.uniform(250.0, 550.0)
        lr0 = rng.uniform(0.006, 0.014)
        lr1 = 0.0015
        try:
            cand2 = smooth_opt(cand, steps=steps, alpha0=a0, alpha1=a1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        try:
            v2 = max_cos(cand2)
        except Exception:
            continue
        need_lp = (v2 < best_val + 0.0015) or (v2 < cur_val + 0.0008)
        if rng.rand() < 0.04 and v2 < best_val + 0.006:
            need_lp = True
        if need_lp:
            sec = 1.2
            if v2 < best_val - 0.0002:
                sec = 2.4
            if remain - sec < 30.0:
                sec = max(0.6, remain - 30.5)
            try:
                cand3 = lp_polish(cand2, seconds=sec)
                v3 = max_cos(cand3)
            except Exception:
                cand3, v3 = cand2, v2
        else:
            cand3, v3 = cand2, v2
        if v3 < best_val - 1e-12:
            best, best_val = cand3, v3
            cur, cur_val = cand3, v3
        elif v3 < cur_val - 1e-12:
            cur, cur_val = cand3, v3
        else:
            try:
                pa = float(np.exp((cur_val - v3) / max(T_acc, 1e-12)))
            except Exception:
                pa = 0.0
            if rng.rand() < pa:
                cur, cur_val = cand3, v3
        if cur_val > best_val + 0.008 and rng.rand() < 0.3:
            cur, cur_val = best.copy(), best_val
        if time.time() - t0 >= BUDGET - 30.0:
            break
    remain = BUDGET - (time.time() - t0)
    final_sec = max(1.0, remain - 2.0)
    final_sec = min(final_sec, 32.0)
    try:
        best = lp_polish(best, seconds=final_sec)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-30
    best = best / nr
    return best
