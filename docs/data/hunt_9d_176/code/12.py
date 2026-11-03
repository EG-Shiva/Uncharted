import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def repulse_anneal(X, steps=450, beta0=25.0, beta1=160.0, lr0=0.025, lr1=0.002):
    n = X.shape[0]
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if gnorm < 1e-16 or not np.isfinite(gnorm):
            continue
        X = X - (lr / gnorm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(nr, 1e-12)
    return X

def lp_polish_deadline(x, deadline, r_init=0.004, max_pairs=2500, fast_exit=False):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    no_imp = 0
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if sel.sum() == 0:
            radius *= 0.5
            no_imp += 1
            if fast_exit and no_imp > 5:
                break
            continue
        if sel.sum() > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel2 = np.zeros(len(c_all), dtype=bool)
            sel2[part] = True
            # keep threshold if enough remain
            if np.sum(sel & sel2) > 300:
                sel = sel & sel2
            else:
                sel = sel2
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
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
            no_imp += 1
            if fast_exit and no_imp > 5:
                break
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            no_imp += 1
            if fast_exit and no_imp > 5:
                break
            continue
        if res.status != 0:
            radius *= 0.5
            no_imp += 1
            if fast_exit and no_imp > 5 and radius < r_init * 0.3:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            no_imp += 1
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
            no_imp = 0
        else:
            radius *= 0.5
            no_imp += 1
            if fast_exit and no_imp > 6 and radius < r_init * 0.3:
                break
    return x

def _find_hole(X_fixed, rng, M=2000, refine=150):
    n_f, d = X_fixed.shape
    R = rng.standard_normal((M, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    dots = R @ X_fixed.T
    worst = dots.max(axis=1)
    r = R[int(np.argmin(worst))].copy()
    beta0, beta1 = 30.0, 220.0
    lr0, lr1 = 0.03, 0.002
    for s in range(refine):
        frac = s / max(1, refine - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        dts = X_fixed @ r
        m = float(dts.max())
        w = np.exp((dts - m) * beta)
        grad = w @ X_fixed
        gt = grad - (float(grad @ r)) * r
        ng = float(np.linalg.norm(gt))
        if ng < 1e-16 or not np.isfinite(ng):
            break
        r = r - (lr / ng) * gt
        nr = float(np.linalg.norm(r))
        if nr < 1e-12 or not np.isfinite(nr):
            break
        r = r / nr
    return r

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LOOP_END = t0 + 148.0
    rng = np.random.default_rng(20261103)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])
    if time.time() < LOOP_END:
        try:
            best_X = lp_polish_deadline(best_X, min(time.time() + 3.0, LOOP_END), r_init=0.0012)
            best_v = max_cos(best_X)
        except Exception:
            pass
    curr_X = best_X.copy()
    curr_v = best_v
    sigmas = [0.005, 0.01, 0.02, 0.04, 0.08]
    T = 5e-4
    trial = 0
    while time.time() < LOOP_END:
        trial += 1
        if trial % 30 == 0:
            curr_X = best_X.copy(); curr_v = best_v
        try:
            Gc = curr_X @ curr_X.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            mcur = float(Gc.max())
            contacts = np.sum(Gc > mcur - 0.002, axis=1)
            order = np.argsort(rowmax)
            order_c = np.argsort(contacts)
        except Exception:
            order = np.arange(n); order_c = np.arange(n)
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 7
        Xp = curr_X.copy()
        try:
            if mode == 0:
                Xp = Xp + sigma * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            elif mode == 1:
                k = 10
                idx = order[-k:]
                Xp = Xp + 0.002 * rng.standard_normal((n, d))
                Xp[idx] = Xp[idx] + (sigma * 1.5) * rng.standard_normal((k, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            elif mode == 2:
                idx = int(order_c[-1])
                Xf = np.delete(Xp, idx, axis=0)
                r = _find_hole(Xf, rng, M=2000, refine=150)
                Xp[idx] = r
                Xp = Xp + 0.003 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            elif mode == 3:
                Xp = Xp + 0.003 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                for idx in list(order_c[-2:]):
                    Xf = np.delete(Xp, int(idx), axis=0)
                    r = _find_hole(Xf, rng, M=1200, refine=120)
                    Xp[int(idx)] = r
                Xp = Xp + 0.001 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            elif mode == 4:
                idx = int(order[-1])
                M = 1500
                R = rng.standard_normal((M, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                dots = R @ Xp.T
                dots[:, idx] = -2.0
                worst = dots.max(axis=1)
                Xp[idx] = R[int(np.argmin(worst))]
                Xp = Xp + 0.002 * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            elif mode == 5:
                k = 24
                idxs = rng.choice(n, size=k, replace=False)
                Xp[idxs] = Xp[idxs] + sigma * rng.standard_normal((k, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            else:
                Xp = Xp + (sigma * 0.7) * rng.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        except Exception:
            continue
        steps = 450 if sigma < 0.05 else 350
        try:
            Xp = repulse_anneal(Xp, steps=steps)
        except Exception:
            pass
        try:
            va = max_cos(Xp)
        except Exception:
            continue
        if not np.isfinite(va):
            continue
        if va < curr_v or rng.random() < np.exp((curr_v - va) / T):
            curr_X = Xp.copy(); curr_v = va
        if va < best_v + 0.002:
            budget = 1.6
            deadline = min(time.time() + budget, LOOP_END, HARD_END - 15.0)
            if deadline <= time.time() + 0.3:
                break
            try:
                r0 = 0.006 if sigma > 0.02 else 0.003
                Xq = lp_polish_deadline(Xp, deadline, r_init=r0, fast_exit=True)
                vq = max_cos(Xq)
            except Exception:
                continue
            if not np.isfinite(vq):
                continue
            if vq < best_v:
                best_v = vq; best_X = Xq.copy()
                curr_X = Xq.copy(); curr_v = vq
            elif vq < curr_v or rng.random() < np.exp((curr_v - vq) / T):
                curr_X = Xq.copy(); curr_v = vq
    try:
        if time.time() < HARD_END - 1.0:
            best_X = lp_polish_deadline(best_X, HARD_END - 1.0, r_init=0.0012)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
