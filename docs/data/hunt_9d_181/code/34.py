import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_radius=2e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.25 > budget and radius < 8e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(np.max(c_all))
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            kth = np.partition(c_all, -3000)[-3000]
            sel = c_all >= kth
            cnt = int(np.sum(sel))
            if cnt > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_free(Y, steps, T0, T1, lr0, lr1, mu=0.6):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= dot * Y
        sc = float(np.mean(np.linalg.norm(Gr, axis=1))) + 1e-12
        Gn = Gr / sc
        V = mu * V + Gn
        Y = Y - lr * V
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _sym_optim(Zn_init, tail, steps, T0, T1, lr0, lr1, mu):
    K = Zn_init.shape[0]
    Zn = Zn_init.copy()
    Zn /= np.linalg.norm(Zn, axis=1, keepdims=True)
    V = np.zeros_like(Zn)
    IDX = (np.arange(9)[None, :] - np.arange(9)[:, None]) % 9
    m = tail.shape[0] if tail is not None else 0
    n = K * 9 + m
    best_val = np.inf
    best_Zn = Zn.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        X = np.empty((n, 9))
        for k in range(K):
            X[k*9:(k+1)*9] = Zn[k][IDX]
        if m > 0:
            X[K*9:] = tail
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cur = float(np.max(G))
        if cur < best_val:
            best_val = cur
            best_Zn = Zn.copy()
        W = (G - cur) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        GrX = W @ X
        dot = np.sum(GrX * X, axis=1, keepdims=True)
        GrX -= dot * X
        Gz = np.empty_like(Zn)
        for k in range(K):
            Gb = GrX[k*9:(k+1)*9]
            acc = np.zeros(9)
            for ss in range(9):
                acc += np.roll(Gb[ss], -ss)
            Gz[k] = acc
        d = np.sum(Gz * Zn, axis=1, keepdims=True)
        Gz -= d * Zn
        sc = float(np.mean(np.linalg.norm(Gz, axis=1))) + 1e-12
        Gn = Gz / sc
        V = mu * V + Gn
        Zn = Zn - lr * V
        Zn /= np.linalg.norm(Zn, axis=1, keepdims=True)
    Xb = np.empty((n, 9))
    for k in range(K):
        Xb[k*9:(k+1)*9] = best_Zn[k][IDX]
    if m > 0:
        Xb[K*9:] = tail
    return Xb, best_val

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    # load fallback
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    b_best = float(vals[bi])

    # symmetry-constrained search: C9 cyclic shifts, 181 = 20*9+1
    f_diag = np.ones(9) / 3.0
    tail1 = f_diag[None, :]
    tail2 = np.vstack([np.eye(9), f_diag[None, :]])
    sym_best = None
    sym_val = np.inf
    # trials: (K, tail, steps, T0,T1,lr0,lr1)
    trials = [
        (20, tail1, 650, 0.05, 0.004, 0.08, 0.010),
        (20, tail1, 650, 0.03, 0.003, 0.06, 0.008),
        (20, tail1, 550, 0.07, 0.006, 0.10, 0.012),
        (19, tail2, 600, 0.05, 0.004, 0.08, 0.010),
        (19, tail2, 550, 0.03, 0.003, 0.06, 0.008),
    ]
    for ti, (K, tail, steps, T0, T1, lr0, lr1) in enumerate(trials):
        if time.time() - t0 > 60.0:
            break
        Zn_init = rng.standard_normal((K, 9))
        Zn_init /= np.linalg.norm(Zn_init, axis=1, keepdims=True)
        try:
            Xb, vb = _sym_optim(Zn_init, tail, steps, T0, T1, lr0, lr1, 0.6)
        except Exception:
            continue
        if vb < sym_val:
            sym_val = vb
            sym_best = Xb.copy()
    # break symmetry: free smooth anneal of symmetric seed
    seeds = []
    if sym_best is not None:
        try:
            # short free relaxation to leave symmetric manifold
            Y = sym_best.copy()
            # keep time bounded
            Y = _smooth_free(Y, 350, 0.03, 0.004, 0.04, 0.008, 0.6)
            seeds.append(Y)
            seeds.append(sym_best.copy())
        except Exception:
            seeds.append(sym_best.copy())
    # perturbed copies with increasing size
    base_list = list(seeds)
    sigmas = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    pert = []
    for b in base_list[:2]:
        for sg in sigmas:
            if len(pert) >= 6:
                break
            try:
                q = b + rng.standard_normal(b.shape) * sg
                q /= np.linalg.norm(q, axis=1, keepdims=True)
                pert.append(q)
            except Exception:
                pass
    # polish queue: perturbed symmetric first, then record best
    queue = pert + [x_best.copy()]
    # quick smooth + LP for each while time remains
    for qi, q in enumerate(queue):
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        try:
            # light smooth
            qs = _smooth_free(q.copy(), 120, 0.02, 0.005, 0.03, 0.008, 0.6)
        except Exception:
            qs = q
        try:
            vq = _max_cos(qs)
        except Exception:
            continue
        # allocate LP budget: more for last (record) item
        is_last = (qi == len(queue) - 1)
        lp_b = min(12.0 if is_last else 6.0, max(2.0, rem - 2.0))
        # if symmetric seed still bad, spend less
        if not is_last and vq > b_best + 0.03 and rem < 40:
            lp_b = min(lp_b, 3.0)
        try:
            qp = _lp_polish(qs, seconds=lp_b, init_radius=3e-3)
            vp = _max_cos(qp)
        except Exception:
            continue
        if vp < b_best:
            b_best = vp
            x_best = qp.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
