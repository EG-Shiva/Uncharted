import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_refine(X_init, steps, b0, b1, lr):
    n, d = X_init.shape
    X = X_init.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    tt = 0
    # precompute diag indices
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = b0 * ((b1 / b0) ** frac)
        G = X @ X.T
        # mask diagonal
        np.fill_diagonal(G, -2.0)
        cur_max = float(np.max(G))
        G -= cur_max
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s_sum = float(G.sum() * 0.5)
        if s_sum < 1e-100:
            continue
        grad = (G @ X) / s_sum
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        grad -= dot * X
        tt += 1
        m *= b1m
        m += (1.0 - b1m) * grad
        v *= b2m
        v += (1.0 - b2m) * (grad * grad)
        c1 = 1.0 - (b1m ** tt)
        c2 = 1.0 - (b2m ** tt)
        vhat = v / c2
        step = (lr / c1) * m / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr
        scale = np.minimum(1.0, cap / sn)
        step = step * scale
        X -= step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
        md = np.einsum('ij,ij->i', m, X)[:, None]
        m -= md * X
    return X

def _lp_polish(x, deadline, k_top=4500, radius_init=4e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    ntri = iu0.shape[0]
    K = min(k_top, ntri)
    while time.time() < deadline - 1.2 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        # top-K active set
        if K < ntri:
            top = np.argpartition(-c_all, K)[:K]
            I = iu0[top]
            J = iu1[top]
            c_sel = c_all[top]
        else:
            I = iu0
            J = iu1
            c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_blocks)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            x, best = y, v
            radius = min(radius * 1.4, 0.01)
        else:
            # check predicted vs actual to adapt
            radius *= 0.5
        # early exit if too little time for another solve
        if time.time() > deadline - 2.0:
            break
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    HOP_END = t0 + 118.0
    rng = np.random.default_rng(7)
    n, d = 250, 6
    cands = []
    try:
        a = np.load("best.npy")
        cands.append(np.asarray(a, dtype=float))
    except Exception:
        pass
    try:
        b = np.load("record.npy")
        cands.append(np.asarray(b, dtype=float))
    except Exception:
        pass
    best = None
    best_val = np.inf
    for c in cands:
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (n, d):
                continue
            c = np.nan_to_num(c)
            nr = np.linalg.norm(c, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            c = c / nr
            s = _max_cos(c)
            if s < best_val:
                best_val = s
                best = c.copy()
        except Exception:
            continue
    if best is None:
        best = rng.standard_normal((n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # Stage 0: immediate squeeze with moderate-high beta
    try:
        Xc = _adam_refine(best, 1500, 100.0, 1200.0, 0.0012)
        vc = _max_cos(Xc)
        if vc < best_val:
            best, best_val = Xc, vc
    except Exception:
        pass
    if time.time() > HOP_END - 5:
        pass
    else:
        # basin hopping with small sigmas (single-bottleneck regime)
        trial = 0
        sigmas = [0.002, 0.004, 0.008, 0.0015, 0.006, 0.012]
        while time.time() < HOP_END and trial < 80:
            trial += 1
            if time.time() > HARD - 60:
                break
            try:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm)
            except Exception:
                order = np.arange(n)
            sigma = sigmas[trial % len(sigmas)]
            mode = trial % 5
            X0 = best.copy()
            if mode == 0:
                # kick worst K harder
                K = 8 + (trial % 10)
                idx = order[:K]
                X0[idx] += (sigma * 3.0) * rng.standard_normal((K, d))
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                X0[mask] += (sigma * 0.3) * rng.standard_normal((n - K, d))
            elif mode == 1:
                # single worst reinsertion to best hole among random cands
                w = int(order[0])
                C = 64
                cand = rng.standard_normal((C, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc = cand @ best.T
                sc[:, w] = -2.0
                mx = sc.max(axis=1)
                bi = int(np.argmin(mx))
                X0[w] = cand[bi]
                X0 += (sigma * 0.5) * rng.standard_normal((n, d))
            elif mode == 2:
                # kick two max-pair points specifically
                # find argmax pair
                ii, jj = np.unravel_index(int(np.argmax(G)), G.shape)
                X0[ii] += sigma * 2.0 * rng.standard_normal(d)
                X0[jj] += sigma * 2.0 * rng.standard_normal(d)
                X0 += (sigma * 0.3) * rng.standard_normal((n, d))
            else:
                X0 += sigma * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            # choose refine schedule alternating
            try:
                if trial % 2 == 0:
                    Xc = _adam_refine(X0, 900, 120.0, 1500.0, 0.0012)
                else:
                    Xc = _adam_refine(X0, 1100, 300.0, 3500.0, 0.0009)
            except Exception:
                continue
            try:
                vc = _max_cos(Xc)
            except Exception:
                continue
            if vc < best_val - 1e-12:
                best, best_val = Xc, vc
            if time.time() >= HOP_END:
                break

    # Stage: LP squeeze + high-beta Adam alternation with remaining time
    try:
        if time.time() < HARD - 6:
            best = _lp_polish(best, min(HARD - 1.0, time.time() + 28.0), k_top=4500, radius_init=4e-4)
            best_val = _max_cos(best)
    except Exception:
        pass
    try:
        if time.time() < HARD - 8:
            # short high-beta squeeze
            Xc = _adam_refine(best, 1200, 800.0, 6000.0, 0.0006)
            vc = _max_cos(Xc)
            if vc < best_val:
                best, best_val = Xc, vc
    except Exception:
        pass
    try:
        if time.time() < HARD - 5:
            best = _lp_polish(best, HARD - 1.0, k_top=5000, radius_init=2.5e-4)
            best_val = _max_cos(best)
    except Exception:
        pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    return best
