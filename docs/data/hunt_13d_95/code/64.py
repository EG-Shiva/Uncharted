import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, seconds, t_final):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if now >= t_final:
            break
        if radius < 1e-14:
            break
        if now - start > seconds - 0.15:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = 1500
        if cnt == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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


def _adam_continuation(X0, T, t0, expl_deadline, rng_seed_unused=None):
    n, d = X0.shape
    X = X0.copy()
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    a_target = float(n) / float(d)
    Eye = np.eye(d)
    best_X = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_true = float(G0.max())
    for s in range(T):
        f = s / max(1, T - 1)
        beta = 25.0 * (24.0 ** f)  # 25 -> 600
        lam = 1e-2 * (1.0 - f)
        mu = 1e-3 * (1.0 - f)
        lr = 0.008 * (0.25 + 0.75 * (1.0 - f))  # 0.008 -> 0.002
        # Gram
        G = X @ X.T
        # global LSE (off-diagonal)
        gmax = float(np.max(G - 2.0 * np.eye(n)))
        # exp stable
        E = np.exp(beta * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            S = 1.0
            E[:] = 0.0
        W = E / S
        Grad_g = 2.0 * (W @ X)
        # frame term
        if lam > 0:
            C = X.T @ X
            Ef = C - a_target * Eye
            Grad_f = 4.0 * (X @ Ef) * lam
        else:
            Grad_f = 0.0
        # row variance term (row-LSE approx of max)
        if mu > 0:
            # row max off-diag
            Gt = G.copy()
            np.fill_diagonal(Gt, -2.0)
            rmax = Gt.max(axis=1)  # (n,)
            Er = np.exp(beta * (Gt - rmax[:, None]))
            # diag was -2 -> exp tiny, force 0
            # set diag exactly 0
            Er[np.arange(n), np.arange(n)] = 0.0
            rowsum = Er.sum(axis=1, keepdims=True)
            rowsum[rowsum < 1e-300] = 1e-300
            P = Er / rowsum  # (n,n)
            r = rmax + np.log(rowsum[:, 0]) / beta
            rmean = float(r.mean())
            cvec = (2.0 * mu / float(n)) * (r - rmean)  # (n,)
            PX = P @ X
            term1 = cvec[:, None] * PX
            term2 = ((cvec[:, None] * P).T @ X)
            Grad_v = term1 + term2
        else:
            Grad_v = 0.0
        Grad = Grad_g + Grad_f + Grad_v
        # tangent projection
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt2 = Grad - dot * X
        # Adam
        M = b1 * M + (1.0 - b1) * Gt2
        V = b2 * V + (1.0 - b2) * (Gt2 * Gt2)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        Mh = M / bc1
        Vh = V / bc2
        step = lr * Mh / (np.sqrt(Vh) + eps)
        # re-project step to tangent
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - sdot * X
        X = X - step_t
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        # track true best every 10 steps + last
        if (s % 10 == 0) or (s == T - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            tv = float(Gc.max())
            if tv < best_true:
                best_true = tv
                best_X = X.copy()
        if (s % 50 == 49):
            if time.time() >= expl_deadline:
                break
    return best_X, best_true


def solve():
    t0 = time.time()
    EXPL_DEADLINE = t0 + 145.0
    T_FINAL = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if np.isfinite(xn).all():
            cands.append(xn)
    if xr is not None and xr.shape == (95, 13):
        xn = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if np.isfinite(xn).all():
            cands.append(xn)
    if len(cands) == 0:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    base = cands[bi].copy()
    incumbent = base.copy()
    best_val = float(vals[bi])
    n, d = 95, 13
    sigmas = [0.03, 0.08, 0.15]
    T = 2000
    best_cont = incumbent.copy()
    # exploration: one copy per sigma
    for sigma in sigmas:
        if time.time() >= EXPL_DEADLINE:
            break
        # tangent perturbation
        Z = rng.standard_normal((n, d))
        dot = np.sum(Z * base, axis=1, keepdims=True)
        Zt = Z - dot * base
        nz = np.linalg.norm(Zt, axis=1, keepdims=True)
        nz[nz == 0] = 1.0
        Zt = Zt / nz
        X0 = base + sigma * Zt
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        try:
            Xopt, vopt = _adam_continuation(X0, T, t0, EXPL_DEADLINE)
        except Exception:
            continue
        if vopt < best_val - 1e-15:
            best_val = vopt
            incumbent = Xopt.copy()
            best_cont = Xopt.copy()
        elif vopt < _max_cos(best_cont):
            # keep best continuation even if not global best (for polish diversity)
            # only replace best_cont if it is worse than new but new is close
            cur_bc = _max_cos(best_cont)
            if vopt < cur_bc:
                best_cont = Xopt.copy()
        # if this copy is the global best, update best_cont too
        # ensure best_cont tracks incumbent
        if best_val == vopt:
            best_cont = incumbent.copy()
    # single trust-region LP polish on best copy for remaining time
    # polish target is the incumbent (best continuation / record)
    # if continuation never improved, still polish incumbent
    remain = T_FINAL - time.time() - 0.8
    if remain > 1.0:
        try:
            Xpol, vpol = _lp_polish(best_cont if _max_cos(best_cont) <= _max_cos(incumbent) else incumbent,
                                    seconds=float(remain), t_final=T_FINAL - 0.5)
            if np.isfinite(vpol) and vpol < best_val:
                incumbent = Xpol / np.linalg.norm(Xpol, axis=1, keepdims=True)
                best_val = float(vpol)
            # else return incumbent unchanged
        except Exception:
            pass
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
