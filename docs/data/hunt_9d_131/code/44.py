import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def adam_tau(X, tau, steps, lr, T0, T1, deadline):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    for k in range(steps):
        if (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        T = float(np.exp(lt0 * (1 - frac) + lt1 * frac))
        G = X @ X.T
        D = (G - tau) / T
        SP = np.logaddexp(0, D)
        Dc = np.clip(D, -60, 60)
        SIG = 1.0 / (1.0 + np.exp(-Dc))
        W = (2.0 * SP * SIG) / T
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** (k + 1)
        bc2 = 1 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def refine_hole(z, Xk, steps=200, T=0.002, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        m = float(np.max(dots))
        E = np.exp((dots - m) / T)
        S = float(np.sum(E)) + 1e-300
        Wn = E / S
        g = Wn @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z

def lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
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
        y = _norm(y)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n, d = 131, 9
    curX = None
    cur_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = max_cos(v)
                if c < cur_best:
                    cur_best = c
                    curX = v.copy()
        except Exception:
            pass
    if curX is None:
        curX = _norm(rng.standard_normal((n, d)))
        cur_best = max_cos(curX)
    feas_deadline = t0 + 150.0
    # target-feasibility loop
    step = 0.0005
    tau = cur_best - step
    # clamp first tau to ~0.4220 if higher
    if tau > 0.4220 and cur_best > 0.4220:
        tau = 0.4220
        step = cur_best - tau
    attempt = 0
    T0, T1 = 3e-3, 2e-4
    while time.time() < feas_deadline:
        if step < 3e-05:
            break
        if time.time() > feas_deadline - 2:
            break
        attempt += 1
        # identify 9 rattlers = smallest per-point maxima
        ppm = per_point_max(curX)
        order = np.argsort(ppm)  # ascending
        ratt_idx = order[:9]
        core_idx = order[9:]
        coreX = curX[core_idx].copy()  # 122 x 9
        # shrink steps if little time
        remain = feas_deadline - time.time()
        if remain < 10:
            cs, js = 400, 250
        elif remain < 40:
            cs, js = 1000, 500
        else:
            cs, js = 1700, 800
        # 1) core relaxation under tau penalty
        coreX = adam_tau(coreX, tau, cs, 0.003, T0, T1, feas_deadline)
        if time.time() > feas_deadline:
            break
        # 2) reinsert rattlers one by one into best holes
        cur_set = coreX.copy()
        for k in range(9):
            # sample >=20000 sphere points
            Z = rng.standard_normal((20000, d))
            Z = _norm(Z)
            D = Z @ cur_set.T
            worst = np.max(D, axis=1)
            bi = int(np.argmin(worst))
            z = Z[bi].copy()
            z = refine_hole(z, cur_set, steps=180, T=0.002, lr=0.03)
            cur_set = np.vstack([cur_set, z.reshape(1, d)])
        Xnew = cur_set  # 131 x 9
        # 3) joint re-polish under tau penalty
        Xnew = adam_tau(Xnew, tau, js, 0.0025, 2e-3, T1, feas_deadline)
        if time.time() > feas_deadline:
            # still evaluate
            pass
        nm = max_cos(Xnew)
        if nm < cur_best - 1e-12:
            # accept: true maximum below previous best (hence below previous tau progression)
            curX = _norm(Xnew)
            cur_best = nm
            # keep step, next tau more ambitious
            tau = cur_best - step
        else:
            # reject: decrease tau-step and retry from last feasible
            step *= 0.5
            tau = cur_best - step
        # safety: if tau extremely close, nudge
        if attempt > 60:
            break
    # final LP polish, reserve ~20s
    remain_total = 178.0 - (time.time() - t0)
    lp_sec = min(20.0, max(2.0, remain_total - 1.0))
    if lp_sec > 0.5:
        try:
            Xlp = lp_polish(curX, lp_sec)
            vlp = max_cos(Xlp)
            if vlp < cur_best:
                curX = Xlp
                cur_best = vlp
        except Exception:
            pass
    curX = _norm(curX)
    return curX
