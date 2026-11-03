import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish_deadline(x, deadline, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        try:
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if v < best - 1e-12:
            pass
    return best_x

# ---- symmetry helpers ----
_ANG = np.array([k * np.pi / 4.0 for k in range(8)])
_CS = np.cos(_ANG)
_SS = np.sin(_ANG)

def expand22(R):
    nO = R.shape[0]
    Xr = np.empty((nO, 8, 9))
    P = R[:, :8].reshape(nO, 4, 2)
    s = R[:, 8]
    for k in range(8):
        c = _CS[k]; si = _SS[k]
        Rx = c * P[:, :, 0] - si * P[:, :, 1]
        Ry = si * P[:, :, 0] + c * P[:, :, 1]
        Xr[:, k, 0] = Rx[:, 0]; Xr[:, k, 1] = Ry[:, 0]
        Xr[:, k, 2] = Rx[:, 1]; Xr[:, k, 3] = Ry[:, 1]
        Xr[:, k, 4] = Rx[:, 2]; Xr[:, k, 5] = Ry[:, 2]
        Xr[:, k, 6] = Rx[:, 3]; Xr[:, k, 7] = Ry[:, 3]
        Xr[:, k, 8] = s
    return Xr.reshape(nO * 8, 9)

def pullback22(gradX, nO):
    G = gradX.reshape(nO, 8, 9)
    Gp = G[:, :, :8].reshape(nO, 8, 4, 2)
    gradR = np.zeros((nO, 9))
    accx = np.zeros((nO, 4)); accy = np.zeros((nO, 4))
    for k in range(8):
        c = _CS[k]; si = _SS[k]
        gx = Gp[:, k, :, 0]; gy = Gp[:, k, :, 1]
        bx = c * gx + si * gy
        by = -si * gx + c * gy
        accx += bx; accy += by
    gradR[:, 0] = accx[:, 0]; gradR[:, 1] = accy[:, 0]
    gradR[:, 2] = accx[:, 1]; gradR[:, 3] = accy[:, 1]
    gradR[:, 4] = accx[:, 2]; gradR[:, 5] = accy[:, 2]
    gradR[:, 6] = accx[:, 3]; gradR[:, 7] = accy[:, 3]
    gradR[:, 8] = G[:, :, 8].sum(axis=1)
    return gradR

def expand11(R):
    nO = R.shape[0]
    Xr = np.empty((nO, 16, 9))
    P = R[:, :8].reshape(nO, 4, 2)
    s = R[:, 8]
    for k in range(8):
        c = _CS[k]; si = _SS[k]
        Rx = c * P[:, :, 0] - si * P[:, :, 1]
        Ry = si * P[:, :, 0] + c * P[:, :, 1]
        for e in (0, 1):
            kk = k * 2 + e
            sgn = 1.0 if e == 0 else -1.0
            Xr[:, kk, 0] = Rx[:, 0]; Xr[:, kk, 1] = Ry[:, 0]
            Xr[:, kk, 2] = Rx[:, 1]; Xr[:, kk, 3] = Ry[:, 1]
            Xr[:, kk, 4] = Rx[:, 2]; Xr[:, kk, 5] = Ry[:, 2]
            Xr[:, kk, 6] = Rx[:, 3]; Xr[:, kk, 7] = Ry[:, 3]
            Xr[:, kk, 8] = sgn * s
    return Xr.reshape(nO * 16, 9)

def pullback11(gradX, nO):
    G = gradX.reshape(nO, 16, 9)
    Gp = G[:, :, :8].reshape(nO, 16, 4, 2)
    gradR = np.zeros((nO, 9))
    accx = np.zeros((nO, 4)); accy = np.zeros((nO, 4))
    accs = np.zeros(nO)
    for k in range(8):
        c = _CS[k]; si = _SS[k]
        for e in (0, 1):
            kk = k * 2 + e
            sgn = 1.0 if e == 0 else -1.0
            gx = Gp[:, kk, :, 0]; gy = Gp[:, kk, :, 1]
            bx = c * gx + si * gy
            by = -si * gx + c * gy
            accx += bx; accy += by
            accs += sgn * G[:, kk, 8]
    gradR[:, 0] = accx[:, 0]; gradR[:, 1] = accy[:, 0]
    gradR[:, 2] = accx[:, 1]; gradR[:, 3] = accy[:, 1]
    gradR[:, 4] = accx[:, 2]; gradR[:, 5] = accy[:, 2]
    gradR[:, 6] = accx[:, 3]; gradR[:, 7] = accy[:, 3]
    gradR[:, 8] = accs
    return gradR

def sym_adam_restart(Rinit, expand_fn, pullback_fn, steps, k0, k1, lr0, deadline):
    R = Rinit / np.linalg.norm(Rinit, axis=1, keepdims=True)
    m1 = np.zeros_like(R); m2 = np.zeros_like(R)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    nO = R.shape[0]
    for t in range(1, steps + 1):
        if t % 50 == 0 and time.time() > deadline - 0.1:
            break
        frac = t / steps
        k = k0 + (k1 - k0) * frac
        lr = lr0 * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        X = expand_fn(R)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        E = np.exp(k * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Pm = E / S
        gradX = Pm @ X
        gradR = pullback_fn(gradX, nO)
        dot = np.sum(gradR * R, axis=1, keepdims=True)
        g = gradR - dot * R
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * g
        m2 = b2 * m2 + (1.0 - b2) * (g * g)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.05
        over = sn > cap
        if np.any(over):
            step = np.where(over, step * (cap / np.maximum(sn, 1e-12)), step)
        R = R - step
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        if not np.all(np.isfinite(R)):
            break
    Xf = expand_fn(R)
    Xf = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
    try:
        v = max_cos(Xf)
    except Exception:
        v = 1e9
    return R, Xf, v

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    SYM1_END = t0 + 70.0
    SYM2_END = min(t0 + 92.0, HARD_END - 80.0)
    rng = np.random.default_rng(0)
    # global best from files
    X_best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (176, 9):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(an)
                if v < best_val:
                    best_val = v; X_best = an.copy()
        except Exception:
            continue
    if X_best is None:
        X_best = rng.standard_normal((176, 9))
        X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
        best_val = max_cos(X_best)
    X_global = X_best.copy(); gval = best_val

    # Phase 1: 22 orbits x 8
    X_sym_best = None; sym_val = 1e9; R_sym_best = None
    trial = 0
    cfgs = [ (30.0, 250.0, 0.004, 2200), (60.0, 350.0, 0.003, 2200), (20.0, 180.0, 0.006, 1800) ]
    while time.time() < SYM1_END - 0.5:
        if HARD_END - time.time() < 90.0:
            break
        cfg = cfgs[trial % len(cfgs)]
        # init: mostly random, occasionally subsample global
        if trial % 6 == 5 and X_global is not None:
            idx = rng.choice(176, size=22, replace=False)
            Rinit = X_global[idx].copy() + 0.05 * rng.standard_normal((22, 9))
        else:
            Rinit = rng.standard_normal((22, 9))
        Rinit /= np.linalg.norm(Rinit, axis=1, keepdims=True)
        _, Xf, v = sym_adam_restart(Rinit, expand22, pullback22, cfg[3], cfg[0], cfg[1], cfg[2], SYM1_END)
        trial += 1
        if v < sym_val:
            sym_val = v; X_sym_best = Xf.copy()
            if v < gval:
                pass
    # Phase 2: 11 orbits x 16 (sign flip) if time permits
    X_sym2_best = None; sym2_val = 1e9
    trial2 = 0
    while time.time() < SYM2_END - 0.5:
        if HARD_END - time.time() < 75.0:
            break
        cfg = cfgs[trial2 % len(cfgs)]
        Rinit = rng.standard_normal((11, 9))
        Rinit /= np.linalg.norm(Rinit, axis=1, keepdims=True)
        _, Xf, v = sym_adam_restart(Rinit, expand11, pullback11, 1800, cfg[0], cfg[1], cfg[2], SYM2_END)
        trial2 += 1
        if v < sym2_val:
            sym2_val = v; X_sym2_best = Xf.copy()
    # choose symmetric candidate
    X_cand = None; cand_val = 1e9
    if X_sym_best is not None and sym_val < cand_val:
        cand_val = sym_val; X_cand = X_sym_best
    if X_sym2_best is not None and sym2_val < cand_val:
        cand_val = sym2_val; X_cand = X_sym2_best
    # break symmetry with 1e-4 tangent perturbation + final LP polish
    # reserve time: split remaining between sym-candidate polish and global polish
    rem = HARD_END - time.time()
    if X_cand is not None and rem > 5.0:
        # tiny tangent perturbation
        noise = rng.standard_normal(X_cand.shape) * 1e-4
        dn = np.sum(noise * X_cand, axis=1, keepdims=True)
        noise = noise - dn * X_cand
        Xp = X_cand + noise
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # polish sym candidate with ~40% of remaining (but at least 5s, at most 35s)
        sym_polish_time = min(35.0, max(5.0, rem * 0.4))
        Xp_pol = lp_polish_deadline(Xp, time.time() + sym_polish_time)
        try:
            vp = max_cos(Xp_pol)
        except Exception:
            vp = 1e9
        if vp < gval:
            X_global = Xp_pol.copy(); gval = vp
        # also keep if sym alone was already better (even if polish didn't beat, keep best seen)
        # if polish failed to beat global but symmetric raw was better than global, adopt raw then continue polishing
        if cand_val < gval:
            # adopt raw symmetric perturbed (already evaluated) - re-evaluate
            try:
                vraw = max_cos(Xp)
                if vraw < gval:
                    X_global = Xp.copy(); gval = vraw
            except Exception:
                pass
    # final LP polish of global best with remaining time
    if time.time() < HARD_END - 0.5:
        X_global = lp_polish_deadline(X_global, HARD_END - 0.2)
        try:
            gval = max_cos(X_global)
        except Exception:
            pass
    X_global = X_global / np.linalg.norm(X_global, axis=1, keepdims=True)
    return X_global
