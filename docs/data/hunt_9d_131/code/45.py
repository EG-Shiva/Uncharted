import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G >= (m - tol)
    deg = C.sum(axis=1)
    return deg, m

def _adam_tau(Xc, tau, steps, lr, T0, T1, rng_seed=0):
    n, d = Xc.shape
    X = Xc.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu = np.triu_indices(n, 1)
    for k in range(steps):
        f = k / max(1, steps - 1)
        # exponential anneal T
        if T0 > 0 and T1 > 0:
            T = T0 * ((T1 / T0) ** f)
        else:
            T = T0
        G = X @ X.T
        # upper pairs
        c = G[iu]
        v = (c - tau) / T
        # stable softplus + sigmoid
        s = np.empty_like(v)
        sig = np.empty_like(v)
        hi = v > 20
        lo = v < -20
        mid = ~(hi | lo)
        s[hi] = v[hi]; sig[hi] = 1.0
        ev = np.exp(v[lo]); s[lo] = ev; sig[lo] = ev
        evm = np.exp(v[mid]); s[mid] = np.log1p(evm); sig[mid] = evm / (1.0 + evm)
        # dP/dc
        w = 2.0 * s * sig / T
        # build full symmetric weight
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # Adam
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (k + 1)
        c2 = 1 - b2 ** (k + 1)
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep tangent part only for stability
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        # clip step norm
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.02
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * scale
        X = X - step
        X = _norm(X)
    return X

def _refine_single(z, Xref, steps=250, T=0.004, lr=0.03):
    z = z / max(1e-12, np.linalg.norm(z))
    for _ in range(steps):
        dots = Xref @ z
        m = float(np.max(dots))
        e = np.exp((dots - m) / T)
        w = e / (np.sum(e) + 1e-300)
        g = w @ Xref
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
        lr *= 0.999
    return z

def _reinsert_rattlers(core, n_total, rng, n_sample=20000):
    n_core = core.shape[0]
    d = core.shape[1]
    n_rat = n_total - n_core
    Xlist = [core.copy()]
    # current full set for evaluation
    cur = core.copy()
    outs = []
    for r in range(n_rat):
        # sample candidates in chunks to save memory
        best_z = None
        best_v = 1e9
        chunk = 5000
        done = 0
        while done < n_sample:
            cs = min(chunk, n_sample - done)
            Z = rng.standard_normal((cs, d))
            Z = _norm(Z)
            D = Z @ cur.T
            worst = D.max(axis=1)
            idx = int(np.argmin(worst))
            if worst[idx] < best_v:
                best_v = float(worst[idx])
                best_z = Z[idx].copy()
            done += cs
        # local refine against cur
        best_z = _refine_single(best_z, cur, steps=300, T=0.004, lr=0.03)
        # second small sampling around best to escape
        outs.append(best_z)
        cur = np.vstack([cur, best_z[None, :]])
    full = np.vstack([core] + [o[None, :] for o in outs])
    return full

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 178.0
    LP_RESERVE = 20.0
    FEAS_DEADLINE = DEADLINE - LP_RESERVE - 2.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    curX = best.copy()
    curVal = best_val
    # identify rattlers: 9 lowest degree with 1e-7 tol
    deg, _ = _degrees(curX, tol=1e-7)
    order = np.argsort(deg)
    # ensure exactly 9 rattlers
    n_rat = 9
    rat_idx = order[:n_rat]
    core_idx = order[n_rat:]
    # sort core_idx to keep deterministic
    core_idx = np.sort(core_idx)
    # target stepping: start ambitious 0.4220 then -0.0005
    # but adapt to curVal: first tau = min(0.4220, curVal-0.0002)
    first_tau = min(0.4220, curVal - 0.0002)
    tau = first_tau
    step = 0.0005
    # feasibility loop
    attempt = 0
    while time.time() < FEAS_DEADLINE and attempt < 12:
        attempt += 1
        if tau >= curVal - 1e-9:
            tau = curVal - step
        # time check: need at least ~8s for an attempt
        if FEAS_DEADLINE - time.time() < 8:
            break
        # split core / rattlers from CURRENT best topology (recompute each attempt)
        deg_c, _ = _degrees(curX, tol=1e-7)
        ord_c = np.argsort(deg_c)
        rat_idx_c = np.sort(ord_c[:n_rat])
        core_idx_c = np.sort(ord_c[n_rat:])
        Xcore0 = curX[core_idx_c]
        # Phase A: core-only Adam under tau penalty, T 3e-3 -> 2e-4
        # steps adapt to remaining time
        steps_core = 1400
        # shorter if late
        if attempt > 4:
            steps_core = 900
        Xcore = _adam_tau(Xcore0, tau, steps=steps_core, lr=0.004, T0=3e-3, T1=2e-4)
        # Phase B: reinsert rattlers with >=20000 samples
        full = _reinsert_rattlers(Xcore, n, rng, n_sample=20000)
        # Phase C: joint re-polish all 131 under tau penalty, small T
        # use two-stage: first larger T then tiny T
        full = _adam_tau(full, tau, steps=500, lr=0.002, T0=1e-3, T1=2e-4)
        full = _adam_tau(full, tau, steps=300, lr=0.001, T0=2e-4, T1=2e-4)
        v = _max_cos(full)
        # accept only if below previous tau target and improves curVal
        # per spec: accept if true max below previous tau (here: tau) and below curVal
        if v < curVal - 2e-9 and v <= tau + 5e-06:
            curX = full
            curVal = v
            # advance tau downward
            step = min(step * 1.15, 0.0006)
            tau = curVal - step
        else:
            # check if any improvement at all (even if not reaching tau) -> take it as new basin
            if v < curVal - 2e-9:
                curX = full
                curVal = v
                step *= 0.6
                tau = curVal - step
            else:
                step *= 0.5
                tau = curVal - step
                if step < 5e-05:
                    step = 5e-05
                    # perturb core slightly to try new basin with same tau
                    # small random kick to rattlers only
                    tmp = curX.copy()
                    # kick the 9 rattlers strongly
                    deg2, _ = _degrees(tmp, tol=1e-7)
                    o2 = np.argsort(deg2)[:n_rat]
                    tmp[o2] = _norm(tmp[o2] + 0.05 * rng.standard_normal((n_rat, d)))
                    curX_try = _adam_tau(tmp, tau, steps=400, lr=0.002, T0=1e-3, T1=3e-4)
                    vt = _max_cos(curX_try)
                    if vt < curVal:
                        curX = curX_try
                        curVal = vt
        # early exit if excellent
        if curVal < 0.415:
            break
    # final LP polish with remaining time (reserve)
    try:
        curX = _lp_polish(curX, DEADLINE - 0.5)
    except Exception:
        pass
    # safety normalize
    curX = _norm(curX)
    return curX
