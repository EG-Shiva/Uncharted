import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def gen_X_from_Vn(Vn, Q, plane_ks, cos_tab, sin_tab):
    R, D = Vn.shape
    n_planes = len(plane_ks)
    N = R * Q
    X = np.empty((N, D))
    for r in range(R):
        vr = Vn[r]
        base = vr.copy()
        for j in range(Q):
            idx = r * Q + j
            # copy fixed part once, rotate planes
            X[idx] = base
            for pi in range(n_planes):
                a = 2 * pi; b = 2 * pi + 1
                c = cos_tab[j, pi]; s = sin_tab[j, pi]
                xa = vr[a]; xb = vr[b]
                X[idx, a] = c * xa - s * xb
                X[idx, b] = s * xa + c * xb
    return X

def optimize_symmetric(Q, plane_ks, R, seed, alphas=(100.0, 400.0), maxiters=(350, 350), time_limit=30.0):
    t0 = time.time()
    D = 13
    n_planes = len(plane_ks)
    N = R * Q
    assert N == 99, (R, Q)
    # precompute cos/sin tables: Q x n_planes
    cos_tab = np.zeros((Q, n_planes))
    sin_tab = np.zeros((Q, n_planes))
    for j in range(Q):
        for pi, k in enumerate(plane_ks):
            th = 2.0 * np.pi * k * j / Q
            cos_tab[j, pi] = np.cos(th)
            sin_tab[j, pi] = np.sin(th)
    iu0, iu1 = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed)
    V0 = rng.standard_normal(size=(R, D))
    V0 /= np.linalg.norm(V0, axis=1, keepdims=True)
    cur = V0.ravel().copy()

    for stage, (alpha, maxiter) in enumerate(zip(alphas, maxiters)):
        if time.time() - t0 > time_limit:
            break
        def fun(vflat, _alpha=alpha):
            V = vflat.reshape(R, D)
            nrm = np.linalg.norm(V, axis=1, keepdims=True)
            nrm = np.maximum(nrm, 1e-12)
            Vn = V / nrm
            X = gen_X_from_Vn(Vn, Q, plane_ks, cos_tab, sin_tab)
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(_alpha * (g - m))
            s = float(e.sum()) + 1e-300
            val = m + np.log(s) / _alpha
            w = e / s
            # Grad_X = Wsym @ X
            # build symmetric weight action without full matrix for speed
            # use accumulation: Grad_X[i] = sum_j w_{ij} X_j
            Wmat = np.zeros((N, N))
            Wmat[iu0, iu1] = w
            Wmat = Wmat + Wmat.T
            Grad_X = Wmat @ X
            # backprop to Vn
            Grad_Vn = np.zeros((R, D))
            fix_start = 2 * n_planes
            for r in range(R):
                acc = np.zeros(D)
                # fixed dims sum
                if fix_start < D:
                    # sum over j
                    for j in range(Q):
                        idx = r * Q + j
                        acc[fix_start:] += Grad_X[idx, fix_start:]
                for j in range(Q):
                    idx = r * Q + j
                    gx = Grad_X[idx]
                    for pi in range(n_planes):
                        a = 2 * pi; b = 2 * pi + 1
                        c = cos_tab[j, pi]; s_ = sin_tab[j, pi]
                        ga = gx[a]; gb = gx[b]
                        acc[a] += c * ga + s_ * gb
                        acc[b] += -s_ * ga + c * gb
                Grad_Vn[r] = acc
            Grad_V = np.zeros((R, D))
            for r in range(R):
                dot = float(np.dot(Grad_Vn[r], Vn[r]))
                Grad_V[r] = (Grad_Vn[r] - Vn[r] * dot) / nrm[r, 0]
            return val, Grad_V.ravel()
        try:
            res = minimize(fun, cur, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-9, "maxfun": int(maxiter*5)})
            cur = res.x
        except Exception:
            break
        if time.time() - t0 > time_limit:
            break
    V = cur.reshape(R, D)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    X = gen_X_from_Vn(V, Q, plane_ks, cos_tab, sin_tab)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, max_cos(X)

def soft_refine(v, schedule, beta=0.9):
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    n, d = cur.shape
    for (alpha, steps, lr0, lr1) in schedule:
        mom[:] = 0.0
        for s in range(steps):
            G = cur @ cur.T
            np.fill_diagonal(G, -10.0)
            m = G.max()
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            Grad = W @ cur
            proj = np.sum(Grad * cur, axis=1, keepdims=True)
            Gt = Grad - proj * cur
            rs = W.sum(axis=1, keepdims=True) + 1e-12
            Gt = Gt / rs
            mom = beta * mom + (1.0 - beta) * Gt
            bc = 1.0 - beta ** (s + 1)
            direction = mom / bc
            frac = s / max(1, steps - 1)
            lr = lr0 + (lr1 - lr0) * frac
            dn = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
            direction = direction / np.maximum(1.0, dn * 5.0)
            cur = cur - lr * direction
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            if (s % 200 == 199) or (s == steps - 1):
                sc = max_cos(cur)
                if sc < best_s:
                    best_s = sc
                    best = cur.copy()
    return best, best_s

def lp_polish(x, seconds, floor=1e-9, factor=20.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(factor * radius, floor)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 6000:
            # keep tightest 6000
            order = np.argsort(-c[sel])[:6000]
            I, J = I[order], J[order]
            p = 6000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != int(sel.sum()) else -c[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def contact_stats(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    per = (G >= m - 1e-9).sum(axis=1)
    ppm = G.max(axis=1)
    return m, per, ppm

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(999)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_x = xb.copy()
    best_val = max_cos(best_x)

    # --- symmetry-restricted search from scratch ---
    sym_pool = []
    # C11: 5 planes k=1..5 + 3 fixed, 9 reps; C9: 4 planes k=1..4 + 5 fixed, 11 reps
    configs = [
        (11, [1, 2, 3, 4, 5], 9, 1101),
        (11, [1, 2, 3, 4, 5], 9, 2202),
        (9, [1, 2, 3, 4], 11, 3303),
        (9, [1, 2, 3, 4], 11, 4404),
    ]
    for (Q, ks, R, seed) in configs:
        if time.time() - t0 > 55.0:
            break
        try:
            Xs, vs = optimize_symmetric(Q, ks, R, seed,
                                        alphas=(100.0, 400.0), maxiters=(400, 400),
                                        time_limit=14.0)
        except Exception:
            continue
        sym_pool.append((vs, Xs))
        if vs < best_val - 1e-12:
            best_val = vs
            best_x = Xs.copy()
    # keep diverse symmetric seeds even if worse (within 5e-3)
    sym_pool = sorted(sym_pool, key=lambda t: t[0])[:3]

    # --- release symmetry: annealed LogSumExp ---
    release_sched = [(100.0, 350, 0.10, 0.05), (200.0, 350, 0.07, 0.035),
                     (400.0, 350, 0.055, 0.03), (600.0, 250, 0.045, 0.025)]
    short_sched = [(150.0, 300, 0.09, 0.045), (450.0, 300, 0.055, 0.03)]
    candidates = []
    for vs, Xs in sym_pool:
        # only release seeds not too bad
        if vs > best_val + 5e-3:
            continue
        candidates.append(Xs)
    candidates.append(best_x.copy())
    # dedupe by score proximity not needed; limit count
    # release each with time guard
    for ci, cand in enumerate(list(candidates)):
        if time.time() - t0 > 128.0:
            break
        # small random tangential shake to break exact symmetry before release
        if ci < len(candidates) - 1:
            pert = cand + rng.standard_normal(size=cand.shape) * 1e-4
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
            cand = pert
        try:
            xr, vr = soft_refine(cand, release_sched if ci == 0 else short_sched)
        except Exception:
            continue
        # contact-degree info for comparison (balanced vs hub)
        try:
            m_, per_, ppm_ = contact_stats(xr)
        except Exception:
            per_ = None
        if vr < best_val - 1e-12:
            best_val = vr
            best_x = xr.copy()
        # extra shake-and-restore to jump basin for symmetric seeds
        if time.time() - t0 < 120.0 and ci < 2:
            try:
                sh = xr + rng.standard_normal(size=xr.shape) * 0.02
                sh /= np.linalg.norm(sh, axis=1, keepdims=True)
                x2, v2 = soft_refine(sh, short_sched)
                if v2 < best_val - 1e-12:
                    best_val = v2
                    best_x = x2.copy()
            except Exception:
                pass

    # --- final wide-active LP then tight LP ---
    remaining = BUDGET - (time.time() - t0)
    remaining = max(6.0, min(38.0, remaining))
    # split: 60% wide, 40% tight
    wide_sec = remaining * 0.6
    tight_sec = remaining - wide_sec
    try:
        best_x = lp_polish(best_x, wide_sec, floor=3e-4, factor=30.0)
    except Exception:
        pass
    try:
        best_x = lp_polish(best_x, tight_sec, floor=1e-9, factor=20.0)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
