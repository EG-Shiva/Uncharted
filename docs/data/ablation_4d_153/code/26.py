import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _build_full(seeds, m, q):
    K = seeds.shape[0]
    j = np.arange(m)
    a1 = 2.0 * np.pi * j / m
    a2 = 2.0 * np.pi * q * j / m
    c1 = np.cos(a1); s1 = np.sin(a1)
    c2 = np.cos(a2); s2 = np.sin(a2)
    N = K * m
    X = np.empty((N, 4))
    for k in range(K):
        s0, s1v, s2v, s3v = seeds[k]
        # first plane
        x0 = c1 * s0 - s1 * s1v
        x1 = s1 * s0 + c1 * s1v
        x2 = c2 * s2v - s2 * s3v
        x3 = s2 * s2v + c2 * s3v
        X[k*m:(k+1)*m, 0] = x0
        X[k*m:(k+1)*m, 1] = x1
        X[k*m:(k+1)*m, 2] = x2
        X[k*m:(k+1)*m, 3] = x3
    return X


def _orbit_smooth(seeds, m, q, steps, tau0, tau1, lr0, lr1, mom=0.9, rng=None):
    K = seeds.shape[0]
    N = K * m
    j = np.arange(m)
    a1 = 2.0 * np.pi * j / m
    a2 = 2.0 * np.pi * q * j / m
    c1 = np.cos(a1); s1 = np.sin(a1)
    c2 = np.cos(a2); s2 = np.sin(a2)
    vel = np.zeros_like(seeds)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    seeds = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        X = _build_full(seeds, m, q)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gm = float(np.max(G))
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        gradX = (W @ X) / row
        # tangent projection on X (helps stability, not strictly needed for chain)
        dotx = np.sum(gradX * X, axis=1, keepdims=True)
        gradX = gradX - dotx * X
        # chain to seeds: grad_seed = sum_j R_j^T gradX_{k,j}
        gs = np.zeros_like(seeds)
        for k in range(K):
            gx = gradX[k*m:(k+1)*m]
            g0 = gx[:, 0]; g1 = gx[:, 1]; g2 = gx[:, 2]; g3 = gx[:, 3]
            gs[k, 0] = np.sum(c1 * g0 + s1 * g1)
            gs[k, 1] = np.sum(-s1 * g0 + c1 * g1)
            gs[k, 2] = np.sum(c2 * g2 + s2 * g3)
            gs[k, 3] = np.sum(-s2 * g2 + c2 * g3)
        gs = gs / float(m)
        # tangent projection on seeds
        dots = np.sum(gs * seeds, axis=1, keepdims=True)
        gs = gs - dots * seeds
        # clip
        gn = np.linalg.norm(gs, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            gs = gs * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * gs
        bc = 1.0 - mom ** (t + 1)
        seeds = seeds - lr * (vel / bc)
        nr = np.linalg.norm(seeds, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        seeds = seeds / nr
    return seeds


def _free_smooth(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        gm = float(np.max(G))
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 2800:
            order = np.argpartition(c[idx], -2800)[-2800:]
            idx = idx[order]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    ORBIT_END = t0 + 60.0
    rng = np.random.default_rng(20261031)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)

    # ---- Phase A: symmetry-constrained search from scratch ----
    orbit_best_x = None
    orbit_best_v = 1e9
    configs = [
        (51, 3, [1, 2, 4, 5, 7, 10]),
        (17, 9, [1, 2, 3, 5]),
        (9, 17, [1, 2, 4]),
    ]
    # restart counts per config type
    for (m, K, qlist) in configs:
        if K * m != n:
            continue
        nrest = 3 if K <= 3 else (2 if K <= 9 else 1)
        for q in qlist:
            for r in range(nrest):
                if time.time() > ORBIT_END:
                    break
                seeds = rng.standard_normal(size=(K, d))
                seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
                # duplicate-orbit rejection on init: ensure seeds well separated
                # annealed orbit optimisation
                try:
                    seeds_opt = _orbit_smooth(seeds, m, q, steps=900,
                                              tau0=0.05, tau1=0.002,
                                              lr0=0.15, lr1=0.015, mom=0.9)
                    # second finer anneal
                    seeds_opt = _orbit_smooth(seeds_opt, m, q, steps=600,
                                              tau0=0.008, tau1=0.0008,
                                              lr0=0.05, lr1=0.008, mom=0.9)
                except Exception:
                    continue
                X = _build_full(seeds_opt, m, q)
                # duplicate-orbit rejection: if any inter-point cos ~1 (distinct points coincide)
                v = _max_cos(X)
                if v > 0.999:
                    continue
                if v < orbit_best_v:
                    orbit_best_v = v
                    orbit_best_x = X.copy()
                if time.time() > ORBIT_END:
                    break
            if time.time() > ORBIT_END:
                break
        if time.time() > ORBIT_END:
            break

    # update global best with raw orbit code
    if orbit_best_x is not None and orbit_best_v < best_v:
        best_x = orbit_best_x.copy()
        best_v = orbit_best_v

    # choose symmetry-breaking seed: prefer orbit best even if slightly worse,
    # to explore new basin, but keep record as fallback at the end
    if orbit_best_x is not None:
        cur_x = orbit_best_x.copy()
        cur_v = orbit_best_v
    else:
        cur_x = best_x.copy()
        cur_v = best_v

    # ---- Phase B: gradual symmetry breaking ----
    scales = [0.001, 0.003, 0.01, 0.03]
    for si, sc in enumerate(scales):
        if time.time() > HARD_END - 35.0:
            break
        y = cur_x.copy()
        noise = rng.standard_normal(size=(n, d)) * sc
        dot = np.sum(noise * y, axis=1, keepdims=True)
        noise = noise - dot * y
        y = y + noise
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # soft-max quench (free)
        # larger noise -> hotter start
        if sc <= 0.003:
            y = _free_smooth(y, steps=500, tau0=0.015, tau1=0.001, lr0=0.08, lr1=0.008)
        elif sc <= 0.01:
            y = _free_smooth(y, steps=700, tau0=0.03, tau1=0.001, lr0=0.10, lr1=0.008)
        else:
            y = _free_smooth(y, steps=900, tau0=0.05, tau1=0.0012, lr0=0.12, lr1=0.008)
        v = _max_cos(y)
        # LP polish at each stage (short)
        dl = min(5.0, max(1.0, (HARD_END - 30.0 - time.time()) / (len(scales) - si + 1)))
        if time.time() + 1.0 < HARD_END - 25.0:
            try:
                y2, v2 = _lp_polish(y, time.time() + dl)
            except Exception:
                y2, v2 = y, v
        else:
            y2, v2 = y, v
        if v2 < cur_v:
            cur_x, cur_v = y2, v2
        # also track global best
        if cur_v < best_v:
            best_x, best_v = cur_x.copy(), cur_v
        # if this stage failed to improve, keep perturbing from cur (not from bad y)
        # add an extra quench from best orbit-free mix on last stage
        if time.time() > HARD_END - 35.0:
            break

    # ---- Phase C: spend remaining time polishing the best ----
    # alternate free quench (small) + LP on the current global best
    if cur_v < best_v + 1e-12:
        # cur is best, polish it
        target = cur_x
    else:
        target = best_x
    # final LP with remaining budget
    try:
        # short free refinement first
        if time.time() < HARD_END - 12.0:
            # tiny shake + quench to escape shallow rattler
            y = target.copy()
            # identify rattler-like points (optional): skip, just tiny global quench
            y = _free_smooth(y, steps=400, tau0=0.006, tau1=0.0006, lr0=0.03, lr1=0.005)
            vy = _max_cos(y)
            if vy < _max_cos(target):
                target = y
        if time.time() < HARD_END - 3.0:
            target, vt = _lp_polish(target, HARD_END - 1.5)
            if vt < best_v:
                best_x, best_v = target, vt
            elif _max_cos(target) < best_v:
                best_x, best_v = target, _max_cos(target)
    except Exception:
        pass

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
