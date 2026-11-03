import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def adj_set(x, thr=1e-3):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - thr
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s, m


def build_modes(xb, mc, rng):
    n, d = xb.shape
    N = n * d
    G = xb @ xb.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    thresh = mc - 0.005
    sel = c > thresh
    cnt = int(sel.sum())
    if cnt > 1500:
        idx = np.argpartition(c, -1500)[-1500:]
        mask = np.zeros_like(sel, dtype=bool)
        mask[idx] = True
        sel = mask
        cnt = 1500
    if cnt < 10:
        return None, iu[0][sel], iu[1][sel], c[sel]
    I = iu[0][sel]
    J = iu[1][sel]
    dots = c[sel]
    # rows: block I: x[J]-dot*x[I], block J: x[I]-dot*x[J]
    BI = xb[J] - dots[:, None] * xb[I]
    BJ = xb[I] - dots[:, None] * xb[J]
    p = cnt
    rows = np.repeat(np.arange(p), 2 * d)
    cI = (I[:, None] * d + np.arange(d)).ravel()
    cJ = (J[:, None] * d + np.arange(d)).ravel()
    cols = np.empty(2 * p * d, dtype=int)
    cols[0::2 * d] = 0  # placeholder
    # interleave: first d entries per row are I-block, next d are J-block
    # build via tiling
    cols = np.concatenate([cI.reshape(p, d), cJ.reshape(p, d)], axis=1).ravel()
    vals = np.concatenate([BI, BJ], axis=1).ravel()
    try:
        from scipy.sparse import coo_matrix as coo
        Jmat = coo((vals, (rows, cols)), shape=(p, N)).tocsr()
    except Exception:
        return None, I, J, dots
    # smallest singular vectors
    k = min(25, min(p, N) - 2)
    if k < 1:
        return None, I, J, dots
    try:
        from scipy.sparse.linalg import svds
        v0 = rng.standard_normal(N)
        # svds smallest magnitude
        _, _, Vh = svds(Jmat, k=k, which='SM', tol=1e-3, maxiter=120, v0=v0)
        # Vh: k x N, rows are right singular vectors; svds returns sorted ascending? ensure order by s
        return Vh, I, J, dots
    except Exception:
        return None, I, J, dots


def tangent_project(vmat, x):
    # vmat: (N,) or (k,N)
    n, d = x.shape
    if vmat.ndim == 1:
        V = vmat.reshape(n, d).copy()
        dot = np.sum(V * x, axis=1, keepdims=True)
        V -= dot * x
        return V
    else:
        k = vmat.shape[0]
        V = vmat.reshape(k, n, d).copy()
        dot = np.sum(V * x[None, :, :], axis=2, keepdims=True)
        V -= dot * x[None, :, :]
        return V


def cluster_rotate(xb, mc, rng):
    n, d = xb.shape
    G = xb @ xb.T
    np.fill_diagonal(G, -2.0)
    # overloaded centre: most neighbours above mc-0.005 else max row
    thr = mc - 0.005
    counts = (G > thr).sum(axis=1)
    # pick among top-5 overloaded randomly
    order = np.argsort(-counts)
    top = order[:5]
    centre = int(top[rng.integers(0, len(top))])
    row = G[centre]
    nbr_order = np.argsort(-row)
    # exclude self (self is -2 so last)
    k_neigh = int(rng.integers(10, 26))  # 10..25
    k_neigh = min(k_neigh, n - 1)
    neigh = nbr_order[:k_neigh].tolist()
    cluster = [centre] + neigh
    # random 2-plane
    A = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]
    v = Q[:, 1]
    ang = float(rng.uniform(0.05, 0.25)) * (1.0 if rng.random() < 0.5 else -1.0)
    co, si = np.cos(ang), np.sin(ang)
    y = xb.copy()
    for idx in cluster:
        p = y[idx]
        a = float(p @ u)
        b = float(p @ v)
        w = p - a * u - b * v
        a2 = co * a - si * b
        b2 = si * a + co * b
        p2 = w + a2 * u + b2 * v
        nrm = np.linalg.norm(p2)
        if nrm > 1e-12:
            p2 /= nrm
        y[idx] = p2
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_T = 122.0
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        x0 = rng.standard_normal((95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = x_best.shape

    # record adjacency for topology comparison
    if xr is not None:
        rec_adj, rec_m = adj_set(xr, thr=1e-3)
        rec_val = float(max_cos(xr))
    else:
        rec_adj, rec_m = adj_set(x_best, thr=1e-3)
        rec_val = best_val

    x_base = x_best.copy()
    base_val = best_val

    alphas = [0.02, 0.05, 0.1, 0.2]

    outer = 0
    while time.time() - t0 < EXPL_T:
        outer += 1
        if time.time() - t0 > BUDGET - 40:
            break
        # recompute base value
        base_val = max_cos(x_base)
        # build soft modes around x_base
        try:
            Vh, Iact, Jact, dact = build_modes(x_base, base_val, rng)
        except Exception:
            Vh = None
        trials = []
        if Vh is not None:
            k = Vh.shape[0]
            Vproj = tangent_project(Vh, x_base)  # k x n x d
            # choose subset: smallest 2 (first rows? svds ascending) + random
            # svds returns ascending order, so row 0 is smallest
            idxs = list(range(min(2, k)))
            perm = rng.permutation(k).tolist()
            for p_ in perm:
                if len(idxs) >= 6:
                    break
                if p_ not in idxs:
                    idxs.append(p_)
            for mi in idxs:
                # two alphas per mode + random sign
                a_choices = rng.choice(len(alphas), size=2, replace=False)
                for ai in a_choices:
                    alpha = alphas[int(ai)]
                    if rng.random() < 0.5:
                        alpha = -alpha
                    V = Vproj[mi]
                    # scale: V has unit-ish norm; use as is
                    xt = x_base + alpha * V
                    xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                    trials.append(('soft', xt))
                if len(trials) >= 12:
                    break
        else:
            # fallback: random tangent kicks
            for _ in range(4):
                V = rng.standard_normal((n, d))
                dot = np.sum(V * x_base, axis=1, keepdims=True)
                V -= dot * x_base
                # normalize overall
                nrm = np.linalg.norm(V)
                if nrm < 1e-12:
                    continue
                V /= nrm
                alpha = float(rng.choice(alphas)) * (1.0 if rng.random() < 0.5 else -1.0)
                xt = x_base + alpha * V
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                trials.append(('rand', xt))
        # cluster moves: 4-6
        nclus = 5
        for _ in range(nclus):
            try:
                yc = cluster_rotate(x_base, base_val, rng)
                trials.append(('clus', yc))
            except Exception:
                continue
        # cap trials to fit time
        # screen each with 0.5s LP
        for typ, xt in trials:
            if time.time() - t0 > EXPL_T:
                break
            if time.time() - t0 > BUDGET - 40:
                break
            try:
                xp = lp_polish(xt, seconds=0.5, radius_init=0.005)
            except Exception:
                continue
            v = max_cos(xp)
            if v < best_val - 1e-12:
                # beat record -> long polish
                remain = BUDGET - (time.time() - t0)
                lp2 = min(4.0, max(1.0, remain - 12.0))
                try:
                    if lp2 > 0.8:
                        xp2 = lp_polish(xp, seconds=lp2, radius_init=1e-3)
                        v2 = max_cos(xp2)
                        if v2 < v:
                            xp, v = xp2, v2
                except Exception:
                    pass
                x_best = xp.copy()
                best_val = float(v)
                x_base = xp.copy()
                base_val = float(v)
                # update rec? keep original rec_adj for diversity check
            elif v < best_val + 1e-4:
                # check topology differing from record's
                try:
                    s, m_ = adj_set(xp, thr=1e-3)
                except Exception:
                    continue
                if s != rec_adj:
                    # valuable near-tie with new topology -> long polish
                    remain = BUDGET - (time.time() - t0)
                    lp2 = min(3.0, max(0.8, remain - 12.0))
                    try:
                        if lp2 > 0.8:
                            xp2 = lp_polish(xp, seconds=lp2, radius_init=8e-4)
                            v2 = max_cos(xp2)
                            if v2 < v + 5e-5:
                                xp, v = xp2, v2
                                s, m_ = adj_set(xp, thr=1e-3)
                    except Exception:
                        pass
                    if v < best_val - 1e-12:
                        x_best = xp.copy()
                        best_val = float(v)
                        x_base = xp.copy()
                    else:
                        # move base to new basin for further exploration
                        # keep best, but explore from here next iteration
                        x_base = xp.copy()
                        base_val = float(v)
                        # occasional: if much worse than best, revert with prob
                        if rng.random() < 0.3:
                            x_base = x_best.copy()
                else:
                    # same topology, no long polish; small chance to drift
                    if rng.random() < 0.02 and v < base_val:
                        x_base = xp.copy()
            # else discard
        # if base drifted too far above best, pull back
        if base_val > best_val + 2e-4 or outer % 6 == 0:
            # mostly stay near best, sometimes stay in new basin
            if rng.random() < 0.7:
                x_base = x_best.copy()
                base_val = best_val

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 1.5, 4.0, 40.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 0.8
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 18.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
