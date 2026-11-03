import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X0, beta, maxiter, maxcor=12):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -35.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    Y = Y / nrm
    return Y

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(np.max(iu_c))
        # include plateau: wider window to tie near-active pairs
        sel = iu_c > m - max(20.0 * radius, 8e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            thr = np.sort(iu_c[sel])[-4500]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
            x, best, radius = y, v, min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _void_candidate(X0, exclude, rng, nc=2048):
    n, d = X0.shape
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X0.T
    if np.isscalar(exclude):
        S[:, int(exclude)] = -2.0
    else:
        for w in exclude:
            S[:, int(w)] = -2.0
    mx = S.max(axis=1)
    bi = int(np.argmin(mx))
    return cand[bi]

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    HOP_DEADLINE = t_start + 132.0
    LP_DEADLINE = t_start + 163.0
    rng = np.random.default_rng(7)
    n, d = 250, 6
    cand_list = []
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
        if x0.shape == (n, d):
            cand_list.append(x0 / np.linalg.norm(x0, axis=1, keepdims=True))
    except Exception:
        pass
    try:
        rec = np.asarray(np.load("record.npy"), dtype=np.float64)
        if rec.shape == (n, d):
            cand_list.append(rec / np.linalg.norm(rec, axis=1, keepdims=True))
    except Exception:
        pass
    if len(cand_list) == 0:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand_list.append(tmp)
    best = min(cand_list, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    iu0, iu1 = np.triu_indices(n, 1)

    # Stage 1: quick anneal from best
    cur = best.copy()
    for beta, it in [(800.0, 120), (4000.0, 120), (15000.0, 100)]:
        if time.time() > t_start + 35.0:
            break
        try:
            cur = _smooth_opt(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            best = cur.copy()

    # Stage 2: basin hopping with pair-aware moves
    trial = 0
    while time.time() < HOP_DEADLINE and trial < 70:
        trial += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order_w = np.argsort(-rowmax)
            # top pairs for separation moves
            flat = G[iu0, iu1]
            topk = np.argpartition(-flat, min(500, flat.size - 1))[:500]
            # sort those 500
            topk = topk[np.argsort(-flat[topk])]
        except Exception:
            order_w = np.arange(n)
            topk = np.array([0])
        X0 = best.copy()
        mode = trial % 7
        try:
            if mode == 0:  # worst-pair separation kick
                # pick a random pair among top 50
                pi = int(rng.integers(0, 50))
                pk = int(topk[pi])
                i, j = int(iu0[pk]), int(iu1[pk])
                # push apart symmetrically + jitter neighbourhood
                step = 0.015 if (trial % 2 == 0) else 0.008
                X0[i] -= step * X0[j]
                X0[j] -= step * X0[i]
                K = 5
                idx = order_w[:K]
                X0[idx] += 0.006 * rng.standard_normal((K, d))
                X0 += 0.001 * rng.standard_normal((n, d))
            elif mode == 1:  # deep-hole void reinsert worst point
                w = int(order_w[0])
                nc = 2048 if (trial % 3 == 0) else 1024
                X0[w] = _void_candidate(X0, w, rng, nc=nc)
                X0 += 0.0012 * rng.standard_normal((n, d))
                # also nudge second worst
                w2 = int(order_w[1])
                X0[w2] += 0.004 * rng.standard_normal(d)
            elif mode == 2:  # double void reinsert (two worst)
                w1, w2 = int(order_w[0]), int(order_w[1])
                X0[w1] = _void_candidate(X0, [w1, w2], rng, nc=1536)
                # recompute with updated X0 for second
                X0[w2] = _void_candidate(X0, [w1, w2], rng, nc=1024)
                X0 += 0.0008 * rng.standard_normal((n, d))
            elif mode == 3:  # kick worst K with larger sigma
                K = 5 + (trial % 8)
                idx = order_w[:K]
                sig = 0.012 if (trial % 2 == 0) else 0.007
                X0[idx] += sig * rng.standard_normal((K, d))
                X0 += 0.001 * rng.standard_normal((n, d))
            elif mode == 4:  # shake random subset + worst
                K = 8
                idx = order_w[:K]
                X0[idx] += 0.010 * rng.standard_normal((K, d))
                rsel = rng.integers(0, n, size=10)
                X0[rsel] += 0.008 * rng.standard_normal((10, d))
            elif mode == 5:  # heavy single + pair neighbours
                pi = int(rng.integers(0, 20))
                pk = int(topk[pi])
                i, j = int(iu0[pk]), int(iu1[pk])
                X0[i] += 0.025 * rng.standard_normal(d)
                X0[j] += 0.010 * rng.standard_normal(d)
                idx = order_w[2:7]
                X0[idx] += 0.005 * rng.standard_normal((5, d))
            else:  # global small shake (basin explore)
                X0 += 0.0025 * rng.standard_normal((n, d))
                K = 4
                idx = order_w[:K]
                X0[idx] += 0.007 * rng.standard_normal((K, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            continue
        b1 = 5000.0 if (trial % 2 == 0) else 12000.0
        try:
            Y2 = _smooth_opt(X0, b1, 90)
        except Exception:
            continue
        v2 = _max_cos(Y2)
        if v2 < best_val + 0.0025 and time.time() < HOP_DEADLINE:
            try:
                Y2b = _smooth_opt(Y2, 22000.0, 50)
                v2b = _max_cos(Y2b)
                if v2b < v2:
                    Y2, v2 = Y2b, v2b
            except Exception:
                pass
        if v2 < best_val:
            best_val = v2
            best = Y2.copy()
        if t_start + TIME_BUDGET - time.time() < 32.0:
            break

    # Stage 3: alternate LP squeeze + high-beta snap
    if time.time() < LP_DEADLINE:
        try:
            polished = _lp_polish(best, min(LP_DEADLINE, time.time() + 18.0))
            vp = _max_cos(polished)
            if vp < best_val:
                best, best_val = polished, vp
        except Exception:
            pass
    if time.time() < t_start + TIME_BUDGET - 4.0:
        try:
            Yf = _smooth_opt(best, 25000.0, 60)
            vf = _max_cos(Yf)
            if vf < best_val:
                best, best_val = Yf, vf
        except Exception:
            pass
        try:
            if time.time() < LP_DEADLINE:
                p2 = _lp_polish(best, time.time() + 5.0)
                v2 = _max_cos(p2)
                if v2 < best_val:
                    best = p2
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
