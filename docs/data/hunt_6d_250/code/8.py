import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _smooth_opt(X0, beta, maxiter):
    n, d = X0.shape
    Y0 = X0.copy()

    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0:
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()

    res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-20,
                            "gtol": 1e-12, "maxfun": maxiter * 5,
                            "maxcor": 20})
    Y = res.x.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(15.0 * radius, 5e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 6000:
            thr = np.sort(iu_c[sel])[-6000]
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    HOP_DEADLINE = t_start + 138.0
    LP_DEADLINE = t_start + 163.0
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n, d = 250, 6
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cand_list = []
    if x0 is not None:
        try:
            x0 = np.asarray(x0, dtype=np.float64)
            if x0.shape == (n, d):
                cand_list.append(x0 / np.linalg.norm(x0, axis=1, keepdims=True))
        except Exception:
            pass
    if rec is not None:
        try:
            rec = np.asarray(rec, dtype=np.float64)
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

    def remain():
        return t_start + TIME_BUDGET - time.time()

    # Stage 1: short anneal from best
    cur = best.copy()
    sched = [(500.0, 150), (2000.0, 150), (8000.0, 180), (20000.0, 120)]
    for beta, it in sched:
        if time.time() > t_start + 40.0:
            break
        try:
            cur = _smooth_opt(cur, beta, it)
        except Exception:
            break
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            best = cur.copy()

    # Stage 2: basin hopping with kicks + void reinsertion
    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    while time.time() < HOP_DEADLINE and trial < 80:
        trial += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order_w = np.argsort(-rowmax)
            order_r = np.argsort(rowmax)
        except Exception:
            order_w = np.arange(n)
            order_r = np.arange(n)
        X0 = best.copy()
        mode = trial % 6
        try:
            if mode == 0:  # kick worst K
                K = 4 + (trial % 9)
                idx = order_w[:K]
                sig = 0.010 if (trial % 2 == 0) else 0.005
                X0[idx] += sig * rng.standard_normal((K, d))
                # also jitter a few rattlers
                ridx = order_r[:3]
                X0[ridx] += 0.003 * rng.standard_normal((3, d))
            elif mode == 1:  # global small shake + worst medium
                K = 6 + (trial % 6)
                idx = order_w[:K]
                X0[idx] += 0.008 * rng.standard_normal((K, d))
                X0 += 0.0015 * rng.standard_normal((n, d))
            elif mode == 2:  # void reinsertion of worst point
                w = int(order_w[0])
                if trial % 2 == 0 and n > 1:
                    w2 = int(order_w[1])
                    ws = [w, w2]
                    nc = 256
                    cand = rng.standard_normal((nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    # score by max dot to all except ws
                    S = cand @ X0.T
                    for ww in ws:
                        S[:, ww] = -2.0
                    mx = S.max(axis=1)
                    bi = int(np.argmin(mx))
                    X0[w] = cand[bi]
                    # jitter second worst slightly
                    X0[w2] += 0.004 * rng.standard_normal(d)
                else:
                    nc = 384
                    cand = rng.standard_normal((nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    S = cand @ X0.T
                    S[:, w] = -2.0
                    mx = S.max(axis=1)
                    bi = int(np.argmin(mx))
                    X0[w] = cand[bi]
                    X0 += 0.001 * rng.standard_normal((n, d))
            elif mode == 3:  # heavy single kick
                w = int(order_w[trial % 3])
                X0[w] += 0.03 * rng.standard_normal(d)
                nb = 4
                idx = order_w[1:1 + nb]
                X0[idx] += 0.006 * rng.standard_normal((nb, d))
            elif mode == 4:  # shake overloaded + random
                K = 10 + (trial % 10)
                idx = order_w[:K]
                X0[idx] += 0.012 * rng.standard_normal((K, d))
                rsel = rng.integers(0, n, size=6)
                X0[rsel] += 0.006 * rng.standard_normal((6, d))
            else:  # rattler reshuffle: move 2 rattlers to voids
                for _ in range(2):
                    w = int(order_r[rng.integers(0, 10)])
                    nc = 128
                    cand = rng.standard_normal((nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    S = cand @ X0.T
                    S[:, w] = -2.0
                    mx = S.max(axis=1)
                    bi = int(np.argmin(mx))
                    X0[w] = cand[bi]
                X0 += 0.001 * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            continue
        # re-optimize: first moderate beta short
        b1 = 3000.0 if (trial % 3 == 0) else 8000.0
        try:
            Y2 = _smooth_opt(X0, b1, 110)
        except Exception:
            continue
        v2 = _max_cos(Y2)
        # if promising, high-beta snap
        if v2 < best_val + 0.002 and time.time() < HOP_DEADLINE:
            try:
                Y2b = _smooth_opt(Y2, 20000.0, 60)
                v2b = _max_cos(Y2b)
                if v2b < v2:
                    Y2, v2 = Y2b, v2b
            except Exception:
                pass
        if v2 < best_val:
            best_val = v2
            best = Y2.copy()
        if remain() < 30.0:
            break

    # Stage 3: final LP squeeze + tiny L-BFGS alternation
    if time.time() < LP_DEADLINE:
        try:
            polished = _lp_polish(best, min(LP_DEADLINE, time.time() + 22.0))
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass
    # final tiny smoothing to clean numerical noise if time left
    if time.time() < t_start + TIME_BUDGET - 3.0:
        try:
            Yf = _smooth_opt(best, 25000.0, 60)
            vf = _max_cos(Yf)
            if vf < best_val:
                best = Yf
                best_val = vf
        except Exception:
            pass
        try:
            if time.time() < LP_DEADLINE:
                polished2 = _lp_polish(best, time.time() + 5.0)
                vp2 = _max_cos(polished2)
                if vp2 < best_val:
                    best = polished2
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    best = best / nrm
    return best
