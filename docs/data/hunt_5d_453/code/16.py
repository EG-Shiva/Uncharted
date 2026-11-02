import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _micro_greedy(X, bestV, bestX, t0, deadline, eps0=5e-7):
    n, d = X.shape
    eps = float(eps0)
    eps_max = 2e-6
    eps_min = 5e-13
    # main loop
    while True:
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
            bestX = X.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(X[i] @ X[j])
        # tangent directions pushing apart
        ti = X[j] - cij * X[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            # degenerate, tiny random kick
            eps *= 0.5
            if eps < eps_min:
                break
            continue
        ti /= ni
        tj = X[i] - cij * X[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            eps *= 0.5
            if eps < eps_min:
                break
            continue
        tj /= nj
        # m excluding i,j for fast eval
        # mask rows/cols i,j in G (G will be recomputed next iter)
        # compute m_excl
        # use copy of row/col? modify G in place is fine
        # save not needed
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_excl = float(np.max(G))
        # row j max vs others for single moves (from original X)
        # we need original dots: recompute quickly from X rows
        # original row maxima excluding i,j:
        # di0 = X @ X[i] etc not needed; m_excl already covers others-others,
        # for single-i move we also need max_{k} Xj.Xk
        # compute it:
        # Use X[j] dot all:
        rj = X @ X[j]
        rj[i] = -2.0; rj[j] = -2.0
        max_rj = float(np.max(rj))
        ri = X @ X[i]
        ri[i] = -2.0; ri[j] = -2.0
        max_ri = float(np.max(ri))
        # base for single moves: max(m_excl, max_rj) is current m without i
        # but m is current worst (i-j), so single-move new max must beat m
        improved = False
        best_try_m = m
        best_try_kind = None
        best_try_vi = None
        best_try_vj = None
        # try scales
        for scale in (eps, eps * 0.5, eps * 0.25):
            vi = X[i] - scale * ti
            vin = float(np.linalg.norm(vi))
            if vin < 1e-12:
                continue
            vi /= vin
            vj = X[j] - scale * tj
            vjn = float(np.linalg.norm(vj))
            if vjn < 1e-12:
                continue
            vj /= vjn
            # fast eval both
            di = X @ vi
            dj = X @ vj
            # mask stale entries
            # di[k] for k!=i,j is valid; di[i],di[j] invalid
            # dj similarly
            # compute maxima
            # manual max to avoid alloc: use numpy max on sliced?
            # set invalid to -2
            di[i] = -2.0; di[j] = -2.0
            dj[i] = -2.0; dj[j] = -2.0
            m1 = float(np.max(di))
            m2 = float(np.max(dj))
            vij = float(vi @ vj)
            new_m = m_excl
            if m1 > new_m:
                new_m = m1
            if m2 > new_m:
                new_m = m2
            if vij > new_m:
                new_m = vij
            if new_m < best_try_m - 1e-15:
                best_try_m = new_m
                best_try_kind = 'both'
                best_try_vi = vi
                best_try_vj = vj
                # prefer largest scale that improves, break scales?
                # keep searching smaller scale only if it is even better
                # for now continue to find best among scales
            # single i only: max = max(m_excl, max_rj, max_{k} vi.Xk, vi.Xj)
            # max_{k} vi.Xk = m1, vi.Xj = vi@X[j]
            vij1 = float(vi @ X[j])
            new_m_i = m_excl
            if max_rj > new_m_i:
                new_m_i = max_rj
            if m1 > new_m_i:
                new_m_i = m1
            if vij1 > new_m_i:
                new_m_i = vij1
            if new_m_i < best_try_m - 1e-15:
                best_try_m = new_m_i
                best_try_kind = 'i'
                best_try_vi = vi
                best_try_vj = None
            vij2 = float(X[i] @ vj)
            new_m_j = m_excl
            if max_ri > new_m_j:
                new_m_j = max_ri
            if m2 > new_m_j:
                new_m_j = m2
            if vij2 > new_m_j:
                new_m_j = vij2
            if new_m_j < best_try_m - 1e-15:
                best_try_m = new_m_j
                best_try_kind = 'j'
                best_try_vi = None
                best_try_vj = vj
        if best_try_kind is not None and best_try_m < m - 1e-15:
            if best_try_kind == 'both':
                X[i] = best_try_vi
                X[j] = best_try_vj
            elif best_try_kind == 'i':
                X[i] = best_try_vi
            else:
                X[j] = best_try_vj
            if best_try_m < bestV:
                bestV = best_try_m
                bestX = X.copy()
            eps = min(eps * 1.08, eps_max)
        else:
            eps *= 0.5
            if eps < eps_min:
                # converged for single-pair moves
                break
    return X, bestV, bestX

def _lp_micro(x, bestV, bestX, t0, deadline, radius0=3e-7):
    n, d = x.shape
    radius = float(radius0)
    iu_r, iu_c = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-13:
        if time.time() > deadline - 1.0:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        if m < bestV:
            bestV = m
            bestX = x.copy()
        tol = 20.0 * radius
        if tol < 2e-9:
            tol = 2e-9
        sel = c > m - tol
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu_r[sel]
        J = iu_c[sel]
        if cnt > 4000:
            cs = c[sel]
            kp = np.argpartition(-cs, 4000)[:4000]
            I = I[kp]; J = J[kp]
            b_ub = -cs[kp]
            p = 4000
        else:
            b_ub = -c[sel]
            p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < m - 1e-15:
            x = y
            if v < bestV:
                bestV = v
                bestX = y.copy()
            radius = min(radius * 1.5, 1e-6)
        else:
            radius *= 0.5
    return x, bestV, bestX

def solve():
    t0 = time.time()
    TIME_END = 172.0
    np.random.seed(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((453, 5))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    bestV = float("inf")
    bestX = None
    for cc in cands:
        vv = _max_cos(cc)
        if vv < bestV:
            bestV = vv
            bestX = cc.copy()
    cur = bestX.copy()
    # Phase 1: micro greedy (single-pair basin)
    cur, bestV, bestX = _micro_greedy(cur, bestV, bestX, t0, t0 + min(80.0, TIME_END - 90.0), eps0=5e-7)
    if time.time() - t0 > TIME_END - 60:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        return bestX
    cur = bestX.copy()
    # Phase 2: tiny LP for coordinated moves
    cur, bestV, bestX = _lp_micro(cur, bestV, bestX, t0, t0 + min(120.0, TIME_END - 45.0), radius0=3e-7)
    if time.time() - t0 > TIME_END - 40:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        return bestX
    cur = bestX.copy()
    # Phase 3: second micro greedy with smaller start
    cur, bestV, bestX = _micro_greedy(cur, bestV, bestX, t0, t0 + (TIME_END - 12.0), eps0=2e-7)
    if time.time() - t0 < TIME_END - 10.0:
        # final tiny LP if time remains
        cur, bestV, bestX = _lp_micro(bestX.copy(), bestV, bestX, t0, t0 + (TIME_END - 1.0), radius0=1e-7)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # guard
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    return bestX
