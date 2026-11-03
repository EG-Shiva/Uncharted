import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_smooth(X, steps, tau0, tau1, lr0, lr1, deadline):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for k in range(steps):
        if time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        # sparse cut to save exp
        # keep pairs within 0.03 of max (exp(-0.03/tau) tiny)
        W = np.exp((G - mg) / tau)
        # threshold
        W[G < mg - 0.03] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _squeeze(X, cur, step0, iters, margin, tau, deadline):
    bestX = X.copy()
    best = cur
    curX = X.copy()
    curV = cur
    step = step0
    for _ in range(iters):
        if time.time() > deadline:
            break
        G = curX @ curX.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp((G - cmax) / tau)
        E[G < cmax - margin] = 0.0
        np.fill_diagonal(E, 0.0)
        if E.sum() == 0:
            break
        grad = E @ curX
        dot = np.sum(grad * curX, axis=1, keepdims=True)
        gt = grad - dot * curX
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-14)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        Xt = _norm(curX - step * direc)
        vt = _maxcos(Xt)
        if vt < curV:
            curX = Xt
            curV = vt
            if vt < best:
                best = vt
                bestX = Xt.copy()
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.5
            if step < 1e-12:
                break
    return bestX, best, curX, curV

def _void_replace(X, rng, n_replace=1, n_cand=4096):
    n, d = X.shape
    Xn = X.copy()
    for _ in range(n_replace):
        G = Xn @ Xn.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        k = int(np.argmax(rowmax))
        C = rng.standard_normal((n_cand, d))
        C = _norm(C)
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xr = Xn[mask]
        D = C @ Xr.T
        worst = D.max(axis=1)
        ib = int(np.argmin(worst))
        Xn[k] = C[ib]
    return _norm(Xn)

def _crossover(A, B, rng):
    # align B to A by orthogonal Procrustes
    try:
        M = B.T @ A
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
        Ba = B @ Q
    except Exception:
        Ba = B
    n = A.shape[0]
    # random mask: take ~half from each
    perm = rng.permutation(n)
    h = n // 2
    C = A.copy()
    C[perm[h:]] = Ba[perm[h:]]
    # small jitter to break exact symmetry
    C = C + 0.002 * rng.standard_normal(C.shape)
    return _norm(C)

def _lp_polish(X, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline_hop = t0 + 138.0
    deadline_final = t0 + BUDGET - 1.0
    rng = np.random.default_rng(42)
    np.random.seed(1)
    n, d = 93, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    # elites population
    elites = [(best_val, best.copy())]

    # quick initial squeeze of loaded point
    try:
        dl = t0 + 8.0
        b1 = _adam_smooth(best.copy(), 600, 0.006, 0.001, 0.01, 0.003, dl)
        v1 = _maxcos(b1)
        b1b, bv, _, _ = _squeeze(b1, v1, 0.002, 250, 0.004, 0.0008, dl)
        if bv < best_val:
            best_val, best = bv, b1b.copy()
            elites = [(best_val, best.copy())]
    except Exception:
        pass

    sigmas = [0.006, 0.015, 0.035, 0.07, 0.12, 0.20]
    trial = 0
    while time.time() < deadline_hop:
        if time.time() - t0 > BUDGET - 35:
            break
        mode = trial % 6
        sigma = sigmas[mode]
        try:
            if mode == 4 and len(elites) >= 2:
                # crossover of two elites
                i1, i2 = 0, 1
                if len(elites) > 2:
                    i1 = int(rng.integers(len(elites)))
                    i2 = int(rng.integers(len(elites)))
                    if i2 == i1:
                        i2 = (i2 + 1) % len(elites)
                A = elites[i1][1]
                B = elites[i2][1]
                Xt = _crossover(A, B, rng)
                Xt = Xt + sigma * 0.3 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
                steps = 1100
            elif mode == 5:
                # void reinsert + shake
                Xt = _void_replace(best, rng, n_replace=2, n_cand=6000)
                Xt = Xt + 0.008 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
                steps = 1200
            elif mode == 3:
                Xt = _void_replace(best, rng, n_replace=1, n_cand=8000)
                Xt = Xt + 0.004 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
                steps = 1000
            elif mode == 2:
                # medium kick + one worst reinsert among candidates
                Xt = best + sigma * rng.standard_normal(best.shape)
                Xt = _norm(Xt)
                # replace worst one with void candidate (cheap)
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                k = int(np.argmax(rm))
                C = _norm(rng.standard_normal((3000, d)))
                others = np.delete(Xt, k, axis=0)
                ib = int(np.argmin((C @ others.T).max(axis=1)))
                Xt[k] = C[ib]
                Xt = _norm(Xt)
                steps = 1200
            else:
                Xt = best + sigma * rng.standard_normal(best.shape)
                Xt = _norm(Xt)
                steps = 800 if mode == 0 else 1000

            dl_inner = min(deadline_hop + 5, deadline_final - 5)
            # two-temperature adam: high then low
            Xt = _adam_smooth(Xt, steps, 0.012, 0.0012, 0.008, 0.002, dl_inner)
            # extra low-temp focus on ~500 near contacts
            Xt = _adam_smooth(Xt, 300, 0.0012, 0.00025, 0.004, 0.0015, dl_inner)
            vc = _maxcos(Xt)
            XtB, vb, _, _ = _squeeze(Xt, vc, 0.0025, 220, 0.003, 0.0007, dl_inner)
            if vb < best_val - 1e-12 and np.all(np.isfinite(XtB)):
                best_val, best = vb, XtB.copy()
                elites.append((best_val, best.copy()))
                elites.sort(key=lambda t: t[0])
                elites = elites[:4]
            elif vc < best_val - 1e-12 and np.all(np.isfinite(Xt)):
                # keep without squeeze? use squeezed best for elites
                best_val, best = vb, XtB.copy()
                elites.append((best_val, best.copy()))
                elites.sort(key=lambda t: t[0])
                elites = elites[:4]
            else:
                # keep diversity: add second-best to elites if good enough
                if vb < elites[-1][0] and len(elites) < 4 and np.all(np.isfinite(XtB)):
                    elites.append((vb, XtB.copy()))
                    elites.sort(key=lambda t: t[0])
        except Exception:
            pass
        trial += 1
        if trial > 500:
            break

    # final intensive polish on best
    try:
        remain = deadline_final - time.time()
        if remain > 8:
            # long low-temp adam
            fast_dl = deadline_final - min(12.0, remain * 0.45)
            best = _adam_smooth(best, 2500, 0.004, 0.00018, 0.006, 0.0012, fast_dl)
            v = _maxcos(best)
            best_b, bv, best_c, cv = _squeeze(best, v, 0.002, 600, 0.002, 0.0005, fast_dl)
            if bv < best_val:
                best_val, best = bv, best_b.copy()
            elif cv < best_val:
                best_val, best = cv, best_c.copy()
            # LP finish with remaining time
            lp_time = deadline_final - time.time()
            if lp_time > 4:
                try:
                    yp = _lp_polish(best, time.time() + min(lp_time, 22.0))
                    vp = _maxcos(yp)
                    if vp < best_val and np.all(np.isfinite(yp)):
                        best_val, best = vp, yp
                except Exception:
                    pass
            # last tiny greedy with very small margin
            try:
                v = _maxcos(best)
                best_b2, bv2, _, _ = _squeeze(best, v, 0.0006, 400, 0.0012, 0.00025, deadline_final)
                if bv2 < best_val:
                    best_val, best = bv2, best_b2.copy()
            except Exception:
                pass
    except Exception:
        pass

    best = _norm(np.nan_to_num(best))
    best = _norm(best)
    return best
