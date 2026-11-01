import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _run_full(X0, betas=(80.0, 250.0, 600.0), maxiter=150):
    n, d = X0.shape
    IU = np.triu_indices(n, 1)
    y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    for b in betas:
        beta = float(b)
        def fg(yf, _beta=beta):
            Y = yf.reshape(n, d)
            nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / nr
            g = (X @ X.T)[IU]
            m = float(np.max(g))
            e = np.exp(_beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            F = m + np.log(S) / _beta
            w = e / S
            P = np.zeros((n, n))
            P[IU] = w
            P = P + P.T
            GX = P @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return F, GY.ravel()
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(n, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish_full(x, deadline, r0=8e-4, cap_p=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.35:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng = np.random.default_rng(987654321)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best:
                        best = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    curX = bestX.copy()
    cur = best

    ks_cycle = [4, 6, 3, 8, 5, 7, 2, 9, 4, 6]
    sig_cycle = [0.12, 0.25, 0.08, 0.35, 0.15]
    it = 0
    # quick initial LP squeeze of incumbent if time allows (1s) -- already polished, skip mostly
    while time.time() < T_END - 18.0:
        it += 1
        time_left = T_END - 18.0 - time.time()
        if time_left < 1.0:
            break
        mode = it % 4  # 0,1,2 block repair ; 3 global shake
        if mode != 3:
            k = ks_cycle[(it - 1) % len(ks_cycle)]
            # victim clique: worst point + nearest neighbours
            try:
                pm, G = _per_max(curX)
            except Exception:
                pm = None
            if pm is None:
                w = int(rng.integers(0, n))
                G = curX @ curX.T
            else:
                # add small random tie-break to vary seed among worst
                order_w = np.argsort(-pm + 1e-6 * rng.standard_normal(n))
                w = int(order_w[0 if (rng.random() < 0.7) else int(rng.integers(0, 3))])
            nbr = np.argsort(-G[w])
            nbr = [int(v) for v in nbr if v != w]
            victims = [w] + nbr[:k-1]
            # sequential far-hole reinsertion
            X0 = curX.copy()
            # temporarily mark victims as invalid
            fixed_mask = np.ones(n, dtype=bool)
            for v in victims:
                fixed_mask[v] = False
            # random fill order
            fill_order = victims.copy()
            rng.shuffle(fill_order)
            ncand = 22000
            for v in fill_order:
                Xf = X0[fixed_mask]
                cand = rng.standard_normal((ncand, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                sc = cand @ Xf.T
                mx = np.max(sc, axis=1)
                # pick among best B for diversity
                B = 24
                idx = np.argpartition(mx, B - 1)[:B]
                pick = idx[int(rng.integers(0, B))]
                X0[v] = cand[pick]
                fixed_mask[v] = True
            # anneal: low beta first to allow topology change
            if k >= 6:
                betas = (60.0, 160.0, 400.0)
                mi = 120
            else:
                betas = (80.0, 250.0, 600.0)
                mi = 130
            try:
                Xopt = _run_full(X0, betas=betas, maxiter=mi)
            except Exception:
                continue
        else:
            sig = sig_cycle[(it // 4) % len(sig_cycle)]
            X0 = curX + sig * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            try:
                Xopt = _run_full(X0, betas=(70.0, 200.0, 500.0), maxiter=120)
            except Exception:
                continue
        s = _max_cos(Xopt)
        # short LP squeeze only if promising (within 3e-3 of best) to save time
        if s < best + 0.003 and time.time() < T_END - 20.0:
            dl = min(time.time() + 2.2, T_END - 18.0)
            try:
                Xopt = _lp_polish_full(Xopt, dl, r0=6e-4)
            except Exception:
                pass
            s = _max_cos(Xopt)
        if s < best - 1e-12:
            best = s
            bestX = Xopt.copy()
            curX = Xopt.copy()
            cur = s
        elif s < cur - 1e-12:
            curX = Xopt.copy()
            cur = s
        else:
            # basin-hopping random walk: occasionally accept uphill to keep moving
            if it % 6 == 0 and s < cur + 0.006:
                curX = Xopt.copy()
                cur = s
            elif s < cur + 0.001 and rng.random() < 0.15:
                curX = Xopt.copy()
                cur = s
            # periodically restart walk from best
            if it % 11 == 0:
                curX = bestX.copy()
                cur = best
        # safety: keep finite
        if not np.all(np.isfinite(curX)):
            curX = bestX.copy()
            cur = best
    # final intensive polish of best with remaining time
    try:
        if time.time() < T_END - 1.0:
            # one more LSE at high beta then LP to convergence
            Xb = _run_full(bestX, betas=(300.0, 800.0), maxiter=120)
            sb = _max_cos(Xb)
            if sb < best:
                best, bestX = sb, Xb
    except Exception:
        pass
    try:
        bestX = _lp_polish_full(bestX, T_END - 0.5, r0=7e-4)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
