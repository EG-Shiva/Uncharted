import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V / n

def _run_lbfgs(fun, x0, maxiter):
    try:
        r = minimize(fun, x0, jac=True, method='L-BFGS-B',
                     options={'maxiter': int(maxiter), 'ftol': 1e-20,
                              'gtol': 1e-7, 'maxfun': int(maxiter)*6, 'maxcor': 10})
        return r.x.copy(), float(r.fun)
    except Exception:
        return x0, float('inf')

def _make_obj(alpha, lam):
    a = float(alpha); l = float(lam)
    def fun(Uf):
        N = 99; d = 13
        U = Uf.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp(a * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L0 = m + np.log(S) / a
        GX0 = (E @ X) / S
        if l <= 0:
            GX = GX0; L = L0
        else:
            Rs = E.sum(axis=1) + 1e-300  # (N,)
            ri = m + np.log(Rs) / a
            rbar = float(np.mean(ri))
            var = float(np.mean((ri - rbar) ** 2)) + 1e-24
            std = float(np.sqrt(var))
            L = L0 + l * std
            coeff = l * (ri - rbar) / (N * std + 1e-300)  # dP/dri
            w = coeff / Rs  # (N,)
            # H_ij = w_i * E_ij  (ordered)
            # grad = (H+H^T) @ X
            HE = E * w[:, None]
            GXp = (HE + HE.T) @ X
            GX = GX0 + GXp
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / nr
        return float(L), GU.ravel().astype(float)
    return fun

def _anneal(X0, schedule, deadline):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    for alpha, lam, mit in schedule:
        if time.time() > deadline:
            break
        fun = _make_obj(alpha, lam)
        U0, _ = _run_lbfgs(fun, U0, mit)
    U = U0.reshape(N, d)
    return _norm_rows(U)

def _far_reinsert(base, idx, rng, n_cand=1500):
    x0 = base.copy()
    for k in list(idx):
        cand = rng.standard_normal(size=(n_cand, base.shape[1]))
        cand = _norm_rows(cand)
        others = np.delete(x0, int(k), axis=0)
        md = (cand @ others.T).max(axis=1)
        x0[int(k)] = cand[np.argmin(md)]
    return x0

def _balanced_gd(x, steps, lr0, delta, topk, deadline):
    cur = x.copy()
    best = _max_cos(cur); bx = cur.copy()
    n = cur.shape[0]
    for s in range(int(steps)):
        if time.time() > deadline:
            break
        if s % 50 == 0 and s > 0:
            sc = _max_cos(cur)
            if sc < best:
                best = sc; bx = cur.copy()
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - delta).astype(float)
        np.fill_diagonal(Mask, 0.0)
        idx = np.argsort(-G, axis=1)[:, :int(topk)]
        Bal = np.zeros_like(Mask)
        # vectorized scatter
        rows = np.repeat(np.arange(n), int(topk))
        cols = idx.ravel()
        Bal[rows, cols] = 1.0
        np.fill_diagonal(Bal, 0.0)
        Bal = np.maximum(Bal, Bal.T)
        Comb = Mask + 0.4 * Bal
        cnt = Comb.sum(axis=1, keepdims=True) + 1e-12
        Gr = Comb @ cur
        pr = np.sum(Gr * cur, axis=1, keepdims=True)
        Gt = (Gr - pr * cur) / cnt
        cur = cur - lr0 * (1.0 - 0.6 * s / max(int(steps), 1)) * Gt
        cur = _norm_rows(cur)
        if s == int(steps) // 2:
            delta = max(delta * 0.3, 2e-06)
    sc = _max_cos(cur)
    if sc < best:
        best = sc; bx = cur.copy()
    return bx, best

def _lp_polish(x, seconds, deadline):
    x = _norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x); bx = x.copy()
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14 and it < 15:
        it += 1
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = max(25.0 * radius, 5e-7)
        sel = c > m - thr
        if int(np.sum(sel)) < 800:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:800]] = True
            sel = sel | s2
        if int(np.sum(sel)) > 2200:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:2200]] = True
            sel = s2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        cs = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={'time_limit': 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.4, 0.006)
        else:
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    T_EXPL = t0 + 132.0
    T_LP = T_END - 8.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.asarray(xb).shape == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.asarray(xr).shape == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if len(cands) == 0:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    cands.sort(key=lambda z: _max_cos(z))
    best_x = cands[0].copy()
    best_val = _max_cos(best_x)
    cur_x = best_x.copy(); cur_val = best_val
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.32]
    sched_sets = [
        [(60.0, 0.6, 70), (220.0, 0.8, 70), (600.0, 0.0, 60)],
        [(100.0, 1.2, 70), (350.0, 0.5, 70), (800.0, 0.0, 60)],
        [(40.0, 0.0, 80), (180.0, 0.6, 80)],
    ]
    it = 0
    while time.time() < T_EXPL and it < 200:
        it += 1
        Gc = cur_x @ cur_x.T
        np.fill_diagonal(Gc, -2.0)
        rm = Gc.max(axis=1)
        order = np.argsort(rm)
        mode = rng.random()
        if mode < 0.45:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            base = cur_x if rng.random() > 0.15 else best_x
            x0 = base + rng.standard_normal(size=(99, 13)) * sigma
            x0 = _norm_rows(x0)
        elif mode < 0.75:
            kk = int(rng.integers(2, 7))
            idx = order[-kk:]
            x0 = cur_x.copy()
            sigma = float(rng.uniform(0.06, 0.28))
            x0[idx] = x0[idx] + rng.standard_normal(size=(kk, 13)) * sigma
            x0 = _norm_rows(x0)
            if rng.random() < 0.55:
                pick = rng.choice(idx, size=min(2, kk), replace=False)
                x0 = _far_reinsert(x0, pick, rng, 1500)
        else:
            kk = int(rng.integers(2, 6))
            idx = order[-kk:]
            x0 = _far_reinsert(cur_x, idx, rng, 1800)
            x0 = x0 + rng.standard_normal(size=(99, 13)) * 0.012
            x0 = _norm_rows(x0)
        sched = sched_sets[int(rng.integers(0, len(sched_sets)))]
        try:
            x1 = _anneal(x0, sched, T_EXPL + 2.0)
        except Exception:
            continue
        v1 = _max_cos(x1)
        # light balancing to equalise row maxima
        try:
            x1b, v1b = _balanced_gd(x1, steps=160, lr0=0.014, delta=7e-05, topk=7, deadline=T_EXPL + 2.0)
        except Exception:
            x1b, v1b = x1, v1
        if v1b < v1:
            x1, v1 = x1b, v1b
        if v1 < best_val:
            best_val = v1; best_x = x1.copy()
            cur_x = x1.copy(); cur_val = v1
        else:
            # simulated-annealing move to new basin
            prog = (time.time() - t0) / max(T_EXPL - t0, 1e-9)
            Temp = 0.0012 * (1 - prog) + 0.00012
            if v1 < cur_val + Temp * float(rng.random() * 2.0):
                cur_x = x1.copy(); cur_val = v1
            # occasionally restart from best
            if rng.random() < 0.08:
                cur_x = best_x.copy(); cur_val = best_val
        if time.time() > T_EXPL:
            break
    # final polish on best
    try:
        if time.time() < T_LP:
            xf = _anneal(best_x, [(150.0, 0.5, 120), (450.0, 0.2, 120), (900.0, 0.0, 100)], T_LP)
            vf = _max_cos(xf)
            if vf < best_val:
                best_val = vf; best_x = xf.copy()
    except Exception:
        pass
    try:
        if time.time() < T_LP:
            xg, vg = _balanced_gd(best_x, steps=220, lr0=0.012, delta=8e-05, topk=7, deadline=T_LP)
            if vg < best_val:
                best_val = vg; best_x = xg.copy()
    except Exception:
        pass
    try:
        if time.time() < T_END - 2.0:
            xl, vl = _lp_polish(best_x, seconds=max(4.0, T_END - time.time() - 3.0), deadline=T_END - 1.0)
            if vl < best_val:
                best_val = vl; best_x = xl.copy()
    except Exception:
        pass
    best_x = _norm_rows(np.asarray(best_x, float))
    return best_x
