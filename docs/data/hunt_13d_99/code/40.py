import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lbfgs_anneal(X0, alphas, miters, t_deadline):
    N, d = X0.shape
    U0 = (X0.copy()).ravel().astype(float)
    Xcur = X0.copy()
    for alpha, mit in zip(alphas, miters):
        if time.time() > t_deadline:
            break
        a = float(alpha)
        def fun(Uflat, _a=a):
            U = Uflat.reshape(N, d)
            nr = np.linalg.norm(U, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = U / nr
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp(_a * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5) + 1e-300
            L = m + np.log(S) / _a
            GX = (E @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GU = (GX - dot * X) / nr
            return float(L), GU.ravel().astype(float)
        try:
            res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                           options={'maxiter': int(mit), 'ftol': 1e-20,
                                    'gtol': 1e-7, 'maxfun': int(mit) * 8, 'maxcor': 10})
            U0 = res.x.copy()
        except Exception:
            break
        U = U0.reshape(N, d)
        Xcur = _norm_rows(U)
    return Xcur

def _lp_polish(x, seconds, t_hard):
    x = _norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 6e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and time.time() < t_hard and radius > 1e-14 and it < 25:
        it += 1
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = max(25.0 * radius, 5e-7)
        sel = c > m - thr
        if int(np.sum(sel)) < 900:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:900]] = True
            sel = sel | s2
        if int(np.sum(sel)) > 2800:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:2800]] = True
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={'time_limit': 5.0})
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
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.4, 0.008)
        else:
            radius *= 0.5
    return bx, best

def _greedy_balance(x, steps, lr0, delta, topk):
    cur = x.copy()
    best = _max_cos(cur)
    bx = cur.copy()
    n = cur.shape[0]
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - delta).astype(float)
        np.fill_diagonal(Mask, 0.0)
        idx = np.argsort(-G, axis=1)[:, :topk].copy()
        Bal = np.zeros_like(Mask)
        for i in range(n):
            Bal[i, idx[i]] = 1.0
        np.fill_diagonal(Bal, 0.0)
        Bal = np.maximum(Bal, Bal.T)
        Comb = Mask + 0.4 * Bal
        cnt = Comb.sum(axis=1, keepdims=True) + 1e-12
        Gr = Comb @ cur
        pr = np.sum(Gr * cur, axis=1, keepdims=True)
        Gt = (Gr - pr * cur) / cnt
        cur = cur - lr0 * (1.0 - 0.6 * s / max(steps, 1)) * Gt
        cur = _norm_rows(cur)
        if s % 150 == 149 or s == steps - 1:
            sc = _max_cos(cur)
            if sc < best:
                best = sc
                bx = cur.copy()
        if s == steps // 2:
            delta = max(delta * 0.3, 2e-06)
    return bx, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_HARD = t0 + BUDGET
    T_EXPL = t0 + 138.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
    xb = _norm_rows(xb)
    best_x = xb.copy()
    best_val = _max_cos(best_x)
    cur_x = best_x.copy()

    # quick LP on incumbent to see if cheap gain (very short)
    # exploration: annealed L-BFGS restarts
    trial = 0
    # predefine schedules
    scheds = [
        ([30.0, 90.0, 250.0, 550.0], [80, 100, 100, 80]),
        ([50.0, 150.0, 400.0, 800.0], [80, 100, 100, 80]),
        ([20.0, 70.0, 180.0, 450.0], [70, 90, 90, 70]),
        ([80.0, 200.0, 600.0], [100, 110, 90]),
    ]
    sigmas = [0.005, 0.02, 0.06, 0.13, 0.25, 0.40]
    while time.time() < T_EXPL and trial < 60:
        trial += 1
        si = sigmas[(trial - 1) % len(sigmas)]
        sch = scheds[(trial - 1) % len(scheds)]
        if trial == 1:
            X0 = best_x.copy()
        elif trial % 7 == 0:
            # fresh random start (a few)
            X0 = _norm_rows(rng.standard_normal(size=(99, 13)))
        else:
            base = best_x if (trial % 3 != 0) else cur_x
            # bias perturbation to worst points
            if rng.random() < 0.5:
                Gc = base @ base.T
                np.fill_diagonal(Gc, -2.0)
                rm = Gc.max(axis=1)
                order = np.argsort(rm)
                X0 = base.copy()
                k = int(rng.integers(3, 12))
                idx = order[-k:]
                X0[idx] += rng.standard_normal(size=(k, 13)) * (si * 2.0 + 0.02)
                mask = np.ones(99, dtype=bool)
                mask[idx] = False
                X0[mask] += rng.standard_normal(size=(np.sum(mask), 13)) * (si * 0.4)
                X0 = _norm_rows(X0)
            else:
                X0 = _norm_rows(base + rng.standard_normal(size=(99, 13)) * si)
        X1 = _lbfgs_anneal(X0, sch[0], sch[1], min(T_EXPL, T_HARD - 25.0))
        sc = _max_cos(X1)
        # light greedy balance to equalise per-point maxima
        if time.time() < T_EXPL and sc < best_val + 5e-04:
            try:
                X1b, scb = _greedy_balance(X1, steps=180, lr0=0.012, delta=8e-05, topk=7)
                if scb < sc:
                    X1 = X1b
                    sc = scb
            except Exception:
                pass
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = X1.copy()
            cur_x = X1.copy()
        elif sc < _max_cos(cur_x) and rng.random() < 0.25:
            cur_x = X1.copy()
        # occasional walk of current
        if trial % 5 == 0 and time.time() < T_EXPL:
            # simulated-annealing accept
            if sc < _max_cos(cur_x) + 2e-04 and rng.random() < 0.6:
                cur_x = X1.copy()

    # final polish: LP + greedy on best
    try:
        if time.time() < T_HARD - 8.0:
            bx2, bv2 = _lp_polish(best_x, seconds=min(22.0, T_HARD - time.time() - 4.0), t_hard=T_HARD - 3.0)
            if bv2 < best_val:
                best_val = bv2
                best_x = bx2.copy()
    except Exception:
        pass
    try:
        if time.time() < T_HARD - 2.0:
            bx3, bv3 = _greedy_balance(best_x, steps=250, lr0=0.008, delta=5e-05, topk=7)
            if bv3 < best_val:
                best_val = bv3
                best_x = bx3.copy()
    except Exception:
        pass
    best_x = _norm_rows(np.asarray(best_x, dtype=float))
    # safety: remove NaN
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    best_x = _norm_rows(best_x)
    return best_x
