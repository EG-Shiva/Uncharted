import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lbfgs_stages(X0, alphas, iters, deadline):
    N, d = X0.shape
    U0 = X0.ravel().astype(float).copy()
    Xcur = X0.copy()
    for alpha, mit in zip(alphas, iters):
        if time.time() > deadline:
            break
        a = float(alpha)
        def fun(Uf, _a=a):
            U = Uf.reshape(N, d)
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
                                    'gtol': 1e-7, 'maxfun': int(mit)*6, 'maxcor': 8})
            U0 = res.x.copy()
            U = U0.reshape(N, d)
            Xcur = _norm_rows(U)
        except Exception:
            break
    return Xcur

def _balance_gd(X, steps, lr0, delta, deadline):
    cur = X.copy()
    best = _max_cos(cur)
    bx = cur.copy()
    n = cur.shape[0]
    for s in range(steps):
        if s % 40 == 0 and time.time() > deadline:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - delta).astype(float)
        np.fill_diagonal(Mask, 0.0)
        # ensure each row has at least 4 partners (top-k symmetrized)
        # use argpartition for speed
        idx = np.argpartition(-G, 4, axis=1)[:, :4]
        Bal = np.zeros_like(Mask)
        Bal[np.repeat(np.arange(n), 4), idx.ravel()] = 1.0
        np.fill_diagonal(Bal, 0.0)
        Bal = np.maximum(Bal, Bal.T)
        Comb = Mask + 0.5 * Bal
        cnt = Comb.sum(axis=1, keepdims=True) + 1e-12
        Gr = Comb @ cur
        pr = np.sum(Gr * cur, axis=1, keepdims=True)
        Gt = (Gr - pr * cur) / cnt
        # normalize step
        gn = np.linalg.norm(Gt) + 1e-12
        step = lr0 * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - step * Gt / max(gn * 0.05, 1.0)
        cur = _norm_rows(cur)
        if s % 50 == 49 or s == steps - 1:
            sc = _max_cos(cur)
            if sc < best:
                best = sc
                bx = cur.copy()
        if s == steps // 2:
            delta = max(delta * 0.3, 3e-06)
    return bx, best

def _void_replace(base, k, rng, n_cand=4000):
    n, d = base.shape
    G = base @ base.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    worst = np.argsort(rm)[-k:]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[worst] = False
    Keep = base[keep_mask]
    # random candidates, pick farthest from Keep
    C = rng.standard_normal(size=(n_cand, d))
    C = _norm_rows(C)
    # score = max dot to Keep (lower better); chunk to save memory
    best_idx = []
    # greedy sequential insertion for k points
    cur_keep = Keep.copy()
    chosen = []
    for _ in range(k):
        Cc = rng.standard_normal(size=(n_cand, d))
        Cc = _norm_rows(Cc)
        S = Cc @ cur_keep.T
        sm = S.max(axis=1)
        # also penalize closeness among chosen (already in cur_keep after pick)
        pick = int(np.argmin(sm))
        chosen.append(Cc[pick])
        cur_keep = np.vstack([cur_keep, Cc[pick][None, :]])
    X0 = base.copy()
    X0[worst] = np.array(chosen)
    return _norm_rows(X0)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_HARD = t0 + BUDGET
    T_EXPL = t0 + 148.0
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
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    best_x = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    cur_x = best_x.copy()
    cur_val = best_val

    sched_A = ([60.0, 200.0, 500.0], [60, 70, 60])
    sched_B = ([100.0, 300.0, 700.0], [60, 70, 60])
    sched_C = ([40.0, 150.0, 400.0], [50, 60, 50])

    sigmas = [0.0, 0.006, 0.02, 0.06, 0.14, 0.30]
    trial = 0
    while time.time() < T_EXPL and trial < 40:
        trial += 1
        si = sigmas[(trial - 1) % len(sigmas)]
        sch = (sched_A, sched_B, sched_C)[(trial - 1) % 3]
        dl = min(T_EXPL, T_HARD - 20.0)
        # build start
        if trial == 1:
            X0 = best_x.copy()
        elif trial % 6 == 0:
            # void surgery on best
            k = int(rng.integers(2, 5))
            try:
                X0 = _void_replace(best_x if rng.random() < 0.7 else cur_x, k, rng)
                # small shake after
                X0 = _norm_rows(X0 + rng.standard_normal(size=(99, 13)) * 0.01)
            except Exception:
                X0 = _norm_rows(best_x + rng.standard_normal(size=(99, 13)) * 0.05)
        elif trial % 9 == 0:
            # fresh random (rare)
            X0 = _norm_rows(rng.standard_normal(size=(99, 13)))
        else:
            base = best_x if (trial % 3 != 0) else cur_x
            if rng.random() < 0.45:
                # targeted shake of worst points
                rm, _ = _row_max(base)
                order = np.argsort(rm)
                k = int(rng.integers(3, 10))
                idx = order[-k:]
                X0 = base.copy()
                X0[idx] += rng.standard_normal(size=(k, 13)) * (si * 2.0 + 0.02)
                msk = np.ones(99, dtype=bool); msk[idx] = False
                X0[msk] += rng.standard_normal(size=(int(np.sum(msk)), 13)) * (si * 0.4 + 0.001)
                X0 = _norm_rows(X0)
            else:
                X0 = _norm_rows(base + rng.standard_normal(size=(99, 13)) * max(si, 0.001))

        X1 = _lbfgs_stages(X0, sch[0], sch[1], dl)
        sc = _max_cos(X1)
        # short balancing to equalise per-point maxima
        if time.time() < T_EXPL and sc < best_val + 8e-04:
            try:
                X1b, scb = _balance_gd(X1, steps=120, lr0=0.02, delta=6e-05, deadline=T_EXPL)
                if scb < sc:
                    X1, sc = X1b, scb
            except Exception:
                pass
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = X1.copy()
            cur_x = X1.copy(); cur_val = sc
        else:
            if sc < cur_val and rng.random() < 0.5:
                cur_x = X1.copy(); cur_val = sc
            elif sc < cur_val + 2e-04 and rng.random() < 0.25:
                cur_x = X1.copy(); cur_val = sc

    # final polish on best with high alpha + balance
    try:
        if time.time() < T_HARD - 10.0:
            Xp = _lbfgs_stages(best_x, [300.0, 800.0], [70, 60], T_HARD - 6.0)
            sp = _max_cos(Xp)
            if sp < best_val:
                best_val = sp; best_x = Xp.copy()
    except Exception:
        pass
    try:
        if time.time() < T_HARD - 4.0:
            Xb, sb = _balance_gd(best_x, steps=150, lr0=0.010, delta=4e-05, deadline=T_HARD - 1.0)
            if sb < best_val:
                best_val = sb; best_x = Xb.copy()
    except Exception:
        pass
    best_x = _norm_rows(np.asarray(best_x, dtype=float))
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    best_x = _norm_rows(best_x)
    return best_x
