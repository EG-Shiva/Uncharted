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

def _ncontacts(X, tol=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - tol).sum(axis=1)
    return C, m

def _homotopy_run(X0, alphas, lambdas, maxiters, t_deadline):
    X_cur = X0.copy()
    U0 = X_cur.ravel().copy()
    N, d = X_cur.shape
    iu = np.triu_indices(N, 1)
    for alpha, lam, mit in zip(alphas, lambdas, maxiters):
        if time.time() > t_deadline:
            break
        def fun(Uflat, _alpha=alpha, _lam=lam):
            U = Uflat.reshape(N, d)
            nr = np.linalg.norm(U, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = U / nr
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G[iu]))
            E = np.exp(_alpha * (G - m))
            np.fill_diagonal(E, 0.0)
            # triu sum
            S = float(np.sum(E[iu])) + 1e-300
            L = m + np.log(S) / _alpha
            W = E / S
            GradL = W @ X
            # per-point balance: r_i = row sum
            r = E.sum(axis=1) + 1e-300
            l = np.log(r)
            mu = float(np.mean(l))
            sd = float(np.std(l))
            if sd < 1e-12 or _lam == 0.0:
                F = L
                GradX = GradL
            else:
                F = L + _lam * sd / _alpha
                u = (l - mu)
                c = u / (N * sd)  # d std/dl
                Q = E / r[:, None]
                QX = Q @ X
                Y = c[:, None] * X
                H = c[:, None] * QX + (Q.T @ Y)
                GradX = GradL + _lam * H
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradU = (GradX - dot * X) / nr
            return float(F), GradU.ravel().astype(float)
        try:
            res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                           options={'maxiter': mit, 'ftol': 1e-20, 'gtol': 1e-7,
                                    'maxfun': mit * 10, 'maxcor': 10})
            U0 = res.x
        except Exception:
            break
        U = U0.reshape(N, d)
        X_cur = _norm_rows(U)
    return X_cur

def _lp_polish(x, seconds):
    x = _norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = max(25.0 * radius, 1e-9)
        sel = c > m - thr
        # ensure at least 800 constraints for balance
        if int(np.sum(sel)) < 800:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:800]] = True
            sel = sel | sel2
        if int(np.sum(sel)) > 3000:
            idx = np.argsort(-c)
            # keep tightest 3000 among selected? just top 3000 overall
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[:3000]] = True
            sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            best_x = y.copy()
            x = y
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
    return best_x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
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
    n, d = 99, 13
    best_val = _max_cos(xb)
    best_x = xb.copy()
    # record baseline contacts
    Cb, _ = _ncontacts(best_x, tol=1e-6)
    base_bal = int(np.sum(Cb >= 5))
    # starts: 3 perturbed + 2 random = 5
    starts = []
    for sig in [0.01, 0.03, 0.08]:
        Y = xb + rng.standard_normal(size=(n, d)) * sig
        starts.append(_norm_rows(Y))
    for _ in range(2):
        Y = rng.standard_normal(size=(n, d))
        starts.append(_norm_rows(Y))
    alphas = [60.0, 140.0, 300.0]
    lambdas = [0.5, 0.2, 0.07]
    maxiters = [100, 100, 130]
    # extra high-alpha short stage params
    alphas2 = [320.0]
    lambdas2 = [0.05]
    maxiters2 = [80]
    explore_end = t0 + 138.0
    best_bal = base_bal
    balanced_best_val = best_val
    balanced_best_x = best_x.copy()
    for si, S0 in enumerate(starts):
        if time.time() > explore_end:
            break
        # small time deadline per start
        remain = explore_end - time.time()
        per = remain / (len(starts) - si)
        dead = time.time() + min(per, 30.0)
        try:
            Xc = _homotopy_run(S0, alphas, lambdas, maxiters, dead)
            # extra polish stage at higher alpha / low lambda
            if time.time() < dead:
                Xc = _homotopy_run(Xc, alphas2, lambdas2, maxiters2, dead)
        except Exception:
            continue
        try:
            v = _max_cos(Xc)
        except Exception:
            continue
        Cc, _ = _ncontacts(Xc, tol=1e-6)
        bal = int(np.sum(Cc >= 5))
        # update balanced pool: keep if more balanced even if up to 1e-6 worse
        if bal > best_bal and v <= balanced_best_val + 1e-6 + 5e-4:
            # allow slightly larger window to actually capture topology
            best_bal = bal
            balanced_best_val = v
            balanced_best_x = Xc.copy()
        elif bal >= best_bal and v < balanced_best_val:
            balanced_best_val = v
            balanced_best_x = Xc.copy()
        if v < best_val:
            best_val = v
            best_x = Xc.copy()
        # quick short LP touch (2s) to evaluate polished potential
        if time.time() < explore_end - 2:
            try:
                Xp, vp = _lp_polish(Xc, min(2.0, max(0.5, explore_end - time.time() - 1)))
                Ccp, _ = _ncontacts(Xp, tol=1e-6)
                balp = int(np.sum(Ccp >= 5))
                if balp > best_bal and vp <= balanced_best_val + 1e-6 + 5e-4:
                    best_bal = balp
                    balanced_best_val = vp
                    balanced_best_x = Xp.copy()
                if vp < best_val:
                    best_val = vp
                    best_x = Xp.copy()
            except Exception:
                pass
    # final: choose target = best overall; if balanced within 1e-6, prefer balanced topology
    target = best_x
    if best_bal > base_bal and balanced_best_val <= best_val + 1e-6:
        target = balanced_best_x
        best_val = min(best_val, balanced_best_val)
    rem = BUDGET - (time.time() - t0)
    rem = max(5.0, min(rem - 1.0, 30.0))
    try:
        Xp, vp = _lp_polish(target, rem)
        if vp < best_val + 1e-12:
            best_x = Xp
            best_val = vp if vp < best_val else best_val
        else:
            best_x = target
    except Exception:
        best_x = target
    best_x = _norm_rows(best_x)
    # safety: if somehow worse than loaded best, return loaded best
    try:
        if _max_cos(best_x) > _max_cos(xb) + 1e-12:
            best_x = xb
    except Exception:
        pass
    return best_x
