import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 1:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx

def _contact_counts(X, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    cnt = np.sum(G > m - tol, axis=1)
    return cnt, m, G

def _smooth_relax(X_init, K, maxiter):
    n, d = X_init.shape
    Y0 = X_init.ravel().copy()
    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = Y / r
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        gmax = float(np.max(G))
        E = np.exp(K * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if S <= 0:
            S = 1e-300
        f = gmax + np.log(S) / K
        W = E / S
        GX = W @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        dY = (GX - dot * X) / r
        return f, dY.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7})
        Y = res.x.reshape(n, d)
    except Exception:
        return X_init
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    X = Y / nrm
    # cleanup non-finite
    if not np.all(np.isfinite(X)):
        return X_init
    return X

def _refine_single(x, Others, steps=300, step0=0.2):
    x = x / np.linalg.norm(x)
    s = step0
    for it in range(steps):
        dots = Others @ x
        m = float(dots.max())
        act = Others[dots > m - 2e-4]
        if len(act) == 0:
            break
        g = np.mean(act, axis=0)
        gt = g - float(g @ x) * x
        ng = np.linalg.norm(gt)
        if ng < 1e-12:
            break
        gt = gt / ng
        # line search-ish decay
        x_new = x - s * gt
        x_new = x_new / np.linalg.norm(x_new)
        m_new = float(np.max(Others @ x_new))
        if m_new < m:
            x = x_new
        else:
            s *= 0.95
            if s < 1e-5:
                # still move a bit
                x = x_new if np.random.rand() < 0.01 else x
                pass
        s *= 0.998
    return x / np.linalg.norm(x)

def solve():
    t0 = time.time()
    TIME_HARD = 172.0
    rng = np.random.default_rng(0)
    # load start
    best_X = None
    best_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            if a.shape == (82, 14):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = rng.standard_normal((82, 14))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_v = _max_cos(best_X)
    n, d = 82, 14
    cur_X = best_X.copy()
    cur_v = best_v
    # initial rattlers
    def get_rattlers(X, v):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cnt = np.sum(G > v - 1e-7, axis=1)
        ratt = np.where(cnt == 0)[0]
        order = np.argsort(cnt)
        return ratt, cnt, order
    # global budget: leave ~35s for final polish
    t_global_end = t0 + 128.0
    trial = 0
    Temp = 0.0006
    while time.time() < t_global_end and time.time() - t0 < TIME_HARD - 45:
        trial += 1
        mode = trial % 7
        ratt, cnt, order = get_rattlers(cur_X, cur_v)
        # also rattlers of best
        _, cntB, _ = get_rattlers(best_X, best_v)
        prop = cur_X.copy()
        if mode == 0:  # small global shake
            sig = 0.025
            prop = prop + sig * rng.standard_normal(prop.shape)
            K1, K2 = 120, 450
            it1, it2 = 50, 70
        elif mode == 1:  # medium shake
            sig = 0.07
            prop = prop + sig * rng.standard_normal(prop.shape)
            K1, K2 = 100, 350
            it1, it2 = 50, 70
        elif mode == 2:  # rattler replacement + small shake others
            # pick 1-2 indices: prefer rattlers of cur, else random
            if len(ratt) > 0:
                k = int(min(len(ratt), 1 + (trial % 2)))
                idx = rng.choice(ratt, size=k, replace=False)
            else:
                idx = rng.choice(n, size=2, replace=False)
                # lowest contacts
                idx = order[:2]
            mask = np.ones(n, bool)
            mask[idx] = False
            prop[mask] += 0.02 * rng.standard_normal((int(np.sum(mask)), d))
            for j in idx:
                prop[j] = rng.standard_normal(d)
            K1, K2 = 100, 500
            it1, it2 = 60, 80
        elif mode == 3:  # kick overloaded
            # top loaded
            k = 5
            idx = order[-k:]
            prop[idx] += 0.25 * rng.standard_normal((k, d))
            # rattlers big kick too
            if len(ratt) > 0:
                prop[ratt] += 0.15 * rng.standard_normal((len(ratt), d))
            prop += 0.01 * rng.standard_normal(prop.shape)
            K1, K2 = 120, 400
            it1, it2 = 50, 70
        elif mode == 4:  # hole filling single point (no LBFGS)
            # choose victim: rattler or most loaded
            if len(ratt) > 0 and (trial % 2 == 0):
                victim = int(rng.choice(ratt))
            else:
                victim = int(order[-1])  # most loaded
            Others = np.delete(prop, victim, axis=0)
            # sample voids
            S = 12000
            cand = rng.standard_normal((S, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            x0 = cand[bi]
            x0 = _refine_single(x0, Others, steps=250, step0=0.2)
            prop[victim] = x0
            # light relax with high K short
            K1, K2 = 300, 600
            it1, it2 = 30, 40
            # small noise to others
            m2 = np.ones(n, bool); m2[victim] = False
            prop[m2] += 0.005 * rng.standard_normal((n - 1, d))
        elif mode == 5:  # large shake of random subset
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            prop[idx] += 0.35 * rng.standard_normal((k, d))
            prop += 0.01 * rng.standard_normal(prop.shape)
            K1, K2 = 90, 350
            it1, it2 = 60, 70
        else:  # mode 6: shake from best (restart diversity)
            if rng.random() < 0.5:
                prop = best_X.copy() + 0.05 * rng.standard_normal(prop.shape)
            else:
                prop = cur_X.copy() + 0.04 * rng.standard_normal(prop.shape)
            K1, K2 = 130, 480
            it1, it2 = 50, 70
        # renorm before opt
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        # two-stage smooth relax
        try:
            X1 = _smooth_relax(prop, K1, it1)
            if time.time() > t_global_end:
                # evaluate X1 only
                v1 = _max_cos(X1)
                if v1 < best_v - 1e-12:
                    best_v = v1; best_X = X1.copy()
                break
            X2 = _smooth_relax(X1, K2, it2)
        except Exception:
            continue
        v2 = _max_cos(X2)
        if v2 < best_v - 2e-13:
            best_v = v2
            best_X = X2.copy()
            cur_X = X2.copy()
            cur_v = v2
        elif v2 < cur_v:
            cur_X = X2.copy()
            cur_v = v2
        else:
            # SA acceptance of worse cur to escape
            if rng.random() < np.exp(-(v2 - cur_v) / max(Temp, 1e-9)):
                cur_X = X2.copy()
                cur_v = v2
            # occasional reset to best if cur drifts
            if (trial % 25 == 0) and (cur_v > best_v + 0.004):
                cur_X = best_X.copy()
                cur_v = best_v
        # decay temp slightly
        Temp *= 0.9985
        Temp = max(Temp, 0.00015)
        if best_v < 0.190:
            break
    # final polish with remaining time
    remain = TIME_HARD - (time.time() - t0) - 2.0
    remain = float(np.clip(remain, 8.0, 45.0))
    try:
        polished = _lp_polish(best_X, remain)
        vp = _max_cos(polished)
        if vp <= best_v + 1e-12:
            # keep polished if not worse (it ties contacts, more stable)
            if vp < best_v:
                best_v = vp
            best_X = polished
        else:
            # polished worse due to numerical? keep best but lightly polish best with short run
            pass
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
