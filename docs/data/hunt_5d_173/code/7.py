import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.02:
        return x
    radius = radius_init
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_opt(X0, tau, maxiter, rng):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)

    def fun_jac(Zflat):
        Z = Zflat.reshape(n, d)
        R = np.linalg.norm(Z, axis=1, keepdims=True)
        R = np.maximum(R, 1e-12)
        X = Z / R
        G = X @ X.T
        D = G[iu0, iu1]
        m = float(D.max())
        E = np.exp((D - m) / tau)
        Ssum = float(E.sum())
        S = m + tau * np.log(Ssum + 1e-300)
        W = E / Ssum
        Wmat = np.zeros((n, n), dtype=np.float64)
        Wmat[iu0, iu1] = W
        Wmat[iu1, iu0] = W
        GX = Wmat @ X
        proj = (GX * X).sum(axis=1, keepdims=True)
        GZ = (GX - proj * X) / R
        return S, GZ.ravel()

    try:
        res = minimize(fun_jac, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": maxiter * 5})
        Z = res.x.reshape(n, d)
        Zn = np.linalg.norm(Z, axis=1, keepdims=True)
        Zn[Zn == 0] = 1
        return Z / Zn
    except Exception:
        return X0


def _hole_for(Y, rng, n_sample=30000, refine_steps=250):
    # Y: (m,d) fixed points, find unit v minimizing max(Y@v)
    m, d = Y.shape
    Z = rng.normal(size=(n_sample, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    dots = Z @ Y.T
    dm = dots.max(axis=1)
    idx = np.argsort(dm)[:6]
    best_v = None
    best_val = 1e9
    for k in idx:
        v = Z[k].copy()
        cur = float((Y @ v).max())
        step = 0.05
        for _ in range(refine_steps):
            j = int(np.argmax(Y @ v))
            yj = Y[j]
            g = yj - float(yj @ v) * v
            gn = float(np.linalg.norm(g))
            if gn < 1e-12:
                break
            vn = v - step * g
            nrm = float(np.linalg.norm(vn))
            if nrm == 0:
                step *= 0.5
                continue
            vn /= nrm
            nv = float((Y @ vn).max())
            if nv < cur:
                v = vn; cur = nv; step *= 1.08
                step = min(step, 0.2)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        if cur < best_val:
            best_val = cur; best_v = v
    return best_v, best_val


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v; best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    def contact_counts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G >= m - 1e-9).sum(axis=1)
        return m, cnt, G

    trial = 0
    # reserve ~10s for final polish
    while time.time() < deadline - 10.0:
        trial += 1
        rem = deadline - 10.0 - time.time()
        if rem < 1.0:
            break
        pt = min(0.9, max(0.35, rem / 8.0))
        mode = trial % 7
        try:
            _, cnt, Gcur = contact_counts(best)
            order_desc = np.argsort(-cnt)
            order_asc = np.argsort(cnt)
            y = best.copy()
            if mode == 0:
                # small global shake + smooth
                y = y + 0.012 * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                tau = float(rng.uniform(0.006, 0.014))
                y = _smooth_opt(y, tau, 120, rng)
                y = _lp_polish(y, pt)
            elif mode == 1:
                # medium shake + smooth
                y = y + 0.045 * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                tau = float(rng.uniform(0.008, 0.020))
                y = _smooth_opt(y, tau, 100, rng)
                y = _lp_polish(y, pt)
            elif mode == 2:
                # kick most jammed points hard
                k = int(rng.integers(3, 6))
                victims = order_desc[:k]
                y[victims] += 0.28 * rng.normal(size=(len(victims), d))
                # also nudge a few random
                rr = rng.choice(n, 3, replace=False)
                y[rr] += 0.06 * rng.normal(size=(3, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _lp_polish(y, pt)
            elif mode == 3:
                # optimal reinsertion of worst point
                victim = int(order_desc[int(rng.integers(0, 3))])
                Y = np.delete(y, victim, axis=0)
                vnew, _ = _hole_for(Y, rng, n_sample=30000)
                y[victim] = vnew
                # small shake neighbours to open hole
                dots = Y @ vnew
                nb = np.argsort(-dots)[:3]
                for bb in nb:
                    bi = int(bb) if int(bb) < victim else int(bb) + 1
                    # map back approx: just shake random near ones
                    pass
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _lp_polish(y, pt)
            elif mode == 4:
                # reinsert rattler/low-contact into best hole + shake jammed a bit
                victim = int(order_asc[int(rng.integers(0, 3))])
                Y = np.delete(y, victim, axis=0)
                vnew, _ = _hole_for(Y, rng, n_sample=25000)
                y[victim] = vnew
                victims = order_desc[:2]
                y[victims] += 0.03 * rng.normal(size=(len(victims), d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _lp_polish(y, pt)
            elif mode == 5:
                # jammed + rattler joint kick + smooth
                victims = np.unique(np.concatenate([order_desc[:3], order_asc[:2], rng.choice(n, 3, replace=False)]))
                y[victims] += 0.12 * rng.normal(size=(len(victims), d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                tau = float(rng.uniform(0.005, 0.012))
                y = _smooth_opt(y, tau, 80, rng)
                y = _lp_polish(y, pt)
            else:
                # large global jump
                y = y + float(rng.uniform(0.08, 0.15)) * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                tau = float(rng.uniform(0.010, 0.025))
                y = _smooth_opt(y, tau, 150, rng)
                y = _lp_polish(y, pt)

            v = _max_cos(y)
            if v < best_val - 1e-12:
                best = y / np.linalg.norm(y, axis=1, keepdims=True)
                best_val = v
        except Exception:
            continue

    # final intensive polish
    try:
        frem = deadline - time.time() - 0.5
        if frem > 1.0:
            best = _lp_polish(best, min(frem, 9.0), radius_init=2e-3)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
