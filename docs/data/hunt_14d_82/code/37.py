import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normed(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lbfgs_stage(X0, p, maxiter, t_end):
    n, d = X0.shape
    Y0 = X0.copy().ravel()
    best = {'x': X0.copy(), 'v': _max_cos(X0)}
    # closure with time check
    def fun_grad(Yf):
        # hard abort via value: if past deadline, raise to stop
        if time.time() >= t_end:
            raise StopIteration()
        Y = Yf.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # stable exp
        S = p * (G - m)
        # clip low tail
        np.clip(S, -50.0, 0.0, out=S)
        E = np.exp(S)
        np.fill_diagonal(E, 0.0)
        sum_u = float(E.sum() * 0.5)
        if sum_u <= 0 or not np.isfinite(sum_u):
            sum_u = 1.0
        f = m + np.log(sum_u) / p
        W = E / sum_u
        Gx = W @ X
        dot = np.sum(Gx * X, axis=1, keepdims=True)
        Gy = (Gx - dot * X) / rn
        return float(f), Gy.ravel()
    try:
        res = minimize(fun_grad, Y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-09,
                                'maxfun': maxiter * 5, 'maxcor': 20})
        Yf = res.x
        Y = Yf.reshape(n, d)
        X = _normed(Y)
        v = _max_cos(X)
        # keep best between start and end (LSE may not monotonic in max)
        if v < best['v']:
            best['x'] = X
            best['v'] = v
        else:
            # even if worse in max, LSE improvement may help next p; return X if its LSE lower?
            # return the optimizer output anyway if its max is not much worse
            # we return best max-wise to guarantee monotonic
            pass
    except StopIteration:
        pass
    except Exception:
        pass
    # also try intermediate? just return best
    return best['x'], best['v']

def _lp_polish(X, t_end, radius_init, cap=4000, use_wide=True):
    X = _normed(X)
    n, d = X.shape
    best = _max_cos(X)
    Xbest = X.copy()
    radius = float(radius_init)
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        if use_wide:
            thr = m - max(5e-6, 20.0 * radius)
        else:
            thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p_ = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p_), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p_, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p_, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        # true max-cosine adaptation
        if v < best:
            X = y
            best = v
            Xbest = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # clock check every iteration (while condition)
    return Xbest, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    internal = t0 + 148.0  # hard internal deadline, reserve ~20s
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        a = rng0.standard_normal((n, d))
        cands.append(_normed(a))
    cands.sort(key=lambda c: _max_cos(c))
    X = cands[0].copy()
    best_val = _max_cos(X)
    X_best = X.copy()
    def keep(Xc):
        nonlocal X_best, best_val
        v = _max_cos(Xc)
        if v < best_val and np.all(np.isfinite(Xc)):
            X_best = _normed(Xc.copy())
            best_val = v
        return v

    # Annealed smooth minimax: p=100 -> 1000 -> 10000
    stages = [(100, 250), (1000, 300), (10000, 400)]
    for p, mi in stages:
        if time.time() >= internal:
            break
        # per-stage time cap ~25s
        stage_end = min(internal, time.time() + 28.0, t_end - 22.0)
        if stage_end - time.time() < 1.0:
            break
        try:
            Xn, vn = _lbfgs_stage(X_best, p, mi, stage_end)
        except Exception:
            continue
        keep(Xn)
        # continue from best
    # Active-set LP with wide threshold
    if time.time() < internal - 1.0:
        # start from current best with moderate radius
        Xlp, vlp = _lp_polish(X_best, min(internal, t_end - 20.0), radius_init=2e-3, cap=4000, use_wide=True)
        keep(Xlp)
        # extra squeeze with smaller radius if time remains
        if time.time() < internal - 1.0:
            Xlp2, vlp2 = _lp_polish(X_best, min(internal, t_end - 20.0), radius_init=2e-4, cap=4000, use_wide=True)
            keep(Xlp2)

    # Reserve final 20s for one capped HiGHS polish
    if time.time() < t_end - 1.5:
        try:
            Xf, vf = _lp_polish(X_best, t_end - 1.0, radius_init=8e-5, cap=3000, use_wide=True)
            keep(Xf)
        except Exception:
            pass
    X_best = _normed(X_best)
    return X_best
