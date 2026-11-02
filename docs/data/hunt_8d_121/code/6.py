import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize, linprog as _linprog
    from scipy.sparse import coo_matrix as _coo

    t0 = _time.time()
    deadline = t0 + 168.0
    rng = _np.random.RandomState(0)

    n, d = 121, 8

    def _normed(a):
        a = _np.asarray(a, dtype=float)
        nr = _np.linalg.norm(a, axis=1, keepdims=True)
        nr = _np.maximum(nr, 1e-12)
        return a / nr

    # load starting points
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = _np.load(fn)
            v = _np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _normed(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _normed(rng.randn(n, d))
        best_val = _max_cos(best)

    # ---------- smooth LSE refine ----------
    def refine(x_init, beta, maxiter):
        x_init = _np.asarray(x_init, dtype=float)
        x0f = x_init.ravel().copy()
        nn, dd = x_init.shape

        def fg(xf):
            X = xf.reshape(nn, dd)
            ns = _np.sqrt(_np.sum(X * X, axis=1))
            ns = _np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            # off-diagonal max for shift
            # set diag to very low for max computation
            # use copy view
            m = _np.max(G - 2.0 * _np.eye(nn))
            A = beta * (G - m)
            # A <= small positive due to roundoff; clip low only
            # exp safe: A<=0 mostly
            _np.clip(A, -700, 0, out=A)
            E = _np.exp(A)
            _np.fill_diagonal(E, 0.0)
            # sum triu
            S = _np.sum(_np.triu(E, 1))
            if S <= 0:
                S = 1e-300
            F = m + _np.log(S) / beta
            EY = E.dot(Y)
            GradY = EY / S
            dot = _np.sum(GradY * Y, axis=1)
            GradX = (GradY - dot[:, None] * Y) / ns[:, None]
            return float(F), GradX.ravel()

        try:
            res = _minimize(fg, x0f, jac=True, method="L-BFGS-B",
                            options={"maxiter": maxiter, "ftol": 1e-20,
                                     "gtol": 1e-7, "maxfun": maxiter * 10,
                                     "maxls": 20})
            Xo = res.x.reshape(nn, dd)
        except Exception:
            Xo = x_init.copy()
        nr = _np.linalg.norm(Xo, axis=1, keepdims=True)
        nr = _np.maximum(nr, 1e-12)
        Xo = Xo / nr
        return Xo

    # ---------- LP polish ----------
    iu0, iu1 = _np.triu_indices(n, 1)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1e-3
        start = _time.time()
        nv = n * d + 1
        while (_time.time() - start < seconds and _time.time() < deadline - 1.0
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[iu0, iu1]
            m = c_all.max()
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            # cap constraints for speed
            if p > 2500:
                # keep tightest 2500
                idx = _np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = _np.repeat(_np.arange(p), 2 * d + 1)
            cols = _np.concatenate(
                [I[:, None] * d + _np.arange(d),
                 J[:, None] * d + _np.arange(d),
                 _np.full((p, 1), n * d)], axis=1).ravel()
            vals = _np.concatenate([x[J], x[I], -_np.ones((p, 1))], axis=1).ravel()
            A_ub = _coo((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = _coo((x.ravel(),
                         (_np.repeat(_np.arange(n), d), _np.arange(n * d))),
                        shape=(n, nv)).tocsr()
            obj = _np.zeros(nv); obj[-1] = 1.0
            try:
                res = _linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                               b_eq=_np.zeros(n),
                               bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                               method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _normed(y)
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    def kick_max_pair(x, alpha):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        k = _np.argmax(G)
        a = k // n; b = k % n
        pa = x[b] - _np.dot(x[b], x[a]) * x[a]
        pb = x[a] - _np.dot(x[a], x[b]) * x[b]
        na = _np.linalg.norm(pa); nb = _np.linalg.norm(pb)
        if na > 1e-12:
            pa /= na
        if nb > 1e-12:
            pb /= nb
        y = x.copy()
        y[a] -= alpha * pa
        y[b] -= alpha * pb
        y += rng.randn(n, d) * (alpha * 0.25)
        return _normed(y)

    # initial short LP squeeze of starting point if time allows
    # main basin-hopping loop
    schedule = [(50, 120), (120, 150), (300, 180), (600, 180)]
    trial = 0
    while _time.time() < deadline - 12.0 and trial < 30:
        if trial == 0:
            x_init = best.copy()
        elif trial % 3 == 1:
            x_init = _normed(best + rng.randn(n, d) * 0.008)
        elif trial % 3 == 2:
            x_init = kick_max_pair(best, 0.008 + rng.rand() * 0.012)
        else:
            x_init = _normed(best + rng.randn(n, d) * 0.03)
            for _k in range(2):
                idx = rng.randint(n)
                v = rng.randn(d); v /= _np.linalg.norm(v)
                x_init[idx] = v
            x_init = _normed(x_init)

        x = x_init
        for beta, mi in schedule:
            if _time.time() > deadline - 10.0:
                break
            x = refine(x, beta, mi)
            v = _max_cos(x)
            if v < best_val:
                best_val = v
                best = x.copy()
        if _time.time() < deadline - 8.0:
            try:
                x2 = lp_polish_timed(x, seconds=min(4.0, deadline - 2.0 - _time.time()))
                v2 = _max_cos(x2)
                if v2 < best_val:
                    best_val = v2
                    best = x2.copy()
                    x = x2
            except Exception:
                pass
        trial += 1
        # keep current best tight: extra polish of best occasionally
        if trial % 5 == 0 and _time.time() < deadline - 15.0:
            try:
                xb = lp_polish_timed(best, seconds=3.0)
                vb = _max_cos(xb)
                if vb < best_val:
                    best_val = vb
                    best = xb.copy()
            except Exception:
                pass

    best = _normed(best)
    return best
