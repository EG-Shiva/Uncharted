import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                           shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    import time as _time
    import numpy as _np
    _np.random.seed(0)
    t0 = _time.time()
    HARD_END = t0 + 175.0
    GLOBAL_END = t0 + 138.0

    n, d = 272, 4
    # load start
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=_np.float64)
            if a.shape == (n, d):
                nr = _np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if _np.all(_np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        a = _np.random.randn(n, d)
        a /= _np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    vals = [_max_cos(a) for a in cand]
    bi = int(_np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])
    cur = best.copy()
    cur_val = best_val

    iu0, iu1 = _np.triu_indices(n, 1)

    b1m = 0.9
    b2m = 0.999
    eps = 1e-8

    def adam_refine(X0, total_steps, b0, b1, lr0, lr1):
        X = X0.copy()
        m = _np.zeros_like(X)
        v = _np.zeros_like(X)
        G0 = X @ X.T
        _np.fill_diagonal(G0, -2.0)
        bv = float(G0.max())
        best_X = X.copy()
        if total_steps <= 0:
            return best_X, bv
        logb = _np.log(b1 / b0) / total_steps
        loglr = _np.log(lr1 / lr0) / total_steps
        for t in range(1, total_steps + 1):
            beta = b0 * _np.exp(logb * t)
            lr = lr0 * _np.exp(loglr * t)
            G = X @ X.T
            c = G[iu0, iu1]
            m0 = float(c.max())
            if m0 < bv:
                bv = m0
                best_X = X.copy()
            e = _np.exp(beta * (c - m0))
            s = float(e.sum())
            if not _np.isfinite(s) or s == 0:
                continue
            w = e / s
            W = _np.zeros((n, n), dtype=X.dtype)
            W[iu0, iu1] = w
            W += W.T
            grad = W @ X
            proj = _np.sum(grad * X, axis=1, keepdims=True)
            g = grad - proj * X
            m *= b1m
            m += (1.0 - b1m) * g
            v *= b2m
            v += (1.0 - b2m) * (g * g)
            bc1 = 1.0 - b1m ** t
            bc2 = 1.0 - b2m ** t
            mhat = m / bc1
            vhat = v / bc2
            step = lr * mhat / (_np.sqrt(vhat) + eps)
            # clip extreme steps for stability
            _np.clip(step, -0.08, 0.08, out=step)
            X -= step
            nr = _np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        Gf = X @ X.T
        _np.fill_diagonal(Gf, -2.0)
        fv = float(Gf.max())
        if fv < bv:
            return X, fv
        return best_X, bv

    trial = 0
    # main basin hopping
    while _time.time() < GLOBAL_END and _time.time() < HARD_END - 20:
        trial += 1
        # sigma schedule: mostly small, periodically large to change topology
        if trial % 17 == 0:
            sigma = 0.20
            steps = 420
        elif trial % 7 == 0:
            sigma = 0.08
            steps = 380
        elif trial % 3 == 0:
            sigma = 0.045
            steps = 340
        else:
            sigma = 0.016
            steps = 300

        # --- perturb cur ---
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        per_max = Gc.max(axis=1)
        gmax = float(Gc.max())
        rattler = per_max < (gmax - 0.003)
        noise = _np.random.randn(n, d) * sigma
        if _np.any(rattler):
            noise[rattler] *= 2.5
        Xt = cur + noise
        # targeted contact break: push one of top-20 pairs apart
        if _np.random.rand() < 0.75:
            c = Gc[iu0, iu1]
            # indices of top 20
            k = 20
            part = _np.argpartition(-c, k)[:k]
            pick = part[_np.random.randint(0, k)]
            a = int(iu0[pick]); b = int(iu1[pick])
            alpha = min(0.25, sigma * 2.0 + 0.05)
            # push apart along difference
            Xt[a] -= alpha * cur[b]
            Xt[b] -= alpha * cur[a]
            # extra random kick to the pair
            Xt[a] += _np.random.randn(d) * sigma * 0.5
            Xt[b] += _np.random.randn(d) * sigma * 0.5
        nr = _np.linalg.norm(Xt, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Xt /= nr

        Xopt, vopt = adam_refine(Xt, steps, 55.0, 420.0, 0.012, 0.0025)

        # update best
        if vopt < best_val - 1e-14:
            best_val = vopt
            best = Xopt.copy()
        # Metropolis move for cur (annealed)
        T = 0.0012 * _np.exp(-trial / 250.0) + 1e-5
        if vopt < cur_val:
            cur, cur_val = Xopt, vopt
        else:
            if _np.random.rand() < _np.exp((cur_val - vopt) / T):
                cur, cur_val = Xopt, vopt
        # if stuck long, restart cur from best with big kick
        if trial % 40 == 0 and _time.time() < GLOBAL_END:
            # shake best strongly to diversify
            if cur_val > best_val + 0.002:
                cur = best.copy()
                cur_val = best_val

        if best_val < 0.890:
            break

    # final polish of best with a short smooth touch then LP
    try:
        if _time.time() < HARD_END - 8:
            best, best_val2 = adam_refine(best, 200, 120.0, 600.0, 0.004, 0.001)
            best_val = min(best_val, best_val2)
    except Exception:
        pass
    try:
        remain = HARD_END - _time.time()
        if remain > 5:
            best = _lp_polish(best, HARD_END - 1.0)
    except Exception:
        pass
    best = best / _np.linalg.norm(best, axis=1, keepdims=True)
    return best
