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
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0, iu1]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-cfull[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        I = iu0[sel]
        J = iu1[sel]
        csel = cfull[sel]
        p = len(I)
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
    _np.random.seed(20260717)
    t0 = _time.time()
    HARD_END = t0 + 175.0
    GLOBAL_END = t0 + 133.0

    n, d = 272, 4
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

    Wbuf = _np.zeros((n, n), dtype=_np.float64)

    def adam_refine(X0, total_steps, b0, b1, lr0, lr1, deadline):
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
        W = Wbuf
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
            e /= s
            W.fill(0.0)
            W[iu0, iu1] = e
            W[iu1, iu0] = e
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
            _np.clip(step, -0.08, 0.08, out=step)
            X -= step
            nr = _np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if (t & 31) == 0 and _time.time() > deadline:
                break
        Gf = X @ X.T
        _np.fill_diagonal(Gf, -2.0)
        fv = float(Gf.max())
        if fv < bv:
            return X, fv
        return best_X, bv

    def hole_move(Xin, n_samples=15000):
        X = Xin
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        order = _np.argsort(-per)
        r = _np.random.rand()
        if r < 0.6:
            move = int(order[0])
        elif r < 0.85:
            move = int(order[_np.random.randint(0, min(3, n))])
        else:
            move = int(order[_np.random.randint(0, min(8, n))])
        Y = _np.delete(X, move, axis=0)
        # sample holes
        best_c = None
        best_s = 2.0
        bs = 3000
        nb = (n_samples + bs - 1) // bs
        for _ in range(nb):
            C = _np.random.randn(bs, d)
            C /= _np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ Y.T
            sm = S.max(axis=1)
            k2 = int(_np.argmin(sm))
            if float(sm[k2]) < best_s:
                best_s = float(sm[k2])
                best_c = C[k2].copy()
        c = best_c / _np.linalg.norm(best_c)
        # local single-point refine
        st = 0.04
        for it in range(120):
            dots = Y @ c
            mm = float(dots.max())
            w = _np.exp(90.0 * (dots - mm))
            sw = float(w.sum())
            if not _np.isfinite(sw) or sw == 0:
                break
            w /= sw
            gg = w @ Y
            gg = gg - float(gg @ c) * c
            nn = float(_np.linalg.norm(gg))
            if nn < 1e-12:
                break
            gg /= nn
            c = c - st * gg
            c /= _np.linalg.norm(c)
            st *= 0.99
        y = X.copy()
        y[move] = c
        return y

    trial = 0
    while _time.time() < GLOBAL_END and _time.time() < HARD_END - 20:
        trial += 1
        # choose move
        if trial % 23 == 0:
            # global random probe
            Xt = _np.random.randn(n, d)
            Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 480, 30.0, 350.0, 0.015, 0.002
        elif trial % 7 == 0:
            try:
                Xt = hole_move(cur)
            except Exception:
                Xt = cur + _np.random.randn(n, d) * 0.05
                Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            # small jitter to others
            jit = _np.random.randn(n, d) * 0.008
            Xt = Xt + jit
            Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 360, 55.0, 420.0, 0.012, 0.0025
        elif trial % 5 == 0:
            # worst-subset reinit
            Gc = cur @ cur.T
            _np.fill_diagonal(Gc, -2.0)
            per = Gc.max(axis=1)
            k = 8 if (trial % 10 == 0) else 5
            worst = _np.argpartition(-per, k)[:k]
            Xt = cur + _np.random.randn(n, d) * 0.01
            R = _np.random.randn(k, d)
            R /= _np.linalg.norm(R, axis=1, keepdims=True)
            Xt[worst] = R
            Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = 360, 45.0, 400.0, 0.012, 0.0025
        else:
            if trial % 17 == 0:
                sigma, steps = 0.20, 400
            elif trial % 7 == 0:
                sigma, steps = 0.08, 340
            elif trial % 3 == 0:
                sigma, steps = 0.045, 320
            else:
                sigma, steps = 0.016, 280
            Gc = cur @ cur.T
            _np.fill_diagonal(Gc, -2.0)
            per_max = Gc.max(axis=1)
            gmax = float(Gc.max())
            rattler = per_max < (gmax - 0.003)
            noise = _np.random.randn(n, d) * sigma
            if _np.any(rattler):
                noise[rattler] *= 2.5
            Xt = cur + noise
            if _np.random.rand() < 0.75:
                c = Gc[iu0, iu1]
                kk = 20
                part = _np.argpartition(-c, kk)[:kk]
                pick = part[_np.random.randint(0, kk)]
                a = int(iu0[pick]); b = int(iu1[pick])
                alpha = min(0.25, sigma * 2.0 + 0.05)
                Xt[a] -= alpha * cur[b]
                Xt[b] -= alpha * cur[a]
                Xt[a] += _np.random.randn(d) * sigma * 0.5
                Xt[b] += _np.random.randn(d) * sigma * 0.5
            Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, b0, b1, lr0, lr1 = steps, 55.0, 420.0, 0.012, 0.0025

        dl = min(GLOBAL_END, HARD_END - 20)
        try:
            Xopt, vopt = adam_refine(Xt, steps, b0, b1, lr0, lr1, dl)
        except Exception:
            continue
        if vopt < best_val - 1e-14:
            best_val = vopt
            best = Xopt.copy()
        T = 0.0012 * _np.exp(-trial / 250.0) + 1e-5
        if vopt < cur_val:
            cur, cur_val = Xopt, vopt
        else:
            if _np.random.rand() < _np.exp((cur_val - vopt) / T):
                cur, cur_val = Xopt, vopt
        if trial % 40 == 0:
            if cur_val > best_val + 0.002:
                cur = best.copy()
                cur_val = best_val
        if best_val < 0.890:
            break

    try:
        if _time.time() < HARD_END - 8:
            best, bv2 = adam_refine(best, 220, 120.0, 650.0, 0.004, 0.001, HARD_END - 6)
            best_val = min(best_val, bv2)
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
