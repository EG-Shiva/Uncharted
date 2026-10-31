import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
try:
    from scipy.linalg import expm
    _HAS_EXPM = True
except Exception:
    _HAS_EXPM = False


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
    _np.random.seed(12345)
    t0 = _time.time()
    HARD_END = t0 + 172.0
    GLOBAL_END = t0 + 138.0

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

    def find_voids(X, n_probe=4000, n_void=3):
        R = _np.random.randn(n_probe, 4)
        R /= _np.linalg.norm(R, axis=1, keepdims=True)
        S = R @ X.T
        mx = S.max(axis=1)
        idx = _np.argpartition(mx, n_void)[:n_void]
        voids = R[idx].copy()
        # refine each void by ascent on sphere minimizing softmax max
        for k in range(n_void):
            u = voids[k].copy()
            for _ in range(40):
                dots = X @ u
                m0 = float(dots.max())
                beta = 120.0
                e = _np.exp(beta * (dots - m0))
                s = float(e.sum())
                if not _np.finite(s) or s == 0:
                    break
                w = e / s
                grad = w @ X
                g = grad - _np.dot(grad, u) * u
                ng = _np.linalg.norm(g)
                if ng < 1e-12:
                    break
                u = u - 0.08 * g
                u /= _np.linalg.norm(u)
            voids[k] = u
        return voids

    def random_small_rotation(scale):
        A = _np.random.randn(4, 4)
        Sk = (A - A.T) * (scale / 2.0)
        if _HAS_EXPM:
            try:
                return expm(Sk)
            except Exception:
                pass
        # fallback Cayley / QR approx: I+S
        Q, _ = _np.linalg.qr(_np.eye(4) + Sk)
        # ensure proper rotation (det +1)
        if _np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        # interpolate toward identity: Q^alpha approx
        # just blend: (1-a)I + aQ then orthonormalize
        a = min(1.0, scale * 3.0)
        M = (1 - a) * _np.eye(4) + a * Q
        Q2, _ = _np.linalg.qr(M)
        if _np.linalg.det(Q2) < 0:
            Q2[:, 0] *= -1
        return Q2

    trial = 0
    while _time.time() < GLOBAL_END and _time.time() < HARD_END - 25:
        trial += 1
        rtype = _np.random.rand()
        Gc = cur @ cur.T
        _np.fill_diagonal(Gc, -2.0)
        per_max = Gc.max(axis=1)
        gmax = float(Gc.max())

        Xt = None
        steps = 300
        # choose perturbation
        if rtype < 0.35:
            # global Gaussian + contact break
            if trial % 17 == 0:
                sigma = 0.20
                steps = 400
            elif trial % 7 == 0:
                sigma = 0.08
                steps = 360
            elif trial % 3 == 0:
                sigma = 0.045
                steps = 320
            else:
                sigma = 0.016
                steps = 280
            rattler = per_max < (gmax - 0.003)
            noise = _np.random.randn(n, d) * sigma
            if _np.any(rattler):
                noise[rattler] *= 2.5
            Xt = cur + noise
            if _np.random.rand() < 0.7:
                c = Gc[iu0, iu1]
                k = 20
                part = _np.argpartition(-c, k)[:k]
                pick = part[_np.random.randint(0, k)]
                a = int(iu0[pick]); b = int(iu1[pick])
                alpha = min(0.25, sigma * 2.0 + 0.05)
                Xt[a] -= alpha * cur[b]
                Xt[b] -= alpha * cur[a]
                Xt[a] += _np.random.randn(d) * sigma * 0.5
                Xt[b] += _np.random.randn(d) * sigma * 0.5
        elif rtype < 0.60:
            # cap destruction / resampling
            steps = 380
            cidx = _np.random.randint(0, n)
            center = cur[cidx].copy()
            dots = cur @ center
            order = _np.argsort(-dots)
            k = int(_np.random.randint(5, 17))
            idx_cap = order[:k]
            Xt = cur + _np.random.randn(n, d) * 0.006
            if _np.random.rand() < 0.5:
                # global re-scatter
                Xt[idx_cap] = _np.random.randn(k, d)
            else:
                s = 0.30 + 0.20 * _np.random.rand()
                Xt[idx_cap] = center + _np.random.randn(k, d) * s
            # extra push apart for the cap points
            Xt[idx_cap] += _np.random.randn(k, d) * 0.02
        elif rtype < 0.80:
            # rattler teleport to void
            steps = 360
            Xt = cur + _np.random.randn(n, d) * 0.008
            try:
                voids = find_voids(cur, n_probe=4000, n_void=3)
            except Exception:
                voids = None
            if voids is not None:
                # pick worst rattlers: smallest per_max
                order = _np.argsort(per_max)
                nr_move = 1 if _np.random.rand() < 0.6 else 2
                # ensure they are relatively rattling, else still move smallest
                for j in range(nr_move):
                    ii = int(order[j])
                    vv = voids[_np.random.randint(0, len(voids))]
                    Xt[ii] = vv + _np.random.randn(d) * 0.015
                # jitter neighbours of teleported points a bit
                Xt += _np.random.randn(n, d) * 0.004
            else:
                Xt = cur + _np.random.randn(n, d) * 0.05
        else:
            # cluster rotation (contact component)
            steps = 360
            Xt = cur.copy()
            delta = 0.004 + 0.004 * _np.random.rand()
            thr = gmax - delta
            # adjacency
            # BFS from random seed
            seed = int(_np.random.randint(0, n))
            visited = set([seed])
            frontier = [seed]
            # build neighbor lists on fly
            while frontier and len(visited) < 30:
                u = frontier.pop()
                nbrs = _np.where(Gc[u] > thr)[0]
                for w_ in nbrs:
                    w_ = int(w_)
                    if w_ not in visited:
                        if len(visited) >= 30:
                            break
                        visited.add(w_)
                        frontier.append(w_)
                    if len(visited) >= 30:
                        break
            cl = _np.array(sorted(visited), dtype=int)
            if len(cl) < 3:
                # fallback: k nearest to seed
                dots = cur @ cur[seed]
                order = _np.argsort(-dots)
                cl = order[:int(_np.random.randint(5, 15))]
            angle = 0.06 + 0.25 * _np.random.rand()
            Q = random_small_rotation(angle)
            Xt[cl] = Xt[cl] @ Q.T
            Xt += _np.random.randn(n, d) * 0.010
            # also break top contact inside cluster
            if _np.random.rand() < 0.5 and len(cl) >= 2:
                a = int(cl[_np.random.randint(0, len(cl))])
                b = int(cl[_np.random.randint(0, len(cl))])
                if a != b:
                    Xt[a] -= 0.08 * cur[b]
                    Xt[b] -= 0.08 * cur[a]

        nr = _np.linalg.norm(Xt, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Xt /= nr

        # two beta regimes randomly to diversify basin
        if _np.random.rand() < 0.5:
            Xopt, vopt = adam_refine(Xt, steps, 55.0, 420.0, 0.012, 0.0025)
        else:
            # exploratory low-beta then convergent (split)
            h = steps // 2
            Xmid, vmid = adam_refine(Xt, h, 18.0, 60.0, 0.018, 0.008)
            Xopt, vopt = adam_refine(Xmid, steps - h, 70.0, 500.0, 0.009, 0.002)

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
            best, bv2 = adam_refine(best, 220, 120.0, 650.0, 0.004, 0.001)
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
