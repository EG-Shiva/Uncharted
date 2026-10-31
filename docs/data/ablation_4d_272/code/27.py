import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _get_H():
    H = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            H.append(v)
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            for c in (-0.5, 0.5):
                for d in (-0.5, 0.5):
                    H.append([a, b, c, d])
    return np.array(H, dtype=np.float64)


def _qmul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw = a[..., 0]; ax = a[..., 1]; ay = a[..., 2]; az = a[..., 3]
    bw = b[..., 0]; bx = b[..., 1]; by = b[..., 2]; bz = b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def _qconj(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a[..., 1:] *= -1.0
    return a


def _build_X(S, P, R, H, cross):
    K = S.shape[0]
    Xorb = np.empty((K * 24, 4), dtype=np.float64)
    for k in range(K):
        y = _qmul(H, S[k])
        t = _qmul(P[k], y)
        z = _qmul(t, R[k])
        Xorb[k * 24:(k + 1) * 24] = z
    X = np.vstack([Xorb, cross])
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X /= nr
    return X


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
    t0 = _time.time()
    HARD_END = t0 + 175.0
    SYM_END = t0 + 55.0
    FULL_END = t0 + 145.0

    n, d, K = 272, 4, 11
    H = _get_H()
    cross0 = _np.vstack([_np.eye(4), -_np.eye(4)]).astype(_np.float64)

    # fallback best
    best_overall = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=_np.float64)
            if a.shape == (n, d):
                nr = _np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if _np.all(_np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best_overall = a.copy()
        except Exception:
            pass
    if best_overall is None:
        best_overall = _np.random.RandomState(0).randn(n, d)
        best_overall /= _np.linalg.norm(best_overall, axis=1, keepdims=True)
        best_val = _max_cos(best_overall)

    rng = _np.random.RandomState(12345)

    # ---- symmetric seed optimisation ----
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    sym_best_X = None
    sym_best_val = 2.0

    # configs: (use_double, rotate_cross)
    configs = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        (False, False),
        (True, False),
    ]

    def rand_unit(m, r):
        a = r.randn(m, 4)
        a /= _np.linalg.norm(a, axis=1, keepdims=True)
        return a

    for cfg_idx, (use_double, rot_cross) in enumerate(configs):
        if _time.time() > SYM_END - 3:
            break
        S0 = rand_unit(K, rng)
        if use_double:
            P = rand_unit(K, rng)
            Rr = rand_unit(K, rng)
        else:
            P = _np.zeros((K, 4)); P[:, 0] = 1.0
            Rr = _np.zeros((K, 4)); Rr[:, 0] = 1.0
        if rot_cross:
            # random left-right rotation of the 8-point cross
            p0 = rand_unit(1, rng)[0]
            r0 = rand_unit(1, rng)[0]
            tmp = _qmul(p0, cross0)
            cross = _qmul(tmp, r0)
            cross /= _np.linalg.norm(cross, axis=1, keepdims=True)
        else:
            cross = cross0.copy()

        S = S0.copy()
        m1 = _np.zeros_like(S)
        v1 = _np.zeros_like(S)
        # steps adapted to remaining time
        steps = 1400 if cfg_idx < 4 else 900
        beta0, beta1 = 45.0, 420.0
        lr0, lr1 = 0.02, 0.003
        logb = _np.log(beta1 / beta0) / steps
        loglr = _np.log(lr1 / lr0) / steps
        # init val
        X0 = _build_X(S, P, Rr, H, cross)
        bv = _max_cos(X0)
        bS = S.copy()
        bX = X0.copy()
        for tt in range(1, steps + 1):
            if tt % 200 == 0 and _time.time() > SYM_END:
                break
            beta = beta0 * _np.exp(logb * tt)
            lr = lr0 * _np.exp(loglr * tt)
            X = _build_X(S, P, Rr, H, cross)
            G = X @ X.T
            _np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            E = _np.exp(beta * (G - m0))
            _np.fill_diagonal(E, 0.0)
            Es = float(E.sum()) * 0.5
            if not _np.isfinite(Es) or Es <= 0:
                continue
            gradX = (E @ X) / Es
            # backprop to seeds
            Gorb = gradX[:K * 24].reshape(K, 24, 4)
            gradS = _np.empty_like(S)
            for k in range(K):
                g = Gorb[k]
                ph = _qmul(P[k], H)
                phi = _qconj(ph)
                ri = _qconj(Rr[k])
                tmp2 = _qmul(phi, g)
                contrib = _qmul(tmp2, ri)
                gradS[k] = contrib.sum(axis=0)
            dot = _np.sum(gradS * S, axis=1, keepdims=True)
            gt = gradS - dot * S
            m1 = b1m * m1 + (1 - b1m) * gt
            v1 = b2m * v1 + (1 - b2m) * (gt * gt)
            bc1 = 1.0 - b1m ** tt
            bc2 = 1.0 - b2m ** tt
            step = lr * (m1 / bc1) / (_np.sqrt(v1 / bc2) + eps)
            _np.clip(step, -0.08, 0.08, out=step)
            S = S - step
            S /= _np.linalg.norm(S, axis=1, keepdims=True)
            if tt % 120 == 0 or tt == steps:
                Xc = _build_X(S, P, Rr, H, cross)
                vc = _max_cos(Xc)
                if vc < bv:
                    bv = vc
                    bS = S.copy()
                    bX = Xc.copy()
        if bv < sym_best_val:
            sym_best_val = bv
            sym_best_X = bX.copy()
        # also consider final S
        Xf = _build_X(S, P, Rr, H, cross)
        vf = _max_cos(Xf)
        if vf < sym_best_val:
            sym_best_val = vf
            sym_best_X = Xf.copy()

    # ---- break symmetry + full annealing ----
    iu0, iu1 = _np.triu_indices(n, 1)

    def adam_full(X0, total_steps, b0, b1, lr0_, lr1_):
        X = X0.copy()
        m = _np.zeros_like(X)
        v = _np.zeros_like(X)
        G0 = X @ X.T
        _np.fill_diagonal(G0, -2.0)
        bv_ = float(G0.max())
        best_X = X.copy()
        if total_steps <= 0:
            return best_X, bv_
        logb_ = _np.log(b1 / b0) / total_steps
        loglr_ = _np.log(lr1_ / lr0_) / total_steps
        for t in range(1, total_steps + 1):
            beta = b0 * _np.exp(logb_ * t)
            lr = lr0_ * _np.exp(loglr_ * t)
            G = X @ X.T
            c = G[iu0, iu1]
            m0 = float(c.max())
            if m0 < bv_:
                bv_ = m0
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
            step = lr * (m / bc1) / (_np.sqrt(v / bc2) + eps)
            _np.clip(step, -0.08, 0.08, out=step)
            X -= step
            nr = _np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        Gf = X @ X.T
        _np.fill_diagonal(Gf, -2.0)
        fv = float(Gf.max())
        if fv < bv_:
            return X, fv
        return best_X, bv_

    if sym_best_X is not None:
        # symmetry-broken descendants with different noise levels
        # include rotated single-orbit mixtures: rotate first orbit by random pair
        cands = []
        # plain breaks
        for sig in (0.008, 0.015, 0.03):
            Xt = sym_best_X + rng.randn(n, d) * sig
            Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
            cands.append(Xt)
        # one-orbit rotated mixture: apply random left-right rotation to first 24 pts
        try:
            pA = rand_unit(1, rng)[0]
            rA = rand_unit(1, rng)[0]
            Xm = sym_best_X.copy()
            block = Xm[:24]
            tmp = _qmul(pA, block)
            block2 = _qmul(tmp, rA)
            block2 /= _np.linalg.norm(block2, axis=1, keepdims=True)
            Xm[:24] = block2
            Xm = Xm + rng.randn(n, d) * 0.012
            Xm /= _np.linalg.norm(Xm, axis=1, keepdims=True)
            cands.append(Xm)
        except Exception:
            pass
        for Xt in cands:
            if _time.time() > FULL_END:
                break
            # short anneal
            Xo, vo = adam_full(Xt, 700, 55.0, 420.0, 0.012, 0.0025)
            if vo < best_val:
                best_val = vo
                best_overall = Xo.copy()
            # longer touch on the most promising
            if vo < sym_best_val + 0.005 and _time.time() < FULL_END - 10:
                Xo2, vo2 = adam_full(Xo, 500, 120.0, 600.0, 0.005, 0.0012)
                if vo2 < best_val:
                    best_val = vo2
                    best_overall = Xo2.copy()

    # extra basin-hopping from best_overall if time remains (topology changes)
    trial = 0
    cur = best_overall.copy()
    cur_val = best_val
    while _time.time() < FULL_END - 5:
        trial += 1
        if trial > 12:
            break
        sigma = 0.02 if (trial % 3) else 0.06
        Xt = cur + rng.randn(n, d) * sigma
        # targeted contact break on a top pair
        try:
            Gc = cur @ cur.T
            _np.fill_diagonal(Gc, -2.0)
            c = Gc[iu0, iu1]
            k = 20
            part = _np.argpartition(-c, k)[:k]
            pick = part[rng.randint(0, k)]
            a = int(iu0[pick]); b = int(iu1[pick])
            alpha = min(0.25, sigma * 2.0 + 0.05)
            Xt[a] -= alpha * cur[b]
            Xt[b] -= alpha * cur[a]
        except Exception:
            pass
        Xt /= _np.linalg.norm(Xt, axis=1, keepdims=True)
        Xo, vo = adam_full(Xt, 450, 55.0, 420.0, 0.010, 0.0022)
        if vo < best_val - 1e-14:
            best_val = vo
            best_overall = Xo.copy()
        if vo < cur_val:
            cur, cur_val = Xo, vo
        elif rng.rand() < _np.exp((cur_val - vo) / 0.0008):
            cur, cur_val = Xo, vo

    # final LP polish
    try:
        remain = HARD_END - _time.time()
        if remain > 6:
            best_overall = _lp_polish(best_overall, HARD_END - 1.0)
    except Exception:
        pass
    best_overall = best_overall / _np.linalg.norm(best_overall, axis=1, keepdims=True)
    return best_overall
