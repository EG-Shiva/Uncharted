import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _hinge_batch(X, steps, band, lr0, lr1, t_end=None):
    X = _norm(X.copy())
    n, d = X.shape
    for it in range(steps):
        if t_end is not None and (it & 31) == 0 and time.time() > t_end:
            break
        f = it / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        thr = M - band
        Mk = G > thr
        # weight linearly by excess to focus on worst
        W = np.zeros_like(G)
        # only upper would double; use full symmetric
        W[Mk] = (G[Mk] - thr) / max(band, 1e-15) + 0.25
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        gn = float(np.sqrt(np.mean(np.sum(Gt*Gt, axis=1)))) + 1e-15
        # normalized step
        step = lr / gn
        # cap per-point move
        pn = np.sqrt(np.sum((step*Gt)**2, axis=1, keepdims=True)) + 1e-15
        cap = 0.03
        scale = np.minimum(1.0, cap / (pn + 1e-15))
        X = X - (step * Gt) * scale
        X = _norm(X)
        band *= 0.9995
        if band < 8e-4:
            band = 8e-4
    return X

def _pair_push(X, steps, s0, s1, t_end=None):
    X = _norm(X.copy())
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    for it in range(steps):
        if t_end is not None and (it & 511) == 0 and time.time() > t_end:
            break
        f = it / max(steps - 1, 1)
        st = s0 + (s1 - s0) * f
        # argmax
        k = int(np.argmax(G))
        i = k // n; j = k % n
        if i == j:
            np.fill_diagonal(G, -2.0)
            continue
        dij = float(G[i, j])
        # tangent components
        gi = X[j] - dij * X[i]
        gj = X[i] - dij * X[j]
        ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
        if ni < 1e-14 or nj < 1e-14:
            # nudge randomly tiny
            continue
        gi /= ni; gj /= nj
        # move both apart
        xi = X[i] - st * gi
        xj = X[j] - st * gj
        xi /= (np.linalg.norm(xi) + 1e-15)
        xj /= (np.linalg.norm(xj) + 1e-15)
        X[i] = xi; X[j] = xj
        # incremental update of G rows/cols i,j
        # recompute dots
        di = X @ xi  # n
        dj = X @ xj
        G[i, :] = di; G[:, i] = di
        G[j, :] = dj; G[:, j] = dj
        G[i, i] = -2.0; G[j, j] = -2.0
        # fix intersection: G[i,j] consistent
        # (di[j] should equal dj[i])
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 2e-15:
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        mm = float(c.max())
        thr = mm - max(25.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.6, 0.03)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _refine_void(y, others):
    y = y / (np.linalg.norm(y) + 1e-15)
    step = 0.07
    for _ in range(30):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - float(others[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= (np.linalg.norm(y) + 1e-15)
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _greedy_fill(kept, k, rng, ntry=1500):
    d = kept.shape[1]
    cur = kept.copy()
    outs = []
    for _ in range(k):
        C = rng.standard_normal((ntry, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-15)
        if len(cur) > 0:
            dots = C @ cur.T
            w = dots.max(axis=1)
            bi = int(np.argmin(w))
            y = C[bi]
            y = _refine_void(y, cur)
        else:
            y = C[0]
        cur = np.vstack([cur, y[None, :]])
        outs.append(y)
    return cur, np.array(outs)

def _rand_rot(rng, d, sigma):
    A = rng.standard_normal((d, d))
    A = A - A.T
    M = np.eye(d) + sigma * A
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R)); s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    EXPL = t0 + 142.0
    rng = np.random.default_rng(20261031)
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    # immediate exact micro-descent on incumbent (cheap, may gain 1e-9)
    try:
        if HARD - time.time() > 15:
            tmp = _pair_push(best_x, 12000, 0.004, 0.00015, HARD - 12)
            vv = _maxcos(tmp)
            if vv < best_v:
                best_v = vv; best_x = tmp.copy()
            # short hinge + push combo
            tmp2 = _hinge_batch(best_x, 120, 0.004, 0.05, 0.005, HARD - 12)
            tmp2 = _pair_push(tmp2, 8000, 0.002, 0.0001, HARD - 12)
            vv2 = _maxcos(tmp2)
            if vv2 < best_v:
                best_v = vv2; best_x = tmp2.copy()
    except Exception:
        pass
    try:
        if HARD - time.time() > 20:
            bx, bv = _lp_polish(best_x, min(time.time() + 2.5, HARD - 10))
            if bv < best_v:
                best_x, best_v = bx, bv
            else:
                # even if not better, pair-push after LP often squeezes
                tmp = _pair_push(bx, 8000, 0.001, 0.00008, HARD - 10)
                vv = _maxcos(tmp)
                if vv < best_v:
                    best_x, best_v = tmp, vv
    except Exception:
        pass
    cur_x = best_x.copy(); cur_v = best_v
    temp0, temp1 = 0.0012, 0.00006
    it = 0
    while time.time() < EXPL:
        it += 1
        if time.time() > HARD - 24:
            break
        frac = (time.time() - t0) / max(EXPL - t0, 1e-9)
        frac = min(1.0, max(0.0, frac))
        temp = temp0 + (temp1 - temp0) * frac
        base = best_x.copy() if (it % 7 == 0 and rng.random() < 0.55) else cur_x.copy()
        mode = rng.random()
        try:
            if mode < 0.30:
                sigma = float(rng.uniform(0.02, 0.22))
                kick = _norm(base + sigma * rng.standard_normal(base.shape))
            elif mode < 0.52:
                kick = base.copy()
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                m = float(G.max())
                deg = (G > m - 0.004).sum(axis=1)
                score = mx + 0.0015 * deg
                order = np.argsort(-score)
                K = int(rng.integers(4, 12))
                victims = order[:K]
                mask = np.ones(116, dtype=bool); mask[victims] = False
                kept = kick[mask]
                _, newpts = _greedy_fill(kept, K, rng, ntry=1500)
                for ii, vi in enumerate(victims):
                    kick[vi] = newpts[ii]
                kick = _norm(kick + float(rng.uniform(0.015, 0.08)) * rng.standard_normal(kick.shape))
            elif mode < 0.70:
                kick = base.copy()
                K = int(rng.integers(18, 55))
                idx = rng.choice(116, size=K, replace=False)
                Q = _rand_rot(rng, 6, float(rng.uniform(0.3, 1.0)))
                kick[idx] = kick[idx] @ Q.T
                kick = _norm(kick + float(rng.uniform(0.015, 0.07)) * rng.standard_normal(kick.shape))
            elif mode < 0.85:
                kick = base.copy()
                u = rng.standard_normal(6); u /= (np.linalg.norm(u) + 1e-15)
                dots = base @ u
                K = int(rng.integers(6, 16))
                # remove one cap, regrow in opposite void
                victims = np.argsort(-dots)[:K]
                mask = np.ones(116, dtype=bool); mask[victims] = False
                kept = kick[mask]
                _, newpts = _greedy_fill(kept, K, rng, ntry=1500)
                for ii, vi in enumerate(victims):
                    kick[vi] = newpts[ii]
                kick = _norm(kick + 0.02 * rng.standard_normal(kick.shape))
            else:
                kick = base.copy()
                K = int(rng.integers(8, 20))
                idx = rng.choice(116, size=K, replace=False)
                kick[idx] = _norm(rng.standard_normal((K, 6)))
        except Exception:
            kick = _norm(base + 0.1 * rng.standard_normal(base.shape))
        q_end = min(time.time() + 1.6, HARD - 20)
        try:
            band = float(rng.uniform(0.002, 0.006))
            kick = _hinge_batch(kick, int(rng.integers(120, 260)), band, 0.06, 0.006, q_end)
        except Exception:
            pass
        try:
            kick = _pair_push(kick, 4000, 0.0025, 0.0002, q_end)
        except Exception:
            pass
        v = _maxcos(kick)
        # selective LP only if promising
        if v < best_v + 0.0025 and time.time() < HARD - 16 and rng.random() < 0.35:
            try:
                tp, vp = _lp_polish(kick, min(time.time() + 0.7, HARD - 14))
                # extra micro squeeze after LP
                tp = _pair_push(tp, 2500, 0.0008, 0.00008, min(time.time() + 0.4, HARD - 14))
                vp = _maxcos(tp)
                kick, v = tp, vp
            except Exception:
                pass
        if v < best_v - 1e-12:
            best_v = v; best_x = kick.copy()
            cur_x = kick.copy(); cur_v = v
        elif v < cur_v or rng.random() < np.exp(-(v - cur_v) / max(temp, 1e-12)):
            if v < cur_v + 0.02:
                cur_x = kick.copy(); cur_v = v
    # final intensive jam of best
    try:
        remain = HARD - time.time() - 0.8
        if remain > 2.0:
            # alternate LP and pair-push
            deadline = time.time() + min(remain, 18.0)
            while time.time() < deadline - 1.0:
                bx, bv = _lp_polish(best_x, min(time.time() + 1.2, deadline - 0.8))
                if bv < best_v:
                    best_v = bv; best_x = bx.copy()
                # micro
                tmp = _pair_push(best_x, 6000, 0.0006, 0.00005, deadline - 0.5)
                vv = _maxcos(tmp)
                if vv < best_v:
                    best_v = vv; best_x = tmp.copy()
                else:
                    if time.time() > deadline - 2.0:
                        break
                    # small random tangent shake to escape LP fixed point
                    shake = 1e-4 * rng.standard_normal(best_x.shape)
                    # project tangent
                    proj = (shake * best_x).sum(axis=1, keepdims=True)
                    shake = shake - proj * best_x
                    best_x = _norm(best_x + shake)
                    vv2 = _maxcos(best_x)
                    if vv2 < best_v:
                        best_v = vv2
                    else:
                        # revert if worse by much
                        if vv2 > best_v + 2e-6:
                            best_x = bx.copy()
                        break
    except Exception:
        pass
    return _norm(best_x)
