import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
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
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8, noise=0.0, rng=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if noise > 0 and rng is not None:
            gt = gt + noise * rng.standard_normal(gt.shape)
            # re-tangentialize noise approx: remove normal comp
            # (already approx tangential plus noise; re-project)
            dot2 = np.sum(gt * x, axis=1, keepdims=True)
            gt = gt - dot2 * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def refine_single(Y, c0, steps=200, s0=0.03, mu=0.5):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    s = s0
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].sum(axis=0)
        gt = g - np.dot(g, c) * c
        n = np.linalg.norm(gt)
        if n < 1e-14:
            break
        gt = gt / n
        vel = mu * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c /= np.linalg.norm(c)
            s *= 0.996
    return c


def hole_reinsert(x, rng, n_samples=15000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    i, j = divmod(k, n)
    Gi = G[i].copy(); Gi[j] = -2
    Gj = G[j].copy(); Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.2:
        move = i if rng.random() < 0.5 else j
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 3000
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    best_c = refine_single(Y, best_c, steps=200, s0=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y


def random_rotation(d, rng, angle):
    A = rng.standard_normal((d, d))
    A = A - A.T
    # exp approx via Cayley or just QR of small skew: use matrix exp by series for tiny angle
    # simple: Q = I + angle*A / ||A|| ; orthonormalize
    nrm = np.linalg.norm(A)
    if nrm < 1e-12:
        return np.eye(d)
    A = A / nrm * angle
    Q, _ = np.linalg.qr(np.eye(d) + A)
    return Q


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(42)
    n, d = 272, 4
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            bx = bx / np.linalg.norm(bx, axis=1, keepdims=True)
            cands.append(bx)
    if rx is not None:
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            rx = rx / np.linalg.norm(rx, axis=1, keepdims=True)
            cands.append(rx)
    if len(cands) == 0:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best_x = min(cands, key=lambda z: max_cos(z)).copy()
    best_v = max_cos(best_x)

    it = 0
    # main basin-hopping; reserve ~22s for final polish
    while time.time() - t0 < BUDGET - 22.0:
        it += 1
        op = it % 10
        y = best_x.copy()
        if op in (0, 1, 2, 3):
            # targeted worst-point kick
            try:
                rm, _ = row_max(best_x)
                order = np.argsort(-rm)
            except Exception:
                order = np.arange(n)
            k = int(rng.integers(4, 36))
            # occasionally larger kick
            if rng.random() < 0.15:
                k = int(rng.integers(30, 80))
            sigma = float(rng.choice([0.02, 0.04, 0.07, 0.12, 0.20, 0.30]))
            idx = order[:k]
            y[idx] = y[idx] + rng.standard_normal((k, d)) * sigma
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # annealed GD
            a0 = float(rng.choice([8.0, 15.0, 25.0]))
            a1 = float(rng.choice([150.0, 300.0, 500.0]))
            s0 = float(rng.choice([0.015, 0.03, 0.05]))
            s1 = float(rng.choice([0.001, 0.003, 0.006]))
            steps = int(rng.choice([250, 400, 600]))
            mu = 0.8 if rng.random() < 0.7 else 0.5
            y = smooth_gd(y, steps, a0, a1, s0, s1, mu=mu)
        elif op == 4 or op == 5:
            # hole reinsertion + short GD
            y = hole_reinsert(best_x, rng, n_samples=12000)
            y = smooth_gd(y, 250, 15.0, 300.0, 0.015, 0.002, mu=0.8)
        elif op == 6:
            # cap rotation
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            dots = best_x @ u
            thr = float(rng.choice([0.75, 0.85, 0.90, 0.93]))
            sel = np.where(dots > thr)[0]
            if len(sel) < 2 or len(sel) > 200:
                # fallback kick
                y = best_x + rng.standard_normal((n, d)) * 0.05
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                ang = float(rng.choice([0.03, 0.06, 0.12]))
                Q = random_rotation(d, rng, ang)
                y[sel] = (y[sel] @ Q.T)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_gd(y, 350, 12.0, 250.0, 0.02, 0.003, mu=0.8)
        elif op == 7:
            # fresh random restart, longer GD
            y = rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_gd(y, 700, 10.0, 400.0, 0.04, 0.002, mu=0.85)
        else:
            # global jitter + Langevin-ish GD
            sigma = float(rng.choice([0.01, 0.03, 0.08]))
            y = best_x + rng.standard_normal((n, d)) * sigma
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_gd(y, 400, 20.0, 350.0, 0.025, 0.002, mu=0.8,
                          noise=0.02, rng=rng)
        v = max_cos(y)
        # accept strict improvement immediately with quick LP lock if time allows
        if v < best_v:
            # lock with short LP if promising and time remains
            rem = BUDGET - 22.0 - (time.time() - t0)
            if rem > 4.0 and v < 0.905:
                # spend 2-3 s
                try:
                    yl = lp_polish(y, min(3.0, max(1.0, rem * 0.15)))
                    vl = max_cos(yl)
                    if vl < v:
                        y, v = yl, vl
                except Exception:
                    pass
                if time.time() - t0 >= BUDGET - 22.0:
                    if v < best_v:
                        best_x, best_v = y, v
                    break
            if v < best_v:
                best_x, best_v = y, v
                continue
        # occasional sideways LP on near-best to change topology
        if it % 12 == 0 and v < best_v + 0.004 and v < 0.905:
            rem = BUDGET - 22.0 - (time.time() - t0)
            if rem > 6.0:
                try:
                    yl = lp_polish(y, 2.0)
                    vl = max_cos(yl)
                    if vl < best_v:
                        best_x, best_v = yl, vl
                except Exception:
                    pass
        if time.time() - t0 >= BUDGET - 22.0:
            break

    # final polish of incumbent with remaining time
    rem = BUDGET - (time.time() - t0)
    if rem > 3.0:
        try:
            yp = lp_polish(best_x, max(3.0, rem - 1.0))
            vp = max_cos(yp)
            if vp < best_v:
                best_x, best_v = yp, vp
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
