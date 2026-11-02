import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: clip argument
        A = alpha * (G - m)
        # cap to avoid inf (exp(50) huge)
        np.clip(A, -50.0, 0.0, out=A)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt, axis=1).mean()) + 1e-12
        X = X - (lr / gn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X


def _quench_sym(Y, steps, a0, a1, lr0, lr1):
    # Y: (29,22), X = [Y;-Y] with symmetrized gradient
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        X = np.concatenate([Y, -Y], axis=0)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        A = alpha * (G - m)
        np.clip(A, -50.0, 0.0, out=A)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # symmetrize: dY from top half minus bottom half
        n2 = Y.shape[0]
        GtY = Gt[:n2] - Gt[n2:]
        gn = float(np.linalg.norm(GtY, axis=1).mean()) + 1e-12
        Y = Y - (lr / gn) * GtY
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16)
    X = np.concatenate([Y, -Y], axis=0)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-16)
    return X


def _hole_replace(X, idxs, rng, n_cand=2500):
    n, d = X.shape
    if np.isscalar(idxs):
        idxs = [int(idxs)]
    else:
        idxs = [int(i) for i in idxs]
    X2 = X.copy()
    for idx in idxs:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        Y = X2[mask]
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-16)
        dots = C @ Y.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        # also try second best with small prob for diversity done by caller
        X2[idx] = C[bi]
    return X2


def _greedy_push(X, steps, eps0, eps1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        if eps0 != eps1:
            eps = eps0 * (eps1 / eps0) ** f
        else:
            eps = eps0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= (np.linalg.norm(X[i]) + 1e-16)
        X[j] /= (np.linalg.norm(X[j]) + 1e-16)
    return X


def _lp_polish(x, seconds, t_deadline, cap=1100, maxit=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-14:
        it += 1
        if it > maxit:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        if v < best - 1e-15:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def _short_lp(x, iters=5, radius0=4e-4, cap=700):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-16)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    iu = np.triu_indices(n, 1)
    for _ in range(iters):
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 5e-5)
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-16)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.3, 0.01)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return x, best


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 136.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)

    # presets: (a0,a1,lr0,lr1,steps)
    presets = [
        (60, 600, 0.020, 0.0020, 320),
        (120, 900, 0.015, 0.0015, 380),
        (40, 500, 0.025, 0.0025, 280),
        (150, 1200, 0.012, 0.0012, 420),
        (80, 700, 0.018, 0.0018, 340),
    ]

    trial = 0
    lp_probes = 0
    while time.time() < HOP_END and time.time() < T_END - 32 and trial < 120:
        trial += 1
        a0, a1, lr0, lr1, steps = presets[trial % len(presets)]
        mode = trial % 12
        y = None
        sym_mode = False
        try:
            if mode == 0:
                sig = 0.008 + 0.004 * ((trial // 12) % 3)
                y = best + sig * rng.standard_normal(best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 1:
                y = best.copy()
                k = 6
                idx = rng.choice(n, size=k, replace=False)
                y[idx] += 0.08 * rng.standard_normal((k, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 2:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = (G > m - 0.004).sum(axis=1)
                order = np.argsort(-cnt, kind="stable")
                k = 2 + (trial % 2)
                victims = order[:k].copy()
                if rng.random() < 0.5:
                    victims = rng.choice(order[:8], size=k, replace=False)
                y = _hole_replace(best, victims, rng, n_cand=2500)
                y += 0.004 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 3:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G))
                i = k // n
                j = k % n
                victim = i if rng.random() < 0.5 else j
                y = _hole_replace(best, victim, rng, n_cand=3500)
                other = j if victim == i else i
                y[other] += 0.01 * rng.standard_normal(d)
                y += 0.002 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 4:
                y = best.copy()
                k = 4
                idx = rng.choice(n, size=k, replace=False)
                z = rng.standard_normal((k, d))
                z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-16)
                y[idx] = z
                y += 0.002 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 5:
                # fresh random restart
                y = rng.standard_normal((n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                # use wider quench
                a0, a1, lr0, lr1, steps = 30, 500, 0.03, 0.003, 420
            elif mode == 6:
                # antipodal symmetric search
                sym_mode = True
                Y0 = rng.standard_normal((29, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
                # 50% seed from best by pairing
                if rng.random() < 0.5 and best is not None:
                    # pick 29 of 58 at random as seeds
                    idx = rng.choice(n, size=29, replace=False)
                    Y0 = best[idx].copy() + 0.01 * rng.standard_normal((29, d))
                    Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            elif mode == 7:
                y = best + 0.015 * rng.standard_normal(best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 8:
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = (G > m - 0.004).sum(axis=1)
                order = np.argsort(-cnt, kind="stable")
                k = 3
                victims = rng.choice(order[:10], size=k, replace=False)
                y = _hole_replace(best, victims, rng, n_cand=2000)
                y += 0.006 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 9:
                y = best + 0.03 * rng.standard_normal(best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 10:
                # excise worst 5 pairs' endpoints (up to 5 distinct) and reinsert greedily
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                iu0 = np.triu_indices(n, 1)
                c = G[iu0]
                order = np.argsort(-c)
                victims = []
                for t in order[:12]:
                    ii = int(iu0[0][t])
                    jj = int(iu0[1][t])
                    for v in (ii, jj):
                        if v not in victims:
                            victims.append(v)
                        if len(victims) >= 4:
                            break
                    if len(victims) >= 4:
                        break
                y = _hole_replace(best, victims, rng, n_cand=2000)
                y += 0.003 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                # random rotation of a subset in a random 2D plane
                y = best.copy()
                k = 8
                idx = rng.choice(n, size=k, replace=False)
                a = rng.standard_normal(d)
                a /= (np.linalg.norm(a) + 1e-16)
                b = rng.standard_normal(d)
                b = b - float(b @ a) * a
                b /= (np.linalg.norm(b) + 1e-16)
                ang = float(rng.uniform(-0.15, 0.15))
                ca, sa = float(np.cos(ang)), float(np.sin(ang))
                for ii in idx:
                    v = y[ii]
                    pa, pb = float(v @ a), float(v @ b)
                    v = v - pa * a - pb * b + (ca * pa - sa * pb) * a + (sa * pa + ca * pb) * b
                    v /= (np.linalg.norm(v) + 1e-16)
                    y[ii] = v
                y += 0.002 * rng.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)

            remain = min(HOP_END - time.time(), T_END - 32 - time.time())
            if remain < 1.0:
                break
            if remain < 12 and steps > 320:
                steps = 320

            if sym_mode:
                q = _quench_sym(Y0, steps, a0, a1, lr0, lr1)
            else:
                q = _quench(y, steps, a0, a1, lr0, lr1)
                q = _greedy_push(q, 60, 0.003, 0.0003)
            v = _max_cos(q)
            # direct improvement
            if v < best_val - 1e-12:
                # probe with short LP to get true basin value
                if time.time() < HOP_END - 4 and lp_probes < 14:
                    try:
                        qp, vp = _short_lp(q, iters=5)
                        lp_probes += 1
                        if vp < best_val:
                            best = qp
                            best_val = vp
                        elif v < best_val:
                            best = q
                            best_val = v
                        continue
                    except Exception:
                        pass
                best = q
                best_val = v
                continue
            # near-miss: if within tolerance, occasionally probe with LP
            if v < best_val + 0.0006 and lp_probes < 14 and (trial % 4 == 0):
                if time.time() < HOP_END - 5:
                    try:
                        qp, vp = _short_lp(q, iters=4)
                        lp_probes += 1
                        if vp < best_val - 1e-12:
                            best = qp
                            best_val = vp
                    except Exception:
                        pass
        except Exception:
            continue

    # final trust-region LP polish with remaining budget
    try:
        budget = min(30.0, max(6.0, T_END - time.time() - 2.0))
        if budget > 2.0:
            best = _lp_polish(best, budget, T_END - 1.0, cap=1100, maxit=60)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    for i in range(best.shape[0]):
        nr = float(np.linalg.norm(best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            vv = rng.standard_normal(d)
            vv /= (np.linalg.norm(vv) + 1e-16)
            best[i] = vv
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
