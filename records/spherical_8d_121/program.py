import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    rng = np.random.default_rng(0)
    np.random.seed(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    # E8 roots (240 x 8), unit norm
    E8 = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    E8.append(v)
    for mask in range(256):
        bits = [(mask >> k) & 1 for k in range(8)]
        if sum(bits) % 2 == 1:
            continue
        v = np.where(np.array(bits) == 1, -0.5, 0.5)
        E8.append(v.astype(float))
    E8 = np.stack(E8, axis=0)  # 240 x 8
    E8 /= np.linalg.norm(E8, axis=1, keepdims=True)

    def make_hinge(t):
        tt = float(t)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            diff = c - tt
            pos = diff > 0
            if np.any(pos):
                f = float((diff[pos] ** 2).sum())
                w = np.zeros_like(c)
                w[pos] = 2.0 * diff[pos]
            else:
                f = 0.0
                w = np.zeros_like(c)
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), np.nan_to_num(GX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        return fun

    def make_lse(beta):
        b = float(beta)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), np.nan_to_num(GX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        return fun

    def run_lbfgs(Xinit, fun, maxiter):
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(maxiter * 5)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
        return X

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 60 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2000:
                kth = np.sort(cf)[-2000]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 2500:
                    idxp = np.argpartition(-cf, 2500)[:2500]
                    mk = np.zeros_like(sel); mk[idxp] = True
                    sel = mk; cnt = 2500
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
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
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def block_rotate(X, nblocks):
        perm = rng.permutation(n)
        parts = np.array_split(perm, nblocks)
        Y = np.empty_like(X)
        for blk in parts:
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            Y[blk] = X[blk] @ Q.T
        return Y / np.linalg.norm(Y, axis=1, keepdims=True)

    def k_holes(base, k, nsample=20000):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        bm = (C @ base.T).max(axis=1)
        picked = []
        Tot = bm.copy()
        for _ in range(k):
            if picked:
                P = np.stack(picked, axis=0)
                pm = (C @ P.T).max(axis=1)
                Tot = np.maximum(bm, pm)
            idx = int(np.argmin(Tot))
            bc = C[idx].copy()
            cur_base = base if not picked else np.concatenate([base, np.stack(picked)], axis=0)
            cur = float((bc @ cur_base.T).max())
            for st in range(60):
                sig = 0.03 * (1.0 - st / 60.0) + 0.001
                prop = bc + sig * rng.standard_normal(d)
                pn = np.linalg.norm(prop)
                if pn == 0:
                    continue
                prop /= pn
                md = float((prop @ cur_base.T).max())
                if md < cur:
                    bc = prop; cur = md
            picked.append(bc)
            Tot = np.maximum(Tot, C @ bc)
        return np.stack(picked, axis=0)

    def e8_start():
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        R = E8 @ Q.T
        # greedy farthest-point sampling to 121
        m = R.shape[0]
        selected = [int(rng.integers(0, m))]
        best_dot = R @ R[selected[0]]
        for _ in range(n):
            if len(selected) >= n:
                break
            # max dot to selected set
            D = R @ R[selected].T
            md = D.max(axis=1)
            md[selected] = 2.0
            selected.append(int(np.argmin(md)))
        X = R[selected[:n]].copy()
        X += 0.015 * rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def joint(Xinit, target):
        Xh = run_lbfgs(Xinit, make_hinge(target), 70)
        if time.time() > DEADLINE - 6:
            return Xh, _max_cos(Xh)
        Xf = Xh.ravel().copy()
        for beta, mi in [(350.0, 50), (1200.0, 70)]:
            if time.time() > DEADLINE - 5:
                break
            fun = make_lse(beta)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 5)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X, _max_cos(X)

    # light initial squeeze not needed much; skip heavy polish
    it = 0
    while time.time() < DEADLINE - 8.0:
        it += 1
        if it > 60:
            break
        kind = it % 5
        try:
            if kind == 0:
                nb = [4, 5, 3, 6][(it // 5) % 4]
                X0 = block_rotate(Xbest, nb)
            elif kind == 1:
                k = [18, 22, 15, 25][(it // 5) % 4]
                # large cap knockout around random direction
                ctr = rng.standard_normal(d); ctr /= np.linalg.norm(ctr)
                dots = Xbest @ ctr
                order = np.argsort(-dots)
                rlist = list(order[:k])
                mask = np.ones(n, dtype=bool); mask[rlist] = False
                base = Xbest[mask]
                newpts = k_holes(base, k, nsample=20000)
                X0 = np.empty_like(Xbest); X0[mask] = base; X0[~mask] = newpts
            elif kind == 2:
                X0 = e8_start()
            elif kind == 3:
                X0 = rng.standard_normal((n, d))
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            else:
                sig = [0.45, 0.30, 0.55][(it // 5) % 3]
                X0 = Xbest + sig * rng.standard_normal((n, d))
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            continue
        target = Vbest - (0.008 if kind in (0, 1, 2) else 0.004)
        try:
            Xo, vo = joint(X0, target)
        except Exception:
            continue
        if vo < Vbest + 0.006 and time.time() < DEADLINE - 10:
            try:
                Xo2, vo2 = lp_polish(Xo, 1.2)
            except Exception:
                Xo2, vo2 = Xo, vo
            if vo2 < vo:
                Xo, vo = Xo2, vo2
        if vo < Vbest:
            Xbest, Vbest = Xo, vo
            # extra squeeze on improvement
            if time.time() < DEADLINE - 10:
                try:
                    Xp, Vp = lp_polish(Xbest, 1.5)
                    if Vp < Vbest:
                        Xbest, Vbest = Xp, Vp
                except Exception:
                    pass

    # final squeeze
    if time.time() < DEADLINE - 4:
        try:
            Xp, Vp = lp_polish(Xbest, max(0.5, DEADLINE - time.time() - 1.5))
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
