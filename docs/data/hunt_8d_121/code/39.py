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

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 80 or radius < 1e-14:
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
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel); mk[idxp] = True
                    sel = mk; cnt = 3000
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

    def make_fun(beta=None, hinge_t=None):
        if hinge_t is not None:
            tt = float(hinge_t)
            def fun(flat):
                X = flat.reshape(n, d)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = X / nr
                G = Y @ Y.T
                c = G[iu]
                diff = c - tt
                pos = diff > 0
                f = float((diff[pos] ** 2).sum()) if np.any(pos) else 0.0
                w = np.zeros_like(c); w[pos] = 2.0 * diff[pos]
                W = np.zeros((n, n)); W[Iall, Jall] = w; W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            return fun
        else:
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
                W = np.zeros((n, n)); W[Iall, Jall] = w; W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            return fun

    def run_lbfgs(Xinit, fun, maxiter):
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(maxiter * 6)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1.0
        X /= nn
        return X

    def joint_optimize(Xinit):
        if time.time() > DEADLINE - 8:
            return Xinit, _max_cos(Xinit)
        # stage 1: hinge slightly below best to force feasibility push
        t_h = Vbest - 0.002
        X1 = run_lbfgs(Xinit, make_fun(hinge_t=t_h), 120)
        if time.time() > DEADLINE - 7:
            return X1, _max_cos(X1)
        # stage 2: LogSumExp annealing increasing beta
        Xf = X1.ravel().copy()
        for beta, mi in [(120.0, 60), (350.0, 80), (900.0, 100), (2000.0, 120)]:
            if time.time() > DEADLINE - 6:
                break
            fun = make_fun(beta=beta)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X, _max_cos(X)

    def k_holes(base, k, nsample=20000):
        # base (m,d), find k diverse holes
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
            else:
                Tot = bm
            idx = int(np.argmin(Tot))
            bc = C[idx].copy()
            # local refine vs base+other picks
            cur_base = base if not picked else np.concatenate([base, np.stack(picked)], axis=0)
            cur = float((bc @ cur_base.T).max())
            for st in range(120):
                sig = 0.04 * (1.0 - st / 120.0) + 0.001
                prop = bc + sig * rng.standard_normal(d)
                pn = np.linalg.norm(prop)
                if pn == 0:
                    continue
                prop /= pn
                md = float((prop @ cur_base.T).max())
                if md < cur:
                    bc = prop; cur = md
            picked.append(bc)
        return np.stack(picked, axis=0)

    # initial quick polish of incumbent
    try:
        Xbest, Vbest = lp_polish(Xbest, 4.0)
    except Exception:
        pass

    it = 0
    sigmas = [0.18, 0.10, 0.05, 0.025, 0.12, 0.06]
    while time.time() < DEADLINE - 6.0:
        it += 1
        if it > 200:
            break
        mode = it % 3
        if mode == 2:
            # global shake basin hop
            sig = sigmas[(it // 3) % len(sigmas)]
            Xn = Xbest + sig * rng.standard_normal((n, d))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xo, vo = joint_optimize(Xn)
            if vo < Vbest:
                Xo2, vo2 = lp_polish(Xo, 2.0)
                if vo2 < vo:
                    Xo, vo = Xo2, vo2
                Xbest, Vbest = Xo, vo
            elif vo < Vbest + 0.003 and time.time() < DEADLINE - 20:
                Xo2, vo2 = lp_polish(Xo, 1.0)
                if vo2 < Vbest:
                    Xbest, Vbest = Xo2, vo2
            continue
        # surgery: choose k
        k = [6, 4, 9, 3, 7, 5][it % 6]
        # analyze contact topology via top pairs
        try:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            cf = G[iu]
            order = np.argsort(-cf)
            TOP = 800
            tidx = order[:TOP]
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, iu[0][tidx], 1)
            np.add.at(deg, iu[1][tidx], 1)
            imax = int(iu[0][order[0]]); jmax = int(iu[1][order[0]])
            worst_node = imax if deg[imax] >= deg[jmax] else jmax
        except Exception:
            deg = np.zeros(n, dtype=int); worst_node = int(rng.integers(0, n))
        remove = set([worst_node])
        if mode == 0:
            # degree-balanced: half overloaded, half underloaded
            desc = np.argsort(-deg)
            asc = np.argsort(deg)
            n_over = k // 2
            n_under = k - 1 - n_over
            for v in desc:
                if len(remove) >= 1 + n_over:
                    break
                remove.add(int(v))
            for v in asc:
                if len(remove) >= k:
                    break
                # add some randomness among low-degree: skip occasionally
                remove.add(int(v))
            # fill random if needed
            while len(remove) < k:
                remove.add(int(rng.integers(0, n)))
        else:
            # cap-cluster knockout around worst edge midpoint
            try:
                mid = Xbest[imax] + Xbest[jmax]
                nm = np.linalg.norm(mid)
                if nm == 0:
                    mid = Xbest[worst_node]
                else:
                    mid /= nm
                dots = Xbest @ mid
                cap_order = np.argsort(-dots)
                for v in cap_order[:k]:
                    remove.add(int(v))
                # ensure worst included
                remove.add(worst_node)
                # trim to k (keep worst + nearest)
                rl = list(remove)
                if len(rl) > k:
                    # keep worst + k-1 nearest
                    keep = [worst_node]
                    for v in cap_order:
                        v = int(v)
                        if v != worst_node and v in remove and len(keep) < k:
                            keep.append(v)
                    remove = set(keep)
                while len(remove) < k:
                    remove.add(int(rng.integers(0, n)))
            except Exception:
                while len(remove) < k:
                    remove.add(int(rng.integers(0, n)))
        rlist = sorted(list(remove))[:k]
        mask = np.ones(n, dtype=bool); mask[rlist] = False
        base = Xbest[mask]
        try:
            newpts = k_holes(base, k, nsample=18000)
        except Exception:
            newpts = rng.standard_normal((k, d))
            newpts /= np.linalg.norm(newpts, axis=1, keepdims=True)
        Xn = np.concatenate([base, newpts], axis=0)
        # random permutation to avoid ordering bias
        perm = rng.permutation(n)
        Xn = Xn[perm]
        Xo, vo = joint_optimize(Xn)
        if vo < Vbest:
            Xo2, vo2 = lp_polish(Xo, 2.5)
            if vo2 < vo:
                Xo, vo = Xo2, vo2
            Xbest, Vbest = Xo, vo
        elif vo < Vbest + 0.002 and time.time() < DEADLINE - 25:
            Xo2, vo2 = lp_polish(Xo, 1.2)
            if vo2 < Vbest:
                Xbest, Vbest = Xo2, vo2

    # final squeeze
    try:
        remain = DEADLINE - time.time() - 0.5
        if remain > 1.0:
            Xbest, Vbest = lp_polish(Xbest, min(remain, 8.0))
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
