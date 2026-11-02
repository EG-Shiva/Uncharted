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
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
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
    Xcur = Xbest.copy()
    Vcur = Vbest

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
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

    def lse_anneal(Xinit, betas, maxiters):
        Xf = Xinit.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 3.0:
                break
            b = float(beta)
            def fun(flat):
                Xr = flat.reshape(n, d)
                nr = np.linalg.norm(Xr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = Xr / nr
                Gm = Y @ Y.T
                c = Gm[iu]
                mm = float(c.max())
                e = np.exp(b * (c - mm))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(mm), np.zeros_like(flat)
                f = mm + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(Xn, axis=1, keepdims=True); nn[nn==0]=1.0
        Xn /= nn
        return Xn

    def contact_counts(X, thr_off=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        B = G >= m - thr_off
        return B.sum(axis=1), m, G

    def pick_knockout(X, k, mode):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if mode == "cap":
            center = int(rng.integers(0, n))
            nn = np.argsort(-G[center])[:k-1]
            idx = [center] + nn.tolist()
            return np.array(idx, dtype=int)
        elif mode == "dense":
            # crowding: sum of top-6 dots
            part = np.sort(G, axis=1)[:, -6:].sum(axis=1)
            cand = np.argsort(-part)[:12]
            center = int(cand[rng.integers(0, len(cand))])
            nn = np.argsort(-G[center])[:k-1]
            return np.array([center] + nn.tolist(), dtype=int)
        elif mode == "low":
            cc, m, _ = contact_counts(X, 1e-6)
            order = np.argsort(cc)  # ascending
            pool = order[:15]
            # pick 2-3 low ones + cap around a random low
            nl = min(3, k)
            lows = rng.choice(pool, size=nl, replace=False)
            center = int(lows[0])
            nn = np.argsort(-G[center])[:k-len(lows)]
            idx = list(np.asarray(lows).tolist()) + np.argsort(-G[center])[:k-len(lows)].tolist()
            # dedup
            idx = list(dict.fromkeys(idx))
            while len(idx) < k:
                r = int(rng.integers(0, n))
                if r not in idx:
                    idx.append(r)
            return np.array(idx[:k], dtype=int)
        elif mode == "high":
            cc, m, _ = contact_counts(X, 1e-6)
            order = np.argsort(-cc)
            pool = order[:15]
            center = int(pool[rng.integers(0, len(pool))])
            nn = np.argsort(-G[center])[:k-1]
            return np.array([center] + nn.tolist(), dtype=int)
        else:
            return rng.choice(n, size=k, replace=False)

    def multi_voids(base, k, nsample=14000, refine_steps=60):
        m = base.shape[0]
        pool = rng.standard_normal((nsample, d))
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        chosen = []
        cur_base = base
        for t in range(k):
            if time.time() > DEADLINE - 3.0:
                break
            D = pool @ cur_base.T
            cm = D.max(axis=1)
            # diversity vs already chosen
            if chosen:
                C = np.stack(chosen, axis=0)
                D2 = pool @ C.T
                cm2 = D2.max(axis=1)
                cm = np.maximum(cm, cm2)
            bi = int(np.argmin(cm))
            bc = pool[bi].copy()
            cur = float(cm[bi])
            # hill refine
            sig0 = 0.05
            for st in range(refine_steps):
                sig = sig0 * (1.0 - st / float(refine_steps)) + 0.001
                prop = bc + sig * rng.standard_normal(d)
                pn = np.linalg.norm(prop)
                if pn == 0:
                    continue
                prop /= pn
                md = float((prop @ cur_base.T).max())
                if chosen:
                    C = np.stack(chosen, axis=0)
                    md = max(md, float((prop @ C.T).max()))
                if md < cur:
                    bc = prop; cur = md
            chosen.append(bc)
            # remove near-duplicates from pool to keep diversity (optional lazy)
        return np.stack(chosen, axis=0) if chosen else np.zeros((0, d))

    # initial polish of best a little
    try:
        Xbest, Vbest = lp_polish(Xbest, 3.0)
        Xcur, Vcur = Xbest.copy(), Vbest
    except Exception:
        pass

    it = 0
    stall = 0
    tabu_sets = []
    modes = ["cap", "dense", "low", "high", "random"]
    while time.time() < DEADLINE - 2.5:
        it += 1
        elapsed = time.time() - t0
        frac = elapsed / BUDGET
        # temperature schedule
        Temp = 0.004 * (1.0 - frac) + 0.0004
        # knockout size schedule: larger early, smaller late
        if frac < 0.5:
            k = int(rng.integers(6, 13))
        elif frac < 0.8:
            k = int(rng.integers(4, 10))
        else:
            k = int(rng.integers(3, 7))
        mode = modes[int(rng.integers(0, len(modes)))]
        # occasionally large global shake instead of surgery
        if it % 9 == 0:
            sig = 0.12 if frac < 0.6 else 0.05
            Xt = Xcur + sig * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xn = lse_anneal(Xt, [120, 350, 900], [60, 60, 90])
            Vn = _max_cos(Xn)
            # accept
            if Vn < Vbest - 1e-12:
                # polish to confirm
                Xp, Vp = lp_polish(Xn, 1.5)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
                    Xcur, Vcur = Xp, Vp
                    stall = 0
                else:
                    if Vn < Vcur or rng.random() < np.exp(-(Vn - Vcur) / Temp):
                        Xcur, Vcur = Xn, Vn
            else:
                if Vn < Vcur or rng.random() < np.exp(-(Vn - Vcur) / Temp):
                    Xcur, Vcur = Xn, Vn
                    stall += 1
                else:
                    stall += 1
            if stall > 12:
                Xcur, Vcur = Xbest.copy(), Vbest
                # big kick from best
                Xcur = Xcur + 0.2 * rng.standard_normal((n, d))
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                Vcur = _max_cos(Xcur)
                stall = 0
            continue

        idx = pick_knockout(Xcur, k, mode)
        key = tuple(sorted(idx.tolist()))
        if key in tabu_sets:
            # perturb to avoid repeat
            idx = rng.choice(n, size=k, replace=False)
        tabu_sets.append(key)
        if len(tabu_sets) > 60:
            tabu_sets.pop(0)
        mask = np.ones(n, dtype=bool); mask[idx] = False
        base = Xcur[mask]  # (n-k, d)
        # find k voids
        try:
            voids = multi_voids(base, k, nsample=12000, refine_steps=50)
        except Exception:
            continue
        if voids.shape[0] != k:
            continue
        Xt = Xcur.copy()
        Xt[idx] = voids
        # joint anneal: two-stage betas
        try:
            Xn = lse_anneal(Xt, [150, 450, 1100], [70, 70, 110])
        except Exception:
            continue
        Vn = _max_cos(Xn)
        if Vn < Vbest - 5e-10:
            try:
                Xp, Vp = lp_polish(Xn, 2.0)
            except Exception:
                Xp, Vp = Xn, Vn
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
                Xcur, Vcur = Xp, Vp
                stall = 0
                continue
            # even if polish didn't beat, move cur to Xn if good
            if Vn < Vcur or rng.random() < np.exp(-(Vn - Vcur) / Temp):
                Xcur, Vcur = Xn, Vn
        else:
            if Vn < Vcur or rng.random() < np.exp(-(Vn - Vcur) / Temp):
                Xcur, Vcur = Xn, Vn
                stall += 1
            else:
                stall += 1
        if stall > 14:
            Xcur, Vcur = Xbest.copy(), Vbest
            Xcur = Xcur + 0.15 * rng.standard_normal((n, d))
            Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
            Vcur = _max_cos(Xcur)
            stall = 0

    # final polish
    try:
        Xp, Vp = lp_polish(Xbest, max(0.5, DEADLINE - time.time() - 0.5))
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
