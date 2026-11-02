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
    DEADLINE = t0 + 170.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

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
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
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

    def lse_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 8.0:
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
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(Xn, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        Xn /= nn
        return Xn

    def hole_insert(base, nsample=15000, refine=120):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy()
        cur = float(cm[k])
        for st in range(refine):
            sig = 0.04 * (1.0 - st / float(refine)) + 0.001
            prop = bc + sig * rng.standard_normal(d)
            pn = np.linalg.norm(prop)
            if pn == 0:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop
                cur = md
        return bc, cur

    # initial quick polish of incumbent
    if time.time() < DEADLINE - 20:
        try:
            Xp, Vp = lp_polish(Xbest, seconds=4.0, radius_init=3e-4)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        except Exception:
            pass

    def contact_counts(X, thr_offset=2e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - thr_offset
        cnt = (G > thr).sum(axis=1)
        mx = G.max(axis=1)
        return cnt, mx, m

    trial = 0
    stall = 0
    # precompute cycle of knockout sizes
    while time.time() < DEADLINE - 9.0:
        trial += 1
        remain = DEADLINE - time.time()
        if remain < 10:
            break
        # every 7th trial: fresh random diverse start (different basin)
        if trial % 7 == 0:
            Xr = rng.standard_normal((n, d))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            # coarse anneal
            Xa = lse_anneal(Xr, betas=[60, 180, 500], maxiters=[120, 130, 160])
            va = _max_cos(Xa)
            if va < Vbest + 0.004 and time.time() < DEADLINE - 12:
                Xap, vap = lp_polish(Xa, seconds=2.0)
                if vap < Vbest:
                    Xbest, Vbest = Xap, vap
                    stall = 0
                elif va < Vbest:
                    Xbest, Vbest = Xa, va
                    stall = 0
                else:
                    stall += 1
            else:
                if va < Vbest:
                    Xbest, Vbest = Xa, va
                    stall = 0
                else:
                    stall += 1
            continue

        # choose knockout size, grow with stall
        base_ks = [10, 14, 18, 22, 26]
        k = base_ks[rng.integers(0, len(base_ks))]
        if stall > 8:
            k = int(rng.integers(20, 30))
        k = min(k, 30)

        mode = trial % 3
        if mode == 0:
            # cap-cluster knockout: remove geographically clustered cap
            cap = rng.standard_normal(d)
            cap /= np.linalg.norm(cap)
            dots = Xbest @ cap
            idx_drop = np.argsort(-dots)[:k]
        elif mode == 1:
            # overload surgery: drop high-contact + low-contact mix
            try:
                cnt, mx, m = contact_counts(Xbest, thr_offset=2e-3)
                order_high = np.argsort(-cnt)
                order_low = np.argsort(cnt)
                kh = k // 2
                kl = k - kh
                idx_drop = np.concatenate([order_high[:kh], order_low[:kl]])
                # add one worst-pair point to ensure change
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                if rng.random() < 0.7:
                    idx_drop[0] = ij[0]
            except Exception:
                dots = Xbest @ (rng.standard_normal(d))
                idx_drop = np.argsort(-dots)[:k]
        else:
            # random cap + worst points mix
            cap = rng.standard_normal(d)
            cap /= np.linalg.norm(cap)
            dots = Xbest @ cap
            idx_cap = np.argsort(-dots)[: k // 2]
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            idx_worst = np.argsort(-rm)[: k - k // 2]
            idx_drop = np.concatenate([idx_cap, idx_worst])
            # unique
            idx_drop = np.unique(idx_drop)
            while len(idx_drop) < k:
                extra = int(rng.integers(0, n))
                if extra not in idx_drop:
                    idx_drop = np.append(idx_drop, extra)
            idx_drop = idx_drop[:k]

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[idx_drop] = False
        base = Xbest[keep_mask].copy()
        # small jitter to kept points to break exact jamming
        jit = 0.004 + 0.002 * rng.random()
        base = base + jit * rng.standard_normal(base.shape)
        base /= np.linalg.norm(base, axis=1, keepdims=True)

        # grow back k points via farthest-point insertion
        cur = base
        for _ in range(k):
            if time.time() > DEADLINE - 10:
                break
            v, _ = hole_insert(cur, nsample=12000, refine=100)
            cur = np.vstack([cur, v[None, :]])
        Xtry = cur
        if Xtry.shape[0] != n:
            # fallback pad
            while Xtry.shape[0] < n:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                Xtry = np.vstack([Xtry, v[None, :]])
            Xtry = Xtry[:n]
        # shuffle order to avoid bias
        perm = rng.permutation(n)
        Xtry = Xtry[perm]

        # anneal: schedule depends on k
        if k <= 14:
            betas = [150, 400, 900]
            mis = [100, 130, 160]
        else:
            betas = [120, 350, 800]
            mis = [110, 140, 170]
        Xa = lse_anneal(Xtry, betas, mis)
        va = _max_cos(Xa)
        # decide polish
        improved_coarse = va < Vbest + 0.0015
        if improved_coarse and time.time() < DEADLINE - 10:
            Xap, vap = lp_polish(Xa, seconds=2.2, radius_init=4e-4)
            if vap < Vbest:
                Xbest, Vbest = Xap, vap
                stall = 0
            elif va < Vbest:
                Xbest, Vbest = Xa, va
                stall = 0
            else:
                # accept slight worsening occasionally to move basin if much better topology? no
                stall += 1
                # global shake attempt from polished version if close
                if vap < Vbest + 0.0008 and rng.random() < 0.4:
                    sig = 0.015
                    Xs = Xap + sig * rng.standard_normal((n, d))
                    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
                    Xs2 = lse_anneal(Xs, [300, 800], [100, 120])
                    v2 = _max_cos(Xs2)
                    if v2 < Vbest:
                        Xbest, Vbest = Xs2, v2
                        stall = 0
        else:
            if va < Vbest:
                Xbest, Vbest = Xa, va
                stall = 0
            else:
                stall += 1
        # occasional big shake if long stall
        if stall >= 6 and time.time() < DEADLINE - 14:
            sig = 0.05 if stall < 12 else 0.09
            Xs = Xbest + sig * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs2 = lse_anneal(Xs, [100, 350, 800], [80, 110, 130])
            v2 = _max_cos(Xs2)
            if v2 < Vbest + 0.002:
                Xsp, vsp = lp_polish(Xs2, seconds=1.8)
                if vsp < Vbest:
                    Xbest, Vbest = Xsp, vsp
                    stall = 0
                elif v2 < Vbest:
                    Xbest, Vbest = Xs2, v2
                    stall = 0
            stall = 0 if v2 < Vbest else stall

    # final polish with remaining time
    try:
        if time.time() < DEADLINE - 2:
            secs = max(1.0, DEADLINE - time.time() - 1.0)
            secs = min(secs, 12.0)
            Xp, Vp = lp_polish(Xbest, seconds=secs, radius_init=3e-4)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
