import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def solve():
    t0 = time.time()
    TIME_HARD = 177.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(20261102)
    n, d = 73, 13

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1] if xb.ndim > 1 else 1, d)
        try:
            tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()

    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0[0], iu0[1]

    # elite population: list of (val, X)
    elites = [(best, Xbest.copy())]

    def smooth_fg(Xflat, s):
        X = Xflat.reshape(n, d)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        U = X / nr
        G = U @ U.T
        c = G[I0, J0]
        cmax = float(c.max())
        e = np.exp(s * (c - cmax))
        ss = float(e.sum())
        if not np.isfinite(ss) or ss == 0:
            # fallback to max subgradient
            sel = c >= cmax - 1e-12
            w = sel.astype(float)
            w /= w.sum()
            f = cmax
        else:
            w = e / ss
            f = cmax + np.log(ss) / s
        gradU = np.zeros_like(U)
        # grad distribution
        np.add.at(gradU, I0, w[:, None] * U[J0])
        np.add.at(gradU, J0, w[:, None] * U[I0])
        dot = (gradU * U).sum(axis=1, keepdims=True)
        gradX = (gradU - dot * U) / nr
        return f, gradX.ravel()

    def local_smooth(X_start, s_list, it_list):
        Xflat = X_start.ravel().copy()
        for s, mit in zip(s_list, it_list):
            if time.time() - t0 > TIME_HARD - 8:
                break
            def fun(xf, _s=s):
                f, g = smooth_fg(xf, _s)
                return f, g
            try:
                res = minimize(fun, Xflat, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mit), "ftol": 1e-14,
                                        "gtol": 1e-9, "maxfun": int(mit * 12),
                                        "maxcor": 12})
                Xflat = res.x
            except Exception:
                break
            # keep finite
            if not np.all(np.isfinite(Xflat)):
                Xflat = X_start.ravel().copy()
                break
        X = Xflat.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def greedy_reinsert(Y, idxs, ntry=3000):
        Y = Y.copy()
        for idx in idxs:
            C = rng.randn(ntry, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            others = Y[mask]
            dots = C @ others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            Y[idx] = C[bi]
        return Y

    def lp_polish(x, seconds, r0=5e-4):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        cur = _max_cos(x)
        radius = r0
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and radius > 1e-13:
            if time.time() - t0 > TIME_HARD - 1:
                break
            G = x @ x.T
            cfull = G[iu]
            m = float(cfull.max())
            sel = cfull > m - max(20.0 * radius, 1e-9)
            # cap number of constraints for speed
            if int(sel.sum()) > 2500:
                # keep most tight
                thr = np.sort(cfull)[-2500]
                sel = cfull >= thr
            I, J = iu[0][sel], iu[1][sel]
            cs = cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                              np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
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
            if v < cur - 1e-15:
                x, cur = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x

    def contact_counts(X, tol=2e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        cnt = (G >= m - tol).sum(axis=1)
        return G, m, cnt

    trial = 0
    # initial quick tighten of loaded best with moderate smoothing
    try:
        if time.time() - t0 < SEARCH_END - 5:
            X0t = local_smooth(Xbest, [150.0, 600.0, 2000.0], [120, 150, 150])
            v0t = _max_cos(X0t)
            if v0t < best:
                best = v0t
                Xbest = X0t
                elites = [(best, Xbest.copy())]
    except Exception:
        pass

    while time.time() - t0 < SEARCH_END:
        trial += 1
        remain = SEARCH_END - (time.time() - t0)
        if remain < 1.5:
            break
        # pick base: mostly best, sometimes 2nd elite, sometimes fresh
        if trial % 17 == 0 and len(elites) > 1:
            base = elites[1][1]
        elif trial % 23 == 0:
            base = rng.randn(n, d)
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        elif trial % 29 == 0 and len(elites) > 2:
            base = elites[rng.randint(len(elites))][1]
        else:
            base = Xbest

        Gc, cm, cnt = contact_counts(base, tol=2e-3)
        order_over = np.argsort(-cnt)
        # exact contacts for rattler detection
        G2 = Gc.copy()
        cnt_exact = (G2 >= cm - 1e-9).sum(axis=1)
        ratt = np.where(cnt_exact == 0)[0]

        mode = trial % 12
        Xp = base.copy()
        if mode == 0:  # heavy global
            Xp = base + 0.13 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [80.0, 350.0, 1200.0]
            it_list = [120, 130, 160]
        elif mode == 1:  # reinsert overloaded + rattlers
            victims = []
            # top overloaded
            for v in order_over[:3]:
                victims.append(int(v))
            for v in ratt[:2]:
                if int(v) not in victims:
                    victims.append(int(v))
            victims = victims[:4]
            if len(victims) == 0:
                victims = [int(rng.randint(n))]
            Xp = greedy_reinsert(base, victims, ntry=3500)
            Xp = Xp + 0.008 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [150.0, 600.0, 2200.0]
            it_list = [120, 140, 170]
        elif mode == 2:  # two random reinsert
            victims = [int(rng.randint(n)) for _ in range(2)]
            Xp = greedy_reinsert(base, victims, ntry=4000)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [120.0, 500.0, 1800.0]
            it_list = [120, 130, 160]
        elif mode == 3:  # cluster kick on overloaded
            Xp = base + 0.03 * rng.randn(n, d)
            for v in order_over[:6]:
                Xp[int(v)] += 0.07 * rng.randn(d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [100.0, 400.0, 1400.0]
            it_list = [110, 130, 150]
        elif mode == 4:  # tiny local
            Xp = base + 0.007 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [250.0, 900.0, 3000.0]
            it_list = [90, 120, 150]
        elif mode == 5:  # medium shake
            Xp = base + 0.035 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [150.0, 600.0, 2200.0]
            it_list = [110, 130, 160]
        elif mode == 6:  # single hole move
            if len(ratt) > 0:
                victims = [int(ratt[0])]
            else:
                # worst point = max row max
                Gr = base @ base.T
                np.fill_diagonal(Gr, -2)
                victims = [int(np.argmax(Gr.max(axis=1)))]
            Xp = greedy_reinsert(base, victims, ntry=6000)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [200.0, 700.0, 2500.0]
            it_list = [100, 130, 160]
        elif mode == 7:  # fresh random start (basin discovery)
            if trial % 36 == 7:
                Xp = rng.randn(n, d)
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                s_list = [60.0, 200.0, 700.0, 1800.0]
                it_list = [150, 150, 150, 150]
            else:
                Xp = base + 0.055 * rng.randn(n, d)
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                s_list = [120.0, 500.0, 1800.0]
                it_list = [110, 130, 150]
        elif mode == 8:  # subset rotation-like noise
            Xp = base.copy()
            sub = rng.choice(n, size=10, replace=False)
            Xp[sub] += 0.06 * rng.randn(10, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [120.0, 500.0, 1800.0]
            it_list = [110, 130, 150]
        elif mode == 9:  # break current max pair explicitly
            Gr = base @ base.T
            np.fill_diagonal(Gr, -2)
            ij = np.argmax(Gr)
            a, b = int(ij // n), int(ij % n)
            # reinsert one end into hole
            Xp = greedy_reinsert(base, [a], ntry=5000)
            # kick the other end a bit
            Xp[b] += 0.02 * rng.randn(d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [180.0, 650.0, 2400.0]
            it_list = [110, 130, 160]
        elif mode == 10:
            Xp = base + 0.09 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [90.0, 350.0, 1300.0]
            it_list = [120, 130, 150]
        else:
            Xp = base + 0.018 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            s_list = [200.0, 800.0, 2800.0]
            it_list = [90, 120, 150]

        if remain < 12:
            # shorten
            it_list = [min(int(v), 70) for v in it_list]
            s_list = s_list[-2:]

        try:
            Xs = local_smooth(Xp, s_list, it_list)
        except Exception:
            continue
        v = _max_cos(Xs)
        if v < best - 1e-13:
            best = v
            Xbest = Xs.copy()
            elites.append((v, Xs.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:6]
            # intensify immediately with sharper smooth
            if time.time() - t0 < SEARCH_END - 4:
                try:
                    Xf = local_smooth(Xbest, [800.0, 3000.0], [100, 120])
                    vf = _max_cos(Xf)
                    if vf < best:
                        best = vf
                        Xbest = Xf.copy()
                        elites[0] = (best, Xbest.copy())
                        elites = sorted(elites, key=lambda z: z[0])[:6]
                except Exception:
                    pass
        elif v < best + 0.004:
            # keep diverse near-best in population
            if len(elites) < 6:
                elites.append((v, Xs.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:6]
            elif v < elites[-1][0] - 1e-6:
                elites[-1] = (v, Xs.copy())
                elites = sorted(elites, key=lambda z: z[0])[:6]

        # occasional random-walk escape from last Xs even if worse
        if trial % 25 == 0 and (time.time() - t0) < SEARCH_END - 8:
            try:
                Xp2 = Xs + 0.04 * rng.randn(n, d)
                Xp2 /= np.linalg.norm(Xp2, axis=1, keepdims=True)
                Xs2 = local_smooth(Xp2, [150.0, 700.0, 2000.0], [80, 100, 120])
                v2 = _max_cos(Xs2)
                if v2 < best - 1e-13:
                    best = v2
                    Xbest = Xs2.copy()
                    elites.append((v2, Xs2.copy()))
                    elites = sorted(elites, key=lambda z: z[0])[:6]
                elif v2 < elites[-1][0] and len(elites) < 6:
                    elites.append((v2, Xs2.copy()))
            except Exception:
                pass

    # final sharpen + LP polish with remaining time
    try:
        if time.time() - t0 < TIME_HARD - 6:
            Xbest = local_smooth(Xbest, [500.0, 1500.0, 3500.0], [120, 130, 150])
            best = _max_cos(Xbest)
    except Exception:
        pass
    # polish top 2 elites briefly and keep best
    try:
        cands = sorted(elites, key=lambda z: z[0])[:2]
    except Exception:
        cands = [(best, Xbest)]
    rem = TIME_HARD - (time.time() - t0)
    if rem > 4:
        per = min(rem - 0.7, 26.0) / max(1, len(cands))
        best_cand = Xbest
        best_v = _max_cos(best_cand)
        for _, Xc in cands:
            if time.time() - t0 > TIME_HARD - 1:
                break
            try:
                Xp2 = lp_polish(Xc, seconds=per, r0=4e-4)
                v2 = _max_cos(Xp2)
                if v2 < best_v:
                    best_v = v2
                    best_cand = Xp2
            except Exception:
                continue
        Xbest = best_cand
        best = best_v
        # extra ultra-fine LP if time left
        rem2 = TIME_HARD - (time.time() - t0)
        if rem2 > 3:
            try:
                Xbest = lp_polish(Xbest, seconds=rem2 - 0.5, r0=1e-4)
            except Exception:
                pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # guard against NaN / zero
    Xbest[~np.isfinite(Xbest)] = 0.0
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        for i in np.where(bad)[0]:
            vv = rng.randn(d)
            vv /= np.linalg.norm(vv)
            Xbest[i] = vv
        Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
