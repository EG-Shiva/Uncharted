import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())


def solve():
    t0 = time.time()
    BUDGET = 169.0
    DEADLINE = t0 + BUDGET
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]
    Jall = iu[1]

    # ---- load start ----
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

    # ---- L-BFGS LogSumExp anneal ----
    def lbfgs_anneal(X, betas, maxiters):
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
                return float(f), GX.ravel()
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 8)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    # ---- trust region LP polish ----
    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            if radius < 1e-14:
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
            I = iu[0][sel]
            J = iu[1][sel]
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

    # ---- greedy single-point relocate (basin-preserving clean) ----
    def greedy_single(Y, trials):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        cur_global = float(G.max())
        order = np.argsort(-rowmax)
        for idx in order[:15]:
            cur_row = float(rowmax[idx])
            C = rng.standard_normal((trials, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Y.T
            D[:, idx] = -2.0
            cm = D.max(axis=1)
            k = int(np.argmin(cm))
            if cm[k] < cur_row - 1e-12:
                Y2 = Y.copy()
                Y2[idx] = C[k]
                ng = _max_cos(Y2)
                if ng < cur_global - 1e-12:
                    return Y2, ng, True
        return Y, cur_global, False

    # ---- multi-point void re-insertion ----
    def multi_reseed(X, k, nsample):
        rm, _ = _rowmax(X)
        worst = np.argsort(-rm)[:k]
        keep = np.ones(n, dtype=bool)
        keep[worst] = False
        fixed_list = [X[keep]]
        new_pts = []
        for t in range(k):
            base = np.vstack([fixed_list[0]] + ([np.stack(new_pts)] if new_pts else []))
            C = rng.standard_normal((nsample, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ base.T
            cm = D.max(axis=1)
            kbest = int(np.argmin(cm))
            bc = C[kbest].copy()
            cur = float(cm[kbest])
            # hill-climb refine single point
            for st in range(250):
                sig = 0.03 * (1.0 - st / 250.0) + 0.001
                prop = bc + sig * rng.standard_normal(d)
                pn = np.linalg.norm(prop)
                if pn == 0:
                    continue
                prop /= pn
                md = float((prop @ base.T).max())
                if md < cur:
                    bc = prop
                    cur = md
                    if st > 60 and cur < float(rm.max()) - 0.005:
                        # good enough, add a few extra refinements
                        pass
            new_pts.append(bc)
            # update base for next point implicitly via loop
        Y = X.copy()
        for ii, wi in enumerate(worst):
            Y[wi] = new_pts[ii]
        return Y

    # ---- jammed-cluster kick ----
    def cluster_kick(X, ksize, sigma_c, sigma_g):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        # seed among worst 20
        order = np.argsort(-rm)
        seed = int(order[rng.integers(0, min(20, n))])
        nbr = np.argsort(-G[seed])[:ksize]
        Y = X.copy() + sigma_g * rng.standard_normal((n, d))
        Y[nbr] += (sigma_c - sigma_g) * rng.standard_normal((len(nbr), d))
        # occasionally kick a second random cluster
        if rng.random() < 0.4:
            s2 = int(rng.integers(0, n))
            n2 = np.argsort(-G[s2])[:ksize // 2]
            Y[n2] += sigma_c * 0.7 * rng.standard_normal((len(n2), d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    # ---- schedules ----
    sched_coarse = ([120.0, 300.0, 700.0, 1500.0], [120, 120, 150, 200])
    sched_fine = ([400.0, 900.0, 1800.0, 3000.0], [100, 120, 150, 250])
    sched_hot = ([60.0, 150.0, 400.0, 900.0], [150, 150, 150, 200])

    # quick multi-reseed attempts on current best (topology jumps first)
    for k, ns in [(3, 25000), (5, 20000), (4, 25000)]:
        if time.time() > DEADLINE - 120:
            break
        try:
            Y = multi_reseed(Xbest, k, ns)
        except Exception:
            continue
        betas, mis = sched_coarse
        Y = lbfgs_anneal(Y, betas, mis)
        v = _max_cos(Y)
        for _ in range(4):
            Y2, v2, ok = greedy_single(Y, 3000)
            if ok and v2 < v:
                Y, v = Y2, v2
            else:
                break
        if v < Vbest:
            Vbest, Xbest = v, Y.copy()

    # ---- main basin-hopping loop ----
    it = 0
    while time.time() < DEADLINE - 30:
        it += 1
        mode = it % 5
        try:
            if mode == 0:
                Y0 = multi_reseed(Xbest, int(rng.integers(2, 6)), 15000)
                betas, mis = sched_coarse
            elif mode == 1:
                ks = int(rng.integers(7, 14))
                sc = float(rng.uniform(0.20, 0.45))
                sg = float(rng.uniform(0.005, 0.02))
                Y0 = cluster_kick(Xbest, ks, sc, sg)
                betas, mis = sched_coarse if (it % 2 == 0) else sched_hot
            elif mode == 2:
                ks = int(rng.integers(5, 10))
                sc = float(rng.uniform(0.10, 0.25))
                sg = float(rng.uniform(0.01, 0.03))
                Y0 = cluster_kick(Xbest, ks, sc, sg)
                betas, mis = sched_fine
            elif mode == 3:
                sig = float(rng.uniform(0.01, 0.05))
                Y0 = Xbest + sig * rng.standard_normal((n, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
                # random subset re-init
                m = int(rng.integers(3, 9))
                idx = rng.choice(n, size=m, replace=False)
                R = rng.standard_normal((m, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                Y0[idx] = R
                betas, mis = sched_hot
            else:
                sig = float(rng.uniform(0.008, 0.03))
                Y0 = Xbest + sig * rng.standard_normal((n, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
                betas, mis = sched_fine
        except Exception:
            continue
        Y = lbfgs_anneal(Y0, betas, mis)
        v = _max_cos(Y)
        # quick clean
        for _ in range(3):
            if time.time() > DEADLINE - 28:
                break
            Y2, v2, ok = greedy_single(Y, 2000)
            if ok and v2 < v:
                Y, v = Y2, v2
            else:
                break
        if v < Vbest - 1e-12:
            Vbest, Xbest = v, Y.copy()
            # squeeze new basin a bit with light LP
            if time.time() < DEADLINE - 35:
                Xp, vp = lp_polish(Xbest, 2.5, 3e-4)
                if vp < Vbest:
                    Vbest, Xbest = vp, Xp
        # occasional LP on non-improving but close candidates to find hidden gain
        if it % 12 == 0 and time.time() < DEADLINE - 40:
            if v < Vbest + 2e-6:
                Xp, vp = lp_polish(Y, 1.5, 2e-4)
                if vp < Vbest - 1e-12:
                    Vbest, Xbest = vp, Xp.copy()

    # ---- final squeeze ----
    # high-beta anneal then LP
    if time.time() < DEADLINE - 20:
        Y = lbfgs_anneal(Xbest, [800.0, 2000.0, 4000.0], [120, 180, 250])
        v = _max_cos(Y)
        if v < Vbest:
            Vbest, Xbest = v, Y
    for _ in range(6):
        if time.time() > DEADLINE - 18:
            break
        Xbest2, v2, ok = greedy_single(Xbest, 5000)
        if ok and v2 < Vbest:
            Xbest, Vbest = Xbest2, v2
        else:
            break
    if time.time() < DEADLINE - 3:
        Xp, vp = lp_polish(Xbest, max(1.0, DEADLINE - 1.5 - time.time()), 4e-4)
        if vp <= Vbest:
            Vbest, Xbest = vp, Xp

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
