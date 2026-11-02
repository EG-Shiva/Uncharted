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
    BUDGET = 170.0
    DEADLINE = t0 + BUDGET
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]
    Jall = iu[1]

    # load start
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

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 120 or radius < 1e-14:
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
                bx = y.copy()
                radius = min(radius * 1.6, 0.02)
                if best < Vbest:
                    pass
            else:
                radius *= 0.5
        return bx, best

    def make_fun(beta):
        beta = float(beta)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(beta * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / beta
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def lbfgs_relax(Xinit, betas=(60.0, 160.0, 420.0), iters=(90, 90, 70)):
        Xf = Xinit.ravel().copy()
        bestX = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 22.0:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = res.x
            except Exception:
                break
            Xc = Xf.reshape(n, d)
            Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
            vc = _max_cos(Xc)
            if vc < bestV:
                bestV = vc
                bestX = Xc.copy()
        # return best-by-max, but continue from last Xf for annealing chain
        # choose to return bestX
        return bestX, bestV

    def greedy_fill(Xrest, k, Rnum=6000):
        # Xrest: (n-k, d), return k new points far from Xrest and each other
        R = rng.standard_normal((Rnum, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Dr = R @ Xrest.T
        rmax = Dr.max(axis=1)
        chosen = []
        chosen_idx = set()
        cur_to_chosen = np.full(Rnum, -2.0)
        for t in range(k):
            tot = np.maximum(rmax, cur_to_chosen)
            # mask chosen
            if chosen_idx:
                # set chosen to inf
                for ci in chosen_idx:
                    tot[ci] = np.inf
            bi = int(np.argmin(tot))
            chosen.append(R[bi].copy())
            chosen_idx.add(bi)
            # update cur_to_chosen
            dcc = R @ chosen[-1]
            cur_to_chosen = np.maximum(cur_to_chosen, dcc)
        return np.stack(chosen, axis=0)

    # quick exploit of current basin with L-BFGS (in case best.npy not fully squeezed for LSE)
    if time.time() < DEADLINE - 30:
        try:
            Xo, vo = lbfgs_relax(Xbest, betas=(80.0, 250.0), iters=(60, 60))
            if vo < Vbest - 1e-12:
                Xbest, Vbest = Xo, vo
        except Exception:
            pass

    it = 0
    # main cluster-ejection basin hopping
    while time.time() < DEADLINE - 22.0 and it < 300:
        it += 1
        # pick cluster size
        # vary k 3..10, occasionally 12-16 for big jump
        r = rng.random()
        if r < 0.7:
            k = int(rng.integers(3, 9))
        elif r < 0.9:
            k = int(rng.integers(8, 13))
        else:
            k = int(rng.integers(12, 17))
        k = max(2, min(k, 20))

        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        # seed among worst 25
        order = np.argsort(-rowmax)
        pool = order[:25]
        seed = int(pool[rng.integers(0, len(pool))])
        # neighbors by inner product
        nbrs = np.argsort(-G[seed])[:k]
        # ensure seed included
        clus = np.unique(nbrs)
        if len(clus) < k:
            # pad with worst
            extra = [o for o in order if o not in set(clus.tolist())]
            need = k - len(clus)
            clus = np.concatenate([clus, np.array(extra[:need], dtype=int)])
        clus = np.array(sorted(clus.tolist()), dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[clus] = False
        Xrest = Xbest[mask]
        try:
            NewPts = greedy_fill(Xrest, len(clus), Rnum=6000)
        except Exception:
            continue
        Y0 = Xbest.copy()
        Y0[clus] = NewPts
        # small jitter to break symmetry
        Y0[clus] += 1e-4 * rng.standard_normal(Y0[clus].shape)
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)

        # relax
        try:
            Yo, vo = lbfgs_relax(Y0, betas=(60.0, 160.0, 420.0), iters=(70, 70, 60))
        except Exception:
            continue
        if vo < Vbest - 1e-10:
            # squeeze with quick LP
            Yo2, vo2 = lp_polish(Yo, seconds=3.0, radius_init=4e-4)
            if vo2 < Vbest:
                Xbest, Vbest = Yo2, vo2
            else:
                Xbest, Vbest = Yo, vo
            continue
        # occasional global shake exploitation every 8 iters
        if it % 9 == 0 and time.time() < DEADLINE - 30:
            sig = 0.12 if (it % 18 == 0) else 0.05
            Yg = Xbest + sig * rng.standard_normal(Xbest.shape)
            Yg /= np.linalg.norm(Yg, axis=1, keepdims=True)
            try:
                Ygr, vgr = lbfgs_relax(Yg, betas=(60.0, 200.0), iters=(60, 60))
            except Exception:
                continue
            if vgr < Vbest - 1e-10:
                Ygr2, vgr2 = lp_polish(Ygr, seconds=3.0, radius_init=4e-4)
                if vgr2 < Vbest:
                    Xbest, Vbest = Ygr2, vgr2
                else:
                    Xbest, Vbest = Ygr, vgr

    # final squeeze
    if time.time() < DEADLINE - 1.0:
        remain = max(0.5, (DEADLINE - 1.0) - time.time())
        remain = min(remain, 18.0)
        try:
            Xbest, Vbest = lp_polish(Xbest, seconds=remain, radius_init=3e-4)
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
