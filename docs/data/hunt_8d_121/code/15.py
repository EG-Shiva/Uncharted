import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _lp_polish(x, deadline, r0=4e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 2e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.sort(c_all)[-2500]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3000:
                idxp = np.argpartition(-c_all, 3000)[:3000]
                mk = np.zeros_like(sel)
                mk[idxp] = True
                sel = mk
                cnt = 3000
        c = c_all[sel]
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8

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

    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    # ---------- full L-BFGS LogSumExp ----------
    def make_full_fun(beta):
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

    def full_anneal(X, betas, iters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 8.0:
                break
            fun = make_full_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = res.x
            except Exception:
                break
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # ---------- subset (k-point) L-BFGS LogSumExp, others fixed ----------
    def subset_opt(Xbase, idx, beta_sched=(150, 600, 2000), it_sched=(60, 60, 80), x0_sub=None):
        idx = np.asarray(idx, dtype=int)
        k = len(idx)
        mask = np.ones(n, dtype=bool); mask[idx] = False
        Yr = Xbase[mask]
        nr_ = Yr.shape[0]
        if x0_sub is None:
            Ys0 = Xbase[idx].copy()
        else:
            Ys0 = x0_sub.copy()
        flat0 = Ys0.ravel().copy()

        def fun(flat):
            Xs = flat.reshape(k, d)
            nrs = np.linalg.norm(Xs, axis=1, keepdims=True)
            nrs = np.maximum(nrs, 1e-12)
            Ys = Xs / nrs
            C1 = Ys @ Yr.T
            # upper part of Ys@Ys.T
            G2 = Ys @ Ys.T
            iu2 = np.triu_indices(k, 1)
            if k > 1:
                c2 = G2[iu2]
                c = np.concatenate([C1.ravel(), c2])
            else:
                c = C1.ravel()
                iu2 = None
            m = float(c.max())
            # use per-beta? we close over beta via attribute
            beta = fun.beta
            e = np.exp(beta * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / beta
            w = e / s
            w1 = w[:k * nr_].reshape(k, nr_)
            grad = w1 @ Yr
            if k > 1:
                w2 = w[k * nr_:]
                Ws = np.zeros((k, k))
                Ws[iu2[0], iu2[1]] = w2
                Ws[iu2[1], iu2[0]] = w2
                grad = grad + Ws @ Ys
            dot = (grad * Ys).sum(axis=1, keepdims=True)
            Gx = (grad - dot * Ys) / nrs
            return float(f), Gx.ravel()
        flat = flat0
        for beta, mi in zip(beta_sched, it_sched):
            fun.beta = float(beta)
            try:
                res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 6)})
                flat = res.x
            except Exception:
                break
        Ys = flat.reshape(k, d)
        Ys /= np.linalg.norm(Ys, axis=1, keepdims=True)
        return Ys

    def try_single_reinsert(X, V, idx, ntrials=4000, do_refine=True):
        others = np.delete(X, idx, axis=0)
        C = rng.standard_normal((ntrials, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ others.T
        cm = D.max(axis=1)
        order = np.argsort(cm)
        best_cm = float(cm[order[0]])
        if best_cm >= V - 1e-12 and not do_refine:
            return None, V
        # refine top few
        bestY = None; bestv = np.inf
        nref = 4 if do_refine else 1
        for t in range(nref):
            cand0 = C[order[t]]
            if do_refine:
                try:
                    Ys = subset_opt(X, [idx], beta_sched=(120, 500, 2500), it_sched=(50, 50, 70), x0_sub=cand0[None, :])
                    y = Ys[0]
                except Exception:
                    y = cand0
            else:
                y = cand0
            v = float((others @ y).max())
            if v < bestv:
                bestv = v; bestY = y
            if bestv < V - 1e-9:
                break
        if bestY is not None and bestv < V - 1e-12:
            Y2 = X.copy(); Y2[idx] = bestY / np.linalg.norm(bestY)
            nv = _max_cos(Y2)
            if nv < V - 1e-12:
                return Y2, nv
        return None, V

    def try_k_reinsert(X, V, idxs, ntrials_pair=2500):
        k = len(idxs)
        idxs = np.asarray(idxs)
        # random search for init
        best_init = None; best_c = np.inf
        for _ in range(3):
            R = rng.standard_normal((min(ntrials_pair, 800), k, d))
            Rn = np.linalg.norm(R, axis=2, keepdims=True); R = R / np.maximum(Rn, 1e-12)
            mask = np.ones(n, dtype=bool); mask[idxs] = False
            Yr = X[mask]
            # evaluate max affected
            # C1: R vs Yr, C2: R vs R
            # vectorised over trials missing -> loop in chunks
            for b in range(R.shape[0]):
                Ys = R[b]
                c1 = (Ys @ Yr.T).max()
                if k > 1:
                    c2 = float((Ys @ Ys.T - 2*np.eye(k)).max())
                    c = c1 if c1 > c2 else c2
                else:
                    c = c1
                if c < best_c:
                    best_c = c; best_init = Ys.copy()
            if best_c < V - 1e-9:
                break
        if best_init is None:
            best_init = X[idxs].copy()
        try:
            Ys_opt = subset_opt(X, idxs, beta_sched=(120, 500, 1800), it_sched=(60, 60, 90), x0_sub=best_init)
        except Exception:
            Ys_opt = best_init
        Y2 = X.copy(); Y2[idxs] = Ys_opt / np.linalg.norm(Ys_opt, axis=1, keepdims=True)
        nv = _max_cos(Y2)
        if nv < V - 1e-12:
            return Y2, nv
        return None, V

    # ---------- Phase A: exact single-point sweeps ----------
    for sweep in range(4):
        if time.time() > DEADLINE - 110:
            break
        rm = _rowmax(Xbest)
        order = np.argsort(-rm)
        improved = False
        for idx in order[:18]:
            if time.time() > DEADLINE - 105:
                break
            Y2, nv = try_single_reinsert(Xbest, Vbest, int(idx), ntrials=4000, do_refine=True)
            if Y2 is not None:
                # quick LP squeeze
                Y2 = _lp_polish(Y2, min(DEADLINE - 5, time.time() + 6), r0=3e-4)
                nv = _max_cos(Y2)
                if nv < Vbest:
                    Xbest, Vbest = Y2, nv
                    improved = True
                    break
        if not improved:
            break

    # ---------- Phase B: pair / triple re-insertion on worst contacts ----------
    if time.time() < DEADLINE - 95:
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        # worst pairs
        flat_idx = np.argsort(-G.reshape(-1))[:8]
        tried = set()
        for f in flat_idx:
            if time.time() > DEADLINE - 90:
                break
            a, b = divmod(int(f), n)
            if a >= b:
                continue
            if (a, b) in tried:
                continue
            tried.add((a, b))
            Y2, nv = try_k_reinsert(Xbest, Vbest, [a, b])
            if Y2 is not None:
                Y2 = _lp_polish(Y2, min(DEADLINE - 5, time.time() + 6), r0=3e-4)
                nv = _max_cos(Y2)
                if nv < Vbest:
                    Xbest, Vbest = Y2, nv
                    break
        # triple around worst point
        if time.time() < DEADLINE - 85:
            rm = _rowmax(Xbest)
            w0 = int(np.argmax(rm))
            G2 = Xbest @ Xbest.T
            np.fill_diagonal(G2, -2.0)
            neigh = np.argsort(-G2[w0])[:3]
            idxs = [w0] + [int(x) for x in neigh[:2]]
            Y2, nv = try_k_reinsert(Xbest, Vbest, idxs)
            if Y2 is not None:
                Y2 = _lp_polish(Y2, min(DEADLINE - 5, time.time() + 5), r0=3e-4)
                nv = _max_cos(Y2)
                if nv < Vbest:
                    Xbest, Vbest = Y2, nv

    # ---------- Phase C: kick + full anneal basin hopping ----------
    it = 0
    while time.time() < DEADLINE - 22:
        it += 1
        rm = _rowmax(Xbest)
        order = np.argsort(-rm)
        # choose kick size
        if it % 4 == 1:
            k = 2
            sel = order[:6]
            idxs = rng.choice(sel, size=k, replace=False)
            Xk = Xbest.copy()
            for ix in idxs:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                Xk[int(ix)] = v
            betas = (60, 180, 500, 1200); iters = (60, 60, 90, 120)
        elif it % 4 == 2:
            k = 3
            sel = order[:12]
            idxs = rng.choice(sel, size=k, replace=False)
            Xk = Xbest.copy()
            for ix in idxs:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                Xk[int(ix)] = v
            betas = (50, 150, 450, 1000); iters = (60, 60, 90, 110)
        elif it % 4 == 3:
            # gaussian shake of all, stronger on worst
            sig = 0.02 if (it % 8 == 3) else 0.06
            Xk = Xbest + sig * rng.standard_normal((n, d))
            Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
            betas = (80, 250, 800); iters = (70, 80, 110)
        else:
            k = 4
            sel = order[:20]
            idxs = rng.choice(sel, size=k, replace=False)
            Xk = Xbest.copy()
            for ix in idxs:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
                Xk[int(ix)] = v
            betas = (50, 150, 400, 1000); iters = (50, 50, 80, 100)
        Xa = full_anneal(Xk, betas, iters)
        va = _max_cos(Xa)
        if va < Vbest - 2e-9:
            Xa2 = _lp_polish(Xa, min(DEADLINE - 5, time.time() + 7), r0=3e-4)
            va2 = _max_cos(Xa2)
            if va2 < va:
                Xa, va = Xa2, va2
            if va < Vbest:
                Xbest, Vbest = Xa, va
        # opportunistic single reinsert on the annealed result even if not better
        if va < Vbest + 0.004 and time.time() < DEADLINE - 30:
            # try one greedy fix on annealed config then re-evaluate
            rm2 = _rowmax(Xa)
            w = int(np.argmax(rm2))
            Y2, nv = try_single_reinsert(Xa, va, w, ntrials=2500, do_refine=True)
            if Y2 is not None and nv < Vbest:
                Y2 = _lp_polish(Y2, min(DEADLINE - 5, time.time() + 5), r0=2e-4)
                nv = _max_cos(Y2)
                if nv < Vbest:
                    Xbest, Vbest = Y2, nv

    # ---------- final LP polish ----------
    if time.time() < DEADLINE - 3:
        Xbest = _lp_polish(Xbest, DEADLINE - 1.0, r0=4e-4)
        Vbest = _max_cos(Xbest)

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
