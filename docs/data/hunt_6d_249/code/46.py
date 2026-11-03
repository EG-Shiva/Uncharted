import time
import numpy as np
from scipy.optimize import minimize
from collections import deque

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_DIV = t0 + 138.0
    T_END = t0 + 178.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    d = 6
    N249 = 249

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (N249, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(N249, d))
        cands.append(_norm(a))
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))

    baseX = bestX.copy()

    DELTA = 2e-4
    def edge_set(X):
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = np.where(c > m - DELTA)[0]
        s = set(((iu[0][sel].astype(np.int64) * n + iu[1][sel].astype(np.int64))).tolist())
        return s, m
    base_edges, _ = edge_set(baseX)

    def jaccard(X):
        if X.shape[0] != N249:
            return 0.0
        s, _ = edge_set(X)
        if not s and not base_edges:
            return 1.0
        inter = len(s.intersection(base_edges))
        union = len(s.union(base_edges))
        if union == 0:
            return 1.0
        return inter / union

    # ---------- single-point hole refinement ----------
    def refine_hole(Xref, y0, tau=0.008, steps=200, lr=0.12):
        y = y0.copy()
        nr = np.linalg.norm(y)
        if nr < 1e-12:
            y = rng.standard_normal(size=(d,))
            nr = np.linalg.norm(y)
        y /= nr
        for s in range(steps):
            if time.time() > T_DIV:
                break
            dd = Xref @ y
            gm = float(np.max(dd))
            e = np.exp((dd - gm) / tau)
            S = float(np.sum(e))
            if S <= 0 or not np.isfinite(S):
                break
            w = e / S
            gvec = w @ Xref
            gt = gvec - float(gvec @ y) * y
            st = lr * (0.97 ** (s / 20.0))
            y = y - st * gt
            nr = np.linalg.norm(y)
            if nr < 1e-12:
                y = rng.standard_normal(size=(d,))
                nr = np.linalg.norm(y)
            y /= nr
        return y

    def deepest_holes(Xref, npool=100000, nbest=20):
        # generate pool
        C = rng.standard_normal(size=(npool, d))
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        # batched max
        bs = 10000
        cmax = np.empty(npool, dtype=float)
        for a in range(0, npool, bs):
            b = min(a + bs, npool)
            D = C[a:b] @ Xref.T
            cmax[a:b] = np.max(D, axis=1)
        idx = np.argpartition(cmax, nbest)[:nbest]
        # refine best 20
        refined = []
        for k in idx:
            y = refine_hole(Xref, C[k])
            v = float(np.max(Xref @ y))
            refined.append((v, y))
        refined.sort(key=lambda t: t[0])
        return refined, C, cmax

    def grow_by(X, k_add):
        Xref = X.copy()
        for _ in range(k_add):
            if time.time() > T_DIV:
                break
            if Xref.shape[0] >= 272:
                break
            refined, _, _ = deepest_holes(Xref, npool=100000, nbest=20)
            # pick best refined; ensure mutual separation if multiple adds done sequentially it's fine
            ybest = refined[0][1]
            Xref = np.vstack([Xref, ybest[None, :]])
        return _norm(Xref)

    def prune_by(X, k_rem):
        Xc = X.copy()
        for _ in range(k_rem):
            n = Xc.shape[0]
            if n <= 5:
                break
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # severity: max + count near max
            mx = np.max(G, axis=1)
            cnt = np.sum(G > (m - 1e-3), axis=1)
            score = mx + 1e-4 * cnt
            rm = int(np.argmax(score))
            # tie-break random among top 3 to diversify
            order = np.argsort(-score)
            if rng.random() < 0.5:
                cand = order[:3]
                rm = int(cand[rng.integers(0, len(cand))])
            Xc = np.delete(Xc, rm, axis=0)
        return Xc

    def full_anneal(Xinit, taus=(0.02, 0.008, 0.003, 0.0012), iters=(45, 55, 60, 70), creg=0.08):
        n = Xinit.shape[0]
        I0, J0 = np.triu_indices(n, 1)
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_DIV:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            Xm = _norm(Xm)
            Xf = Xm.ravel().copy()
        Xo = Xf.reshape(n, d)
        return _norm(Xo)

    def pick_delete_cap(X, m):
        n = X.shape[0]
        # random direction or hub direction
        if rng.random() < 0.5:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mm = float(np.max(G))
            deg = np.sum(G > (mm - 5e-4), axis=1)
            hub = int(np.argmax(deg))
            u = X[hub] + 0.35 * rng.standard_normal(size=(d,))
        else:
            u = rng.standard_normal(size=(d,))
        u /= max(np.linalg.norm(u), 1e-12)
        dots = X @ u
        idx = np.argsort(-dots)[:m]
        return np.asarray(idx, dtype=int)

    # archive of 249 codes + work pool of varied N
    arch249 = [(bestVal, bestX.copy())]
    work = []  # list of (val, X) with varied N

    # init varied-N seeds: underfull 235-248, overfull 255-270
    try:
        # underfull by pruning best
        for Nt in (240, 235, 245):
            Xp = prune_by(bestX, N249 - Nt)
            work.append((_max_cos(Xp), Xp))
        # overfull by growing best
        Xg = grow_by(bestX, 11)  # 260
        work.append((_max_cos(Xg), Xg))
        Xg2 = grow_by(bestX, 6)  # 255
        work.append((_max_cos(Xg2), Xg2))
    except Exception:
        pass
    # light jam of seeds if time
    for i in range(len(work)):
        if time.time() > T_DIV - 60:
            break
        try:
            v, X = work[i]
            Xj = full_anneal(X)
            work[i] = (_max_cos(Xj), Xj)
        except Exception:
            pass

    ACCEPT = 0.715
    it = 0
    while time.time() < T_DIV:
        it += 1
        # choose source
        r = rng.random()
        if r < 0.30 and arch249:
            # best-ish 249
            arch249.sort(key=lambda t: t[0])
            src = arch249[0][1].copy() if rng.random() < 0.7 else arch249[rng.integers(0, len(arch249))][1].copy()
        elif r < 0.55 and arch249:
            src = arch249[rng.integers(0, len(arch249))][1].copy()
        elif work:
            src = work[rng.integers(0, len(work))][1].copy()
        else:
            src = bestX.copy()
        nsrc = src.shape[0]
        op = rng.random()
        try:
            if nsrc == N249:
                if op < 0.30:
                    # large delete-refill 40-65
                    m = int(rng.integers(40, 66))
                    didx = pick_delete_cap(src, m)
                    keep = np.ones(nsrc, dtype=bool); keep[didx] = False
                    Xkeep = src[keep]
                    Xgrown = grow_by(Xkeep, m)
                    Xnew = full_anneal(Xgrown)
                elif op < 0.50:
                    # 1-for-2 : remove 1 worst, add 2
                    G = src @ src.T; np.fill_diagonal(G, -2.0)
                    mx = np.max(G, axis=1)
                    rm = int(np.argmax(mx))
                    Xt = np.delete(src, rm, axis=0)
                    Xt = grow_by(Xt, 2)  # now 250
                    # prune to 249 by removing worst
                    Xt = prune_by(Xt, 1)
                    Xnew = full_anneal(Xt)
                elif op < 0.70:
                    # 2-for-1 : remove 2, add 1
                    G = src @ src.T; np.fill_diagonal(G, -2.0)
                    mx = np.max(G, axis=1)
                    order = np.argsort(-mx)[:4]
                    rms = rng.choice(order, size=2, replace=False)
                    Xt = np.delete(src, sorted(rms), axis=0)  # 247
                    Xt = grow_by(Xt, 2)  # 249
                    Xnew = full_anneal(Xt)
                elif op < 0.85:
                    # go underfull then regrow (cardinality homotopy down)
                    Nt = int(rng.integers(235, 249))
                    Xt = prune_by(src, N249 - Nt)
                    Xt = grow_by(Xt, N249 - Nt)
                    Xnew = full_anneal(Xt)
                else:
                    # go overfull then prune
                    k = int(rng.integers(6, 18))
                    Xt = grow_by(src, k)  # 255-266
                    Xnew_tmp = full_anneal(Xt)
                    Xnew = prune_by(Xnew_tmp, k)
                    Xnew = full_anneal(Xnew)
            elif nsrc < N249:
                # grow toward 249 or beyond
                if op < 0.6:
                    need = N249 - nsrc
                    Xt = grow_by(src, need)
                    Xnew = full_anneal(Xt)
                else:
                    # 1-for-2 expansion
                    Xt = grow_by(src, 2)
                    # if still <249 keep as varied-N, jam lightly
                    if Xt.shape[0] != N249:
                        Xnew = full_anneal(Xt)
                    else:
                        Xnew = full_anneal(Xt)
            else:  # nsrc > N249
                if op < 0.6:
                    need = nsrc - N249
                    Xt = prune_by(src, need)
                    Xnew = full_anneal(Xt)
                else:
                    # prune partially then jam as overfull survivor
                    k = int(rng.integers(1, max(2, nsrc - N249 + 1)))
                    Xt = prune_by(src, k)
                    Xnew = full_anneal(Xt)
        except Exception:
            continue
        try:
            nnew = Xnew.shape[0]
            vnew = _max_cos(Xnew)
        except Exception:
            continue
        if not np.isfinite(vnew):
            continue
        if nnew == N249:
            if vnew < bestVal - 1e-12:
                bestVal = vnew
                bestX = Xnew.copy()
                arch249.append((vnew, Xnew.copy()))
                arch249 = sorted(arch249, key=lambda t: t[0])[:10]
                base_edges, _ = edge_set(bestX)
            else:
                if vnew < ACCEPT:
                    try:
                        jc = jaccard(Xnew)
                    except Exception:
                        jc = 1.0
                    if jc < 0.72 and vnew < bestVal + 0.011:
                        arch249.append((vnew, Xnew.copy()))
                        arch249 = sorted(arch249, key=lambda t: t[0])[:10]
                    # also allow near-best without diversity occasionally
                    elif vnew < bestVal + 2e-4 and len(arch249) < 10:
                        arch249.append((vnew, Xnew.copy()))
                        arch249 = sorted(arch249, key=lambda t: t[0])[:10]
            # also feed work: perturb to varied N occasionally
            if rng.random() < 0.08 and time.time() < T_DIV - 20:
                try:
                    if rng.random() < 0.5:
                        Xv = prune_by(Xnew, int(rng.integers(2, 10)))
                    else:
                        Xv = grow_by(Xnew, int(rng.integers(2, 10)))
                    work.append((_max_cos(Xv), Xv))
                    if len(work) > 10:
                        work = sorted(work, key=lambda t: t[0])[:10]
                except Exception:
                    pass
        else:
            # varied-N survivor: keep if promising (allow worse absolute since smaller N easier)
            # normalize expectation: keep diverse N values
            if vnew < 0.72:
                work.append((vnew, Xnew.copy()))
                # keep at most 10, prefer diversity in N and value
                # deduplicate by N: keep best per N range
                work = sorted(work, key=lambda t: t[0])[:12]
                if len(work) > 10:
                    # drop random non-best to maintain diversity
                    del work[rng.integers(1, len(work))]

    # ---------- final re-jam diverse 249 survivors + LP polish ----------
    def lp_jam(Xinit, time_limit=8.0, slack=2e-3):
        try:
            from scipy.optimize import linprog as _lp
            from scipy.sparse import coo_matrix as _coo
        except Exception:
            return Xinit, _max_cos(Xinit)
        n = Xinit.shape[0]; k = 5; Nvar = n * k + 1
        R0 = np.random.RandomState(0).standard_normal(size=(6, 5))
        X = Xinit.copy()
        cur = _max_cos(X)
        Iall, Jall = np.triu_indices(n, 1)
        obj = np.zeros(Nvar); obj[-1] = 1.0
        radius = 0.008
        deadline = time.time() + time_limit
        it2 = 0
        while it2 < 60 and time.time() < deadline and time.time() < T_END - 2:
            it2 += 1
            if radius < 1e-12:
                break
            Gf = X @ X.T
            cf = Gf[Iall, Jall]
            m = float(np.max(cf))
            sel = cf > (m - slack)
            p = int(np.sum(sel))
            if p < 20:
                sel = cf > (m - 1e-6); p = int(np.sum(sel))
            if p > 5000:
                idx = np.argpartition(cf, -5000)[-5000:]
                sel = np.zeros_like(sel, dtype=bool); sel[idx] = True; p = 5000
            I = Iall[sel]; J = Jall[sel]; cs = cf[sel]
            # tangent bases
            B = np.empty((n, 6, 5))
            for i in range(n):
                xi = X[i]
                Rp = R0 - np.outer(xi, (xi @ R0))
                Q, _ = np.linalg.qr(Rp)
                B[i] = Q
            BI = B[I]; BJ = B[J]; XJ = X[J]; XI = X[I]
            cI = np.einsum('pkm,pk->pm', BI, XJ)
            cJ = np.einsum('pkm,pk->pm', BJ, XI)
            rows = np.repeat(np.arange(p), 2 * k + 1)
            colI = (I[:, None] * k + np.arange(k)).ravel()
            colJ = (J[:, None] * k + np.arange(k)).ravel()
            colT = np.full(p, Nvar - 1)
            cols = np.concatenate([colI.reshape(p, k), colJ.reshape(p, k), colT[:, None]], axis=1).ravel()
            data = np.concatenate([cI, cJ, -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = _coo((data, (rows, cols)), shape=(p, Nvar)).tocsr()
                b_ub = -cs
                bounds = [(-radius, radius)] * (n * k) + [(None, None)]
                res = _lp(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5; continue
            if res.status != 0:
                radius *= 0.5; continue
            Av = res.x[:-1].reshape(n, k)
            DX = np.einsum('nkm,nm->nk', B, Av)
            Y = X + DX
            Y = _norm(Y)
            v = _max_cos(Y)
            if v < cur - 1e-15:
                X = Y; cur = v
                if n == N249 and v < bestVal:
                    bestVal_local = v  # noqa
                radius = min(radius * 1.4, 0.012)
            else:
                radius *= 0.5
        return X, cur

    # final jam: take most diverse + best survivors
    arch249 = sorted(arch249, key=lambda t: t[0])
    # add current best if missing
    finals = []
    seen = set()
    for v, X in arch249[:6]:
        finals.append(X.copy())
    # ensure bestX included
    finals.append(bestX.copy())
    # quick full anneal + lp for top 3 within remaining time
    bestF = bestX.copy(); bestFv = _max_cos(bestF)
    for Xc in finals[:4]:
        if time.time() > T_END - 6:
            break
        try:
            # diverse re-jam: small random tangent kick then anneal
            if _max_cos(Xc) > bestFv + 1e-9:
                Xn = Xc + 1e-3 * rng.standard_normal(size=Xc.shape)
                Xc = _norm(Xn)
                Xc = full_anneal(Xc, taus=(0.008, 0.003, 0.001), iters=(40, 50, 60))
            Xlp, vlp = lp_jam(Xc, time_limit=5.0)
            if vlp < bestFv:
                bestFv = vlp; bestF = Xlp.copy()
        except Exception:
            continue
    bestF = _norm(bestF)
    # tiny shrink not needed (unit rows already); ensure finite
    bestF[~np.isfinite(bestF)] = 0.0
    # renormalize zero rows if any
    nr = np.linalg.norm(bestF, axis=1, keepdims=True)
    bad = (nr[:, 0] < 1e-9)
    if np.any(bad):
        bestF[bad] = _norm(rng.standard_normal(size=(int(np.sum(bad)), d)))
    bestF = _norm(bestF)
    return bestF
