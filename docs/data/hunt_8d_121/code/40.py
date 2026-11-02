import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    n, d = 121, 8
    iu_all = np.triu_indices(n, 1)
    Iall, Jall = iu_all[0], iu_all[1]
    NP = len(Iall)

    def _normed(a):
        a = np.asarray(a, dtype=float)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        return a / nr

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rs0 = np.random.RandomState(0)
        cands = [_normed(rs0.randn(n, d))]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))
    Xinit0 = Xbest.copy()
    Vinit0 = float(Vbest)

    THR = 2e-5
    def edge_mask(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        c = G[Iall, Jall]
        return (c > m - THR), m

    E0mask, m0 = edge_mask(Xinit0)

    def jaccard_dist(X):
        try:
            Em, _ = edge_mask(X)
            inter = np.logical_and(Em, E0mask).sum()
            union = np.logical_or(Em, E0mask).sum()
            if union == 0:
                return 0.0
            return 1.0 - float(inter) / float(union)
        except Exception:
            return 1.0

    # ---- spectral communities from incumbent contact graph ----
    def spectral_labels(A, k, rs):
        deg = A.sum(axis=1)
        di = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        S = (di[:, None] * A) * di[None, :]
        try:
            w, V = np.linalg.eigh(S)
        except Exception:
            return np.array([i % k for i in range(n)])
        idx = np.argsort(w)[::-1][:k]
        U = V[:, idx]
        rn = np.linalg.norm(U, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        U = U / rn
        cent = U[rs.choice(n, k, replace=False)].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(60):
            dd = ((U[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            nl = np.argmin(dd, axis=1)
            if np.array_equal(nl, lab) and _ > 0:
                lab = nl
                break
            lab = nl
            for c in range(k):
                pts = U[lab == c]
                if len(pts) > 0:
                    cent[c] = pts.mean(axis=0)
                else:
                    cent[c] = U[rs.randint(n)]
        return lab

    communities = []
    try:
        G0 = Xinit0 @ Xinit0.T
        np.fill_diagonal(G0, -2.0)
        mm0 = float(np.max(G0))
        A0 = np.zeros((n, n))
        cc = G0[Iall, Jall] > mm0 - THR
        A0[Iall[cc], Jall[cc]] = 1.0
        A0[Jall[cc], Iall[cc]] = 1.0
        seen = set()
        for kk in (6, 7, 8):
            for sd in (0, 1, 2):
                rs = np.random.RandomState(1000 + kk * 50 + sd)
                lab = spectral_labels(A0, kk, rs)
                for c in range(kk):
                    idx = np.where(lab == c)[0]
                    if 12 <= len(idx) <= 20:
                        key = frozenset(idx.tolist())
                        if key not in seen:
                            seen.add(key)
                            communities.append(idx.copy())
        # degrees for outlier info
        deg0 = A0.sum(axis=1)
    except Exception:
        communities = []
        deg0 = np.zeros(n)
    if len(communities) < 6:
        rsf = np.random.RandomState(7)
        while len(communities) < 12:
            sz = int(rsf.randint(12, 21))
            idx = rsf.choice(n, sz, replace=False)
            key = frozenset(idx.tolist())
            # allow duplicates here
            communities.append(np.sort(idx))

    # ---- helpers for optimisation ----
    def lse_all_refine(Xin, beta, maxiter):
        nn = Xin.shape[0]
        x0 = Xin.ravel().copy()
        iu = np.triu_indices(nn, 1)
        def fg(xf):
            X = xf.reshape(nn, d)
            ns = np.sqrt(np.sum(X * X, axis=1))
            ns = np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            c = G[iu]
            m = float(np.max(c))
            A = beta * (c - m)
            np.clip(A, -700, 0, out=A)
            E = np.exp(A)
            S = float(np.sum(E))
            if S <= 0:
                S = 1e-300
            F = m + np.log(S) / beta
            w = E / S
            W = np.zeros((nn, nn))
            W[iu] = w
            W.T[iu] = w
            GY = W.dot(Y)
            dot = np.sum(GY * Y, axis=1)
            GX = (GY - dot[:, None] * Y) / ns[:, None]
            return float(F), GX.ravel()
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8),
                                    "maxls": 20})
            Xo = res.x.reshape(nn, d)
        except Exception:
            Xo = Xin.copy()
        return _normed(Xo)

    def subset_opt(Rinit, Fmat, betas, maxiter_each):
        r = Rinit.shape[0]
        m = Fmat.shape[0]
        if r == 0:
            return Rinit.copy()
        Iar, Jar = np.triu_indices(r, 1)
        Xcur = Rinit.copy()
        for beta in betas:
            x0 = Xcur.ravel().copy()
            def fg(xf):
                X = xf.reshape(r, d)
                ns = np.sqrt(np.sum(X * X, axis=1))
                ns = np.maximum(ns, 1e-12)
                Y = X / ns[:, None]
                Grf = Y.dot(Fmat.T)
                Grr = Y.dot(Y.T)
                v1 = Grf.ravel()
                v2 = Grr[Iar, Jar] if r > 1 else np.zeros(0)
                v = np.concatenate([v1, v2]) if v2.size else v1
                mm = float(np.max(v))
                A = beta * (v - mm)
                np.clip(A, -700, 0, out=A)
                E = np.exp(A)
                S = float(np.sum(E))
                if S <= 0:
                    S = 1e-300
                Fv = mm + np.log(S) / beta
                w = E / S
                Wrf = w[:r * m].reshape(r, m)
                Wfull = np.zeros((r, r))
                if r > 1 and v2.size:
                    wrr = w[r * m:]
                    Wfull[Iar, Jar] = wrr
                    Wfull[Jar, Iar] = wrr
                GY = Wrf.dot(Fmat) + Wfull.dot(Y)
                dot = np.sum(GY * Y, axis=1)
                GX = (GY - dot[:, None] * Y) / ns[:, None]
                return float(Fv), GX.ravel()
            try:
                res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(maxiter_each), "ftol": 1e-20,
                                        "gtol": 1e-7, "maxfun": int(maxiter_each * 8),
                                        "maxls": 20})
                Xcur = res.x.reshape(r, d)
                Xcur = _normed(Xcur)
            except Exception:
                pass
            if time.time() > DEADLINE - 22:
                break
        return _normed(Xcur)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.0
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = c_all.max()
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2500:
                idx = np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
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
            y = _normed(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    survivors = []  # (v, X, jdist)

    def consider(X):
        nonlocal Xbest, Vbest
        try:
            v = _max_cos(X)
        except Exception:
            return
        if not np.isfinite(v):
            return
        Xn = _normed(X)
        if v < Vbest:
            Vbest = float(v)
            Xbest = Xn.copy()
        jd = jaccard_dist(Xn)
        if v < 0.470 and jd < 0.75:
            if len(survivors) < 30:
                survivors.append((float(v), Xn.copy(), float(jd)))
            else:
                # replace worst value
                survivors.sort(key=lambda z: z[0])
                if float(v) < survivors[-1][0]:
                    survivors[-1] = (float(v), Xn.copy(), float(jd))

    trial = 0
    # time-boxed community excision trials
    while time.time() < DEADLINE - 26:
        C = communities[trial % len(communities)]
        # current contact graph for outlier detection
        try:
            Gc = Xbest @ Xbest.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(np.max(Gc))
            Ac = (Gc > mc - THR).astype(float)
            np.fill_diagonal(Ac, 0.0)
            degc = Ac.sum(axis=1)
            Cset = set(C.tolist())
            # adjacent low-contact outliers
            neigh = []
            for j in range(n):
                if j in Cset:
                    continue
                # adjacent if connected to any in C
                if np.any(Ac[j, list(Cset)] > 0):
                    neigh.append(j)
            neigh = sorted(neigh, key=lambda j: degc[j])
            med = float(np.median(degc))
            extra = [j for j in neigh if degc[j] <= med][: max(0, min(3, 20 - len(Cset)))]
            Rset = list(Cset) + extra
            # enforce 12..20
            if len(Rset) < 12:
                # add lowest-degree non-members
                rest = [j for j in range(n) if j not in set(Rset)]
                rest = sorted(rest, key=lambda j: degc[j])
                need = 12 - len(Rset)
                Rset = Rset + rest[:need]
            if len(Rset) > 20:
                Rset = Rset[:20]
            Rset = np.array(sorted(Rset), dtype=int)
        except Exception:
            Rset = np.array(sorted(C.tolist()), dtype=int)
            extra = []
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[Rset] = False
        Frozen = Xbest[keep_mask].copy()
        r = int(n - Frozen.shape[0])
        rs = np.random.RandomState(20000 + trial * 131)
        # ---- generate >=50000 candidates ----
        NS = 50000
        try:
            Cand = rs.randn(NS, d)
            Cand /= np.maximum(np.linalg.norm(Cand, axis=1, keepdims=True), 1e-12)
            # greedy placement in largest holes with mutual separation
            Df = Cand @ Frozen.T
            fmax = Df.max(axis=1)
            placed = []
            # incremental placed-max
            pmax = np.full(NS, -2.0)
            order_sel = []
            avail = np.ones(NS, dtype=bool)
            for k in range(r):
                tot = np.maximum(fmax, pmax)
                # avoid re-picking same
                tot[~avail] = 2.0
                bi = int(np.argmin(tot))
                order_sel.append(bi)
                avail[bi] = False
                placed.append(Cand[bi].copy())
                if k < r - 1:
                    dp = Cand @ placed[-1]
                    pmax = np.maximum(pmax, dp)
            Rinit = np.stack(placed, axis=0)
        except Exception:
            Rinit = _normed(rs.randn(r, d))
        # ---- jointly optimize only replacements vs frozen ----
        try:
            Ropt = subset_opt(Rinit, Frozen, betas=[120.0, 320.0, 700.0], maxiter_each=60)
        except Exception:
            Ropt = Rinit
        # assemble full code
        try:
            Xfull = np.zeros((n, d))
            Xfull[keep_mask] = Frozen
            Xfull[~keep_mask] = Ropt
            Xfull = _normed(Xfull)
        except Exception:
            trial += 1
            continue
        # ---- short all-point anneal ----
        try:
            # tiny jitter then LSE anneal
            Xa = _normed(Xfull + rs.randn(n, d) * 0.002)
            Xa = lse_all_refine(Xa, beta=250.0, maxiter=70)
            consider(Xa)
            # also consider pre-anneal
            consider(Xfull)
            # extra anneal at higher beta if promising
            vnow = _max_cos(Xa)
            if vnow < Vbest + 0.005 and time.time() < DEADLINE - 30:
                Xb = lse_all_refine(Xa, beta=600.0, maxiter=50)
                consider(Xb)
        except Exception:
            try:
                consider(Xfull)
            except Exception:
                pass
        trial += 1
        if trial > 200:
            break

    # ---- LP-polish 2-3 most distinct survivors ----
    try:
        if survivors:
            # most distinct = largest jdist, but prefer good values
            survivors.sort(key=lambda z: z[2], reverse=True)
            # take top 3 distinct
            top = survivors[:3]
            # also ensure best overall gets a chance: include current best
            for (v, Xs, jd) in top:
                if time.time() > DEADLINE - 2:
                    break
                try:
                    Xp = lp_polish_timed(Xs, seconds=5.0)
                    consider(Xp)
                except Exception:
                    pass
        # final polish of global best briefly
        if time.time() < DEADLINE - 2:
            try:
                Xp = lp_polish_timed(Xbest, seconds=max(1.0, min(6.0, DEADLINE - time.time() - 1.0)))
                # only accept if better
                if _max_cos(Xp) < Vbest:
                    Xbest = _normed(Xp)
                    Vbest = float(_max_cos(Xbest))
            except Exception:
                pass
    except Exception:
        pass

    Xbest = _normed(Xbest)
    return Xbest
