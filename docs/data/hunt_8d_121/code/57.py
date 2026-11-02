import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return a / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SCREEN_DEAD = t0 + 90.0
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    # load references
    Xbest0 = None
    Vbest0 = 1e9
    Rref = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if fn == "record.npy":
                    Rref = a.copy()
                if v < Vbest0:
                    Vbest0 = v
                    Xbest0 = a.copy()
        except Exception:
            pass
    if Xbest0 is None:
        Xbest0 = _norm(rng.standard_normal((n, d)))
        Vbest0 = _max_cos(Xbest0)
    if Rref is None:
        Rref = Xbest0.copy()
    # record contact mask for Hamming distance (threshold 1e-3)
    try:
        Gr = Rref @ Rref.T
        mr = float(Gr[np.triu_indices(n, 1)].max())
        rec_mask = (Gr > mr - 1e-3)
        rec_mask = rec_mask[np.triu_indices(n, 1)]
    except Exception:
        rec_mask = None
        mr = Vbest0

    # ---------- build structured pools ----------
    pools = []
    # E8 roots 240
    try:
        E = []
        # type I: (±1,±1,0^6)/sqrt2
        for i in range(8):
            for j in range(i+1, 8):
                for si in (1.0, -1.0):
                    for sj in (1.0, -1.0):
                        v = np.zeros(8)
                        v[i] = si / np.sqrt(2.0)
                        v[j] = sj / np.sqrt(2.0)
                        E.append(v)
        # type II: (±0.5^8) even #minus
        for k in range(128):
            # enumerate via bits, take those with even parity
            bits = [(k >> b) & 1 for b in range(7)]
            parity = sum(bits) % 2
            # last bit chosen to make even number of -1
            # map 0->+0.5, 1->-0.5
            full = bits + [parity]  # ensures sum even
            v = np.array([ -0.5 if b else 0.5 for b in full ])
            E.append(v)
        E = np.array(E, dtype=float)  # 112+128=240
        E = _norm(E)
        pools.append(E)
    except Exception:
        pass
    # D8-type + random pool
    try:
        D = []
        for i in range(8):
            for s in (1.0, -1.0):
                v = np.zeros(8); v[i] = s
                D.append(v)
        D = np.array(D)
        R1 = rng.standard_normal((600, 8)); R1 = _norm(R1)
        # add E8 type I subset for diversity
        if len(pools) > 0:
            Dpool = np.vstack([D, pools[0][:112], R1])
        else:
            Dpool = np.vstack([D, R1])
        pools.append(_norm(Dpool))
    except Exception:
        pass
    # product pools (a u, b v), u,v in S^3
    try:
        # 24-cell in R4
        U24 = []
        for i in range(4):
            for s in (1.0, -1.0):
                v = np.zeros(4); v[i] = s
                U24.append(v)
        for a in range(16):
            v = np.array([0.5 if ((a >> k) & 1) == 0 else -0.5 for k in range(4)], dtype=float)
            U24.append(v)
        U24 = np.array(U24, dtype=float)
        U24 = _norm(U24)
        Ur = _norm(rng.standard_normal((32, 4)))
        Ubig = np.vstack([U24, Ur])  # 56
        Vbig = Ubig.copy()
        # extra random S3 codes
        for acoef in (0.5, 1.0/np.sqrt(2.0), 0.8):
            bcoef = float(np.sqrt(max(1e-12, 1 - acoef*acoef)))
            # full cartesian would be 56*56=3136, subsample to ~900
            prod = []
            for uu in Ubig:
                for vv in Vbig:
                    prod.append(np.concatenate([acoef*uu, bcoef*vv]))
            prod = np.array(prod)
            # subsample deterministic
            idx = rng.choice(prod.shape[0], size=min(900, prod.shape[0]), replace=False)
            pools.append(_norm(prod[idx]))
    except Exception:
        pass
    # random pool
    try:
        pools.append(_norm(rng.standard_normal((900, 8))))
    except Exception:
        pass
    if len(pools) == 0:
        pools.append(_norm(rng.standard_normal((900, 8))))

    # ---------- greedy max-min selection ----------
    def greedy_one(pool):
        M = pool.shape[0]
        s0 = int(rng.integers(0, M))
        sel = [s0]
        # max dot to selected
        md = pool @ pool[s0]  # (M,)
        md[s0] = np.inf
        # to avoid O(n^2) python, loop
        for _ in range(1, n):
            # pick min md among unselected: argmin works since selected are inf
            k = int(np.argmin(md))
            sel.append(k)
            md[k] = np.inf
            # update: md = max(md, pool@pool[k]) for unselected
            nd = pool @ pool[k]
            # only where nd > md and not inf
            # vectorised
            upd = nd > md
            # keep inf for selected
            md[upd] = nd[upd]
            # keep selected inf
            # (already)
        return sel

    def maxcos_of_sel(pool, sel):
        X = pool[np.array(sel, dtype=int)]
        return _max_cos(X), X

    # 200 random restarts distributed over pools
    total_restarts = 200
    per_pool = max(1, total_restarts // max(1, len(pools)))
    cand_X = []
    cand_V = []
    # cheap loop with time guard
    rest_done = 0
    for pi, pool in enumerate(pools):
        if time.time() > SCREEN_DEAD - 25:
            break
        R = per_pool if pi < len(pools)-1 else (total_restarts - rest_done)
        R = max(1, int(R))
        for _ in range(R):
            if time.time() > SCREEN_DEAD - 20:
                break
            try:
                sel = greedy_one(pool)
                # local swaps: try ~25 random swaps
                X = pool[np.array(sel, dtype=int)].copy()
                cur = _max_cos(X)
                # pool indices not selected
                mask = np.ones(pool.shape[0], dtype=bool)
                mask[np.array(sel, dtype=int)] = False
                unsel_idx = np.where(mask)[0]
                for _s in range(25):
                    si = int(rng.integers(0, n))
                    ui = int(unsel_idx[int(rng.integers(0, len(unsel_idx)))])
                    old = sel[si]
                    sel[si] = ui
                    Xt = pool[np.array(sel, dtype=int)]
                    nv = _max_cos(Xt)
                    if nv < cur:
                        cur = nv
                        X = Xt.copy()
                        mask = np.ones(pool.shape[0], dtype=bool)
                        mask[np.array(sel, dtype=int)] = False
                        unsel_idx = np.where(mask)[0]
                        if len(unsel_idx) == 0:
                            break
                    else:
                        sel[si] = old
                cand_X.append(X)
                cand_V.append(cur)
                rest_done += 1
            except Exception:
                continue
    if len(cand_X) == 0:
        # fallback single random
        cand_X = [_norm(rng.standard_normal((n, d)))]
        cand_V = [_max_cos(cand_X[0])]
    cand_V = np.array(cand_V)
    order = np.argsort(cand_V)
    # take at least 32 seeds: best 32
    K = min(32, len(order))
    # if more time, take up to 48 but at least 32
    if len(order) >= 40 and time.time() < SCREEN_DEAD - 40:
        K = min(40, len(order))
    seeds = [cand_X[order[k]].copy() for k in range(K)]
    # ensure at least 32 by duplicating with noise if needed
    while len(seeds) < 32:
        seeds.append(_norm(seeds[0] + 0.01*rng.standard_normal((n, d))))

    # ---------- batched vectorised Adam LogSumExp (no LP) ----------
    def adam_lse(X0, steps=1000, lr=0.012):
        X = X0.copy()
        m1 = np.zeros_like(X); m2 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps+1):
            if time.time() > SCREEN_DEAD:
                break
            beta = 50.0 * (20.0 ** ((t-1)/max(1, steps-1)))  # 50->1000
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[Iall, Jall]
            m0 = c.max()
            e = np.exp(beta*(c-m0))
            s = e.sum()
            if not np.isfinite(s) or s <= 0:
                break
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY*Y).sum(axis=1, keepdims=True)
            Gr = (GY - dot*Y) / nr
            # Adam
            m1 = b1*m1 + (1-b1)*Gr
            m2 = b2*m2 + (1-b2)*(Gr*Gr)
            c1 = 1 - b1**t; c2 = 1 - b2**t
            mh = m1/c1; vh = m2/c2
            X = X - lr*mh/(np.sqrt(vh)+eps)
            # renormalize to sphere
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn = np.maximum(nn, 1e-12)
            X = X/nn
        return _norm(X)

    screened = []
    screened_V = []
    for s in seeds:
        if time.time() > SCREEN_DEAD:
            break
        try:
            Xo = adam_lse(s, steps=1000, lr=0.012)
            v = _max_cos(Xo)
            screened.append(Xo)
            screened_V.append(v)
        except Exception:
            continue
    if len(screened) == 0:
        screened = seeds[:1]
        screened_V = [_max_cos(screened[0])]

    screened_V = np.array(screened_V, dtype=float)
    # ---------- filter <0.475 and Hamming >100 ----------
    def hamming_to_record(X):
        try:
            G = X @ X.T
            m = float(G[Iall, Jall].max() if hasattr(G, '__getitem__') else _max_cos(X))
            # G[Iall,Jall]
            cm = (G > m - 1e-3)[np.triu_indices(n, 1)]
            if rec_mask is None:
                return 10**9
            return int(np.sum(cm != rec_mask))
        except Exception:
            return 10**9
    promoted = []
    promoted_V = []
    for X, v in zip(screened, screened_V):
        if v < 0.475:
            h = hamming_to_record(X)
            if h > 100:
                promoted.append(X); promoted_V.append(v)
    # if none promoted, take best screened with h>100 or just best screened
    if len(promoted) == 0:
        # relax threshold to allow L-BFGS on best few
        idx = np.argsort(screened_V)
        for k in idx[:5]:
            h = hamming_to_record(screened[k])
            # accept even if >=0.475 but closest
            promoted.append(screened[k]); promoted_V.append(float(screened_V[k]))
            if len(promoted) >= 3:
                break
    promoted_V = np.array(promoted_V, dtype=float)
    best_prom = promoted[int(np.argmin(promoted_V))].copy() if len(promoted)>0 else screened[int(np.argmin(screened_V))].copy()
    best_prom_V = float(np.min(promoted_V)) if len(promoted)>0 else float(np.min(screened_V))

    # ---------- short L-BFGS anneal on promoted ----------
    def lse_fun_factory(beta):
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[Iall, Jall]
            m0 = c.max()
            e = np.exp(beta*(c-m0))
            s = e.sum()
            f = m0 + np.log(max(s, 1e-300))/beta
            w = e/max(s, 1e-300)
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY*Y).sum(axis=1, keepdims=True)
            GX = (GY - dot*Y)/nr
            g = GX.ravel()
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            return float(f), g
        return fun
    # anneal only best promoted (spec) but try up to 2 if time
    anneal_betas = [500.0, 2500.0, 12000.0]
    Xann = best_prom.copy()
    Vann = best_prom_V
    for beta in anneal_betas:
        if time.time() > DEADLINE - 30:
            break
        try:
            fun = lse_fun_factory(beta)
            res = minimize(fun, Xann.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": 80, "ftol": 1e-14, "gtol": 1e-8, "maxfun": 120})
            Xn = np.asarray(res.x, dtype=float).reshape(n, d)
            Xn = _norm(Xn)
            if np.all(np.isfinite(Xn)):
                Xann = Xn
                Vann = _max_cos(Xann)
        except Exception:
            break
    best_prom = Xann.copy()
    best_prom_V = Vann

    # ---------- LP polish only single best promoted seed <=20s ----------
    def lp_polish(x, seconds):
        x = _norm(x)
        best = _max_cos(x)
        radius = 1e-3
        start = time.time()
        iu0 = np.triu_indices(n, 1)
        it = 0
        while time.time() - start < seconds and radius > 1e-14 and it < 120:
            it += 1
            if time.time() > DEADLINE - 2:
                break
            G = x @ x.T
            cfull = G[iu0]
            m = cfull.max()
            thresh = max(20*radius, 1e-9)
            sel = cfull > m - thresh
            cnt = int(sel.sum())
            if cnt > 2500:
                idx = np.argpartition(-cfull, 2500)[:2500]
                mask = np.zeros_like(sel); mask[idx] = True
                sel = mask; cnt = 2500
            if cnt == 0:
                radius *= 0.5
                continue
            c = cfull[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = cnt
            nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p, 1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.05)
            else:
                radius *= 0.5
        return x
    try:
        rem_lp = min(20.0, max(0.0, DEADLINE - time.time() - 4.0))
        if rem_lp > 1.0:
            Xlp = lp_polish(best_prom, rem_lp)
            Vlp = _max_cos(Xlp)
            if Vlp < best_prom_V:
                best_prom, best_prom_V = Xlp, Vlp
    except Exception:
        pass

    # final choice vs lab best (ensure no regression)
    Xfinal = best_prom
    Vfinal = best_prom_V
    # lightly polish lab best if time remains and compare
    try:
        if time.time() < DEADLINE - 3:
            # quick LP on lab best with remaining time (small)
            rem = DEADLINE - time.time() - 1.5
            Xc = Xbest0.copy()
            Vc = Vbest0
            if rem > 2.0:
                Xc2 = lp_polish(Xc, min(8.0, rem))
                Vc2 = _max_cos(Xc2)
                if Vc2 < Vc:
                    Xc, Vc = Xc2, Vc2
            if Vc < Vfinal:
                Xfinal, Vfinal = Xc, Vc
    except Exception:
        pass
    Xfinal = _norm(Xfinal)
    return Xfinal
