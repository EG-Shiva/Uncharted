import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, cap_p=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            order = np.argsort(-csel)[:cap_p]
            I, J = I[order], J[order]
            csel = csel[order]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _single_fg_factory(core, beta):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = core @ y
        m = float(np.max(s))
        e = np.exp(beta * (s - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = core.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    return fg

def _refine_single(y0, core, beta, maxiter):
    fg = _single_fg_factory(core, beta)
    try:
        r = minimize(fg, y0.copy(), jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        return y0 / (np.linalg.norm(y0) + 1e-300)

def _joint_fg_factory(core, k, beta):
    mcore = core.shape[0]
    iu_ff = np.triu_indices(k, 1) if k > 1 else None
    def fg(yf):
        Y = yf.reshape(k, -1)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Xn = Y / nr
        Gfc = Xn @ core.T  # k x mc
        if k > 1:
            Gff_full = Xn @ Xn.T
            gff = Gff_full[iu_ff]
            m = float(max(np.max(Gfc), np.max(gff)))
            efc = np.exp(beta * (Gfc - m))
            eff = np.exp(beta * (gff - m))
            S = float(np.sum(efc) + np.sum(eff)) + 1e-300
            F = m + np.log(S) / beta
            wfc = efc / S
            wff = eff / S
            Wsym = np.zeros((k, k))
            Wsym[iu_ff] = wff
            Wsym = Wsym + Wsym.T
            GX = wfc @ core + Wsym @ Xn
        else:
            m = float(np.max(Gfc))
            efc = np.exp(beta * (Gfc - m))
            S = float(np.sum(efc)) + 1e-300
            F = m + np.log(S) / beta
            wfc = efc / S
            GX = wfc @ core
        dot = np.sum(GX * Xn, axis=1, keepdims=True)
        GY = (GX - dot * Xn) / nr
        return F, GY.ravel()
    return fg

def _joint_opt(core, Y0, betas=(100.0, 300.0, 600.0), maxiter=180):
    y = Y0.ravel().copy()
    for beta in betas:
        fg = _joint_fg_factory(core, Y0.shape[0], beta)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(Y0.shape)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def solve():
    t_start = time.time()
    T_END = t_start + 170.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < best_score and np.all(np.isfinite(a)):
                    best_score = s
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)

    KS = [3, 4, 5, 6, 8]
    explore_end = T_END - 34.0
    trial = 0
    # quick initial squeeze to establish incumbent
    # main multi-point surgery loop
    while time.time() < explore_end:
        trial += 1
        if time.time() > T_END - 40.0:
            break
        # contact load ranking with beta=300
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # sum exp(300*(g-m)) per point
        try:
            W = np.exp(300.0 * (G - m))
        except Exception:
            W = np.ones_like(G)
        np.fill_diagonal(W, 0.0)
        loads = np.sum(W, axis=1)
        order = np.argsort(-loads)
        k = KS[int(rng.integers(0, len(KS)))]
        # choose deletion set: random subset of high-load vs cluster neighbours
        if rng.random() < 0.5:
            # random subset biased to high load: pool top 20
            poolsz = min(20, n)
            pool = order[:poolsz].copy()
            # weighted sample by loads
            w = loads[pool].astype(float)
            w = w - np.min(w) + 0.1 * (np.max(w) - np.min(w) + 1e-12)
            w = w / np.sum(w)
            if k >= len(pool):
                del_idx = pool.copy()
            else:
                del_idx = rng.choice(pool, size=k, replace=False, p=w)
        else:
            # point plus its near-contact neighbours
            top10 = order[:min(10, n)]
            w10 = loads[top10].astype(float)
            w10 = np.maximum(w10, 1e-12)
            w10 = w10 / np.sum(w10)
            hub = int(rng.choice(top10, p=w10))
            neigh = np.argsort(-G[hub])
            neigh = neigh[neigh != hub]
            del_idx = [hub] + neigh[:k-1].tolist()
            del_idx = np.array(del_idx, dtype=int)
        del_idx = np.unique(del_idx.astype(int))
        # if duplicate collapse, pad from high-load pool
        if len(del_idx) < k:
            for cand in order:
                if cand not in del_idx:
                    del_idx = np.append(del_idx, cand)
                if len(del_idx) >= k:
                    break
        del_idx = del_idx[:k]
        mask = np.ones(n, dtype=bool)
        mask[del_idx] = False
        core = bestX[mask]  # (n-k,d)

        # dense hole search: 50000 candidates scored against core
        ncand = 50000
        cand = rng.standard_normal((ncand, d))
        cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-300)
        scores = np.max(cand @ core.T, axis=1)
        # top 20
        part = np.argpartition(scores, 20)[:20]
        top20 = cand[part]
        # refine top 20 against frozen core (single-point LSE beta 300)
        refined = np.zeros_like(top20)
        for ti in range(20):
            # small time guard
            if time.time() > explore_end:
                break
            refined[ti] = _refine_single(top20[ti], core, 300.0, 50)
        refined /= (np.linalg.norm(refined, axis=1, keepdims=True) + 1e-300)
        # greedily pick k diverse inits from refined pool scored vs core + each other
        chosen = []
        # precompute refined vs core max
        r_vs_core = np.max(refined @ core.T, axis=1)
        remaining = list(range(20))
        for p in range(k):
            best_c = -1
            best_v = 1e9
            for c in remaining:
                v = float(r_vs_core[c])
                if chosen:
                    ch = refined[chosen] @ refined[c]
                    v = max(v, float(np.max(ch)))
                # tiny tie-breaker prefer lower core score
                if v < best_v:
                    best_v = v
                    best_c = c
            if best_c < 0:
                break
            chosen.append(best_c)
            remaining.remove(best_c)
        if len(chosen) < k:
            # fill with greedy from big pool if needed
            need = k - len(chosen)
            # pick best remaining from cand pool diverse
            Y0_extra = []
            placed = refined[chosen] if chosen else np.zeros((0, d))
            for _ in range(need):
                # sample a chunk to keep cheap: use scores order
                # evaluate a subset of 5000 best candidates
                idx_sorted = np.argsort(scores)[:5000]
                bv = 1e9
                bc = None
                # subsample 2000 for speed
                sub = rng.choice(idx_sorted, size=min(2000, len(idx_sorted)), replace=False)
                for c in sub:
                    v = float(scores[c])
                    if len(placed) > 0:
                        v = max(v, float(np.max(placed @ cand[c])))
                    if len(Y0_extra) > 0:
                        v = max(v, float(np.max(np.array(Y0_extra) @ cand[c])))
                    if v < bv:
                        bv = v
                        bc = c
                Y0_extra.append(cand[bc].copy())
                placed = np.vstack([placed, cand[bc][None, :]]) if len(placed) > 0 else cand[bc][None, :]
            Y0 = np.vstack([refined[chosen], np.array(Y0_extra)]) if chosen else np.array(Y0_extra)
        else:
            Y0 = refined[chosen].copy()
        # small perturbation for diversity across trials
        Y0 = Y0 + 0.002 * rng.standard_normal(Y0.shape)
        Y0 /= (np.linalg.norm(Y0, axis=1, keepdims=True) + 1e-300)
        # jointly re-optimise all k deleted points vs frozen core + each other, beta 100->600
        Yopt = _joint_opt(core, Y0, betas=(100.0, 300.0, 600.0), maxiter=160)
        Xtrial = bestX.copy()
        Xtrial[del_idx] = Yopt
        Xtrial /= (np.linalg.norm(Xtrial, axis=1, keepdims=True) + 1e-300)
        s_trial = _max_cos(Xtrial)
        # accept only if subsequent LP polish beats incumbent: polish when promising
        # threshold: within 0.003 of best (allows squeeze to win)
        if s_trial < best_score + 0.003:
            # 5-10 s LP polish (use 5s, 7s if very promising)
            lp_budget = 5.0 if s_trial >= best_score else 7.0
            if time.time() + lp_budget > T_END - 28.0:
                lp_budget = max(1.5, T_END - 28.0 - time.time())
            Xpol = _lp_polish(Xtrial, time.time() + lp_budget)
            s_pol = _max_cos(Xpol)
            if np.all(np.isfinite(Xpol)) and s_pol < best_score - 1e-12:
                best_score = s_pol
                bestX = Xpol.copy()
        if trial > 300:
            break

    # final long LP squeeze on incumbent
    remain = T_END - time.time()
    if remain > 6.0:
        lp_time = min(28.0, remain - 2.0)
        bestX = _lp_polish(bestX, time.time() + lp_time)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
