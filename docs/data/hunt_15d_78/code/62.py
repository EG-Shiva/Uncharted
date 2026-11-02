import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _lp_polish(X, deadline, r_init=0.004, r_max=0.03, cap=2500, max_it=60):
    X = _norm(np.asarray(X, dtype=np.float64))
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.6 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def _adam_smooth(X, iters, beta, lr, b1=0.9, b2=0.999, eps=1e-8):
    X = _norm(X.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    t = 0
    for it in range(iters):
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        S = float(e.sum())
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Grad = Gr - dot * X
        t += 1
        m1 = b1 * m1 + (1 - b1) * Grad
        m2 = b2 * m2 + (1 - b2) * (Grad * Grad)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        X = X - step
        X = _norm(X)
    return X

def _refine_candidates(Cand, Xfix, beta=500.0, iters=250, lr=0.05):
    C = _norm(Cand.copy())
    k = C.shape[0]
    m1 = np.zeros_like(C); m2 = np.zeros_like(C)
    t = 0
    for it in range(iters):
        # linear decay of lr
        lrc = lr * (1.0 - 0.7 * it / max(1, iters))
        D = C @ Xfix.T
        sm = D.max(axis=1, keepdims=True)
        E = np.exp(beta * (D - sm))
        S = E.sum(axis=1, keepdims=True)
        W = E / np.maximum(S, 1e-300)
        Gr = W @ Xfix
        dot = (Gr * C).sum(axis=1, keepdims=True)
        Grad = Gr - dot * C
        t += 1
        m1 = 0.9 * m1 + 0.1 * Grad
        m2 = 0.999 * m2 + 0.001 * (Grad * Grad)
        m1h = m1 / (1 - 0.9 ** t)
        m2h = m2 / (1 - 0.999 ** t)
        step = lrc * m1h / (np.sqrt(m2h) + 1e-8)
        # limit step norm to keep on sphere stable
        C = _norm(C - step)
    return C

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    FINAL_LP = 20.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    # load incumbent
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = _norm(rng.standard_normal((n, d)))
        bestV = _max_cos(bestX)
    incumbent = bestX.copy(); incV = bestV

    # ---- vertex cover of tight graph (max-1e-7) ----
    try:
        G0 = incumbent @ incumbent.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        tight = G0 > (m0 - 1e-7)
        # symmetric bool, zero diag
        # greedy cover
        cover = []
        # uncovered edge list via matrix
        Unc = tight.copy()
        np.fill_diagonal(Unc, False)
        total_edges = int(Unc.sum() // 2)
        # degree w.r.t uncovered
        for _ in range(18):
            if len(cover) >= 14:
                cov = total_edges - int(Unc.sum() // 2)
                if total_edges == 0 or cov >= 0.85 * total_edges:
                    break
            deg = Unc.sum(axis=1)
            deg[cover] = -1  # don't reselect (allow but penalize)
            v = int(np.argmax(deg))
            if deg[v] <= 0:
                # pick highest tight degree among remaining
                break
            cover.append(v)
            # remove edges incident to v
            Unc[v, :] = False; Unc[:, v] = False
        if len(cover) < 14:
            # fill with highest degree nodes
            deg0 = tight.sum(axis=1)
            order = np.argsort(-deg0)
            for v in order:
                if v not in cover:
                    cover.append(int(v))
                if len(cover) >= 14:
                    break
        cover = sorted(set(cover))[:18]
        if len(cover) < 14:
            cover = list(np.argsort(-tight.sum(axis=1))[:16])
        ndel = len(cover)
    except Exception:
        cover = list(range(62, 78))
        ndel = len(cover)

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[cover] = False
    Xcur = incumbent[keep_mask].copy()
    # ---- polish remainder with softmax Adam + LP ----
    try:
        # two-stage Adam
        if time.time() + 8 < HARD - FINAL_LP - 10:
            Xcur = _adam_smooth(Xcur, iters=800, beta=200.0, lr=0.01)
        if time.time() + 6 < HARD - FINAL_LP - 10:
            Xcur = _adam_smooth(Xcur, iters=800, beta=600.0, lr=0.004)
        if time.time() + 4 < HARD - FINAL_LP - 8:
            dl = min(HARD - FINAL_LP - 8 - (time.time()), 5.0)
            if dl > 1.0:
                Xcur, _ = _lp_polish(Xcur, time.time() + dl, r_init=0.004)
    except Exception:
        pass
    try:
        mcur = _max_cos(Xcur) if len(Xcur) > 1 else -1.0
    except Exception:
        mcur = incV

    # ---- reinsert deleted points one at a time ----
    NSAMP = 30000
    NKEEP = 20
    for kk in range(ndel):
        if time.time() + 3.0 > HARD - FINAL_LP - 4:
            break
        mfix = len(Xcur)
        # sample
        try:
            C = rng.standard_normal((NSAMP, d))
            C = _norm(C)
            D = C @ Xcur.T
            s = D.max(axis=1)
            # best NKEEP
            idx = np.argpartition(s, NKEEP)[:NKEEP]
            Cand = C[idx].copy()
            # refine vectorized
            CandR = _refine_candidates(Cand, Xcur, beta=500.0, iters=220, lr=0.06)
            # score refined
            DR = CandR @ Xcur.T
            sR = DR.max(axis=1)
            # tie-breaker: number of near contacts
            best_j = -1; best_key = None
            for j in range(NKEEP):
                sj = float(sR[j])
                # count contacts of this candidate within 1e-6 of global new max
                gmax = sj if sj > mcur else mcur
                cnt_close = int(np.count_nonzero(DR[j] > gmax - 1e-6))
                # also internal near-max count is constant, ignore
                # secondary: number within 1e-3 of sj (bushiness)
                bush = int(np.count_nonzero(DR[j] > sj - 1e-3))
                key = (gmax, cnt_close, bush, sj)
                if best_key is None or key < best_key:
                    best_key = key; best_j = j
            winner = CandR[best_j:best_j+1].copy()
            Xcur = np.vstack([Xcur, winner])
            mcur = _max_cos(Xcur)
        except Exception:
            continue
    # if we broke early, fill up greedily to 78
    try:
        while len(Xcur) < 78 and time.time() + 2 < HARD - FINAL_LP:
            mfix = len(Xcur)
            C = _norm(rng.standard_normal((8000, d)))
            D = C @ Xcur.T
            s = D.max(axis=1)
            j = int(np.argmin(s))
            Xcur = np.vstack([Xcur, C[j:j+1]])
    except Exception:
        pass
    # if still short (shouldn't), pad with incumbent rows
    if len(Xcur) < 78:
        need = 78 - len(Xcur)
        extra = incumbent[:need].copy()
        Xcur = np.vstack([Xcur, extra])
    if len(Xcur) > 78:
        Xcur = Xcur[:78].copy()
    Xcur = _norm(Xcur)

    curV = _max_cos(Xcur)
    if curV < incV:
        incumbent = Xcur.copy(); incV = curV
    else:
        # keep Xcur for further joint optimization anyway, but remember incumbent
        pass

    # ---- alternating best-hole relocation + smoothing ----
    # work on Xcur (size 78)
    try:
        cycles = 0
        while time.time() + 4 < HARD - FINAL_LP and cycles < 6:
            cycles += 1
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # involved points
            rows, cols = np.where(G > m - 1e-12)
            involved = sorted(set(rows.tolist()) | set(cols.tolist()))
            if len(involved) == 0:
                break
            # limit to worst few per cycle
            # contact counts
            cnt = (G > m - 1e-9).sum(axis=1)
            involved_sorted = sorted(involved, key=lambda i: -cnt[i])[:6]
            improved = False
            for vi in involved_sorted:
                if time.time() + 3 > HARD - FINAL_LP:
                    break
                Xfix = np.delete(Xcur, vi, axis=0)
                C = _norm(rng.standard_normal((6000, d)))
                D = C @ Xfix.T
                s = D.max(axis=1)
                idx = np.argpartition(s, 8)[:8]
                Cand = C[idx].copy()
                CandR = _refine_candidates(Cand, Xfix, beta=500.0, iters=150, lr=0.05)
                DR = CandR @ Xfix.T
                sR = DR.max(axis=1)
                j = int(np.argmin(sR))
                sj = float(sR[j])
                mfix = _max_cos(Xfix)
                gnew = sj if sj > mfix else mfix
                if gnew < m - 1e-12:
                    Xcur[vi] = CandR[j] / np.linalg.norm(CandR[j])
                    m = gnew
                    improved = True
                elif abs(gnew - m) <= 1e-12:
                    # accept if reduces contacts
                    Gold = Xcur @ Xcur.T
                    np.fill_diagonal(Gold, -2.0)
                    c_old = int(np.count_nonzero(Gold > m - 1e-6))
                    Xtry = Xcur.copy(); Xtry[vi] = CandR[j] / np.linalg.norm(CandR[j])
                    Gtry = Xtry @ Xtry.T
                    np.fill_diagonal(Gtry, -2.0)
                    c_new = int(np.count_nonzero(Gtry > gnew - 1e-6))
                    if c_new < c_old:
                        Xcur = Xtry; improved = True
            # joint smoothing
            if time.time() + 3 < HARD - FINAL_LP:
                try:
                    Xcur = _adam_smooth(Xcur, iters=400, beta=550.0, lr=0.003)
                except Exception:
                    pass
            nv = _max_cos(Xcur)
            if nv < incV:
                incumbent = Xcur.copy(); incV = nv
            if not improved:
                # one L-BFGS-free extra smoothing with higher beta then break if stuck
                if cycles >= 3:
                    break
    except Exception:
        pass

    # ---- final LP polishing on the better of Xcur/incumbent ----
    # start final from the current best among the two
    try:
        vcur = _max_cos(Xcur)
        if vcur < incV:
            incumbent = Xcur.copy(); incV = vcur
    except Exception:
        pass
    try:
        deadline = HARD
        # reserve a bit: use incumbent as start
        Xf, vf = _lp_polish(incumbent, deadline, r_init=0.004, r_max=0.03, cap=2500, max_it=80)
        if vf < incV and np.all(np.isfinite(Xf)):
            incumbent = Xf; incV = vf
    except Exception:
        pass
    incumbent = _norm(incumbent)
    # tiny shrink guard not needed; ensure finite
    incumbent = np.where(np.isfinite(incumbent), incumbent, 0.0)
    # renormalize, replace any zero row with random
    nr = np.linalg.norm(incumbent, axis=1, keepdims=True)
    bad = (nr[:, 0] < 1e-9)
    if np.any(bad):
        R = rng.standard_normal((int(bad.sum()), d))
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        incumbent[bad] = R
    incumbent = _norm(incumbent)
    return incumbent
