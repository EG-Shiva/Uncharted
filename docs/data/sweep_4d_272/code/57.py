import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    mask = g >= m - thr
    # upper triangle pairs
    iu = np.triu_indices(n, 1)
    sel = mask[iu]
    I = iu[0][sel]; J = iu[1][sel]
    deg = (mask).sum(axis=1).astype(float)
    # mask includes diagonal? diagonal is -2 so not counted
    var = float(deg.var())
    return m, I, J, deg, var, g

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac if tau1 > 0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gg = grad - dot * x
        m *= b1; m += (1 - b1) * gg
        v *= b2; v += (1 - b2) * (gg * gg)
        bc1 = 1.0 - b1 ** (k + 1); bc2 = 1.0 - b2 ** (k + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish_forced(x, seconds, deadline, forced_pairs=None, forced_target=None):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    # map forced pairs to dict
    fI = np.zeros(0, dtype=int); fJ = np.zeros(0, dtype=int)
    if forced_pairs is not None and len(forced_pairs) > 0:
        fI = np.array([p[0] for p in forced_pairs], dtype=int)
        fJ = np.array([p[1] for p in forced_pairs], dtype=int)
        if forced_target is None:
            forced_target = best - 2e-4
        # allow per-pair target array
        if np.isscalar(forced_target):
            fT = np.full(len(fI), float(forced_target))
        else:
            fT = np.asarray(forced_target, dtype=float)
    else:
        fT = np.zeros(0)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        cfull = cmat[iu]
        mcur = float(cfull.max())
        sel = cfull > mcur - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        # exclude forced pairs from t-set (they have their own cap); keep them out to force break
        if len(fI) > 0:
            fset = set(zip(fI.tolist(), fJ.tolist())) | set(zip(fJ.tolist(), fI.tolist()))
            keep = np.ones(len(I), dtype=bool)
            for k in range(len(I)):
                if (int(I[k]), int(J[k])) in fset:
                    keep[k] = False
            I = I[keep]; J = J[keep]
            csub = cfull[sel][keep]
        else:
            csub = cfull[sel]
        p = len(I); f = len(fI)
        if p == 0 and f == 0:
            break
        # current forced cosines
        if f > 0:
            cf = cmat[fI, fJ]
        tot = p + f
        nv = n * d + 1
        # build rows: first p rows with -1 on t, next f rows with 0 on t
        rows_list = []; cols_list = []; vals_list = []; bub = []
        # vectorized for p rows
        if p > 0:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            rows_list.append(rows); cols_list.append(cols); vals_list.append(vals)
            bub.append(-csub)
        if f > 0:
            rows2 = np.repeat(np.arange(p, p + f), 2 * d)
            cols2 = np.concatenate([fI[:, None] * d + np.arange(d), fJ[:, None] * d + np.arange(d)], 1).ravel()
            vals2 = np.concatenate([x[fJ], x[fI]], 1).ravel()
            rows_list.append(rows2); cols_list.append(cols2); vals_list.append(vals2)
            bub.append(fT - cf)
        rowsA = np.concatenate(rows_list); colsA = np.concatenate(cols_list); valsA = np.concatenate(vals_list)
        bubA = np.concatenate(bub)
        A_ub = coo_matrix((valsA, (rowsA, colsA)), shape=(tot, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=bubA, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        # for forced mode, require forced pairs below target (with small slack) to accept?
        # we accept if overall max improves or stays and forced satisfied
        if f > 0:
            gf = (y[fI] * y[fJ]).sum(axis=1).max() if f > 0 else -2
            # if forced not yet broken, still allow step if it reduces forced max or overall
            forced_ok = gf <= float(fT.max()) + 1e-9
            if vv < best and (forced_ok or gf < (x[fI] * x[fJ]).sum(axis=1).max()):
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            elif vv <= best + 1e-12 and forced_ok:
                # accept balanced move that keeps max but breaks hub
                x, best = y, min(vv, best) if vv < best else best
                # keep x even if vv==best to move topology
                if vv <= best + 5e-12:
                    x = y
                radius = min(radius * 1.2, 0.05)
            else:
                # check if y at least improves forced even if max slightly worse? allow small worsening
                if gf < (x[fI] * x[fJ]).sum(axis=1).max() - 1e-7 and vv < best + 5e-5:
                    x = y
                    best = min(best, vv)
                    radius = min(radius * 1.2, 0.05)
                else:
                    radius *= 0.5
        else:
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
    return x

def _rigidity_stress(x, I, J):
    n, d = x.shape
    p = len(I)
    if p == 0:
        return np.zeros(0)
    # tangent bases 4x3 per point
    T = np.zeros((n, 4, 3))
    for i in range(n):
        xi = x[i]
        # pick smallest component to build stable basis
        # Gram-Schmidt on 4x4 identity minus xi
        # Use QR of (I - xi xi^T) truncated: do SVD-lite via random
        # simple: take 3 vectors orthogonalized
        M = np.eye(4) - np.outer(xi, xi)
        # eigen via svd of M (symmetric) -> use QR on random
        # deterministic: QR of M with column pivoting via numpy qr
        Q, _ = np.linalg.qr(M)
        # Q[:,0] ~ +/- xi, remaining span tangent (since M rank 3)
        # find column most aligned with xi
        dots = np.abs(Q.T @ xi)
        idx = int(np.argmax(dots))
        cols = [c for c in range(4) if c != idx][:3]
        T[i] = Q[:, cols]
    ncol = 3 * n
    R = np.zeros((p, ncol))
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        R[k, 3*i:3*i+3] = T[i].T @ x[j]
        R[k, 3*j:3*j+3] = T[j].T @ x[i]
    try:
        # full U for left null
        U, S, _ = np.linalg.svd(R, full_matrices=True)
        tol = max(R.shape) * np.finfo(float).eps * (S.max() if S.size else 1.0)
        tol = max(tol, 1e-8)
        rank = int((S > tol).sum())
        # null basis columns rank..p-1
        if rank >= p:
            return np.zeros(p)
        Nb = U[:, rank:]  # p x (p-rank)
        stress = (Nb * Nb).sum(axis=1)
        return stress
    except Exception:
        return np.zeros(p)

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    x_cur = x0.copy()
    _, _, _, _, cur_var, _ = _contacts(x_cur)
    _, _, _, _, best_var, _ = _contacts(x_best)
    # quick rigidity-guided cycles
    cycle = 0
    # per-polish budgets
    while time.time() < deadline - 18:
        cycle += 1
        mcur, Ic, Jc, deg, var, _ = _contacts(x_cur)
        # build neighbor lists
        n = 272
        neigh = [[] for _ in range(n)]
        for a, b in zip(Ic.tolist(), Jc.tolist()):
            neigh[a].append(b); neigh[b].append(a)
        rattlers = [i for i in range(n) if deg[i] <= 0.5]
        if len(rattlers) == 0:
            # pseudo-rattlers: lowest degree
            order = np.argsort(deg)
            rattlers = order[:3].tolist()
        hubs = np.argsort(-deg)
        # stress
        stress = _rigidity_stress(x_cur, Ic, Jc)
        # score each contact: hub-related + stress
        pair_scores = []
        for k in range(len(Ic)):
            a = int(Ic[k]); b = int(Jc[k])
            ds = float(deg[a] + deg[b])
            st = float(stress[k]) if k < len(stress) else 0.0
            # prefer high-degree pairs
            pair_scores.append((ds, st, k))
        pair_scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
        # take top ~30 hub contacts with deg>=8 endpoint
        cand_pairs = []
        for ds, st, k in pair_scores:
            a = int(Ic[k]); b = int(Jc[k])
            if deg[a] >= 8 or deg[b] >= 8:
                cand_pairs.append((a, b, st))
            if len(cand_pairs) >= 30:
                break
        if len(cand_pairs) == 0:
            # fallback: top pairs regardless
            for ds, st, k in pair_scores[:30]:
                cand_pairs.append((int(Ic[k]), int(Jc[k]), st))
        if len(cand_pairs) == 0:
            break
        # limit candidates per cycle by remaining time
        remain = deadline - time.time()
        max_cands = 12 if remain > 60 else (8 if remain > 30 else 4)
        cand_pairs = cand_pairs[:max_cands]
        improved = False
        for ci, (h0, j0, st) in enumerate(cand_pairs):
            if time.time() > deadline - 16:
                break
            # choose hub = higher-degree endpoint, other = contact to break
            if deg[h0] >= deg[j0]:
                h, j = h0, j0
            else:
                h, j = j0, h0
            r = rattlers[ci % len(rattlers)]
            # avoid r == h or j
            if r == h or r == j:
                r = rattlers[(ci + 1) % len(rattlers)]
                if r == h or r == j:
                    continue
            xt = x_cur.copy()
            # relocate rattler to centroid of hub neighbourhood
            nb = neigh[h]
            if len(nb) > 0:
                cent = xt[nb].sum(axis=0)
                # if hub has many contacts, centroid ~ hub direction; use it
                if np.linalg.norm(cent) < 1e-9:
                    cent = xt[h]
                cent = cent / np.linalg.norm(cent)
                # small random tangent jitter
                jr = rng.standard_normal(4)
                jr -= (jr @ cent) * cent
                jr /= (np.linalg.norm(jr) + 1e-12)
                cent = cent + 0.005 * jr
                cent /= np.linalg.norm(cent)
                xt[r] = cent
            else:
                # random void-ish: opposite of hub
                v = -xt[h] + 0.01 * rng.standard_normal(4)
                xt[r] = v / np.linalg.norm(v)
            # displace hub tangentially away from j by 0.02-0.05
            delta = 0.02 + 0.03 * rng.random()
            # alternate magnitude deterministically
            delta = [0.025, 0.035, 0.05, 0.02][(ci + cycle) % 4]
            c = float(xt[h] @ xt[j])
            gvec = xt[j] - c * xt[h]
            ng = np.linalg.norm(gvec)
            if ng < 1e-12:
                tv = rng.standard_normal(4); tv -= (tv @ xt[h]) * xt[h]
                ng = np.linalg.norm(tv); gvec = tv
            u = gvec / ng
            xt[h] = xt[h] - delta * u
            xt[h] /= np.linalg.norm(xt[h])
            # brief anneal (joint, short)
            # vary temp slightly per candidate
            tau0 = 0.012 if (ci % 2 == 0) else 0.008
            xt = _anneal(xt, 220, tau0, 0.001, 0.008, 0.0008)
            # LP polish with forced break h-j below mcur-2e-4
            target = mcur - 2e-4
            # also ensure target < current best - something? use min with best
            target = min(target, best_val - 5e-5) if np.isfinite(best_val) else target
            xt = _lp_polish_forced(xt, 2.2, deadline - 1.0, forced_pairs=[(h, j)], forced_target=target)
            mnew = _max_cos(xt)
            _, _, _, degn, varn, _ = _contacts(xt)
            # acceptance
            if mnew < best_val - 1e-12:
                x_best = xt.copy(); best_val = mnew; best_var = varn
                x_cur = xt.copy(); cur_var = varn
                improved = True
                break
            elif mnew <= best_val + 2e-9 and varn < cur_var - 0.03:
                # balanced move at equal max
                x_cur = xt.copy(); cur_var = varn
                # also adopt as best if equal-or-better (keeps balanced graph)
                if mnew <= best_val:
                    x_best = xt.copy(); best_val = mnew; best_var = varn
                improved = True
                # continue scanning from new cur? break to recompute rigidity
                break
            elif mnew <= best_val + 1e-9 and varn < best_var - 0.1:
                x_cur = xt.copy(); cur_var = varn
                improved = True
                break
        if not improved:
            # diversification: small global kick + short anneal + plain polish on best
            if time.time() > deadline - 16:
                break
            # one random kick attempt to avoid stall
            xk = x_cur + 0.015 * rng.standard_normal((272, 4))
            xk = _norm(xk)
            xk = _anneal(xk, 200, 0.01, 0.001, 0.006, 0.0006)
            xk = _lp_polish_forced(xk, 1.5, deadline - 1.0, forced_pairs=None)
            mv = _max_cos(xk)
            _, _, _, _, vv, _ = _contacts(xk)
            if mv < best_val:
                x_best = xk.copy(); best_val = mv; best_var = vv
                x_cur = xk.copy(); cur_var = vv
            elif mv <= best_val + 2e-9 and vv < cur_var - 0.03:
                x_cur = xk.copy(); cur_var = vv
            # stop after a few fruitless cycles to save time for final polish
            if cycle >= 8:
                break
    # final snap on best without forced constraint
    remain = deadline - time.time() - 1.0
    if remain > 3:
        xf = _lp_polish_forced(x_best, min(remain, 14.0), deadline - 0.5, forced_pairs=None)
        mv = _max_cos(xf)
        if mv <= best_val:
            x_best = xf
    x_best = _norm(x_best)
    return x_best
