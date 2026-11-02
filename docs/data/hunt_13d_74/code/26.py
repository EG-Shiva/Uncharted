import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _degrees(X, thr=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > (m - thr)
    deg = mask.sum(axis=1).astype(int)
    rowmax = G.max(axis=1)
    return deg, rowmax, float(m)

def _refine_hole_cands(C, Xc, iters=60):
    # C: (K,d) candidates, Xc: (m,d) frozen core, minimize max_j <c,xj>
    K = C.shape[0]
    lr0, lr1 = 0.03, 0.003
    for it in range(iters):
        S = C @ Xc.T
        idx = np.argmax(S, axis=1)
        dots = S[np.arange(K), idx]
        Gs = Xc[idx]
        proj = (Gs * C).sum(axis=1, keepdims=True)
        Gt = Gs - proj * C
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        lr = lr1 + (lr0 - lr1) * (1.0 - it / max(1, iters))
        C = C - lr * (Gt / nr)
        C = _normalize(C)
    return C

def _lp_polish(X, best, deadline):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 1.0e-3
    while time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(25.0 * radius, 2e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt > 2200:
            thr = np.sort(cfull)[-2200]
            sel = cfull >= thr
            cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cc = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _normalize(Y)
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    LIM = 172.0
    DEAD = t0 + LIM
    rng = np.random.default_rng(20261102)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    def refine_lse(X, sched):
        m = np.zeros_like(X); v = np.zeros_like(X); cnt = 0
        for tau, steps, lr0 in sched:
            for s in range(steps):
                if time.time() > DEAD - 20:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt += 1
                c1 = 1.0 - 0.9 ** cnt
                c2 = 1.0 - 0.999 ** cnt
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() > DEAD - 20:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 6e-7)
            mask = G > (gmax - delta)
            Gr = mask.astype(np.float64) @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    full_sched = [(0.008, 300, 0.007), (0.0025, 400, 0.0045), (0.0009, 300, 0.0025)]
    fast_sched = [(0.008, 250, 0.007), (0.0025, 350, 0.0045), (0.0009, 250, 0.0025)]
    tiny_sched = [(0.0025, 200, 0.004), (0.0009, 200, 0.002)]

    # light initial squeeze
    X0 = best.copy()
    X0 = refine_lse(X0, tiny_sched)
    X0, v0 = greedy(X0, 200, 0.0015)
    if v0 < best_val:
        best, best_val = X0.copy(), float(v0)

    basin = 0
    targets = [6, 10, 14]
    hub_ns = [2, 3, 4]
    while time.time() < DEAD - 20:
        remain = DEAD - 20 - time.time()
        if remain < 10:
            sched = tiny_sched
        elif remain < 30:
            sched = fast_sched
        else:
            sched = fast_sched if (basin % 2 == 0) else full_sched
        T = targets[basin % 3]
        Hn = hub_ns[(basin // 3) % 3]
        Xcur = best.copy()
        # contact analysis at 1e-7
        deg, rowmax, gmax = _degrees(Xcur, thr=1e-7)
        order_hub = np.argsort(-deg, kind="stable")
        # hubs: top Hn with deg>=9 if possible else top Hn
        hubs = []
        for ii in order_hub:
            if len(hubs) >= Hn:
                break
            hubs.append(int(ii))
        # also enforce deg>=9 preference: if some top have low deg, still keep them
        ratt = [int(i) for i in range(n) if deg[i] == 0]
        # build delete set: rattlers + hubs
        Dset = []
        for h in hubs:
            if h not in Dset:
                Dset.append(h)
        for r in ratt:
            if r not in Dset:
                Dset.append(r)
        # adjust to size T
        if len(Dset) > T:
            # keep all hubs, truncate rattlers (keep worst rattlers by rowmax)
            keep_hubs = set(hubs)
            ratt_in = [x for x in Dset if x not in keep_hubs]
            # sort rattlers by rowmax descending (worst first)
            ratt_in.sort(key=lambda i: -rowmax[i])
            need = T - len(keep_hubs)
            Dset = list(keep_hubs) + ratt_in[:max(0, need)]
        elif len(Dset) < T:
            # add worst points by rowmax not already in set (prefer high deg)
            worst = np.argsort(-rowmax, kind="stable")
            for w in worst:
                w = int(w)
                if w not in Dset:
                    Dset.append(w)
                if len(Dset) >= T:
                    break
        Dset = Dset[:T]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.array(Dset, dtype=int)] = False
        Xcore_list = [Xcur[keep_mask].copy()]  # will grow
        Xfrozen = Xcore_list[0]  # (n-T,d) frozen initially
        # successively reinsert
        cur_set = Xfrozen.copy()
        new_pts = []
        for k in range(T):
            if time.time() > DEAD - 20:
                break
            # sample 20000 candidates
            Cand = _normalize(rng.standard_normal((20000, d)))
            S = Cand @ cur_set.T
            sc = S.max(axis=1)
            # best 20
            part = np.argpartition(sc, 20)[:20]
            Ctop = Cand[part].copy()
            # locally refine best 20 by tangent subgradient on score
            Ctop = _refine_hole_cands(Ctop, cur_set, iters=60)
            S2 = Ctop @ cur_set.T
            sc2 = S2.max(axis=1)
            j = int(np.argmin(sc2))
            chosen = Ctop[j:j+1].copy()
            # slight random choice among top-3 occasionally for diversity
            if (basin % 5 == 4) and k == 0:
                # pick 2nd best to diversify every 5th basin
                o = np.argsort(sc2)
                chosen = Ctop[o[1]:o[1]+1].copy()
            new_pts.append(chosen)
            cur_set = np.vstack([cur_set, chosen])
        if len(new_pts) != T:
            basin += 1
            continue
        Xnew = np.vstack([Xfrozen] + new_pts)
        # unfreeze: small jitter on reinserted part only to break symmetry
        # (keep core frozen exactly, jitter new points tiny)
        if T > 0:
            Xnew[n-T:] = _normalize(Xnew[n-T:] + 0.0015 * rng.standard_normal((T, d)))
        Xnew = refine_lse(Xnew, sched)
        Xnew, vnew = greedy(Xnew, 220, 0.0018)
        if vnew < best_val - 1e-12:
            best = Xnew.copy()
            best_val = float(vnew)
        basin += 1
        if basin > 200:
            break

    if time.time() < DEAD - 1:
        Xf = best.copy()
        Xf, vf = greedy(Xf, 200, 0.0006)
        if vf < best_val:
            best, best_val = Xf.copy(), float(vf)
    if time.time() < DEAD - 0.5:
        try:
            Xp, vp = _lp_polish(best.copy(), best_val, DEAD - 0.3)
            if vp < best_val:
                best, best_val = Xp, vp
        except Exception:
            pass
    best = _normalize(best)
    return best
