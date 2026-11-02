import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = c >= m - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu0[sel], 1)
    np.add.at(cnt, iu1[sel], 1)
    # nearest-neighbor value per point for hub ranking
    G2 = X @ X.T
    np.fill_diagonal(G2, -2.0)
    nn = G2.max(axis=1)
    return m, cnt, nn, G

def _refine_exact_max_adam(Y0, F, steps=350, s0=200.0, s1=800.0, lr0=0.015, lr1=0.002, seed=0):
    Y = Y0.copy()
    k, d = Y.shape
    Nf = F.shape[0]
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    l0, l1 = np.log(lr0), np.log(lr1)
    iu0, iu1 = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    nmove = len(iu0)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        s = s0 + (s1 - s0) * frac
        lr = np.exp(l0 + (l1 - l0) * frac)
        # normalize (stay on sphere)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = Y / nr
        C = Y @ F.T  # k x Nf
        if k > 1:
            Mm = Y @ Y.T
            mv = Mm[iu0, iu1]
            allmax = C.max() if C.size else -2.0
            if mv.size:
                allmax = max(allmax, float(mv.max()))
            # softmax weights = hinge-like exact-max smoothing
            eC = np.exp(s * (C - allmax))
            eM = np.exp(s * (mv - allmax)) if mv.size else np.array([])
            se = float(eC.sum() + eM.sum())
            wC = eC / se
            wM = eM / se if mv.size else eM
            # gradient
            Grad = wC @ F  # k x d
            if nmove > 0:
                W = np.zeros((k, k))
                W[iu0, iu1] = wM
                W = W + W.T
                Grad = Grad + W @ Y
        else:
            allmax = float(C.max())
            eC = np.exp(s * (C - allmax))
            se = float(eC.sum())
            wC = eC / se
            Grad = wC @ F
        sprod = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - sprod * Y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip
        Y = Y - lr * step
    nr = np.linalg.norm(Y, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Y = Y / nr
    return Y

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
        if np.count_nonzero(sel) > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    SURG_END = t0 + 158.0
    n, d = 73, 13
    rng = np.random.default_rng(20261102)
    # load incumbent
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc
                        Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    NHOLE = 20000
    POOL = 500
    trial = 0
    # surgery loop: many cheap replacements, no LP inside
    while time.time() < SURG_END and time.time() < HARD_END - 20:
        trial += 1
        m, cnt, nn, G = _contacts(Xbest, tol=1e-9)
        # identify rattlers (0 contacts) and hubs (highest contacts / worst nn)
        ratt_idx = np.where(cnt == 0)[0]
        hub_order = np.argsort(-cnt - 1e-6 * nn)  # contacts first, then worst nn
        nn_order = np.argsort(-nn)
        # choose k = 3..5 cycling
        k = [3, 4, 4, 5][trial % 4]
        # build removal: always include top hubs to kill active pair
        n_hub = min(2 + (trial % 2), k - 1)  # 2 or 3 hubs
        n_hub = min(n_hub, k - 1)
        removal = []
        for h in hub_order[:n_hub]:
            removal.append(int(h))
        # add rattlers for the rest
        need = k - len(removal)
        if len(ratt_idx) > 0 and need > 0:
            # random rattlers not already in removal
            cand_r = [int(i) for i in ratt_idx if i not in removal]
            if len(cand_r) > 0:
                perm = rng.permutation(len(cand_r))
                for t in range(min(need, len(cand_r))):
                    removal.append(int(cand_r[int(perm[t])]))
                    need -= 1
        # fill any remainder with worst-nn points
        if need > 0:
            for h in nn_order:
                h = int(h)
                if h not in removal:
                    removal.append(h)
                    need -= 1
                    if need == 0:
                        break
        removal = np.array(sorted(removal), dtype=int)
        keep = np.array([i for i in range(n) if i not in set(removal.tolist())], dtype=int)
        F = Xbest[keep, :]
        # frozen internal max
        Gf = F @ F.T
        np.fill_diagonal(Gf, -2.0)
        frozen_max = float(Gf.max())
        if frozen_max >= best:
            # removal did not drop peak; force include max-pair endpoints
            iu0, iu1 = np.triu_indices(n, 1)
            # find current global max pair
            Gall = Xbest @ Xbest.T
            np.fill_diagonal(Gall, -2.0)
            am = int(np.argmax(Gall))
            ai, aj = am // n, am % n
            removal = np.array(sorted(set([ai, aj] + removal.tolist()))[:k] if len(set([ai, aj] + removal.tolist())) >= k else sorted(set([ai, aj] + removal.tolist()))), dtype=int)
            # if we truncated weirdly, rebuild
            if len(removal) < k:
                for h in nn_order:
                    h = int(h)
                    if h not in removal:
                        removal = np.append(removal, h)
                        if len(removal) == k:
                            break
                removal = np.array(sorted(removal), dtype=int)
            keep = np.array([i for i in range(n) if i not in set(removal.tolist())], dtype=int)
            F = Xbest[keep, :]
            Gf = F @ F.T
            np.fill_diagonal(Gf, -2.0)
            frozen_max = float(Gf.max())
            if frozen_max >= best:
                continue
        # --- hole sampling: >=20000 random points ---
        C = rng.standard_normal(size=(NHOLE, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ F.T  # NHOLE x Nf
        srow = S.max(axis=1)
        order = np.argsort(srow)  # best holes first
        pool_idx = order[:POOL]
        P = C[pool_idx, :]  # POOL x d
        pcross = srow[pool_idx]  # POOL
        # precompute pool-pool dots
        PP = P @ P.T
        # greedy constructions: try ~24 random greedy builds
        best_g_score = 1e9
        best_g_pick = None
        n_try = 24
        for g in range(n_try):
            # start: random among top 20 to diversify
            start_choice = int(rng.integers(0, min(20, POOL)))
            chosen = [start_choice]
            # randomize order of subsequent selection among top candidates
            for pos in range(1, k):
                # score each pool candidate: max(pcross, max dot with chosen)
                dots_ch = PP[:, chosen].max(axis=1)  # POOL
                score = np.maximum(pcross, dots_ch)
                score[chosen] = 2.0  # exclude already chosen
                # pick best, with occasional randomization among top-5
                if (g % 3 == 0):
                    j = int(np.argmin(score))
                else:
                    tb = np.argsort(score)[:5]
                    j = int(tb[int(rng.integers(0, len(tb)))])
                chosen.append(j)
            Y0 = P[np.array(chosen), :]
            # exact global score of this greedy pick
            cm = float((Y0 @ F.T).max())
            if k > 1:
                Gm = Y0 @ Y0.T
                np.fill_diagonal(Gm, -2.0)
                mm = float(Gm.max())
                cm = max(cm, mm)
            glob = max(cm, frozen_max)
            if glob < best_g_score:
                best_g_score = glob
                best_g_pick = np.array(chosen)
                if best_g_score < frozen_max + 1e-12:
                    pass
        if best_g_pick is None:
            continue
        # take top 2 distinct greedy picks for refinement: best + one more diverse
        # reconstruct best Y0
        Y0_best = P[best_g_pick, :].copy()
        cands_Y0 = [Y0_best]
        # second candidate: perturb start
        try:
            alt_start = int(order[POOL + int(rng.integers(0, 200))] % POOL) if False else None
        except Exception:
            pass
        # simple second candidate: shift best pick by neighbour in pool ordering
        if k >= 2:
            alt = best_g_pick.copy()
            alt[0] = (int(alt[0]) + int(rng.integers(1, 30))) % POOL
            # ensure distinct
            if alt[0] not in alt[1:]:
                cands_Y0.append(P[alt, :].copy())
        # refine best candidates with exact-max (hinge-loss) Adam
        improved = False
        for ci, Y0c in enumerate(cands_Y0):
            if time.time() > SURG_END or time.time() > HARD_END - 20:
                break
            # cheap refine: fewer steps for screening
            steps = 300 if ci == 0 else 200
            Yref = _refine_exact_max_adam(Y0c, F, steps=steps, s0=200.0, s1=900.0, lr0=0.015, lr1=0.002)
            cm = float((Yref @ F.T).max())
            if k > 1:
                Gm = Yref @ Yref.T
                np.fill_diagonal(Gm, -2.0)
                mm = float(Gm.max())
                cm = max(cm, mm)
            glob = max(cm, frozen_max)
            if glob < best - 1e-12:
                # accept surgery without LP: lowers unpolished global max
                Xnew = np.empty((n, d))
                Xnew[keep, :] = F
                Xnew[removal, :] = Yref
                # renormalize (already unit)
                Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
                vv = _max_cos(Xnew)
                if vv < best:
                    Xbest = Xnew
                    best = float(vv)
                    improved = True
                    break
        # if first trial batch failed, try one more aggressive refine on best with more steps
        if (not improved) and (best_g_score < best):
            if time.time() < SURG_END and time.time() < HARD_END - 20:
                Yref2 = _refine_exact_max_adam(Y0_best, F, steps=500, s0=150.0, s1=1200.0, lr0=0.02, lr1=0.001)
                cm = float((Yref2 @ F.T).max())
                if k > 1:
                    Gm = Yref2 @ Yref2.T
                    np.fill_diagonal(Gm, -2.0)
                    mm = float(Gm.max())
                    cm = max(cm, mm)
                glob = max(cm, frozen_max)
                if glob < best - 1e-12:
                    Xnew = np.empty((n, d))
                    Xnew[keep, :] = F
                    Xnew[removal, :] = Yref2
                    Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
                    vv = _max_cos(Xnew)
                    if vv < best:
                        Xbest = Xnew
                        best = float(vv)
        # loop continues; time check inside
        if trial > 800:
            break
    # one short LP polish on best accepted configuration
    try:
        remain = HARD_END - time.time() - 1.0
        if remain > 2.0:
            secs = min(remain, 12.0)
            Xp, vp = _lp_polish(Xbest, secs, HARD_END - 0.5)
            if vp < best:
                Xbest = Xp
                best = vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
