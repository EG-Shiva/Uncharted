import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-300
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, deadline, pcap=1500):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    xb = X.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if time.time() > deadline - 1.5:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([X[J].ravel(), X[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(Y)
        if v < best:
            best = v; xb = Y.copy(); X = Y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    n, d = 184, 8
    rng = np.random.default_rng(0)
    # load refs
    best_ref = None
    best_ref_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                v = _max_cos(c)
                if v < best_ref_val:
                    best_ref_val = v; best_ref = c.copy()
        except Exception:
            pass
    if best_ref is None:
        rng0 = np.random.default_rng(0)
        best_ref = _norm(rng0.standard_normal((n, d)))
        best_ref_val = _max_cos(best_ref)
    incumbent_val = float(best_ref_val)
    gbest = best_ref.copy()
    gval = incumbent_val
    X = best_ref.copy()

    targets = [0.49997, 0.49995, 0.49992, 0.49990, 0.49985, 0.49980]
    b1, b2, eps = 0.9, 0.999, 1e-8

    # reserve time for LP
    for ti, tau in enumerate(targets):
        if time.time() > DEADLINE - 40:
            break
        tau_eff = tau - 0.002
        active_thr = tau - 0.02
        m_ad = np.zeros((n, d))
        v_ad = np.zeros((n, d))
        t_ad = 0
        best_E = 1e300
        stag = 0
        kicks = 0
        # iterations per target
        if ti < 2:
            N = 1400
        elif ti < 4:
            N = 1100
        else:
            N = 900
        # Adam lr per target: slightly larger early
        lr = 0.008 if ti < 2 else 0.006
        satisfied = False
        for it in range(N):
            if time.time() > DEADLINE - 40:
                break
            # active set update every 20 (mask)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            # sparse hinge energy with active set
            # mask retains only pairs above tau-0.02
            mask = G > active_thr
            V = G - tau_eff
            V[~mask] = 0.0
            np.maximum(V, 0.0, out=V)
            np.fill_diagonal(V, 0.0)
            # energy triu
            E = float(np.sum(V * V) * 0.5)
            cur_max = float(np.max(G))
            if cur_max < gval:
                gval = cur_max; gbest = X.copy()
            if cur_max <= tau:
                satisfied = True
                break
            if E == 0.0:
                satisfied = True
                break
            # stagnation check on energy
            if E < best_E * (1.0 - 1e-12) - 1e-18:
                best_E = E; stag = 0
            else:
                stag += 1
            # gradient of sum_{i<j} viol^2 : grad = 2*V @ X
            grad = 2.0 * (V @ X)
            # tangent projection
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            t_ad += 1
            m_ad = b1 * m_ad + (1 - b1) * gt
            v_ad = b2 * v_ad + (1 - b2) * (gt * gt)
            mhat = m_ad / (1 - b1 ** t_ad)
            vhat = v_ad / (1 - b2 ** t_ad)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # limit step norm per point for stability
            sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-300
            cap = 0.05
            scale = np.minimum(1.0, cap / (sn + 1e-300))
            step = step * scale
            X = X - step
            X = _norm(X)
            if stag >= 250:
                # kick restricted to endpoints of violated pairs + largest dual-stress proxy
                # stress proxy: row sum of violations (LP dual analogue)
                stress = np.sum(V, axis=1)
                k_top = 8
                idx_stress = np.argsort(-stress, kind="stable")[:k_top]
                # endpoints of largest violations
                # find top 4 pairs from G
                iu = np.triu_indices(n, 1)
                gv = G[iu]
                topk = min(4, len(gv))
                pidx = np.argpartition(-gv, topk - 1)[:topk] if topk > 0 else np.array([], dtype=int)
                extra = []
                for pi in pidx:
                    if gv[pi] > tau - 0.005:
                        extra.append(int(iu[0][pi])); extra.append(int(iu[1][pi]))
                kick_idx = list(idx_stress)
                for e in extra:
                    if e not in kick_idx:
                        kick_idx.append(e)
                kick_idx = np.array(kick_idx, dtype=int)
                sigma = 0.02 if (kicks % 2 == 0) else 0.035
                kicks += 1
                noise = rng.standard_normal((len(kick_idx), d))
                # tangentialize noise
                for ii, gi in enumerate(kick_idx):
                    vv = noise[ii]
                    vv = vv - np.dot(vv, X[gi]) * X[gi]
                    nv = np.linalg.norm(vv) + 1e-300
                    vv = vv / nv * sigma * (0.7 + 0.6 * rng.random())
                    X[gi] = X[gi] + vv
                X = _norm(X)
                # reset adam moments for kicked points (retry, not return to incumbent)
                m_ad[kick_idx] = 0.0
                v_ad[kick_idx] = 0.0
                stag = 0
                # keep best_E as current to force retry progress
                # recompute E next iter; allow slight increase
                best_E = min(best_E, E)
        # carry X forward (do not revert to incumbent) to change topology
        # if satisfied, continue to next lower tau immediately
        continue

    # LP only after first satisfying below incumbent (lab best)
    thresh = incumbent_val - 1e-12
    # gval is best seen; also check current X
    cur_max = _max_cos(X)
    if cur_max < gval:
        gval = cur_max; gbest = X.copy()
    if gval < thresh:
        remaining = DEADLINE - time.time() - 2.0
        if remaining > 5:
            lp_sec = min(remaining, 30.0)
            try:
                Xp, vp = _lp_polish(gbest, seconds=lp_sec, deadline=DEADLINE - 1.0)
                if vp < gval:
                    gval = vp; gbest = Xp.copy()
            except Exception:
                pass
    # safety fallback
    try:
        if not np.all(np.isfinite(gbest)) or _max_cos(gbest) > incumbent_val + 1e-9:
            gbest = best_ref.copy()
    except Exception:
        gbest = best_ref.copy()
    gbest = _norm(np.asarray(gbest, dtype=np.float64))
    return gbest
