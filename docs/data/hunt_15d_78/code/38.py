import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts_deg(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    return (G > m - thr).sum(axis=1), float(m)

def _get_duals(res, p, n):
    cands = []
    try:
        if hasattr(res, 'ineqlin'):
            il = res.ineqlin
            if hasattr(il, 'marginals'):
                try:
                    cands.append(np.asarray(il.marginals, dtype=float).ravel())
                except: pass
            if hasattr(il, 'dual'):
                try:
                    cands.append(np.asarray(il.dual, dtype=float).ravel())
                except: pass
            try:
                if hasattr(il, 'get'):
                    mm = il.get('marginals', None)
                    if mm is not None:
                        cands.append(np.asarray(mm, dtype=float).ravel())
            except: pass
    except: pass
    for attr in ('marginals', 'duals', 'pi', 'con_marginals', 'lambda_'):
        try:
            if hasattr(res, attr):
                cands.append(np.asarray(getattr(res, attr), dtype=float).ravel())
        except: pass
    try:
        if hasattr(res, 'get'):
            mm = res.get('marginals', None)
            if isinstance(mm, dict):
                for k in ('ineqlin', 'A_ub', 'inequality', 'ub'):
                    if k in mm:
                        try:
                            cands.append(np.asarray(mm[k], dtype=float).ravel())
                        except: pass
            elif mm is not None:
                try:
                    cands.append(np.asarray(mm, dtype=float).ravel())
                except: pass
            il2 = res.get('ineqlin', None)
            if il2 is not None:
                try:
                    if hasattr(il2, 'get'):
                        mm2 = il2.get('marginals', None)
                        if mm2 is not None:
                            cands.append(np.asarray(mm2, dtype=float).ravel())
                    elif hasattr(il2, 'marginals'):
                        cands.append(np.asarray(il2.marginals, dtype=float).ravel())
                except: pass
    except: pass
    for v in cands:
        if v.size == p:
            return v.copy()
    for v in cands:
        if v.size == p + n:
            return v[:p].copy()
    for v in cands:
        if v.size > p and v.size < p + n + 5:
            return v[:p].copy()
    return None

def _lp_polish(X, seconds, excluded=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 2e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    last_duals = None
    last_I = None
    last_J = None
    exc_mat = None
    if excluded is not None:
        try:
            exc_mat = np.zeros((n, n), dtype=bool)
            for (a, b) in excluded:
                exc_mat[a, b] = True
                exc_mat[b, a] = True
        except:
            exc_mat = None
    while time.time() - start < seconds and radius > 1e-13 and it < 40:
        it += 1
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu0[sel], iu1[sel]
        csel = c[sel]
        if exc_mat is not None and len(I) > 0:
            keep = ~exc_mat[I, J]
            # keep at least some constraints
            if np.any(keep):
                I, J, csel = I[keep], J[keep], csel[keep]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = max(0.3, seconds - (time.time() - start))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dd = _get_duals(res, p, n)
        if dd is not None:
            last_duals = dd
            last_I = I.copy()
            last_J = J.copy()
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def _stress_once(X, radius=8e-4, band=1e-8):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0, iu1]
    m = c.max()
    sel = c > m - band
    # cap size for speed
    if np.sum(sel) > 900:
        idx = np.argsort(-c)
        sel = np.zeros_like(sel, dtype=bool)
        sel[idx[:900]] = True
    I, J = iu0[sel], iu1[sel]
    csel = c[sel]
    p = len(I)
    if p == 0:
        return I, J, None, float(m)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                      options={"time_limit": 4.0})
        if res.status == 0:
            dd = _get_duals(res, p, n)
            return I, J, dd, float(m)
    except Exception:
        pass
    return I, J, None, float(m)

def solve():
    t0 = time.time()
    LIMIT = 172.0
    DEADLINE = 150.0
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # initial LP polish once
    if LIMIT - (time.time() - t0) > 10:
        bestX, bestV = _lp_polish(bestX, 3.0)
    curX = bestX.copy()
    curV = bestV
    bestEverX = bestX.copy()
    bestEverV = bestV
    deg0, _ = _contacts_deg(bestEverX, 1e-9)
    bestSeq = tuple(sorted(deg0.tolist()))
    niter = 0
    while time.time() - t0 < DEADLINE and niter < 12:
        niter += 1
        if LIMIT - (time.time() - t0) < 15:
            break
        I, J, dd, m = _stress_once(curX, radius=8e-4, band=1e-8)
        p = len(I)
        if p < 10:
            break
        if dd is None:
            w = np.ones(p, dtype=float)
        else:
            w = np.asarray(dd, dtype=float).ravel()
            if w.size != p:
                w = np.ones(p, dtype=float)
            else:
                # fix sign convention: supporting should be positive
                try:
                    med = float(np.median(w))
                except:
                    med = 0.0
                if med < 0:
                    w = -w
                w = np.where(np.isfinite(w), w, 0.0)
        # release 10-30 lowest-stress pairs
        nrel = int(min(30, max(10, p // 18)))
        nrel = min(nrel, p - 5)
        order = np.argsort(w, kind='stable')
        rel_idx = order[:nrel]
        release_set = set()
        for k in rel_idx:
            a0, b0 = int(I[k]), int(J[k])
            if a0 > b0:
                a0, b0 = b0, a0
            release_set.add((a0, b0))
        surv_mask = np.ones(p, dtype=bool)
        surv_mask[rel_idx] = False
        SI, SJ, Sw = I[surv_mask], J[surv_mask], w[surv_mask]
        # total stress per point (use full w, clip negatives to 0 for totals)
        tot = np.zeros(n, dtype=float)
        wpos = np.where(Sw > 0, Sw, 0.0)
        # map surviving weights
        for k in range(len(SI)):
            tot[int(SI[k])] += wpos[k]
            tot[int(SJ[k])] += wpos[k]
        if tot.max() <= 1e-18:
            # fallback to degrees
            deg, _ = _contacts_deg(curX, 1e-8)
            tot = deg.astype(float)
        kpts = 8
        # 5-15 points with largest total stress
        S = np.argsort(-tot, kind='stable')[:kpts]
        S = np.sort(S)
        kpts = len(S)
        gpos = {int(v): k for k, v in enumerate(S)}
        # tangent gradients
        # precompute dots for surviving pairs
        dim = kpts * d
        R = np.zeros((dim, dim), dtype=float)
        # normalize weights to avoid scale issues
        wscale = np.median(np.abs(Sw)) if len(Sw) > 0 else 1.0
        if not np.isfinite(wscale) or wscale <= 1e-18:
            wscale = 1.0
        for k in range(len(SI)):
            i = int(SI[k]); j = int(SJ[k])
            ww = float(Sw[k]) / wscale
            if ww < 0:
                ww = 0.0
            if ww == 0.0:
                continue
            # clamp huge
            if ww > 1e4:
                ww = 1e4
            xi = curX[i]; xj = curX[j]
            cij = float(xi @ xj)
            gi = xj - cij * xi
            gj = xi - cij * xj
            i_in = i in gpos
            j_in = j in gpos
            if i_in and j_in:
                a = gpos[i] * d
                b = gpos[j] * d
                R[a:a+d, a:a+d] += ww * np.outer(gi, gi)
                R[b:b+d, b:b+d] += ww * np.outer(gj, gj)
                R[a:a+d, b:b+d] += ww * np.outer(gi, gj)
                R[b:b+d, a:a+d] += ww * np.outer(gj, gi)
            elif i_in:
                a = gpos[i] * d
                R[a:a+d, a:a+d] += ww * np.outer(gi, gi)
            elif j_in:
                b = gpos[j] * d
                R[b:b+d, b:b+d] += ww * np.outer(gj, gj)
        # regularize slightly
        R = 0.5 * (R + R.T)
        try:
            vals_e, vecs_e = np.linalg.eigh(R)
        except Exception:
            continue
        # take lowest 4 eigenvectors
        neig = min(4, dim)
        trials = []
        for ei in range(neig):
            v = vecs_e[:, ei].copy()
            D0 = v.reshape(kpts, d).copy()
            for a2, idx in enumerate(S):
                xi = curX[int(idx)]
                D0[a2] -= float(D0[a2] @ xi) * xi
            nrm = float(np.linalg.norm(D0))
            if nrm < 1e-12 or not np.isfinite(nrm):
                continue
            D0 /= nrm
            for alpha in (0.01, 0.03, 0.1, -0.01, -0.03, -0.1):
                Xt = curX.copy()
                Xt[S] += alpha * D0
                nr = np.linalg.norm(Xt[S], axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Xt[S] /= nr
                vv = _max_cos(Xt)
                trials.append((vv, alpha, ei, Xt))
        if not trials:
            continue
        trials.sort(key=lambda z: z[0])
        # polish top 3
        moved = False
        cand_curs = []
        for tt in trials[:3]:
            if time.time() - t0 > DEADLINE:
                break
            Xt = tt[3]
            # restricted polish then full polish
            Xp, _ = _lp_polish(Xt, 0.9, excluded=release_set)
            Xp, vp = _lp_polish(Xp, 0.9, excluded=None)
            deg, _ = _contacts_deg(Xp, 1e-9)
            seq = tuple(sorted(deg.tolist()))
            seq_changed = (seq != bestSeq)
            cand_curs.append((vp, Xp, seq, seq_changed))
            if vp < bestEverV - 1e-12:
                bestEverV = vp
                bestEverX = Xp.copy()
                bestSeq = seq
                curX, curV = Xp.copy(), vp
                moved = True
                break
        if moved:
            continue
        # no strict improvement: move to topology-changed candidate if not much worse
        if cand_curs:
            # prefer changed sequence with smallest vp
            cand_curs.sort(key=lambda z: z[0])
            best_changed = None
            for (vp, Xp, seq, ch) in cand_curs:
                if ch and vp < bestEverV + 0.002:
                    best_changed = (vp, Xp)
                    break
            if best_changed is not None:
                curX, curV = best_changed[1].copy(), best_changed[0]
                # keep bestEver unchanged but continue from new basin
            else:
                # take least-worse polished if it improves cur
                if cand_curs[0][0] < curV - 5e-9:
                    curX, curV = cand_curs[0][1].copy(), cand_curs[0][0]
                else:
                    # force unpolished best trial as new cur to change basin (still renormalized)
                    # only if it changes contacts meaningfully
                    Xt0 = trials[0][3]
                    degt, _ = _contacts_deg(Xt0, 1e-9)
                    seqt = tuple(sorted(degt.tolist()))
                    if seqt != tuple(sorted(_contacts_deg(curX, 1e-9)[0].tolist())):
                        # light polish to avoid drift
                        Xp2, vp2 = _lp_polish(Xt0, 0.7, excluded=None)
                        curX, curV = Xp2.copy(), vp2
                    else:
                        # try next eigen direction trial that differs
                        found = False
                        for (vv, al, ei, Xt_) in trials:
                            deg_, _ = _contacts_deg(Xt_, 1e-9)
                            seq_ = tuple(sorted(deg_.tolist()))
                            if seq_ != bestSeq:
                                Xp2, vp2 = _lp_polish(Xt_, 0.7, excluded=None)
                                curX, curV = Xp2.copy(), vp2
                                found = True
                                break
                        if not found:
                            break
            # update bestEver if cur improved (already checked) else continue loop
            if curV < bestEverV:
                bestEverV = curV
                bestEverX = curX.copy()
                degb, _ = _contacts_deg(bestEverX, 1e-9)
                bestSeq = tuple(sorted(degb.tolist()))
        else:
            break
    # final squeeze of best checkpoint
    remain = max(0.5, min(8.0, LIMIT - 1.0 - (time.time() - t0)))
    if remain > 0.4:
        try:
            bestEverX, bestEverV = _lp_polish(bestEverX, remain)
        except Exception:
            pass
    bestEverX = bestEverX / np.linalg.norm(bestEverX, axis=1, keepdims=True)
    return bestEverX
