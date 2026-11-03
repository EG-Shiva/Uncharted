import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _gd(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip huge steps
        nm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-300
        big = nm > 2.0
        if np.any(big):
            Gt = np.where(big, Gt / nm * 2.0, Gt)
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-300
    return X

def _refine_hole(v, others, steps=250, lr0=0.08, lr1=0.002):
    v = v / np.linalg.norm(v)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    best_v = v.copy()
    dts = others @ v
    best = float(dts.max())
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dts = others @ v
        j = int(np.argmax(dts))
        dj = float(dts[j])
        if dj < best:
            best = dj
            best_v = v.copy()
        g = others[j] - dj * v
        v = v - lr * g
        nrm = np.linalg.norm(v)
        if nrm == 0 or not np.isfinite(nrm):
            v = best_v.copy()
            continue
        v /= nrm
    dts = others @ v
    cur = float(dts.max())
    if cur < best:
        return v, cur
    return best_v, best

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            order = np.argsort(-c[sel])[:2200]
            I, J = I[order], J[order]
            csel = c[sel][order]
            p = 2200
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if _max_cos(x) > best + 1e-12:
                x = best_x.copy()
    return best_x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    phase1_end = t0 + 140.0
    rng = np.random.default_rng(0)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (95, 13):
            cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (95, 13):
            cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    except Exception:
        pass
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = 95, 13
    results = [(best_val, Xbest.copy())]
    Xcur = Xbest.copy()
    tabu_vecs = []  # list of deleted vectors
    it = 0
    NSAMP = 50000
    NHOLE = 20
    while time.time() < phase1_end and time.time() < deadline - 35:
        it += 1
        base = Xbest  # star search from best
        rm = _row_max(base)
        if it == 1:
            # delete rattler: minimal row max
            del_idx = int(np.argmin(rm))
        else:
            # delete high-contact hub not in tabu
            order = np.argsort(-rm)
            del_idx = int(order[0])
            # check tabu by vector distance
            found = False
            for cand_idx in order:
                ci = int(cand_idx)
                v = base[ci]
                skip = False
                for tv in tabu_vecs:
                    if abs(float(v @ tv)) > 0.995:
                        skip = True
                        break
                if not skip:
                    del_idx = ci
                    found = True
                    break
            if not found:
                del_idx = int(order[(it - 1) % n])
        del_vec = base[del_idx].copy()
        Y = np.delete(base, del_idx, axis=0)  # 94 x 13
        # LP/softmax-polish remaining 94: fast softmax anneal (gain slack)
        # vary temperature slightly per iteration for diversity
        if it % 3 == 0:
            Yp = _gd(Y.copy(), 500, 0.010, 0.0007, 0.07, 0.005, mu=0.85)
        elif it % 3 == 1:
            Yp = _gd(Y.copy(), 450, 0.007, 0.0005, 0.06, 0.004, mu=0.8)
        else:
            Yp = _gd(Y.copy(), 550, 0.012, 0.0008, 0.08, 0.005, mu=0.85)
        # largest empty cap: sample 50000 unit vectors
        C = rng.standard_normal((NSAMP, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-300
        dots = C @ Yp.T
        md = dots.max(axis=1)
        # best 20
        part = np.argpartition(md, NHOLE)[:NHOLE]
        # sort them by md
        part = part[np.argsort(md[part])]
        best_h = None
        best_hv = 1e9
        for k in part:
            v0 = C[int(k)]
            vr, vv = _refine_hole(v0, Yp, steps=250, lr0=0.06, lr1=0.002)
            if vv < best_hv:
                best_hv = vv
                best_h = vr
            if time.time() > phase1_end:
                break
        if best_h is None:
            continue
        Xnew = np.vstack([Yp, best_h.reshape(1, d)])
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True) + 1e-300
        # relax all 95
        if it % 2 == 0:
            Xnew = _gd(Xnew, 650, 0.008, 0.0005, 0.06, 0.004, mu=0.82)
        else:
            Xnew = _gd(Xnew, 650, 0.010, 0.0006, 0.07, 0.005, mu=0.85)
        vnew = _max_cos(Xnew)
        results.append((vnew, Xnew.copy()))
        # update tabu
        tabu_vecs.append(del_vec / np.linalg.norm(del_vec))
        if len(tabu_vecs) > 25:
            tabu_vecs.pop(0)
        if vnew < best_val:
            best_val = vnew
            Xbest = Xnew.copy()
            Xcur = Xnew.copy()
        # safety: if too many results keep only top few to save memory (keep arrays)
        if len(results) > 60:
            results.sort(key=lambda t: t[0])
            results = results[:30]
        # quick time check
        if time.time() >= phase1_end:
            break
    # rank distinct 95-point results
    results.sort(key=lambda t: t[0])
    # deduplicate roughly by value
    uniq = []
    for v, X in results:
        dup = False
        for vu, _ in uniq:
            if abs(vu - v) < 1e-12:
                dup = True
                break
        if not dup:
            uniq.append((v, X))
        if len(uniq) >= 10:
            break
    if uniq:
        best_val_u, Xbest_u = uniq[0]
        if best_val_u < best_val:
            best_val = best_val_u
            Xbest = Xbest_u.copy()
        else:
            # ensure Xbest is top
            Xbest = uniq[0][1].copy()
            best_val = uniq[0][0]
    # last 30 seconds LP-polish the best one
    remain = deadline - time.time() - 1.0
    if remain > 2:
        budget = min(30.0, remain)
        # brief cold softmax before LP if time allows
        if remain > 12:
            Xbest = _gd(Xbest.copy(), 500, 0.002, 0.0002, 0.02, 0.003, mu=0.7)
        remain2 = deadline - time.time() - 0.8
        if remain2 > 2:
            Xbest = _lp_polish(Xbest, seconds=min(30.0, remain2), t_deadline=deadline - 0.6)
            Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
