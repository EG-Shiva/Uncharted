import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=2e-3, cap=1400):
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + cd, J[:, None]*d + cd,
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _rowmax_and_deg(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    m = float(rm.max())
    deg = np.sum(G >= m - tol, axis=1) - 0  # diag is -2 so not counted
    return rm, deg, m, G

def _best_void(x, rng, n_samples=6000, refine=25):
    n, d = x.shape
    cand = rng.normal(size=(n_samples, d))
    cand = _norm(cand)
    scores = (cand @ x.T).max(axis=1)
    i0 = int(np.argmin(scores))
    v = cand[i0].copy()
    # refine: move away from neighbours
    for _ in range(refine):
        dots = x @ v
        dm = float(dots.max())
        w = np.exp(80.0 * (dots - dm))
        w = w / (w.sum() + 1e-30)
        g = w @ x
        gt = g - float(g @ v) * v
        nr = float(np.linalg.norm(gt))
        if nr < 1e-12:
            break
        gt /= nr
        step = 0.06
        v = v - step * gt
        v = v / max(np.linalg.norm(v), 1e-12)
    return v

def _split_point(x, s_idx, rng, eps=0.20):
    # split x[s] into two points offset along void-directed tangent
    n, d = x.shape
    xs = x[s_idx].copy()
    # direction away from close neighbours in tangent plane
    G = x @ xs
    G[s_idx] = -2.0
    # take ~10 nearest neighbours
    nn = np.argsort(-G)[-20:][::-1]  # not needed; get largest
    nn = np.argsort(-G)[:12]
    # exclude self already; nn includes closest
    C = x[nn].sum(axis=0)
    # component orthogonal to xs
    dvec = C - float(C @ xs) * xs
    # we want to move away, so d = -dvec direction; if small use random
    nr = float(np.linalg.norm(dvec))
    if nr < 1e-8:
        r = rng.normal(size=4)
        r = r - float(r @ xs) * xs
        nr2 = float(np.linalg.norm(r))
        if nr2 < 1e-12:
            r = np.array([1.0, 0, 0, 0])
            r = r - float(r @ xs) * xs
            nr2 = float(np.linalg.norm(r))
        dvec = r / nr2
    else:
        dvec = -dvec / nr
    # two offspring symmetric
    p1 = xs + eps * dvec
    p2 = xs - eps * dvec
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    y = np.delete(x, s_idx, axis=0)
    y = np.vstack([y, p1[None, :], p2[None, :]])
    return y

def _removal_candidates_164(x, topk=3):
    # for N=164/165, find indices whose removal minimizes remaining max
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iu0, iu1]
    order = np.argsort(-c)
    # consider first 40 pairs
    K = min(60, len(order))
    topI = iu0[order[:K]]
    topJ = iu1[order[:K]]
    topC = c[order[:K]]
    # for each i, removal max = first topC whose pair avoids i
    rm_val = np.empty(n)
    for i in range(n):
        v = topC[0]
        for k in range(K):
            if topI[k] != i and topJ[k] != i:
                v = topC[k]
                break
        rm_val[i] = v
    # smallest rm_val are best to remove
    idx = np.argsort(rm_val)[:topk]
    return idx, rm_val

def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(7)
    n4, d = 163, 4
    # load incumbent
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (n4, d):
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                c = _norm(c)
                v = _max_cos(c)
                if v < best_val:
                    best_val = v
                    best = c.copy()
        except Exception:
            pass
    if best is None:
        z = rng.normal(size=(n4, d))
        best = _norm(z)
        best_val = _max_cos(best)
    archive = best.copy()
    arch_val = best_val

    def time_left():
        return HARD - (time.time() - t0)

    def try_update(x):
        nonlocal best, best_val, archive, arch_val
        v = _max_cos(x)
        if v < best_val - 1e-15:
            best = x.copy()
            best_val = v
        if v < arch_val - 1e-15:
            archive = x.copy()
            arch_val = v
        return v

    # ---------- deletion path ----------
    # choose removal set: high rowmax + rattlers/low degree
    try:
        rm, deg, m0, G0 = _rowmax_and_deg(best)
        order_rm = np.argsort(-rm)
        rattlers = np.where(deg == 0)[0]
        del_list = []
        for i in order_rm[:4]:
            if int(i) not in del_list:
                del_list.append(int(i))
        for i in rattlers[:2]:
            if int(i) not in del_list:
                del_list.append(int(i))
        # also add a hub (max degree)
        hub = int(np.argmax(deg))
        if hub not in del_list:
            del_list.append(hub)
        del_list = del_list[:4]
    except Exception:
        del_list = [0, 1, 2, 3]

    for di, rem in enumerate(del_list):
        if time_left() < 55:
            break
        try:
            cur = best.copy()
            # recompute rm on current best (it may have improved)
            rm2, deg2, m2, _ = _rowmax_and_deg(cur)
            # map rem to valid index (clamp)
            ridx = int(rem) % 163
            x162 = np.delete(cur, ridx, axis=0)
            # repolish 162 substantially but cheaply
            x162 = _smooth(x162, iters=220, T0=0.012, T1=0.0012, lr0=0.007, lr1=0.001)
            if time_left() > 40:
                x162 = _lp_polish(x162, seconds=min(1.2, time_left()-38.0))
            else:
                x162 = _lp_polish(x162, seconds=0.5)
            v162 = _max_cos(x162)
            # two insertion branches
            # (a) void insertion
            if time_left() > 30:
                v = _best_void(x162, rng, n_samples=5000, refine=20)
                x163a = np.vstack([x162, v[None, :]])
                x163a = _smooth(x163a, iters=200, T0=0.012, T1=0.001, lr0=0.007, lr1=0.0009)
                x163a = _lp_polish(x163a, seconds=min(1.2, max(0.4, time_left()-28.0)))
                try_update(x163a)
            # (b) split overloaded point
            if time_left() > 28:
                rm162, deg162, _, _ = _rowmax_and_deg(x162)
                # overloaded = max rm, tie by max deg
                s_idx = int(np.argmax(rm162))
                eps = 0.16 + 0.05 * (di % 2)
                x163b = _split_point(x162, s_idx, rng, eps=eps)
                # x163b has 163 points (162-1+2)
                x163b = _smooth(x163b, iters=200, T0=0.014, T1=0.001, lr0=0.008, lr1=0.0009)
                x163b = _lp_polish(x163b, seconds=min(1.2, max(0.4, time_left()-26.0)))
                try_update(x163b)
        except Exception:
            continue

    # ---------- addition path ----------
    n_add_cycles = 3
    for ac in range(n_add_cycles):
        if time_left() < 32:
            break
        try:
            cur = best.copy()
            # insert 1 void -> 164; on second cycle insert 2 -> 165
            want165 = (ac == 1)
            v1 = _best_void(cur, rng, n_samples=5000, refine=20)
            xbig = np.vstack([cur, v1[None, :]])
            if want165:
                v2 = _best_void(xbig, rng, n_samples=4000, refine=15)
                xbig = np.vstack([xbig, v2[None, :]])
            # polish crowded problem
            xbig = _smooth(xbig, iters=180, T0=0.010, T1=0.001, lr0=0.006, lr1=0.0009)
            xbig = _lp_polish(xbig, seconds=min(1.0, max(0.4, time_left()-26.0)))
            # choose deletions back to 163
            nbig = xbig.shape[0]
            ndiff = nbig - 163
            if ndiff == 1:
                cand_idx, _ = _removal_candidates_164(xbig, topk=3)
                for ci in cand_idx:
                    if time_left() < 26:
                        break
                    x163 = np.delete(xbig, int(ci), axis=0)
                    x163 = _smooth(x163, iters=160, T0=0.010, T1=0.001, lr0=0.006, lr1=0.0009)
                    x163 = _lp_polish(x163, seconds=min(1.0, max(0.4, time_left()-24.0)))
                    try_update(x163)
                    # only try first two to save time
                    break if False else None
                    # limit: try at most 2
                    if int(ci) != int(cand_idx[0]):
                        break
            else:  # 165 -> need remove 2; try greedy best pair
                cand_idx, rmv = _removal_candidates_164(xbig, topk=5)
                # try removing top-2 combination among cand set by brute force on max
                G = xbig @ xbig.T
                np.fill_diagonal(G, -2.0)
                best_pair = None
                best_pv = 1e9
                cl = list(map(int, cand_idx))
                for a in range(len(cl)):
                    for b in range(a+1, len(cl)):
                        i, j = cl[a], cl[b]
                        # max without i,j
                        # fast: mask
                        tmp = G.copy()
                        tmp[i, :] = -2.0; tmp[:, i] = -2.0
                        tmp[j, :] = -2.0; tmp[:, j] = -2.0
                        pv = float(tmp.max())
                        if pv < best_pv:
                            best_pv = pv
                            best_pair = (i, j)
                if best_pair is None:
                    best_pair = (cl[0], cl[1])
                keep = [k for k in range(nbig) if k not in best_pair]
                x163 = xbig[keep]
                x163 = _smooth(x163, iters=160, T0=0.010, T1=0.001, lr0=0.006, lr1=0.0009)
                x163 = _lp_polish(x163, seconds=min(1.0, max(0.4, time_left()-24.0)))
                try_update(x163)
        except Exception:
            continue

    # final polish of best
    try:
        left = time_left() - 1.0
        if left > 2.0:
            best = _lp_polish(best, seconds=min(left, 22.0), r0=1.5e-3)
            try_update(best)
    except Exception:
        pass
    # return best verified
    out = best / np.linalg.norm(best, axis=1, keepdims=True)
    out = np.nan_to_num(out)
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    # safety: if somehow archive better, use it
    try:
        if _max_cos(archive) < _max_cos(out):
            out = archive
    except Exception:
        pass
    return out
