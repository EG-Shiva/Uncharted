import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3500
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _forced_single(x, radius, delta, fI, fJ):
    n, d = x.shape
    G = x @ x.T
    iuI, iuJ = np.triu_indices(n, 1)
    cfull = G[iuI, iuJ]
    m = float(cfull.max())
    thresh = m - max(20.0 * radius, 2e-3)
    sel = cfull > thresh
    # map forced pairs to set for exclusion
    fset = set(zip(np.asarray(fI).tolist(), np.asarray(fJ).tolist()))
    # build active list excluding forced
    I_all = iuI[sel]; J_all = iuJ[sel]; c_all = cfull[sel]
    mask_keep = np.ones(len(I_all), dtype=bool)
    # quick exclusion via dictionary
    if len(fset) > 0:
        for k in range(len(I_all)):
            a = int(I_all[k]); b = int(J_all[k])
            if a > b:
                a, b = b, a
            if (a, b) in fset:
                mask_keep[k] = False
    I = I_all[mask_keep]; J = J_all[mask_keep]; ca = c_all[mask_keep]
    # cap active
    if len(I) > 3000:
        part = np.argpartition(ca, -3000)[-3000:]
        I = I[part]; J = J[part]; ca = ca[part]
    fI = np.asarray(fI, dtype=np.int64); fJ = np.asarray(fJ, dtype=np.int64)
    # forced cos values
    cf = G[fI, fJ] if len(fI) > 0 else np.zeros(0)
    p1 = len(I); p2 = len(fI); p = p1 + p2
    if p == 0:
        return None
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.empty(p * (2 * d + 1), dtype=np.int64)
    vals = np.empty(p * (2 * d + 1), dtype=np.float64)
    Cb = cols.reshape(p, 2 * d + 1); Vb = vals.reshape(p, 2 * d + 1)
    ar = np.arange(d)
    if p1 > 0:
        Cb[:p1, :d] = I[:, None] * d + ar
        Cb[:p1, d:2 * d] = J[:, None] * d + ar
        Cb[:p1, 2 * d] = n * d
        Vb[:p1, :d] = x[J]
        Vb[:p1, d:2 * d] = x[I]
        Vb[:p1, 2 * d] = -1.0
    if p2 > 0:
        Cb[p1:, :d] = fI[:, None] * d + ar
        Cb[p1:, d:2 * d] = fJ[:, None] * d + ar
        Cb[p1:, 2 * d] = n * d
        Vb[p1:, :d] = x[fJ]
        Vb[p1:, d:2 * d] = x[fI]
        Vb[p1:, 2 * d] = 0.0
    b_ub = np.empty(p)
    if p1 > 0:
        b_ub[:p1] = -ca
    if p2 > 0:
        b_ub[p1:] = (m - delta) - cf
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    y = x + res.x[:-1].reshape(n, d)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    yn[yn == 0] = 1.0
    y = y / yn
    return y

def _lse_relax(x, steps, tau0, tau1, lr):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        G = x @ x.T
        m = float(G.max())
        # exp weights, exclude diagonal
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0:
            break
        W = E / S  # rows sum to ~ small; grad = W @ X
        Gr = W @ x  # (n,d) matrix product, vectorised
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        # clip step for stability
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        # avoid huge steps
        # scale lr adaptively: limit max move to 0.02
        maxmove = float((lr * gn).max()) if gn.size else 0.0
        step_lr = lr
        if maxmove > 0.02:
            step_lr = lr * (0.02 / maxmove)
        x = x - step_lr * Gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a; b0 = _max_cos(a)
    x_best = x0.copy(); best = float(b0)
    n, d = x_best.shape
    deltas = [3e-4, 1e-3, 3e-3]
    it = 0
    # cache contact edges of best; refresh when best improves
    while True:
        if time.time() - t0 > 148:
            break
        # compute contacts of x_best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = np.sum(G > m - 1e-9, axis=1)
        order_hub = np.argsort(-cnt)
        hubs = order_hub[:12]
        iuI, iuJ = np.triu_indices(n, 1)
        # use full triu for edge list sorted by cosine
        cfull = (x_best @ x_best.T)[iuI, iuJ]
        # contact edge list (tight)
        cmask = cfull > m - 1e-9
        EI = iuI[cmask]; EJ = iuJ[cmask]; EC = cfull[cmask]
        srt = np.argsort(-EC)
        EI = EI[srt]; EJ = EJ[srt]
        ne = len(EI)
        if ne == 0:
            break
        # choose forced set by rotation
        delta = deltas[it % 3]
        mode = it % 3
        fI = []; fJ = []
        if mode == 0:
            # matching: greedy with rotating offset
            used = np.zeros(n, dtype=bool)
            start = (it * 37) % ne
            picked = 0
            for k in range(ne):
                idx = (start + k) % ne
                a = int(EI[idx]); b = int(EJ[idx])
                if not used[a] and not used[b]:
                    # canonical order
                    if a > b: a, b = b, a
                    fI.append(a); fJ.append(b)
                    used[a] = True; used[b] = True
                    picked += 1
                    if picked >= 40:
                        break
        elif mode == 1:
            # hub-incident edges
            h = int(hubs[(it // 3) % len(hubs)])
            # collect edges incident to h (from contact list)
            for k in range(ne):
                a = int(EI[k]); b = int(EJ[k])
                if a == h or b == h:
                    aa, bb = (a, b) if a < b else (b, a)
                    fI.append(aa); fJ.append(bb)
                    if len(fI) >= 20:
                        break
            # pad with matching edges disjoint from these to reach 30
            used = np.zeros(n, dtype=bool)
            used[h] = True
            for v in fI + fJ:
                pass
            # mark vertices already used (approx: mark hub neighbours)
            start = (it * 53) % ne
            for k in range(ne):
                if len(fI) >= 30:
                    break
                idx = (start + k) % ne
                a = int(EI[idx]); b = int(EJ[idx])
                if a == h or b == h:
                    continue
                if not used[a] and not used[b]:
                    aa, bb = (a, b) if a < b else (b, a)
                    fI.append(aa); fJ.append(bb)
                    used[a] = True; used[b] = True
        else:
            # strided subset of top edges
            stride = 7
            start = (it * 11) % stride
            for k in range(start, min(ne, 400), stride):
                a = int(EI[k]); b = int(EJ[k])
                aa, bb = (a, b) if a < b else (b, a)
                fI.append(aa); fJ.append(bb)
                if len(fI) >= 35:
                    break
        if len(fI) == 0:
            it += 1
            continue
        x = x_best.copy()
        # forced LP step with trust region
        radius_f = 0.025
        try:
            yf = _forced_single(x, radius_f, float(delta), np.array(fI), np.array(fJ))
        except Exception:
            yf = None
        if yf is None:
            # fallback: small kick and LSE only
            yf = x + rng.standard_normal(size=x.shape) * 0.015
            yf /= np.linalg.norm(yf, axis=1, keepdims=True)
        # fast vectorised LogSumExp tangential descent
        try:
            yf = _lse_relax(yf, steps=250, tau0=0.008, tau1=0.0025, lr=0.9)
        except Exception:
            pass
        remain = 148 - (time.time() - t0)
        if remain < 1.2:
            break
        budget = min(4.0, remain - 0.2)
        try:
            y = _lp_polish(yf, seconds=budget, init_radius=0.006)
        except Exception:
            it += 1
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            best = v; x_best = y
        it += 1
    remain = 178 - (time.time() - t0)
    if remain > 2:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain - 0.5, 20), init_radius=2e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
