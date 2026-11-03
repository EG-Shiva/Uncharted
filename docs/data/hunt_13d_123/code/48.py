import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_free(V0, steps, b0, b1, l0, l1, mu=0.85):
    V = V0.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = V @ V.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ V
        dot = np.sum(grad * V, axis=1, keepdims=True)
        gt = (grad - dot * V) / row
        # momentum-less simple: use velocity
        if s == 0:
            vel = np.zeros_like(V)
        vel = mu * vel + gt
        V = V - lr * vel
        V = _norm(V)
    return V

def _repack(X0, free_idx, frozen_idx, steps=400, b0=30.0, b1=500.0, l0=0.008, l1=0.0008, mu=0.85):
    X = X0.copy()
    mf = len(free_idx)
    if mf == 0:
        return X
    vel = np.zeros((mf, X.shape[1]))
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        F = X[free_idx]
        G = F @ X.T  # mf x n
        # mask self dots (free point vs itself in full set)
        # self columns correspond to positions free_idx in X
        # set them to very low so they don't dominate
        for ii, gi in enumerate(free_idx):
            G[ii, gi] = -10.0
        gm = float(G.max())
        W = np.exp(beta * (G - gm))  # mf x n
        # gradient w.r.t F: W @ X
        grad = W @ X
        # tangential projection per free point
        dot = np.sum(grad * F, axis=1, keepdims=True)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt = (grad - dot * F) / row
        vel = mu * vel + gt
        F = F - lr * vel
        F = _norm(F)
        X[free_idx] = F
    return X

def _refine_hole(c0, Y, steps=350, b0=40.0, b1=600.0, l0=0.05, l1=0.005):
    c = c0.copy() / (np.linalg.norm(c0) + 1e-12)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        d = Y @ c  # m
        dm = float(d.max())
        w = np.exp(beta * (d - dm))
        # weighted sum
        g = w @ Y  # d-dim
        # tangential
        gt = g - float(g @ c) * c
        nrm = float(np.linalg.norm(gt)) + 1e-12
        gt = gt / (w.sum() + 1e-12)
        c = c - lr * gt
        c = c / (np.linalg.norm(c) + 1e-12)
    return c

def _lp_polish(x, seconds, max_pairs=800):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if seconds - (time.time() - start) < 0.4:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        tol = max(12.0 * radius, 2e-9)
        sel = c > m - tol
        if int(sel.sum()) > max_pairs:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            tight = c > m - 2e-9
            keep[tight] = True
            cnt = int(keep.sum())
            for ii in idx:
                if cnt >= max_pairs:
                    break
                if not keep[ii]:
                    keep[ii] = True
                    cnt += 1
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _deep_holes(Y, rng, n_total=200000, n_pre=40, chunk=20000):
    d = Y.shape[1]
    scores = np.empty(n_total, dtype=float)
    # generate and score in chunks, store candidates of pre-selection?
    # store all candidates would be 200k*13=2.6M floats ok, keep them
    S_all = None
    # to save memory, generate full matrix at once (20MB)
    S = _norm(rng.randn(n_total, d))
    pos = 0
    # chunked scoring
    for a in range(0, n_total, chunk):
        b = min(a + chunk, n_total)
        dots = S[a:b] @ Y.T
        scores[a:b] = dots.max(axis=1)
    # pick n_pre smallest
    part = np.argpartition(scores, n_pre - 1)[:n_pre]
    # sort them
    order = part[np.argsort(scores[part])]
    return S[order], scores[order]

def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    rng = np.random.RandomState(12345)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(n, d)
    xb = np.array(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = _norm(xb)
    best = _mc(xb)
    xbest = xb.copy()

    # identify rattler and hubs
    G0 = xbest @ xbest.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    deg = (G0 > m0 - 1e-7).sum(axis=1)
    rattler = int(np.argmin(deg))
    hub_order = np.argsort(-deg)
    # build deletion triples: rattler + 2 hubs, varying hubs
    triples = []
    hubs = [h for h in hub_order if h != rattler][:10]
    # different hub/rattler deletion triples
    if len(hubs) >= 2:
        triples.append((rattler, int(hubs[0]), int(hubs[1])))
    if len(hubs) >= 4:
        triples.append((rattler, int(hubs[2]), int(hubs[3])))
    if len(hubs) >= 6:
        triples.append((rattler, int(hubs[4]), int(hubs[5])))
    if len(hubs) >= 8:
        triples.append((rattler, int(hubs[0]), int(hubs[5])))
    if len(hubs) >= 3:
        triples.append((rattler, int(hubs[1]), int(hubs[4])))
    # also a triple without rattler to diversify (two hubs + random low-degree)
    low_order = np.argsort(deg)
    triples.append((int(hubs[0]), int(hubs[1]), int(low_order[1])))

    for ti, triple in enumerate(triples):
        if time.time() - t0 > 125:
            break
        triple = tuple(int(t) for t in triple)
        keep = np.ones(n, dtype=bool)
        for t in triple:
            if 0 <= t < n:
                keep[t] = False
        Y = xbest[keep].copy()  # ~120 pts
        # optimize reduced code quickly
        # scale steps with remaining time
        Y = _anneal_free(Y, steps=650, b0=25.0, b1=500.0, l0=0.009, l1=0.0008, mu=0.85)
        # locate deepest holes with >=200k candidates
        cand, sc = _deep_holes(Y, rng, n_total=200000, n_pre=40, chunk=20000)
        # local minimization of each candidate's max dot vs frozen Y
        refined = np.empty_like(cand)
        rscores = np.empty(len(cand))
        for i in range(len(cand)):
            rc = _refine_hole(cand[i], Y, steps=300)
            refined[i] = rc
            rscores[i] = float(Y @ rc.reshape(-1, 1).ravel() if False else (Y @ rc).max())
        order = np.argsort(rscores)
        refined = refined[order]
        rscores = rscores[order]
        # greedily pick 3 distinct holes (sep 0.70)
        picks = []
        for i in range(len(refined)):
            ok = True
            for pj in picks:
                if float(refined[i] @ refined[pj]) > 0.70:
                    ok = False
                    break
            if ok:
                picks.append(i)
                if len(picks) == 3:
                    break
        if len(picks) < 3:
            picks = list(range(min(3, len(refined))))
        new_pts = refined[picks]
        # build full code
        Xnew = np.vstack([Y, new_pts])
        Xnew = _norm(Xnew)
        nnew = Xnew.shape[0]
        # neighbourhood: reinserted indices + neighbours with cosine >0.10
        reins_idx = list(range(Y.shape[0], nnew))
        # if nnew !=123 due to rounding, pad/trim (should be 123)
        if nnew != n:
            # adjust by trimming or padding (should not happen)
            if nnew > n:
                Xnew = Xnew[:n]
                nnew = n
                reins_idx = reins_idx[:n - Y.shape[0]]
            else:
                extra = _norm(rng.randn(n - nnew, d))
                Xnew = np.vstack([Xnew, extra])
                nnew = n
        Gfull = Xnew @ Xnew.T
        free_mask = np.zeros(nnew, dtype=bool)
        free_mask[reins_idx] = True
        for ri in reins_idx:
            neigh = np.where(Gfull[ri] > 0.10)[0]
            free_mask[neigh] = True
        # exclude frozen? free_idx list
        free_idx = np.where(free_mask)[0].tolist()
        # repack neighbourhood only
        Xnew = _repack(Xnew, free_idx, None, steps=450, b0=30.0, b1=550.0, l0=0.008, l1=0.0007)
        v_before_lp = _mc(Xnew)
        # short global LP polish (time-capped)
        remain = TIME_BUDGET - (time.time() - t0)
        if remain < 12:
            # still try tiny polish but skip if almost out of time
            lp_sec = max(1.0, remain - 6)
        else:
            lp_sec = 4.0
        # only polish if promising (within 0.004 of best) to save time
        if v_before_lp < best + 0.004:
            Xnew = _lp_polish(Xnew, seconds=lp_sec, max_pairs=800)
        v = _mc(Xnew)
        if v < best:
            best = v
            xbest = Xnew.copy()

    remain = TIME_BUDGET - (time.time() - t0)
    if remain > 7:
        xbest = _lp_polish(xbest, seconds=min(remain - 2, 25.0), max_pairs=900)
    xbest = _norm(xbest)
    return xbest
