import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    mx = g.max()
    tight = g > mx - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return mx, cnt, iu

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _patch_hinge_opt(x0, Dset, t_target, steps, lr0, lr1, rng=None):
    # optimize only points in Dset with sum relu(dot-t)^2 over pairs involving Dset
    x = x0.copy()
    n, d = x.shape
    D = np.asarray(Dset, dtype=int)
    k = len(D)
    isD = np.zeros(n, dtype=bool)
    isD[D] = True
    m1 = np.zeros((k, d)); v1 = np.zeros((k, d))
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        # hinge matrix for pairs involving D
        # compute grad for D points
        # H[i,j] = 2*relu(G[i,j]-t) for i in D, j!=i
        GD = G[D]  # k x n
        H = np.maximum(GD - t_target, 0.0) * 2.0  # k x n
        # zero diagonal self (not present since GD includes self but self dot=1 -> huge; must exclude)
        for a, idx in enumerate(D):
            H[a, idx] = 0.0
        # grad_a = sum_j H[a,j] * x[j] ; plus symmetric term when both in D:
        # pair (a in D, b in D) contributes to both, our H covers a-side; need also b-side? No, grad for a already includes H[a,b]*x[b].
        # That's sufficient (factor 2? loss counts each unordered pair once, gradient exactly H[a,j]*x[j]).
        # To count once, halve D-D? Keep as is (slightly stronger) - fine.
        grad = H @ x  # k x d
        # project to tangent
        xd = x[D]
        dot = (grad * xd).sum(axis=1, keepdims=True)
        gt = grad - dot * xd
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        xd2 = xd - lr * step
        xd2 /= np.maximum(np.linalg.norm(xd2, axis=1, keepdims=True), 1e-12)
        x[D] = xd2
    return x

def _find_void(Xcur, rng, ncand=20000, refine=250):
    # Xcur: (m,4) unit
    cand = rng.normal(size=(ncand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xcur.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    best = cand[bi].copy()
    bestv = float(md[bi])
    sig = 0.05
    for it in range(refine):
        # decay sigma
        s = sig * (0.98 ** it) + 0.002
        prop = best + rng.normal(0, s, size=4)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        v = float((prop @ Xcur.T).max())
        if v < bestv:
            bestv = v
            best = prop
            # intensify around improvement
        # occasional random restart of sigma handled by decay
    return best

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    ks = [8, 12, 16]  # delete sizes (1 + 7,11,15)
    deltas = [1e-4, 3e-4, 1e-3]
    it = 0
    # precompute hub ordering each iter
    while time.time() - t0 < 138.0:
        it += 1
        if time.time() - t0 > HARD - 32:
            break
        mx, cnt, _ = _contacts(x_best, tol=1e-7)
        ratt = np.where(cnt == 0)[0]
        order = np.argsort(-cnt)
        hubs = order[:12]
        # choose centre
        mod = it % 4
        if mod == 1 and len(ratt) > 0:
            c = int(ratt[(it // 4) % len(ratt)])
        elif mod == 2:
            c = int(hubs[it % len(hubs)])
        elif mod == 3:
            c = int(rng.integers(n))
        else:
            # random hub-biased: pick among top-40 contacts randomly
            top40 = order[:40]
            c = int(top40[rng.integers(len(top40))])
        k = ks[it % 3]
        delta = deltas[(it // 2) % 3]
        t_target = best_val - delta
        # deletion set: k points with largest cosine to c
        row = (x_best @ x_best[c])
        idx = np.argsort(-row)[:k]
        # ensure centre included (it is, dot=1 max)
        D = np.asarray(idx, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[D] = False
        Xkeep = x_best[keep_mask]
        # refill sequentially at voids
        Xcur = Xkeep.copy()
        newpts = []
        for s in range(k):
            if time.time() - t0 > HARD - 30:
                break
            bp = _find_void(Xcur, rng, ncand=20000, refine=200)
            newpts.append(bp)
            Xcur = np.vstack([Xcur, bp.reshape(1, 4)])
        if len(newpts) != k:
            continue
        y = x_best.copy()
        y[D] = np.vstack(newpts)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # patch-only hinge anneal
        try:
            y2 = _patch_hinge_opt(y, D, t_target, steps=600, lr0=0.01, lr1=0.0008)
        except Exception:
            y2 = y
        # short all-free softmax anneal
        try:
            y3 = _smooth_opt(y2, 400, 40.0, 280.0, 0.005, 0.0005, iu)
        except Exception:
            y3 = y2
        v3 = _max_cos(y3)
        thresh = best_val + 0.004
        if v3 > thresh:
            continue
        remain = HARD - (time.time() - t0) - 24.0
        if remain < 1.0:
            break
        b = min(2.2, max(1.0, remain / 10.0))
        yp, vp = _lp_polish(y3, b)
        improved = False
        if vp < best_val - 1e-12:
            x_best, best_val = yp, vp
            improved = True
            continue
        # keep near-miss for extra all-free anneal (changed graph event)
        if vp < best_val + 0.002:
            if time.time() - t0 > HARD - 28:
                # no time for extra
                continue
            try:
                y4 = _smooth_opt(yp, 700, 30.0, 320.0, 0.004, 0.0003, iu)
            except Exception:
                y4 = yp
            remain2 = HARD - (time.time() - t0) - 22.0
            if remain2 < 0.8:
                continue
            b2 = min(2.0, max(0.8, remain2 / 10.0))
            yp2, vp2 = _lp_polish(y4, b2)
            if vp2 < best_val - 1e-12:
                x_best, best_val = yp2, vp2
                improved = True
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 1.5:
        remain = min(remain, 24.0)
        try:
            x_best, best_val = _lp_polish(x_best, remain)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
