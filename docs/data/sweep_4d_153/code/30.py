import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gt = x @ x.T
        cfull = Gt[iu0]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.empty((p, 2 * d + 1), dtype=float)
        # cols
        ccols = np.empty((p, 2 * d + 1), dtype=int)
        # fill cols: I*d + k, J*d + k, last
        ar = np.arange(d)
        # build cols via broadcasting
        # I part
        # vals
        vals = np.empty((p, 2 * d + 1), dtype=float)
        vals[:, :d] = x[J]
        vals[:, d:2*d] = x[I]
        vals[:, 2*d] = -1.0
        ccols[:, :d] = I[:, None] * d + ar[None, :]
        ccols[:, d:2*d] = J[:, None] * d + ar[None, :]
        ccols[:, 2*d] = n * d
        A_ub = coo_matrix((vals.ravel(), (rows, ccols.ravel())), shape=(p, nv)).tocsr()
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
    return x

def _smooth(x, iters, k0, k1, lr0, lr1, mu, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        b1 = np.load("best.npy")
    except Exception:
        b1 = None
    try:
        b2 = np.load("record.npy")
    except Exception:
        b2 = None
    cands0 = []
    for b in (b1, b2):
        if b is not None:
            try:
                b = np.asarray(b, dtype=float)
                if b.shape == (n, d):
                    b = b / np.linalg.norm(b, axis=1, keepdims=True)
                    cands0.append(b)
            except Exception:
                pass
    if len(cands0) == 0:
        incumbent = rng.standard_normal((n, d))
        incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    else:
        vals = [_max_cos(b) for b in cands0]
        incumbent = cands0[int(np.argmin(vals))].copy()
    inc_val = _max_cos(incumbent)
    elites = [(inc_val, incumbent.copy())]

    SEARCH_END = t0 + 118
    trial = 0
    # pre-allocate iu not needed
    while time.time() < SEARCH_END:
        trial += 1
        # choose base
        rb = rng.random()
        if rb < 0.70:
            base = incumbent
        else:
            idxb = int(rng.integers(len(elites)))
            base = elites[idxb][1]
        x = base.copy()
        op = rng.random()
        if op < 0.28:
            sigma = float(rng.choice(np.array([0.03, 0.06, 0.10, 0.16, 0.25])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif op < 0.58:
            # worst-k removal + farthest-void reinsertion
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(4, 11))
            order = np.argsort(-maxper, kind="stable")
            idx = order[:k]
            # randomize insertion order
            idx = idx[rng.permutation(k)]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            for j in idx:
                nc = 400
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xkeep.T
                scores = dots.max(axis=1)
                # pick among best 3 for diversity
                best3 = np.argsort(scores, kind="stable")[:3]
                pick = best3[int(rng.integers(len(best3)))]
                chosen = cand[pick]
                # few local repulsion refinements of void point
                for _ in range(15):
                    dd = Xkeep @ chosen
                    # focus on nearest
                    w = np.exp(60.0 * (dd - dd.max()))
                    gvec = (w[:, None] * Xkeep).sum(axis=0)
                    gvec = gvec - np.dot(gvec, chosen) * chosen
                    gn = np.linalg.norm(gvec)
                    if gn < 1e-12:
                        break
                    chosen = chosen - 0.05 * gvec / gn
                    chosen /= np.linalg.norm(chosen)
                x[j] = chosen
                Xkeep = np.vstack([Xkeep, chosen[None, :]])
            # tiny jitter to keeps
            x = x + 0.005 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif op < 0.82:
            # coordinated cluster drift
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            seed = int(rng.integers(n))
            s = int(rng.integers(6, 19))
            nn = np.argsort(-G[seed], kind="stable")[:s]
            # random drift direction
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            delta = float(rng.uniform(0.15, 0.50))
            for i in nn:
                xi = x[i]
                tang = u - np.dot(u, xi) * xi
                x[i] = xi + delta * tang
            x[nn] += 0.02 * rng.standard_normal((len(nn), d))
            # light global jitter
            x += 0.005 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # rattler / low-contact relocation + hub jitter
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            contacts = (G > m - 1e-7).sum(axis=1)
            low = np.where(contacts <= 2)[0]
            high = np.where(contacts >= 7)[0]
            # jitter hubs
            if len(high) > 0:
                x[high] += 0.06 * rng.standard_normal((len(high), d))
            if len(low) > 0:
                # relocate up to 4 lowest
                k2 = min(len(low), int(rng.integers(1, 5)))
                # lowest contacts first, tie by worst maxper
                maxper = G.max(axis=1)
                loword = low[np.argsort(maxper[low], kind="stable")[::-1][:k2]]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[loword] = False
                Xkeep = x[keep_mask]
                # need current normalized x for keep part
                Xn = x / np.linalg.norm(x, axis=1, keepdims=True)
                Xkeep = Xn[keep_mask]
                for j in loword:
                    nc = 400
                    cand = rng.standard_normal((nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    scores = (cand @ Xkeep.T).max(axis=1)
                    best3 = np.argsort(scores, kind="stable")[:3]
                    pick = best3[int(rng.integers(len(best3)))]
                    chosen = cand[pick]
                    x[j] = chosen
                    Xkeep = np.vstack([Xkeep, chosen[None, :]])
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        iters = int(rng.integers(250, 550))
        k0 = float(rng.uniform(12, 30))
        k1 = float(rng.choice(np.array([150.0, 250.0, 400.0])))
        lr0 = float(rng.uniform(0.008, 0.025))
        lr1 = float(rng.uniform(0.0015, 0.005))
        mu = float(rng.uniform(0.6, 0.9))
        x = _smooth(x, iters, k0, k1, lr0, lr1, mu, rng)
        val = _max_cos(x)
        do_lp = False
        if val < inc_val - 1e-6:
            do_lp = True
            lp_t = 1.6
        elif val < inc_val + 0.001 and rng.random() < 0.12:
            do_lp = True
            lp_t = 1.2
        elif val < inc_val + 0.004 and rng.random() < 0.03:
            do_lp = True
            lp_t = 1.0
        if do_lp:
            if time.time() + lp_t > SEARCH_END + 2:
                # skip if no time
                pass
            else:
                y = _lp_polish(x, lp_t)
                v2 = _max_cos(y)
                if v2 < inc_val - 1e-12:
                    incumbent = y
                    inc_val = v2
                    elites.append((v2, y.copy()))
                    elites = sorted(elites, key=lambda e: e[0])[:5]
                elif len(elites) < 5 and v2 < elites[-1][0] + 0.001:
                    elites.append((v2, y.copy()))
                    elites = sorted(elites, key=lambda e: e[0])[:5]
        else:
            if val < elites[-1][0] - 2e-4 and len(elites) < 5:
                elites.append((val, x.copy()))
                elites = sorted(elites, key=lambda e: e[0])[:5]

    # final long polish of incumbent within remaining budget
    remain = 175 - (time.time() - t0)
    remain = max(5.0, min(30.0, remain))
    incumbent = _lp_polish(incumbent, remain)
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
