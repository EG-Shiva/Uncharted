import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 249, 6
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        # fallback reshape if needed
        a = np.zeros((n, d))
        m = min(xb.shape[0], n)
        a[:m, :min(xb.shape[1], d)] = xb[:m, :min(xb.shape[1], d)]
        xb = a
    nr = np.linalg.norm(xb, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    xb /= nr
    bestX = xb.copy()
    bestVal = _max_cos(bestX)
    cur = bestX.copy()
    curV = bestVal

    iu_all = np.triu_indices(n, 1)
    Iall, Jall = iu_all
    Np = Iall.size
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)

    def smooth_adam(x, steps, tau0, tau1, lr):
        mt = np.zeros_like(x)
        vt = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = 1.0, 1.0
        for t in range(1, steps + 1):
            if (t % 200 == 0) and time.time() > DEAD - 25:
                break
            frac = t / steps
            tau = tau0 + (tau1 - tau0) * frac
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            mval = G.max()
            W = np.exp((G - mval) / tau)
            np.fill_diagonal(W, 0.0)
            S = W.sum()
            if S < 1e-200:
                break
            grad = (W @ x) / S
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gtan = grad - dot * x
            b1p *= b1
            b2p *= b2
            mt = b1 * mt + (1.0 - b1) * gtan
            vt = b2 * vt + (1.0 - b2) * (gtan * gtan)
            mhat = mt / (1.0 - b1p)
            vhat = vt / (1.0 - b2p)
            upd = lr * mhat / (np.sqrt(vhat) + eps)
            x = x - upd
            nn = np.linalg.norm(x, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            x /= nn
        return x

    def lp_solve(X, radius, b_extra):
        # b_extra: array length p added to -c (i.e. -delta for forced)
        Gfull = X @ X.T
        c_all = Gfull[iu_all]
        m = float(c_all.max())
        # select top K
        K = 5000
        if Np > K:
            part = np.argpartition(-c_all, K)[:K]
            sel = part
        else:
            sel = np.arange(Np)
        I = Iall[sel]
        J = Jall[sel]
        c_sel = c_all[sel]
        p = sel.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_block = np.concatenate(
            [(I[:, None] * d + ar), (J[:, None] * d + ar),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals_block = np.concatenate(
            [X[J], X[I], -np.ones((p, 1))], axis=1
        ).ravel()
        A_ub = coo_matrix((vals_block, (rows, cols_block)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        b_ub = -c_sel + b_extra  # b_extra = -delta for forced, 0 else
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                      b_eq=np.zeros(n), bounds=bounds, method="highs")
        return res, (I, J, c_sel, sel)

    def lp_polish_small(X, seconds):
        Xc = X.copy()
        bc = _max_cos(Xc)
        radius = 2e-3
        start = time.time()
        while time.time() < DEAD - 2 and (time.time() - start) < seconds and radius > 1e-13:
            Gfull = Xc @ Xc.T
            c_all = Gfull[iu_all]
            m = float(c_all.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = np.where(c_all > thresh)[0]
            if sel.size == 0:
                break
            if sel.size > 4000:
                part = np.argpartition(-c_all, 4000)[:4000]
                sel = part
            I = Iall[sel]; J = Jall[sel]; c_sel = c_all[sel]
            p = sel.size
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols_block = np.concatenate(
                [(I[:, None] * d + ar), (J[:, None] * d + ar),
                 np.full((p, 1), n * d)], axis=1
            ).ravel()
            vals_block = np.concatenate(
                [Xc[J], Xc[I], -np.ones((p, 1))], axis=1
            ).ravel()
            try:
                A_ub = coo_matrix((vals_block, (rows, cols_block)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((Xc.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                              b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = Xc + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < bc:
                Xc, bc = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return Xc, bc

    # precompute deltas / radii / sizes cycles
    deltas = [1e-4, 5e-4, 1e-3, 5e-4, 1e-4, 1e-3]
    radiiF = [0.01, 0.02, 0.05, 0.03, 0.015, 0.04]
    sizes = [120, 180, 250, 150, 300, 100, 200]

    it = 0
    # main forced-open cycles, leave ~30s for final polish
    while time.time() < DEAD - 32:
        it += 1
        # build contact graph on cur with slack 2e-4
        try:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            cmax = float(Gc.max())
            mask = Gc > (cmax - 2e-4)
            # degrees
            deg = mask.sum(axis=1)
            hubs = np.argsort(-deg)
        except Exception:
            hubs = np.arange(n)
            cmax = curV
        # pick hub cyclically among top 12
        topH = hubs[:12]
        hub = int(topH[(it - 1) % len(topH)])
        # BFS cluster to collect S edges
        target = sizes[(it - 1) % len(sizes)]
        # adjacency from mask
        # build neighbor lists on fly using mask row
        visited = np.zeros(n, dtype=bool)
        visited[hub] = True
        queue = [hub]
        qpos = 0
        Sset = set()
        # to avoid O(n^2) per iter, use Gc row threshold
        # precompute neighbor lists for visited expansion
        while len(Sset) < target and qpos < len(queue):
            u = queue[qpos]; qpos += 1
            # neighbors of u
            nbrs = np.where(mask[u])[0]
            for v in nbrs:
                if len(Sset) >= target:
                    break
                a = int(u) if u < v else int(v)
                b = int(v) if u < v else int(u)
                code = a * n + b
                if code not in Sset:
                    Sset.add(code)
                if not visited[v]:
                    visited[v] = True
                    queue.append(int(v))
            if qpos > 400:
                break
        if len(Sset) < 30:
            # fallback: take worst pairs globally as S
            try:
                Gall = cur @ cur.T
                np.fill_diagonal(Gall, -2.0)
                call = Gall[iu_all]
                idx = np.argpartition(-call, target)[:target]
                Sset = set((int(Iall[k]) * n + int(Jall[k])) for k in idx)
            except Exception:
                pass
        delta = float(deltas[(it - 1) % len(deltas)])
        radiusF = float(radiiF[(it - 1) % len(radiiF)])
        # forced LP on cur
        try:
            # need b_extra aligned with selected top-K; compute selection first inside?
            # replicate selection to know forced flags
            Gfull = cur @ cur.T
            c_all = Gfull[iu_all]
            K = 5000
            if Np > K:
                part = np.argpartition(-c_all, K)[:K]
                sel = part
            else:
                sel = np.arange(Np)
            # forced flag
            codes_sel = Iall[sel] * n + Jall[sel]
            # Sset lookup
            isF = np.array([1.0 if int(c) in Sset else 0.0 for c in codes_sel])
            b_extra = -delta * isF
            res, _ = lp_solve(cur, radiusF, b_extra)
            if res.status != 0:
                # try smaller radius
                res2 = None
                for rr in [0.01, 0.005]:
                    try:
                        r2, _ = lp_solve(cur, rr, b_extra * 0.5)
                        if r2.status == 0:
                            res = r2
                            break
                    except Exception:
                        continue
                if res.status != 0:
                    continue
            dx = res.x[:-1].reshape(n, d)
            Xf = cur + dx
            Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
            # step even if worsens: polish from Xf
            # short softmax
            Xp = smooth_adam(Xf, steps=600, tau0=0.006, tau1=0.0015, lr=0.005)
            vp = _max_cos(Xp)
            # tiny LP polish (1-2 sec)
            Xp2, vp2 = lp_polish_small(Xp, seconds=2.0)
            # update best
            if vp2 < bestVal:
                bestVal = vp2
                bestX = Xp2.copy()
            # update cur: accept if not too bad (allow worsening to change basin)
            # keep cur near best but allow exploration
            if vp2 < curV or vp2 < bestVal + 0.008:
                cur, curV = Xp2.copy(), vp2
            else:
                # occasional random hub shake from best to diversify
                if it % 7 == 0:
                    cur = bestX.copy()
                    # small shake on cluster nodes
                    nodes = list(set([c // n for c in list(Sset)[:30]] + [c % n for c in list(Sset)[:30]]))
                    if len(nodes) > 0:
                        cur[nodes] += rng.normal(scale=0.01, size=(len(nodes), d))
                        cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                        curV = _max_cos(cur)
        except Exception:
            continue
        # opportunistic: if cur is best, do a quick greedy from best occasionally
        if it % 5 == 0 and time.time() < DEAD - 35:
            try:
                if curV > bestVal + 0.002:
                    cur = bestX.copy()
                    curV = bestVal
            except Exception:
                pass

    # final squeeze of best
    try:
        if time.time() < DEAD - 2:
            # softmax anneal
            bestX = smooth_adam(bestX, steps=1500, tau0=0.005, tau1=0.0012, lr=0.004)
            bestVal = _max_cos(bestX)
    except Exception:
        pass
    try:
        if time.time() < DEAD - 1:
            bestX, bestVal = lp_polish_small(bestX, seconds=max(2.0, DEAD - time.time() - 1.0))
    except Exception:
        pass
    nn = np.linalg.norm(bestX, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    bestX /= nn
    bestX = np.nan_to_num(bestX, nan=0.0, posinf=0.0, neginf=0.0)
    nn2 = np.linalg.norm(bestX, axis=1)
    for i in np.where(nn2 < 1e-12)[0]:
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        bestX[i] = v
    return bestX
