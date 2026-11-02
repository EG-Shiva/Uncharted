import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.05:
        return X
    start = time.time()
    radius = 1.5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
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
            tl = seconds - (time.time() - start)
            if tl <= 0.05:
                break
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return X

def _adam_quench(X0, steps=600, beta0=50.0, beta1=400.0, lr0=0.015):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu0, iu1 = np.triu_indices(n, 1)
    mt = np.zeros_like(X)
    vt = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = X @ X.T
        gv = G[iu0, iu1]
        m = gv.max()
        e = np.exp(beta * (gv - m))
        S = e.sum()
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        mt = b1 * mt + (1.0 - b1) * Gt
        vt = b2 * vt + (1.0 - b2) * (Gt * Gt)
        mhat = mt / (1.0 - b1 ** t)
        vhat = vt / (1.0 - b2 ** t)
        lr = lr0 * (1.0 - 0.7 * t / steps)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _near_info(X, tol):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    deg = mask.sum(axis=1)
    pmax = G.max(axis=1)
    cnt = int(mask.sum() // 2)
    return m, G, deg, pmax, cnt

def solve():
    t0 = time.time()
    HARD = 150.0
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        rng0 = np.random.default_rng(0)
        bestX = rng0.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    baseX = bestX.copy()
    baseV = bestV
    _, _, _, _, baseNear = _near_info(baseX, 1e-4)
    bestNear = baseNear

    nAtt = 0
    # several seeds / cluster choices
    while time.time() - t0 < HARD:
        remain = t0 + HARD - time.time()
        if remain < 4.0:
            break
        nAtt += 1
        # cycle parameters deterministically
        K = 10 + ((nAtt - 1) % 5)  # 10..14
        tol_near = [1e-3, 5e-4, 8e-4][(nAtt - 1) % 3]
        sigma = [0.005, 0.008, 0.012, 0.015][(nAtt - 1) % 4]
        seed = 1000 + nAtt * 97
        rng = np.random.default_rng(seed)

        # score points by degree + personal max on baseX
        try:
            m0, G0, deg, pmax, _ = _near_info(baseX, tol_near)
        except Exception:
            continue
        # order seeds: highest degree, tie by pmax
        order = np.lexsort((-pmax, -deg))
        # try different starting seed among top few to diversify
        top_pool = order[:6]
        start_node = int(top_pool[(nAtt - 1) % len(top_pool)])
        # build connected cluster by most-shared neighbours
        in_clu = np.zeros(n, dtype=bool)
        in_clu[start_node] = True
        # adjacency for near contacts (symmetric, no diag)
        Adj = (G0 > m0 - tol_near)
        np.fill_diagonal(Adj, False)
        while int(in_clu.sum()) < K:
            # candidates not in cluster: score = connections to cluster + deg bonus
            conn = Adj[:, in_clu].sum(axis=1).astype(float)
            conn[in_clu] = -1
            # need connectivity: prefer conn>0, but if isolated allow best deg
            # add small personal-max tie break
            score = conn * 10.0 + deg * 0.1 + (pmax - m0 + tol_near) * 50.0
            score[in_clu] = -1e9
            # if no connection (max conn==0), just take highest deg
            nxt = int(np.argmax(score))
            in_clu[nxt] = True
        clu_idx = np.where(in_clu)[0]
        keep_idx = np.where(~in_clu)[0]
        # jitter kept core
        kept = baseX[keep_idx].copy()
        kept = kept + rng.standard_normal(kept.shape) * sigma
        kept /= np.linalg.norm(kept, axis=1, keepdims=True)

        # sequential greedy reinsertion from 8192 candidates
        reinserted = []
        # random order of cluster slots
        slot_order = np.arange(K)
        rng.shuffle(slot_order)
        ok = True
        for s in range(K):
            try:
                C = rng.standard_normal((8192, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
            except Exception:
                ok = False
                break
            # max dot vs kept + already reinserted
            # kept part
            Dk = C @ kept.T
            mx = Dk.max(axis=1)
            if len(reinserted) > 0:
                R = np.stack(reinserted, axis=0)
                Dr = C @ R.T
                mx = np.maximum(mx, Dr.max(axis=1))
            bi = int(np.argmin(mx))
            reinserted.append(C[bi].copy())
        if not ok:
            continue
        # assemble trial
        Xtry = np.empty((n, d), dtype=np.float64)
        Xtry[keep_idx] = kept
        # map reinserted in slot_order to clu_idx
        for rank, ci in enumerate(clu_idx):
            # slot_order[rank] corresponds? simpler: assign in order
            Xtry[int(ci)] = reinserted[rank]
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # cheap tangent-Adam log-sum-exp quench beta 50->400 ~600 steps
        # adapt steps to remaining time
        steps = 600
        if remain < 12:
            steps = 350
        try:
            Xq = _adam_quench(Xtry, steps=steps, beta0=50.0, beta1=400.0, lr0=0.015)
        except Exception:
            continue
        vq = _max_cos(Xq)
        # only LP-polish candidates whose smooth max is below ~0.175
        if vq > 0.175:
            # still consider as base diversification if near count improved? need polish-free check
            # compute near count at vq-1e-4
            try:
                _, _, _, _, cntq = _near_info(Xq, 1e-4)
            except Exception:
                continue
            # accept as base if much fewer near contacts and not much worse (for exploration)
            if vq < baseV + 5e-4 and cntq < baseNear:
                baseX = Xq.copy()
                baseV = vq
                baseNear = cntq
            continue
        # LP polish with small budget
        left = t0 + HARD - time.time()
        if left < 1.5:
            break
        budget = min(3.0, max(0.8, left - 0.6))
        # give larger budget occasionally
        if nAtt % 5 == 0:
            budget = min(4.0, max(1.0, left - 0.5))
        try:
            Xp = _lp_polish(Xq, budget)
        except Exception:
            Xp = Xq
        try:
            vp = _max_cos(Xp)
            _, _, _, _, cntp = _near_info(Xp, 1e-4)
        except Exception:
            continue
        # accept by post-polish max first, near-count second
        if vp < bestV - 1e-12:
            bestX = Xp.copy()
            bestV = vp
            bestNear = cntp
            baseX = Xp.copy()
            baseV = vp
            baseNear = cntp
        elif vp < baseV + 5e-4 and cntp < baseNear:
            # move base to different basin even if not global best
            baseX = Xp.copy()
            baseV = vp
            baseNear = cntp
        # also if stuck for long, occasionally reset base to best
        if nAtt % 8 == 0 and baseV > bestV + 3e-4:
            baseX = bestX.copy()
            baseV = bestV
            baseNear = bestNear

    # final intensive polish of champion with remaining time
    try:
        remain = 179.0 - (time.time() - t0)
        if remain > 2.0:
            bestX = _lp_polish(bestX, min(remain - 0.5, 12.0))
            bestV = _max_cos(bestX)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
