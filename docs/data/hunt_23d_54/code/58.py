import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0=None, hard=1e9):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > hard:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:1000]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def _adam_movable(X, mov, steps, k0, k1, lr, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    k = len(mov)
    mov = np.asarray(mov, dtype=int)
    is_mov = np.zeros(n, dtype=bool); is_mov[mov] = True
    mt = np.zeros((k, d)); vt = np.zeros((k, d))
    lk0 = np.log(k0); lk1 = np.log(k1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        ke = float(np.exp(lk0 + (lk1 - lk0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        # weights over all pairs (frozen-frozen gives constant but harmless small? exclude)
        # build mask excluding frozen-frozen
        # W = exp(ke*(G-m0)) except frozen-frozen and diag
        E = np.exp(ke * (G - m0))
        # zero diag
        # zero frozen-frozen
        fr = np.where(~is_mov)[0]
        # mask frozen-frozen: set to 0
        # efficient: set rows/cols? just zero submatrix
        if len(fr) > 0:
            E[np.ix_(fr, fr)] = 0.0
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        W = E / S
        # symmetrize (E already symmetric)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Gt_all = F - dot * X
        Gt = Gt_all[mov]
        t = s + 1
        mt = b1 * mt + (1 - b1) * Gt
        vt = b2 * vt + (1 - b2) * (Gt * Gt)
        mh = mt / (1 - b1 ** t)
        vh = vt / (1 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        X[mov] = X[mov] - lr * upd
        nr = np.linalg.norm(X[mov], axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X[mov] /= nr
    return X

def _adam_all(X, steps, ke, lr):
    n, d = X.shape
    mt = np.zeros_like(X); vt = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        E = np.exp(ke * (G - m0))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        W = E / S
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Gt = F - dot * X
        t = s + 1
        mt = b1 * mt + (1 - b1) * Gt
        vt = b2 * vt + (1 - b2) * (Gt * Gt)
        mh = mt / (1 - b1 ** t)
        vh = vt / (1 - b2 ** t)
        upd = mh / (np.sqrt(vh) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _cluster_of_size(adj, a, b, k, rng):
    n = adj.shape[0]
    S = set([int(a), int(b)])
    while len(S) < k:
        # candidates neighbours of S
        cand = set()
        for s in list(S):
            nbrs = np.where(adj[s])[0]
            for nb in nbrs:
                nb = int(nb)
                if nb not in S:
                    cand.add(nb)
        if not cand:
            # disconnected: add lowest-degree outside (deterministic random)
            outside = [i for i in range(n) if i not in S]
            # pick random outside via rng
            S.add(int(rng.choice(outside)))
            continue
        cand = list(cand)
        # score by connections to S
        scores = []
        for c in cand:
            sc = sum(1 for s in S if adj[c, s])
            scores.append(sc)
        scores = np.array(scores)
        mx = scores.max()
        tied = [cand[i] for i in range(len(cand)) if scores[i] == mx]
        # deterministic cycle: pick via rng shuffle
        pick = int(rng.choice(tied)) if len(tied) > 1 else int(tied[0])
        S.add(pick)
    return sorted(S)

def solve():
    t0 = time.time()
    HARD = 171.0
    EXPLORE = 150.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d); tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best_val = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)
    it = 0
    ks = [6, 8, 10, 12]
    while time.time() - t0 < EXPLORE and time.time() - t0 < HARD - 28.0:
        it += 1
        remain = HARD - 28.0 - (time.time() - t0)
        if remain < 4.0:
            break
        k = ks[(it - 1) % len(ks)]
        # contact analysis on x_best
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        adj = G0 > (m0 - 1e-4)
        np.fill_diagonal(adj, False)
        # contact counts tight threshold for low-contact ranking
        tight = G0 > (m0 - 1e-9)
        cnt = tight.sum(axis=1)
        order_lo = np.argsort(cnt, kind="stable")
        # top seed pairs
        flat = G0[iu]
        seed_order = np.argsort(-flat, kind="stable")[:8]
        seed_pairs = [(int(iu[0][s]), int(iu[1][s])) for s in seed_order]
        mode = (it - 1) % 3
        if mode == 0:
            sp = seed_pairs[(it // 3) % len(seed_pairs)]
            del_idx = _cluster_of_size(adj, sp[0], sp[1], k, rng)
        elif mode == 1:
            del_idx = sorted(order_lo[:k].tolist())
        else:
            sp = seed_pairs[(it // 3 + 3) % len(seed_pairs)]
            del_idx = _cluster_of_size(adj, sp[0], sp[1], k, rng)
            # mix: replace last one with a low-contact point if not already in
            for c in order_lo:
                c = int(c)
                if c not in del_idx:
                    del_idx = sorted(del_idx)
                    del_idx[-1] = c
                    break
        del_idx = sorted([int(v) for v in del_idx])
        mask_f = np.ones(n, dtype=bool); mask_f[del_idx] = False
        Xf = x_best[mask_f]
        # greedy reinsert from 30000 candidates
        NPOOL = 30000
        # adapt pool if time low
        if HARD - (time.time() - t0) < 50:
            NPOOL = 15000
        C = rng.randn(NPOOL, d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        Df = C @ Xf.T
        runmax = Df.max(axis=1)
        chosen = []
        chosen_vecs = []
        for t in range(k):
            idx = int(np.argmin(runmax))
            cv = C[idx].copy()
            chosen.append(idx); chosen_vecs.append(cv)
            dots = C @ cv
            runmax = np.maximum(runmax, dots)
            # prevent repick: set picked to large
            runmax[idx] = 2.0
        Xnew = x_best.copy()
        for ti, di in enumerate(del_idx):
            Xnew[di] = chosen_vecs[ti] / np.linalg.norm(chosen_vecs[ti])
        # small jitter on chosen to break symmetry ties
        Xnew[del_idx] += rng.randn(k, d) * 1e-4
        Xnew[del_idx] /= np.linalg.norm(Xnew[del_idx], axis=1, keepdims=True)
        # movable Adam
        steps_mov = 1500
        if remain < 40:
            steps_mov = 700
        try:
            Xnew = _adam_movable(Xnew, del_idx, steps=steps_mov, k0=80.0, k1=800.0, lr=0.01)
        except Exception:
            pass
        # unfreeze short phase
        try:
            short_steps = 200
            if remain < 40:
                short_steps = 100
            Xnew = _adam_all(Xnew, steps=short_steps, ke=800.0, lr=0.005)
        except Exception:
            pass
        v_mid = _max_cos(Xnew)
        # skip LP if hopelessly bad to save time
        if v_mid > best_val + 0.01:
            continue
        # LP polish 5-8 s
        lp_sec = 6.0
        if remain < 30:
            lp_sec = 4.0
        # cap by remaining explore time
        left_explore = EXPLORE - (time.time() - t0)
        lp_sec = min(lp_sec, max(1.0, left_explore - 0.5))
        if lp_sec < 1.0:
            continue
        try:
            Xpol = _lp_polish(Xnew, seconds=lp_sec, t0=t0, hard=HARD - 22.0)
        except Exception:
            continue
        v = _max_cos(Xpol)
        if v < best_val:
            best_val = v
            x_best = Xpol.copy()
        # loop
    # final 20 s LP polish
    left = HARD - (time.time() - t0)
    fsec = min(20.0, max(1.0, left - 1.0))
    if fsec > 0.5:
        try:
            x_best = _lp_polish(x_best, seconds=fsec, t0=t0, hard=HARD - 0.5)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
