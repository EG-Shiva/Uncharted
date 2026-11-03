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
    return G.max(axis=1), G


def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _hard_push(X, steps, lr0, lr1, tol=1e-9):
    n, d = X.shape
    for t in range(steps):
        frac = t / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # active contacts near max
        thr = m - 2e-3
        # limit to top pairs for speed if too many
        mask = G >= thr
        # gradient: sum of tangential components
        # Gt_i = sum_{j in contact} (Xj - dot*Xi)
        # move opposite to reduce dot
        # compute via masked matrix
        W = np.where(mask, 1.0, 0.0)
        # symmetrize (mask already symmetric)
        S = W @ X  # sum of Xj
        cnt = W.sum(axis=1, keepdims=True)
        # sum of dot*Xi = (sum dot)*Xi ; need per-pair dot weighting
        # compute sum_j dot_ij * Xi = Xi * sum_j (G_ij * W_ij)
        sdot = (G * W).sum(axis=1, keepdims=True)
        Gt = S - sdot * X
        # tangential already, but re-project
        dot2 = (Gt * X).sum(axis=1, keepdims=True)
        Gt -= dot2 * X
        # normalize step per point by contacts
        denom = np.maximum(cnt, 1.0)
        Gt /= denom
        X = X - lr * Gt
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 175.0
    LP_FINAL = 28.0
    explore_end = t0 + BUDGET - LP_FINAL - 1.5

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val

    rng = np.random.RandomState(20261103)
    iu0 = np.triu_indices(n, 1)
    it = 0
    # annealing temperature for sideways moves (scale of differences ~1e-9)
    T0 = 2e-9
    T1 = 2e-11

    while time.time() < explore_end:
        it += 1
        elapsed = time.time() - t0
        frac_global = elapsed / max(1e-9, (explore_end - t0))
        Temp = T0 * (T1 / T0) ** frac_global
        r = rng.rand()
        try:
            if r < 0.38:
                # global Gaussian kick + softmax quench
                sigma = [0.005, 0.012, 0.025, 0.05, 0.09][rng.randint(5)]
                X = cur + rng.randn(n, d) * sigma
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                steps = 700 + int(rng.randint(900))
                beta0 = 25.0 + rng.rand() * 45.0
                beta1 = 250.0 + rng.rand() * 550.0
                lr0 = 0.004 + rng.rand() * 0.010
                lr1 = lr0 * (0.08 + rng.rand() * 0.12)
                Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
                # occasional hard refinement
                if rng.rand() < 0.4:
                    Xq = _hard_push(Xq, 120, 0.004, 0.0005)
            elif r < 0.68:
                # void re-insertion of worst / low-contact / overloaded points
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rmax = G.max(axis=1)
                order = np.argsort(rmax)
                k = 1 + int(rng.randint(3))  # 1..3
                if rng.rand() < 0.55:
                    victims = order[-k:]  # worst
                elif rng.rand() < 0.5:
                    # fewest contacts within 1e-7 of max
                    m = G.max()
                    cnt = (G >= m - 1e-7).sum(axis=1)
                    victims = np.argsort(cnt)[:k]
                else:
                    victims = rng.randint(n, size=k)
                X = cur.copy()
                alive = np.ones(n, dtype=bool)
                alive[victims] = False
                for idx in victims:
                    # candidates far from alive set
                    Nc = 2500
                    cand = rng.randn(Nc, d)
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    D = cand @ X[alive].T
                    scores = D.max(axis=1)
                    pick = int(np.argmin(scores))
                    # small jitter around best void to avoid exact symmetry
                    jitter = 1e-3
                    v = cand[pick] + rng.randn(d) * jitter
                    v /= np.linalg.norm(v)
                    X[idx] = v
                    alive[idx] = True
                steps = 900 + int(rng.randint(900))
                beta0 = 30.0 + rng.rand() * 40.0
                beta1 = 300.0 + rng.rand() * 500.0
                lr0 = 0.004 + rng.rand() * 0.008
                lr1 = lr0 * 0.12
                Xq = _quench(X, steps, beta0, beta1, lr0, lr1)
            elif r < 0.85:
                # contact-pair splitting: pick a top pair and push apart
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                flat = G[iu0]
                K = min(40, flat.size)
                top = np.argpartition(flat, -K)[-K:]
                # choose one at random weighted to largest
                ti = top[rng.randint(K)]
                a = int(iu0[0][ti]); b = int(iu0[1][ti])
                X = cur.copy()
                # push a away from b and b away from a along tangent
                amp = 0.015 + rng.rand() * 0.05
                for (p_, q_) in ((a, b), (b, a)):
                    dot = float(X[p_] @ X[q_])
                    tang = X[q_] - dot * X[p_]
                    nrm = np.linalg.norm(tang)
                    if nrm > 1e-12:
                        tang /= nrm
                        X[p_] = X[p_] - amp * tang
                        X[p_] /= np.linalg.norm(X[p_])
                # also shake their neighborhoods slightly
                neigh_sig = 0.005
                X += rng.randn(n, d) * 0.0  # placeholder
                # small noise on the two points' close neighbors
                G2 = X @ X.T
                np.fill_diagonal(G2, -2.0)
                for p_ in (a, b):
                    nbrs = np.where(G2[p_] > G2[p_].max() - 5e-4)[0][:6]
                    for nb in nbrs:
                        if nb == p_:
                            continue
                        X[nb] = X[nb] + rng.randn(d) * neigh_sig * 0.5
                        X[nb] /= np.linalg.norm(X[nb])
                steps = 700 + int(rng.randint(700))
                beta0 = 30.0 + rng.rand() * 40.0
                beta1 = 300.0 + rng.rand() * 500.0
                lr0 = 0.003 + rng.rand() * 0.007
                Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.1)
            else:
                # cluster shake of most jammed point + neighbors
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                m = G.max()
                cnt = (G >= m - 1e-7).sum(axis=1)
                hub = int(np.argmax(cnt + rng.randn(n) * 0.5))
                nbrs = np.where(G[hub] >= m - 2e-3)[0]
                if len(nbrs) > 8:
                    nbrs = nbrs[np.argsort(G[hub, nbrs])[-8:]]
                X = cur.copy()
                sig = 0.02 + rng.rand() * 0.04
                sel = np.concatenate([[hub], nbrs])
                X[sel] += rng.randn(len(sel), d) * sig
                X[sel] /= np.linalg.norm(X[sel], axis=1, keepdims=True)
                steps = 800 + int(rng.randint(800))
                beta0 = 25.0 + rng.rand() * 40.0
                beta1 = 250.0 + rng.rand() * 500.0
                lr0 = 0.004 + rng.rand() * 0.008
                Xq = _quench(X, steps, beta0, beta1, lr0, lr0 * 0.12)
        except Exception:
            continue
        try:
            vq = _max_cos(Xq)
        except Exception:
            continue
        if vq < best_val - 1e-15:
            best = Xq.copy()
            best_val = vq
            cur = Xq.copy()
            cur_val = vq
        elif vq < cur_val:
            cur = Xq.copy()
            cur_val = vq
        else:
            # Metropolis sideways move
            if Temp > 0 and rng.rand() < np.exp(-(vq - cur_val) / Temp):
                cur = Xq.copy()
                cur_val = vq
        # opportunistic quick LP if clearly better and time allows
        if best_val < cur_val - 1e-9:
            pass
        if time.time() + 1.0 > explore_end:
            break

    # final polish of best
    remain = (t0 + BUDGET) - time.time() - 0.5
    if remain > 3:
        try:
            best = _lp_polish(best, min(remain, LP_FINAL))
            best_val = _max_cos(best)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
