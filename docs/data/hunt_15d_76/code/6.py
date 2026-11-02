import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _smooth_refine(x, steps, T0, T1, e0, e1):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T0 > 0 and T1 > 0 else T0
        eta = e0 * (e1 / e0) ** frac if e0 > 0 and e1 > 0 else e0
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        # weights in (0,1], max=1, no overflow
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        norms = np.linalg.norm(gt, axis=1, keepdims=True)
        rs = np.sum(W, axis=1, keepdims=True)
        mx = np.max(rs)
        if mx <= 0:
            break
        scale = rs / mx
        # avoid division by zero
        step = eta * scale * gt / (norms + 1e-15)
        x = x - step
        x = _normed(x)
    return x


def _lp_polish(x, deadline):
    x = _normed(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = float(np.max(c_full))
        sel = c_full > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: need rows repeat structure to match
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::(2 * d + 1)] = np.full(p, 0)  # placeholder, rebuilt below
        # build correctly row by row vectorised:
        base = np.arange(d)
        col_block = np.concatenate(
            [np.tile(base, p), np.tile(base, p), np.zeros(p, dtype=int)], axis=0
        )
        # Instead build with repeat logic:
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)).reshape(-1),
            # we need interlaced per row, so rebuild rows/cols jointly:
        ])
        # Simpler: build dense rows/cols via loops in numpy:
        rr = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.empty(p * (2 * d + 1), dtype=int)
        vv = np.empty(p * (2 * d + 1), dtype=float)
        # x[J] and x[I] blocks
        xJ = x[J]  # p,d
        xI = x[I]
        for k in range(p):
            s = k * (2 * d + 1)
            cc[s:s + d] = I[k] * d + np.arange(d)
            vv[s:s + d] = xJ[k]
            cc[s + d:s + 2 * d] = J[k] * d + np.arange(d)
            vv[s + d:s + 2 * d] = xI[k]
            cc[s + 2 * d] = n * d
            vv[s + 2 * d] = -1.0
        A_ub = coo_matrix((vv, (rr, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (np.tile(x.ravel(), 1),
             (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_full[sel],
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            # keep x at best region, continue shrinking
            x = best_x.copy() if radius < 5e-4 else x
    return best_x


def solve():
    t0 = time.time()
    LIMIT = 178.0
    rng = np.random.default_rng(0)
    # load start
    best_x = None
    best_v = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (76, 15):
                a = _normed(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _normed(rng.standard_normal((76, 15)))
        best_v = _max_cos(best_x)
    n, d = 76, 15
    cur_x = best_x.copy()
    cur_v = best_v

    explore_deadline = t0 + 132.0
    stuck = 0
    trial = 0
    while time.time() < explore_deadline:
        trial += 1
        tx = cur_x.copy()
        r = rng.random()
        if r < 0.38:
            sigma = [0.05, 0.12, 0.25, 0.40][int(rng.integers(0, 4))]
            tx = tx + sigma * rng.standard_normal(size=tx.shape)
            tx = _normed(tx)
        elif r < 0.80:
            # hole-optimized re-insertion of 1-3 points
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            per_max = np.max(G, axis=1)
            order = np.argsort(-per_max)
            k = int(rng.integers(1, 4))
            chosen = []
            if rng.random() < 0.8:
                chosen.append(int(order[0]))
            while len(chosen) < k:
                if rng.random() < 0.5:
                    # pick among top 15 worst
                    cand_idx = int(order[int(rng.integers(0, min(15, n)))])
                else:
                    cand_idx = int(rng.integers(0, n))
                if cand_idx not in chosen:
                    chosen.append(cand_idx)
            for idx in chosen:
                # candidates far from others
                C = 1500
                cand = _normed(rng.standard_normal((C, d)))
                # use current tx others
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                others = tx[mask]
                dots = cand @ others.T
                md = np.max(dots, axis=1)
                tx[idx] = cand[int(np.argmin(md))]
            # small shake of the rest to allow adjustment
            tx = tx + 0.02 * rng.standard_normal(size=tx.shape)
            tx = _normed(tx)
        else:
            # kick a max pair
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ai, aj = ij // n, ij % n
            sigma = float([0.15, 0.3][int(rng.integers(0, 2))])
            tx[ai] = tx[ai] + sigma * rng.standard_normal(d)
            tx[aj] = tx[aj] + sigma * rng.standard_normal(d)
            # also shake all a little
            tx = tx + 0.03 * rng.standard_normal(size=tx.shape)
            tx = _normed(tx)

        steps = int(rng.integers(350, 900))
        # randomize annealing slightly but deterministic via rng
        T0 = float(rng.choice([0.015, 0.02, 0.03]))
        T1 = float(rng.choice([0.003, 0.004, 0.006]))
        e0 = float(rng.choice([0.015, 0.02, 0.03]))
        e1 = float(rng.choice([0.001, 0.002, 0.003]))
        tx = _smooth_refine(tx, steps, T0, T1, e0, e1)
        v = _max_cos(tx)
        if v < best_v - 1e-12:
            best_v = v
            best_x = tx.copy()
            cur_x = tx.copy()
            cur_v = v
            stuck = 0
        elif v < cur_v:
            cur_x = tx.copy()
            cur_v = v
            stuck = 0
        else:
            stuck += 1
            # sideways / uphill to escape
            if stuck > 25:
                # restart from best with big kick
                tx2 = best_x + 0.35 * rng.standard_normal(size=tx.shape)
                tx2 = _normed(tx2)
                tx2 = _smooth_refine(tx2, 400, 0.02, 0.005, 0.02, 0.002)
                v2 = _max_cos(tx2)
                if v2 < best_v - 1e-12:
                    best_v = v2
                    best_x = tx2.copy()
                cur_x = tx2
                cur_v = v2
                stuck = 0
            elif rng.random() < 0.06:
                cur_x = tx.copy()
                cur_v = v

        if best_v < 0.15:  # early huge win
            break

    # final polish of best (and cur if different) with LP
    deadline = t0 + LIMIT - 1.0
    # quick smooth final at low T before LP
    try:
        best_x = _smooth_refine(best_x.copy(), 600, 0.008, 0.002, 0.008, 0.0008)
        best_v = _max_cos(best_x)
    except Exception:
        pass
    if time.time() < deadline - 2.0:
        try:
            px = _lp_polish(best_x, deadline)
            pv = _max_cos(px)
            if pv < best_v:
                best_v = pv
                best_x = px
        except Exception:
            pass
    best_x = _normed(best_x)
    return best_x
