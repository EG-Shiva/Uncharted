import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def smooth_polish(X, steps, T0=0.02, T1=0.004, lr0=0.08, lr1=0.01, mom=0.8):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        s = float(np.sum(W)) / n + 1e-12
        grad = grad / s
        V = mom * V + (1.0 - mom) * grad
        X = X - lr * V
        X = _norm(X)
    return X


def lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def build_pool(rng):
    # all nonzero {-1,0,1}^9 : 3^9-1 = 19682
    N = 3 ** 9
    a = np.arange(N)
    pw = 3 ** np.arange(9)
    digits = ((a[:, None] // pw[None, :]) % 3).astype(np.int8) - 1  # -1,0,1? mapping 0->-1,1->0,2->1
    # digits== -1 from 0, 0 from 1, 1 from 2 -> actually values in {-1,0,1}
    mask = np.any(digits != 0, axis=1)
    tern = digits[mask].astype(float)
    tern = tern / np.linalg.norm(tern, axis=1, keepdims=True)
    # subsampled {-2,-1,0,1,2}^9
    S = 22000
    q = rng.integers(-2, 3, size=(S, 9)).astype(float)
    nz = np.linalg.norm(q, axis=1) > 1e-12
    q = q[nz]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    # dedup directions approximately: round to 1e-8 and use set via abs? keep simple: stack
    pool = np.vstack([tern, q])
    return pool


def greedy_select(pool, rng, topk=8):
    P = pool.shape[0]
    best_score = np.full(P, -2.0)
    selected = []
    # random start
    start = int(rng.integers(0, P))
    selected.append(start)
    best_score = np.maximum(best_score, pool @ pool[start])
    best_score[start] = np.inf
    n_target = 131
    for _ in range(1, n_target):
        # find topk smallest among unselected
        # argpartition for topk
        k = min(topk, P - len(selected))
        part = np.argpartition(best_score, k)[:k]
        # choose randomly among them
        pick = int(rng.choice(part))
        selected.append(pick)
        d = pool @ pool[pick]
        # update
        np.maximum(best_score, d, out=best_score)
        for s in selected:
            best_score[s] = np.inf
    return np.array(selected, dtype=int)


def swap_improve(sel_vecs, pool, rng, sweeps=4, sample_pool=4000):
    n, d = sel_vecs.shape
    S = sel_vecs.copy()
    cur_max = _max_cos(S)
    P = pool.shape[0]
    for _ in range(sweeps):
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # degree at near max
        deg = np.sum(G >= m - 1e-7, axis=1)
        order = np.argsort(-deg)[:12]
        # sample pool
        samp_idx = rng.choice(P, size=min(sample_pool, P), replace=False)
        C = pool[samp_idx] @ S.T  # (Sp, n)
        improved = False
        for i in order:
            # max excluding i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Mex = float(np.max(G[np.ix_(mask, mask)]))
            # candidate max if swap i with each sampled p
            rest_max = np.max(np.delete(C, i, axis=1), axis=1)
            cand = np.maximum(Mex, rest_max)
            bk = int(np.argmin(cand))
            if cand[bk] < m - 1e-12:
                S[i] = pool[samp_idx[bk]]
                m = float(cand[bk])
                cur_max = m
                improved = True
                break
        if not improved:
            # try reduce count of near-max pairs
            cnt_now = int(np.sum(G >= m - 1e-7))
            best_cnt = cnt_now
            best_swap = None
            for i in order[:6]:
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                Mex = float(np.max(G[np.ix_(mask, mask)]))
                rest_max = np.max(np.delete(C, i, axis=1), axis=1)
                # only consider swaps not increasing max beyond 1e-12
                feas = np.where(rest_max <= m + 1e-12)[0]
                if len(feas) == 0:
                    continue
                for bk in feas[:2000]:
                    # estimate new count roughly: count pairs excl i + count new dots near max
                    cvec = C[bk]
                    new_close = int(np.sum(cvec[mask] >= m - 1e-7))
                    excl_close = int(np.sum(G[np.ix_(mask, mask)] >= m - 1e-7))
                    tot = excl_close + 2 * new_close
                    if tot < best_cnt:
                        best_cnt = tot
                        best_swap = (i, samp_idx[feas[0] if False else bk])
                        break
                if best_swap is not None:
                    break
            if best_swap is not None:
                i, pidx = best_swap
                # find actual pool vector
                S[i] = pool[pidx]
            else:
                break
    return S


def contact_stats(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G >= m - 1e-9
    deg = C.sum(axis=1)
    return m, deg


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(20261102)
    n, d = 131, 9
    # load incumbent
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()
    incumbent_val = best_val

    pool = build_pool(rng)

    seeds = []
    # greedy discrete seeds
    n_seeds = 7
    for s in range(n_seeds):
        if time.time() > t0 + 40:
            break
        topk = 6 + (s % 4) * 2  # 6,8,10,12 variety
        sel_idx = greedy_select(pool, rng, topk=topk)
        S = pool[sel_idx]
        S = swap_improve(S, pool, rng, sweeps=5, sample_pool=5000)
        # random orthogonal transform + tiny jitter to leave exact lattice
        # (preserves Gram, jitter breaks it slightly for diversity)
        try:
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            Srot = S @ Q
            # keep both original and rotated as candidates (rotated + jitter)
            if s % 2 == 1:
                S = _norm(Srot + 1e-3 * rng.standard_normal(S.shape))
        except Exception:
            pass
        seeds.append(S)

    # quick smooth polish of each seed
    polished = []
    for S in seeds:
        if time.time() > t0 + 75:
            break
        X = S.copy()
        # short repulsion
        X = smooth_polish(X, steps=250, T0=0.025, T1=0.005, lr0=0.09, lr1=0.015)
        v = _max_cos(X)
        m, deg = contact_stats(X)
        n_zero = int(np.sum(deg == 0))
        # retain if no zero-contact preferred, but keep all sorted by value
        polished.append((v, n_zero, X))

    if len(polished) == 0:
        return incumbent

    polished.sort(key=lambda t: (t[0], t[1]))
    # keep top 3 diverse: check edge overlap vs incumbent later, just take top 2
    cur_best = polished[0][2].copy()
    cur_val = polished[0][0]

    if cur_val < best_val:
        best_val = cur_val
        best = cur_best.copy()

    # basin hopping around discrete-derived best for a while
    # leave ~45s for final LP
    hop_deadline = min(DEADLINE - 45.0, t0 + 125.0)
    it = 0
    while time.time() < hop_deadline:
        it += 1
        # pick base: mostly cur_best, sometimes 2nd best
        if len(polished) > 1 and it % 5 == 0:
            base = polished[1][2]
        else:
            base = cur_best if cur_val < polished[0][0] + 0.01 else polished[0][2]
        Y = base.copy()
        r = rng.random()
        if r < 0.35:
            Y = Y + 0.015 * rng.standard_normal(Y.shape)
        elif r < 0.6:
            Y = Y + 0.035 * rng.standard_normal(Y.shape)
        elif r < 0.8:
            # kick worst points
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G >= m - 1e-6, axis=1)
            order = np.argsort(-deg)[:5]
            k = int(rng.integers(1, 4))
            idx = rng.choice(order, size=k, replace=False)
            Y[idx] += 0.15 * rng.standard_normal((k, d))
        else:
            # hole reinsertion: replace worst point by random far vector
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            deg = np.sum(G >= float(np.max(G)) - 1e-6, axis=1)
            i = int(np.argmax(deg))
            z = rng.standard_normal(d)
            z = z / np.linalg.norm(z)
            # refine hole a bit: move away from others
            for _ in range(60):
                dots = Y @ z
                # weight close points
                W = np.exp((dots - float(np.max(dots))) / 0.01)
                g = W @ Y
                g = g - float(np.dot(g, z)) * z
                z = z - 0.05 * g
                z = z / np.linalg.norm(z)
            Y[i] = z
        Y = _norm(Y)
        Y = smooth_polish(Y, steps=180, T0=0.018, T1=0.004, lr0=0.07, lr1=0.012)
        v = _max_cos(Y)
        if v < cur_val:
            cur_val = v
            cur_best = Y.copy()
            if v < best_val:
                best_val = v
                best = Y.copy()
        # early exit if time
        if time.time() > hop_deadline:
            break

    # final LP squeeze on the best found (prefer discrete-derived if competitive)
    # if discrete-derived is much worse than incumbent, squeeze incumbent instead as fallback
    # choose the better of cur_best and incumbent for final polish
    if cur_val <= incumbent_val + 0.002:
        final_x = cur_best.copy()
    else:
        final_x = incumbent.copy()
    # give LP remaining time minus small margin
    lp_deadline = DEADLINE - 2.0
    final_x = lp_polish_deadline(final_x, lp_deadline)
    fv = _max_cos(final_x)
    if fv < best_val:
        best_val = fv
        best = final_x.copy()

    # safety: return the absolute best among best and incumbent
    if _max_cos(best) > incumbent_val and _max_cos(final_x) > incumbent_val:
        return incumbent
    # choose minimum
    cands = [(incumbent_val, incumbent), (best_val, best), (fv, final_x)]
    cands.sort(key=lambda t: t[0])
    out = _norm(cands[0][1])
    return out
