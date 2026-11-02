import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        c = c_all[sel]
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_blocks = np.empty((p, 2 * d + 1), dtype=int)
        cols_blocks[:, :d] = I[:, None] * d + np.arange(d)
        cols_blocks[:, d:2*d] = J[:, None] * d + np.arange(d)
        cols_blocks[:, 2*d] = n * d
        cols = cols_blocks.ravel()
        vals_blocks = np.empty((p, 2 * d + 1))
        vals_blocks[:, :d] = x[J]
        vals_blocks[:, d:2*d] = x[I]
        vals_blocks[:, 2*d] = -1.0
        vals = vals_blocks.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def adam_subset(Y, F, iters, k0, k1, lr):
    # Y (k,d) movable, F (nf,d) frozen
    k = Y.shape[0]
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(iters):
        ke = k0 + (k1 - k0) * (it / max(1, iters - 1))
        Gmm = Y @ Y.T
        np.fill_diagonal(Gmm, -2.0)
        Gmf = Y @ F.T
        vmax = Gmm.max()
        mfmax = Gmf.max() if Gmf.size else -2.0
        if mfmax > vmax:
            vmax = mfmax
        Emm = np.exp(ke * (Gmm - vmax))
        np.fill_diagonal(Emm, 0.0)
        Emf = np.exp(ke * (Gmf - vmax))
        sum_tri = Emm.sum() * 0.5
        sum_mf = Emf.sum()
        S = sum_tri + sum_mf
        if S <= 0 or not np.isfinite(S):
            break
        Wmm = Emm / S
        Wmf = Emf / S
        grad = Wmm @ Y + Wmf @ F
        dot = (grad * Y).sum(axis=1, keepdims=True)
        g = grad - dot * Y
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        Y = Y - step
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def adam_full(y, iters, k0, k1, lr):
    n, d = y.shape
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for it in range(iters):
        ke = k0 + (k1 - k0) * (it / max(1, iters - 1))
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        E = np.exp(ke * (G - gmax))
        np.fill_diagonal(E, 0.0)
        s = E.sum() * 0.5
        if s <= 0 or not np.isfinite(s):
            break
        W = E / s
        grad = W @ y
        dot = (grad * y).sum(axis=1, keepdims=True)
        g = grad - dot * y
        t += 1
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * (g * g)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t_start = time.time()
    HARD = 178.0
    EXPL_CUTOFF = 150.0
    rng = np.random.default_rng(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        r = min(xb.size, tmp.size)
        tmp.ravel()[:r] = xb.ravel()[:r]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    ks = [6, 8, 10, 12]
    trial = -1
    # precompute nothing; loop
    while time.time() - t_start < EXPL_CUTOFF:
        trial += 1
        elapsed = time.time() - t_start
        remain_expl = EXPL_CUTOFF - elapsed
        if remain_expl < 9.0:
            break
        k = ks[trial % len(ks)]
        # current best gram analysis
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # tight contacts for low-contact ranking
        tight = G > (m - 1e-9)
        cnt_tight = tight.sum(axis=1)
        # loose adjacency for cluster
        loose = G > (m - 1e-4)
        # sorted pairs descending
        iu0 = np.triu_indices(n, 1)
        order = np.argsort(-G[iu0])
        seed_rank = (trial // len(ks)) % 12
        if seed_rank >= len(order):
            seed_rank = seed_rank % len(order)
        a = int(iu0[0][order[seed_rank]])
        b = int(iu0[1][order[seed_rank]])
        mode = trial % 3
        if mode in (0, 1):
            # jammed cluster from seed pair + shared neighbours
            neigh_a = set(np.where(loose[a])[0].tolist())
            neigh_b = set(np.where(loose[b])[0].tolist())
            shared = list((neigh_a & neigh_b) - {a, b})
            # sort shared by tight degree descending
            shared.sort(key=lambda v: -int(cnt_tight[v]))
            cluster = [a, b]
            for v in shared:
                if len(cluster) >= k:
                    break
                if v not in cluster:
                    cluster.append(v)
            # BFS expansion if needed
            if len(cluster) < k:
                # candidates: neighbors of cluster sorted by loose degree
                in_c = set(cluster)
                cand_pool = set()
                for v in cluster:
                    cand_pool.update(np.where(loose[v])[0].tolist())
                cand_pool -= in_c
                # remove self indices confusion: loose diag is False so fine
                cands = sorted(list(cand_pool), key=lambda v: -int(loose[v].sum()))
                for v in cands:
                    if len(cluster) >= k:
                        break
                    cluster.append(int(v))
            if len(cluster) < k:
                # fill by highest max (worst points)
                per = G.max(axis=1)
                worst = np.argsort(-per)
                for v in worst:
                    if int(v) not in cluster:
                        cluster.append(int(v))
                    if len(cluster) >= k:
                        break
            # for mode==1 perturb seed choice slightly: swap one member with low-contact point
            if mode == 1 and len(cluster) == k:
                # replace last element with a low-contact point not in cluster
                lo = np.argsort(cnt_tight)
                for v in lo:
                    if int(v) not in cluster:
                        cluster[-1] = int(v)
                        break
            del_idx = np.array(cluster[:k], dtype=int)
        else:
            # lowest-contact points, biased to indices 17-35
            if (trial // 3) % 2 == 0:
                # global lowest
                del_idx = np.argsort(cnt_tight)[:k]
            else:
                # restricted to 17..35 inclusive, plus fill if needed
                sub = np.arange(17, 36)
                sub_cnt = cnt_tight[sub]
                o = np.argsort(sub_cnt)
                chosen = sub[o[:min(k, len(sub))]].tolist()
                if len(chosen) < k:
                    glo = np.argsort(cnt_tight)
                    for v in glo:
                        if int(v) not in chosen:
                            chosen.append(int(v))
                        if len(chosen) >= k:
                            break
                del_idx = np.array(chosen[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        frozen = x_best[keep_mask]  # (nf,d)
        # greedy reinsertion from 30000 candidates
        M = 30000
        C = rng.standard_normal((M, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        Sf = C @ frozen.T
        runmax = Sf.max(axis=1)
        chosen_list = []
        for tt in range(k):
            bi = int(np.argmin(runmax))
            ch = C[bi].copy()
            chosen_list.append(ch)
            if tt + 1 < k:
                dots = C @ ch
                # running max against frozen + already chosen
                np.maximum(runmax, dots, out=runmax)
                # prevent reselection: set selected to large
                runmax[bi] = 2.0
        Y0 = np.stack(chosen_list, axis=0)
        # small jitter for diversity on some trials
        if trial % 4 == 3:
            Y0 = Y0 + 0.02 * rng.standard_normal(Y0.shape)
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        # optimise movable only
        # check time: need at least ~6s for adam+lp; adam_subset 1500 steps
        Yopt = adam_subset(Y0, frozen, 1500, 80.0, 800.0, 0.01)
        y = np.empty_like(x_best)
        y[keep_mask] = frozen
        y[~keep_mask] = Yopt
        # short unfreeze Adam phase (200 steps)
        # adapt k range for full phase
        try:
            y = adam_full(y, 200, 120.0, 900.0, 0.008)
        except Exception:
            pass
        v_pre = max_cos(y)
        # only LP-polish if promising to save time: threshold
        # frozen-frozen max is fixed = m (old best); if v_pre huge skip LP
        if v_pre > best + 0.02:
            continue
        # LP polish 5-8 s
        sec = 5.0 if (trial % 2 == 0) else 7.0
        # clamp to remaining exploration time
        sec = min(sec, EXPL_CUTOFF - (time.time() - t_start) - 0.5)
        if sec < 2.0:
            # still evaluate without polish
            if v_pre < best - 1e-12:
                best = v_pre
                x_best = y / np.linalg.norm(y, axis=1, keepdims=True)
            continue
        try:
            y3 = lp_polish(y, sec)
        except Exception:
            continue
        v3 = max_cos(y3)
        if v3 < best - 1e-12:
            best = v3
            x_best = y3
        # time guard for hard limit
        if time.time() - t_start > HARD - 25:
            break
    # final 20 s LP polish
    rem = HARD - (time.time() - t_start) - 1.0
    if rem > 2:
        rem = min(rem, 20.0)
        try:
            x_best = lp_polish(x_best, rem)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
