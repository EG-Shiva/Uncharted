import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 160.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 59, 22

    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(np.min(vals))

    def adam_run(X0, steps, a0, a1, lr0, lr1, mask=None):
        # mask: None => all rows; else boolean or index list of rows to update
        X = X0.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        cur_best = X.copy()
        cur_val = _max_cos(X)
        la0, la1 = np.log(a0), np.log(a1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        if mask is None:
            upd_mask = None
        else:
            upd_mask = np.zeros(n, dtype=bool)
            upd_mask[np.asarray(mask, dtype=int)] = True
        for t in range(1, steps + 1):
            if (t & 63) == 0 and (time.time() - t0) > BUDGET:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            G -= M
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (G @ X) / S
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            if upd_mask is not None:
                # zero grad for frozen rows so m/v decay there is harmless
                Gt = Gt.copy()
                Gt[~upd_mask, :] = 0.0
            m *= b1
            m += (1.0 - b1) * Gt
            v *= b2
            v += (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - (b1 ** t)
            c2 = 1.0 - (b2 ** t)
            mc = m / c1
            vc = v / c2
            upd = mc / (np.sqrt(vc) + eps)
            if upd_mask is None:
                X -= lr * upd
            else:
                X[upd_mask] -= lr * upd[upd_mask]
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
            if (t % 300 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv
            cur_best = X.copy()
        return cur_best, cur_val

    # short initial squeeze
    if (time.time() - t0) < BUDGET - 10:
        best, best_val = adam_run(best, steps=2500, a0=100.0, a1=1500.0, lr0=0.004, lr1=0.0003)

    # precomputed pool >=20000
    POOL = 22000
    pool = rng.standard_normal((POOL, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)

    def top_neighbours(X, center, k):
        row = X[center] @ X.T
        row[center] = -2.0
        idx = np.argsort(-row)[:k]
        return idx.tolist()

    # order centers by current rowmax (high-contact first)
    def center_order(X, bv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        # contacts within 0.005 of max as load measure
        cnt = np.sum(G > (bv - 0.005), axis=1)
        order = np.argsort(-(rm + 0.001 * cnt))
        return order.tolist(), rm, cnt

    attempt = 0
    # cycle sizes 6..10 (neighbours 5..9)
    sizes = [7, 8, 6, 9, 7, 10, 6, 8]
    si = 0
    # main cluster loop
    while (time.time() - t0) < BUDGET - 8:
        order, rowmax, cnt = center_order(best, best_val)
        # try a few centers per outer round
        progressed = False
        for ci in order[:12]:
            if (time.time() - t0) > BUDGET - 8:
                break
            K = sizes[si % len(sizes)]
            si += 1
            # need K points: center + K-1 neighbours
            neigh = top_neighbours(best, ci, K - 1)
            cluster = [int(ci)] + [int(x) for x in neigh if int(x) != int(ci)]
            # ensure size K (pad with next best if duplicate)
            if len(cluster) < K:
                for cand in order:
                    if cand not in cluster:
                        cluster.append(int(cand))
                    if len(cluster) >= K:
                        break
            cluster = cluster[:K]
            # time check every cluster (and inside adam every 50)
            attempt += 1
            Xtry = best.copy()
            # random order of vacancies (deterministic via rng)
            vac_order = cluster.copy()
            # shuffle
            perm = rng.permutation(len(vac_order))
            vac_order = [vac_order[p] for p in perm]
            frozen_complement = [i for i in range(n) if i not in cluster]
            done = []
            # adapt subset steps to remaining time
            remain = BUDGET - (time.time() - t0)
            if remain < 20:
                sub_steps, full_steps = 150, 700
            elif remain < 60:
                sub_steps, full_steps = 300, 1200
            else:
                sub_steps, full_steps = 350, 1500
            ok = True
            for vi, v in enumerate(vac_order):
                if (time.time() - t0) > BUDGET - 6:
                    ok = False
                    break
                Fidx = frozen_complement + done
                F = Xtry[np.array(Fidx, dtype=int)]
                # one matmul: pool vs frozen
                S = pool @ F.T
                # max over frozen
                smax = np.max(S, axis=1)
                # best 100 candidates
                part = np.argpartition(smax, 100)[:100]
                # sort those 100
                sub = smax[part]
                o = np.argsort(sub)
                top100 = part[o]
                # pick among top 5 randomly for diversity (deterministic)
                pick = int(rng.integers(0, 5))
                chosen = int(top100[pick])
                Xtry[v] = pool[chosen]
                done.append(int(v))
                # locally optimise only reinserted points against frozen set
                # use mid alpha anneal
                Xtry, _ = adam_run(Xtry, steps=sub_steps, a0=80.0, a1=600.0,
                                   lr0=0.006, lr1=0.0006, mask=done)
                if (time.time() - t0) > BUDGET - 6:
                    ok = False
                    break
            if not ok:
                break
            # unfreeze everything for short polish
            Xpol, vpol = adam_run(Xtry, steps=full_steps, a0=120.0, a1=1200.0,
                                  lr0=0.003, lr1=0.0003, mask=None)
            if vpol < best_val - 1e-12:
                best = Xpol
                best_val = vpol
                progressed = True
            # time check every 50 iterations is inside adam_run (every 64);
            # also check here
            if (time.time() - t0) > BUDGET - 5:
                break
        if (time.time() - t0) > BUDGET - 5:
            break
        # if no progress for a full round, add a small global shake to get new order?
        # keep cycling; rng state advances so picks differ
        if not progressed:
            # occasional tiny shake of worst point to diversify centres without losing best
            pass

    # final tiny squeeze if time remains
    if (time.time() - t0) < BUDGET - 2:
        try:
            b2, v2 = adam_run(best, steps=1200, a0=800.0, a1=2000.0, lr0=0.0012, lr1=0.00012)
            if v2 < best_val:
                best, best_val = b2, v2
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
