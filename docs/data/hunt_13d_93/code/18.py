import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if np.isfinite(v) and v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)

    def adam_topk(X, steps, tau0, tau1, lr0, topK=400, b1=0.9, b2=0.999, eps=1e-8):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        t = 0
        if tau0 == tau1:
            lrat = 0.0
        else:
            lrat = np.log(tau1 / tau0)
        for s in range(steps):
            if time.time() - t0 > LIMIT - 2.0:
                break
            frac = s / max(1, steps - 1)
            tau = tau0 * np.exp(lrat * frac)
            lr = lr0 * (0.35 + 0.65 * (1.0 - frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mg = float(np.max(G))
            # topK mask from upper triangle
            if topK is not None and topK < P:
                flat = G[iu0, iu1]
                # kth largest
                kth = np.partition(flat, P - topK)[P - topK]
                # mask symmetric
                M = (G >= kth)
                np.fill_diagonal(M, False)
                W = np.exp((G - mg) / tau) * M
            else:
                W = np.exp((G - mg) / tau)
                np.fill_diagonal(W, 0.0)
            # zero tiny to sparsify
            # gradient
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            t += 1
            m = b1 * m + (1.0 - b1) * gt
            vv = b2 * v + (1.0 - b2) * (gt * gt)
            v[:] = vv
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            step = lr * mh / (np.sqrt(vh) + eps)
            sd = np.sum(step * X, axis=1, keepdims=True)
            step = step - sd * X
            # clip large steps
            sn = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
            clip = 0.02
            over = sn > clip
            if np.any(over):
                step = np.where(over, step * (clip / sn), step)
            X = _norm(X - step)
        return X

    def pair_stochastic(X, iters, sigma0=3e-4):
        X = X.copy()
        cur = _maxcos(X)
        sigma = sigma0
        for it in range(iters):
            if time.time() - t0 > LIMIT - 2.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            ii = k // n
            jj = k % n
            if ii == jj:
                flat = np.argmax(G[iu0, iu1])
                ii, jj = int(iu0[flat]), int(iu1[flat])
            # propose tangent noise on ii,jj only
            Y = X.copy()
            for idx in (ii, jj):
                xi = X[idx]
                rr = rng.standard_normal(d)
                rr = rr - np.dot(rr, xi) * xi
                nn = np.linalg.norm(rr) + 1e-12
                rr = rr / nn * sigma * (0.5 + rng.random())
                y = xi - rr if idx == ii else xi - rr
                # note: direction random; we want to explore, acceptance decides
                # bias away from the other point half the time
                if rng.random() < 0.7:
                    other = X[jj] if idx == ii else X[ii]
                    away = other - np.dot(other, xi) * xi
                    an = np.linalg.norm(away) + 1e-12
                    away = away / an
                    y = xi - sigma * away * (0.5 + rng.random()) - 0.3 * rr
                y = y / (np.linalg.norm(y) + 1e-12)
                Y[idx] = y
            val = _maxcos(Y)
            if val < cur:
                X = Y
                cur = val
                sigma = min(sigma * 1.1, 5e-3)
            else:
                sigma *= 0.985
                if sigma < 2e-5:
                    sigma = sigma0 * 0.5
        return X

    def void_reinsert(X):
        # find victim: worst pair + degree
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(np.max(G))
        # degree in near-active band
        deg = np.sum(G > mg - 5e-4, axis=1)
        # candidate victims: top degree
        order = np.argsort(-deg)
        # pick among top 6 at random to diversify
        cand_v = order[:6]
        victim = int(cand_v[rng.integers(0, len(cand_v))])
        # also sometimes pick one end of worst pair
        if rng.random() < 0.5:
            k = int(np.argmax(G))
            ii = k // n; jj = k % n
            victim = int(ii if rng.random() < 0.5 else jj)
        others = np.delete(X, victim, axis=0)  # 92x13
        # search random candidates
        K = 3000
        C = rng.standard_normal((K, d))
        C = _norm(C)
        dots = C @ others.T  # K x 92
        mvals = np.max(dots, axis=1)
        # best 5
        bidx = np.argpartition(mvals, 5)[:5]
        pick = bidx[rng.integers(0, 5)]
        Y = X.copy()
        Y[victim] = C[pick]
        return Y

    def lp_polish(X, secs, r_init=0.0015, p_cap=2200):
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        t_start = time.time()
        X = _norm(X.copy())
        cur = _maxcos(X)
        radius = r_init
        while (time.time() - t_start < secs and time.time() - t0 < LIMIT - 1.0 and radius > 1e-14):
            G = X @ X.T
            c_all = G[iu0, iu1]
            m = float(np.max(c_all))
            delta = max(12.0 * radius, 1.5e-4)
            sel_idx = np.where(c_all > m - delta)[0]
            if len(sel_idx) == 0:
                sel_idx = np.array([int(np.argmax(c_all))])
            if len(sel_idx) > p_cap:
                part = np.argpartition(-c_all[sel_idx], p_cap)[:p_cap]
                sel_idx = sel_idx[part]
            I = iu0[sel_idx]; J = iu1[sel_idx]
            p = len(sel_idx)
            c_sel = c_all[sel_idx]
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cb = np.empty((p, 2 * d + 1), dtype=np.int64)
            cb[:, :d] = I[:, None] * d + np.arange(d)
            cb[:, d:2 * d] = J[:, None] * d + np.arange(d)
            cb[:, 2 * d] = n * d
            cols = cb.ravel()
            vals2 = np.empty((p, 2 * d + 1), dtype=np.float64)
            vals2[:, :d] = X[J]
            vals2[:, d:2 * d] = X[I]
            vals2[:, 2 * d] = -1.0
            vals = vals2.ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                er = np.repeat(np.arange(n), d)
                ec = np.arange(n * d)
                ev = X.ravel()
                A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                bnds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bnds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _norm(Y)
            vv = _maxcos(Y)
            if vv < cur:
                X, cur = Y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
            if time.time() - t_start > secs - 0.4:
                break
        return X

    # quick initial tie attempt on loaded best
    cur = best.copy()
    cur_val = best_val
    # short sharp relax to ensure local opt
    if time.time() - t0 < LIMIT - 20:
        cur = adam_topk(cur, steps=400, tau0=0.004, tau1=0.0004, lr0=0.002, topK=500)
        cur = pair_stochastic(cur, iters=120, sigma0=2e-4)
        v = _maxcos(cur)
        if v < best_val:
            best, best_val = cur.copy(), v
        else:
            cur, cur_val = best.copy(), best_val

    hop_end = t0 + 128.0
    trial = 0
    Temp = 4e-4
    while time.time() < hop_end and time.time() - t0 < LIMIT - 45 and trial < 200:
        trial += 1
        r = rng.random()
        # restart from best periodically
        if trial % 7 == 0:
            cur, cur_val = best.copy(), best_val
        Y = cur.copy()
        if r < 0.30:
            sig = 0.006
            Y = _norm(Y + sig * rng.standard_normal((n, d)))
            Y = adam_topk(Y, steps=500, tau0=0.008, tau1=0.0006, lr0=0.003, topK=600)
            Y = adam_topk(Y, steps=400, tau0=0.0006, tau1=0.00008, lr0=0.001, topK=120)
            Y = pair_stochastic(Y, iters=60, sigma0=1.5e-4)
        elif r < 0.50:
            sig = 0.018
            Y = _norm(Y + sig * rng.standard_normal((n, d)))
            Y = adam_topk(Y, steps=500, tau0=0.012, tau1=0.0008, lr0=0.0035, topK=800)
            Y = adam_topk(Y, steps=400, tau0=0.0008, tau1=0.0001, lr0=0.0012, topK=150)
            Y = pair_stochastic(Y, iters=60, sigma0=2e-4)
        elif r < 0.65:
            sig = 0.045
            Y = _norm(Y + sig * rng.standard_normal((n, d)))
            Y = adam_topk(Y, steps=550, tau0=0.015, tau1=0.001, lr0=0.004, topK=1000)
            Y = adam_topk(Y, steps=350, tau0=0.001, tau1=0.00015, lr0=0.0012, topK=200)
        elif r < 0.82:
            # pair-only kick
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            ii = k // n; jj = k % n
            for idx in (ii, jj):
                rr = rng.standard_normal(d)
                rr = rr - np.dot(rr, Y[idx]) * Y[idx]
                rr = rr / (np.linalg.norm(rr) + 1e-12)
                Y[idx] = _norm((Y[idx] + 0.012 * rr).reshape(1, -1))[0]
            # plus small global noise
            Y = _norm(Y + 0.003 * rng.standard_normal((n, d)))
            Y = adam_topk(Y, steps=500, tau0=0.006, tau1=0.0003, lr0=0.0025, topK=400)
            Y = adam_topk(Y, steps=400, tau0=0.0003, tau1=0.00006, lr0=0.0008, topK=60)
            Y = pair_stochastic(Y, iters=80, sigma0=2e-4)
        else:
            # void jump
            Y = void_reinsert(Y)
            Y = adam_topk(Y, steps=600, tau0=0.010, tau1=0.0007, lr0=0.0035, topK=800)
            Y = adam_topk(Y, steps=400, tau0=0.0007, tau1=0.0001, lr0=0.001, topK=150)
            Y = pair_stochastic(Y, iters=60, sigma0=2e-4)

        v = _maxcos(Y)
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            best, best_val = Y.copy(), v
            cur, cur_val = Y.copy(), v
            Temp = max(Temp * 0.9, 5e-5)
        elif v < cur_val:
            cur, cur_val = Y.copy(), v
        else:
            # Metropolis accept worse to walk across plateau
            prog = (time.time() - t0) / LIMIT
            T = Temp * (1.0 - 0.7 * prog)
            if rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                cur, cur_val = Y.copy(), v

    # final intensification on best
    remain = LIMIT - (time.time() - t0)
    if remain > 10:
        best = adam_topk(best, steps=700, tau0=0.001, tau1=0.00008, lr0=0.0008, topK=120)
        best_val = _maxcos(best)
        best = pair_stochastic(best, iters=200, sigma0=1.2e-4)
        best_val = _maxcos(best)
    remain = LIMIT - (time.time() - t0)
    if remain > 8:
        lp_t = min(remain - 1.5, 38.0)
        best = lp_polish(best, secs=lp_t * 0.6, r_init=8e-4, p_cap=2200)
        best_val = _maxcos(best)
        remain2 = LIMIT - (time.time() - t0)
        if remain2 > 5:
            best = adam_topk(best, steps=300, tau0=0.0002, tau1=0.00005, lr0=0.0004, topK=80)
            best = lp_polish(best, secs=remain2 - 1.2, r_init=2e-4, p_cap=2500)
            best_val = _maxcos(best)

    best = _norm(best)
    return best
