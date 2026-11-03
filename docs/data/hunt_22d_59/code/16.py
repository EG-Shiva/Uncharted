import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    import time as _tm
    import numpy as np
    from scipy.optimize import minimize
    t0 = _tm.time()
    HARD = t0 + 172.0
    n, d = 59, 22
    rng = np.random.default_rng(7)
    iu0, iu1 = np.triu_indices(n, 1)

    def make_fun(beta):
        def fun(Yflat):
            Y = Yflat.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            v = G[iu0, iu1]
            M = float(v.max())
            e = np.exp(beta * (v - M))
            S = float(e.sum())
            loss = M + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            grad_X = W @ X
            dot = (grad_X * X).sum(axis=1, keepdims=True)
            grad_Y = (grad_X - dot * X) / nr
            return loss, grad_Y.ravel()
        return fun

    def lbfgs(X0, beta, maxiter, tight=False):
        fun = make_fun(float(beta))
        x0 = X0.ravel().copy()
        try:
            if tight:
                opt = {'maxiter': maxiter, 'maxfun': maxiter*30, 'ftol': 1e-15, 'gtol': 1e-10}
            else:
                opt = {'maxiter': maxiter, 'maxfun': maxiter*20, 'ftol': 1e-13, 'gtol': 1e-7}
            res = minimize(fun, x0, jac=True, method='L-BFGS-B', options=opt)
            Y = res.x.reshape(n, d)
        except Exception:
            Y = X0.copy()
        return _normed(Y)

    def greedy(X0, steps, lr0, t_lim):
        cur = X0.copy()
        best = X0.copy()
        bv = _max_cos(best)
        lr = float(lr0)
        for s in range(steps):
            if _tm.time() > t_lim:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < bv:
                bv = m
                best = cur.copy()
            tol = max(0.0012, 6.0*lr)
            mask = G > (m - tol)
            np.fill_diagonal(mask, False)
            grad = mask.astype(np.float64) @ cur
            dot = (grad * cur).sum(axis=1, keepdims=True)
            grad_t = grad - dot * cur
            deg = mask.sum(axis=1, keepdims=True).astype(np.float64)
            deg = np.maximum(deg, 1.0)
            step = grad_t / deg
            cur = cur - lr * step
            cur = _normed(cur)
            if (s+1) % 400 == 0:
                v = _max_cos(cur)
                if v < bv - 1e-12:
                    bv = v
                    best = cur.copy()
                else:
                    lr *= 0.55
                    cur = _normed(0.72*cur + 0.28*best)
                    if lr < 8e-7:
                        lr = 8e-7
        v = _max_cos(cur)
        if v < bv:
            bv = v
            best = cur.copy()
        return best, bv

    # load seeds
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1)
                if np.all(nr > 1e-9):
                    cands.append(a / nr[:, None])
        except Exception:
            pass
    if len(cands) == 0:
        X0 = _normed(rng.standard_normal((n, d)))
    else:
        vals = [_max_cos(c) for c in cands]
        X0 = cands[int(np.argmin(vals))].copy()
    best_X = _normed(X0)
    best_val = _max_cos(best_X)

    # initial polish
    try:
        cur = lbfgs(best_X, 300, 90, False)
        cur = lbfgs(cur, 1200, 90, False)
        cur, v = greedy(cur, 900, 7e-4, HARD-30)
        if v < best_val:
            best_val = v
            best_X = cur.copy()
    except Exception:
        pass

    elite = [(best_val, best_X.copy())]

    def elite_bases():
        # return sorted elite vals
        elite.sort(key=lambda z: z[0])
        return elite

    it = 0
    # precompute pattern
    while _tm.time() < HARD - 14:
        it += 1
        time_left = HARD - 14 - _tm.time()
        if time_left < 2.5:
            break
        elite_bases()
        # pick base: mostly best, sometimes 2nd/3rd for diversity
        r = rng.random()
        if len(elite) > 1 and r < 0.25:
            bi = int(rng.integers(0, len(elite)))
            base = elite[bi][1]
            base_val = elite[bi][0]
        else:
            base = elite[0][1]
            base_val = elite[0][0]
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        deg = (G > (float(G.max()) - 0.002)).sum(axis=1)
        order_worst = np.argsort(-rowmax)
        order_deg = np.argsort(-deg)

        mode = it % 8
        Xt = base.copy()
        if mode == 0:
            # global moderate shake
            sigma = 0.06 + 0.04 * rng.random()
            Xt = Xt + sigma * rng.standard_normal((n, d))
            Xt = _normed(Xt)
        elif mode == 1:
            # subset strong shake of high-degree hubs
            k = int(rng.integers(3, 9))
            # pick mostly from top-degree
            pool = order_deg[:18]
            ch = rng.choice(pool, size=min(k, len(pool)), replace=False)
            # add 1 random too
            if rng.random() < 0.5:
                ch = np.append(ch, rng.integers(0, n))
                ch = np.unique(ch)
            Xt[ch] = Xt[ch] + 0.35 * rng.standard_normal((len(ch), d))
            Xt = _normed(Xt)
        elif mode == 2:
            # kill worst 1-2 and reinsert at largest hole (optimized)
            k = 1 if rng.random() < 0.6 else 2
            kill = order_worst[:k]
            # also consider high deg
            if rng.random() < 0.4:
                kill = order_deg[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[kill] = False
            Xkeep = Xt[keep_mask]
            # sample candidates, pick best hole
            Nc = 4000
            C = rng.standard_normal((Nc, d))
            C = C / np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Xkeep.T
            # score = max abs? we minimize max inner product (not abs, since cosine max only positive side matters, but antipode also matters via other points? use max)
            scores = D.max(axis=1)
            # take best few and add noise diversity
            bidx = np.argpartition(scores, 4)[:4]
            # choose best
            best_c = C[bidx[int(np.argmin(scores[bidx]))]]
            for j, idx in enumerate(kill):
                if j == 0:
                    Xt[idx] = best_c
                else:
                    # second kill: second best distinct
                    Xt[idx] = C[bidx[(j) % len(bidx)]] + 0.02*rng.standard_normal(d)
            # small shake of rest
            Xt = Xt + 0.008 * rng.standard_normal((n, d))
            Xt = _normed(Xt)
        elif mode == 3:
            # large global jump to new basin
            sigma = 0.12 + 0.08 * rng.random()
            Xt = Xt + sigma * rng.standard_normal((n, d))
            Xt = _normed(Xt)
        elif mode == 4:
            # shake least-constrained + hubs mix
            order_best = np.argsort(rowmax)  # smallest rowmax first
            k1 = int(rng.integers(2, 5))
            ch = np.concatenate([order_deg[:k1], order_best[:2]])
            ch = np.unique(ch)
            Xt[ch] = Xt[ch] + 0.28 * rng.standard_normal((len(ch), d))
            Xt = _normed(Xt)
        elif mode == 5:
            # random fresh replacement of single hub + moderate shake
            kill = int(order_deg[int(rng.integers(0, 6))])
            Xt[kill] = rng.standard_normal(d)
            Xt = Xt + 0.015 * rng.standard_normal((n, d))
            Xt = _normed(Xt)
        elif mode == 6:
            # correlated block rotation: add rank-1 perturbation along random direction
            u = rng.standard_normal(d)
            u = u / np.linalg.norm(u)
            coeff = rng.standard_normal(n) * 0.09
            Xt = Xt + coeff[:, None] * u[None, :]
            Xt = _normed(Xt)
        else:
            sigma = 0.03 + 0.03 * rng.random()
            Xt = Xt + sigma * rng.standard_normal((n, d))
            Xt = _normed(Xt)

        # quick polish
        try:
            # adapt depth to time
            if mode in (2, 5):
                cur = lbfgs(Xt, 350, 55, False)
                cur = lbfgs(cur, 1100, 55, False)
            elif mode == 3:
                cur = lbfgs(Xt, 250, 60, False)
                cur = lbfgs(cur, 900, 60, False)
            else:
                cur = lbfgs(Xt, 400, 50, False)
                cur = lbfgs(cur, 1300, 50, False)
            v = _max_cos(cur)
            # light greedy squeeze if promising
            if v < best_val + 0.0015 and _tm.time() < HARD - 16:
                cur2, v2 = greedy(cur, 350, 5e-4, HARD-14)
                cur, v = cur2, v2
            if v < elite[0][0] - 1e-12:
                elite.insert(0, (v, cur.copy()))
                elite = elite[:5]
                best_val = v
                best_X = cur.copy()
                # deep extra squeeze immediately on new record
                if _tm.time() < HARD - 16:
                    try:
                        c2 = lbfgs(best_X, 2200, 60, True)
                        vv = _max_cos(c2)
                        if vv < best_val:
                            best_val = vv
                            best_X = c2.copy()
                            elite[0] = (best_val, best_X.copy())
                    except Exception:
                        pass
            elif v < elite[-1][0] - 2e-6 and len(elite) < 5:
                # diverse second basin: check distance
                keep = True
                for _, E in elite:
                    if np.linalg.norm(np.abs(E @ cur.T).max(axis=1).mean() - 1.0) < 1e-12:
                        pass
                # simple Gram distance
                for _, E in elite:
                    d2 = float(np.linalg.norm(E - cur)) 
                    if d2 < 0.4:
                        keep = False
                        break
                if keep:
                    elite.append((v, cur.copy()))
                    elite.sort(key=lambda z: z[0])
                    elite = elite[:5]
            elif len(elite) < 5 and v < elite[0][0] + 0.002 and (it % 5 == 0):
                # keep some diversity
                elite.append((v, cur.copy()))
                elite.sort(key=lambda z: z[0])
                elite = elite[:5]
        except Exception:
            continue

    # final deep squeeze of best
    try:
        elite.sort(key=lambda z: z[0])
        best_val, best_X = elite[0][0], elite[0][1].copy()
        if _tm.time() < HARD - 6:
            cur = lbfgs(best_X, 2500, 110, True)
            v = _max_cos(cur)
            if v < best_val:
                best_val = v
                best_X = cur.copy()
        if _tm.time() < HARD:
            best_X, best_val = greedy(best_X, 2600, 4e-4, HARD+6)
    except Exception:
        pass
    best_X = _normed(best_X)
    best_X = np.nan_to_num(best_X, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best_X = best_X / nr
    return best_X
