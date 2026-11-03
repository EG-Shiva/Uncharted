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

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def solve():
    t0 = time.time()
    DEADLINE = 170.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    iu0, iu1 = np.triu_indices(n, 1)

    def smooth_opt(X, steps, T0, T1, lr0, lr1, mu=0.85):
        V = np.zeros_like(X)
        # log schedules
        lt = np.log(T1 / T0) if T0 != T1 else 0.0
        ll = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for k in range(steps):
            if (k & 63) == 0 and time.time() - t0 > DEADLINE:
                break
            f = k / max(1, steps - 1)
            T = T0 * np.exp(lt * f)
            lr = lr0 * np.exp(ll * f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            # global softmax weights
            E = np.exp((G - m) / T)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = np.einsum('ij,ij->i', Gr, X)[:, None]
            Gt = Gr - dot * X
            V = mu * V + (1.0 - mu) * Gt
            # re-project velocity to tangent
            dv = np.einsum('ij,ij->i', V, X)[:, None]
            Vt = V - dv * X
            X = X - lr * Vt
            # normalize
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def hard_polish(X, iters, step0, K=500):
        cur_val = _maxcos(X)
        bestX = X.copy()
        bv = cur_val
        step = step0
        grad = np.empty_like(X)
        for it in range(iters):
            if (it & 15) == 0 and time.time() - t0 > DEADLINE:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            m = float(c.max())
            if K < len(c):
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(len(c))
            I = iu0[idx]
            J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng == 0:
                step *= 0.5
                if step < 1e-14:
                    break
                continue
            Xt = X - step * gt
            Xt = _norm(Xt)
            v = _maxcos(Xt)
            if v < cur_val:
                X = Xt
                cur_val = v
                if v < bv:
                    bv = v
                    bestX = Xt.copy()
                step = min(step * 1.25, 0.01)
            else:
                step *= 0.5
                if step < 1e-14:
                    break
        return bestX, bv

    def void_replace(X, k_idx, n_cand=8000):
        mask = np.ones(n, dtype=bool)
        mask[k_idx] = False
        Xr = X[mask]
        C = _norm(rng.standard_normal((n_cand, d)))
        D = C @ Xr.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        Xn = X.copy()
        Xn[k_idx] = C[bi]
        return Xn

    # initial squeeze
    try:
        best = smooth_opt(best, steps=3500, T0=0.002, T1=4e-05, lr0=0.04, lr1=0.002, mu=0.85)
        best, best_val2 = hard_polish(best, iters=400, step0=0.002, K=500)
        best_val = float(min(best_val, best_val2))
    except Exception:
        pass

    sigmas = [0.012, 0.006, 0.025, 0.004, 0.015, 0.008]
    trial = 0
    # basin hopping
    while time.time() - t0 < 150.0:
        trial += 1
        if time.time() - t0 > DEADLINE - 12:
            break
        mode = trial % 5
        sigma = sigmas[trial % len(sigmas)]
        try:
            if mode == 0:
                Xt = best + sigma * rng.standard_normal(best.shape)
                Xt = _norm(Xt)
                Xt = smooth_opt(Xt, steps=1800, T0=0.0015, T1=3e-05, lr0=0.03, lr1=0.0015, mu=0.85)
                Xt, pv = hard_polish(Xt, iters=250, step0=0.0015, K=400)
            elif mode == 1:
                rm = _rowmax(best)
                worst = np.argsort(rm)[-10:]
                Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
                Xt[worst] += sigma * rng.standard_normal((len(worst), d))
                Xt = _norm(Xt)
                Xt = smooth_opt(Xt, steps=1800, T0=0.001, T1=3e-05, lr0=0.03, lr1=0.0015, mu=0.85)
                Xt, pv = hard_polish(Xt, iters=250, step0=0.0015, K=400)
            elif mode == 2:
                rm = _rowmax(best)
                ibad = int(np.argmax(rm))
                Xt = void_replace(best, ibad, n_cand=8000)
                # tiny shake of rest
                Xt += 0.001 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
                Xt = smooth_opt(Xt, steps=2000, T0=0.0012, T1=2.5e-05, lr0=0.035, lr1=0.0012, mu=0.85)
                Xt, pv = hard_polish(Xt, iters=300, step0=0.0012, K=500)
            elif mode == 3:
                Xt = best.copy()
                nb = 22
                idx = rng.choice(n, size=nb, replace=False)
                Xt[idx] += (sigma * 2.0) * rng.standard_normal((nb, d))
                Xt = _norm(Xt)
                Xt = smooth_opt(Xt, steps=1800, T0=0.002, T1=4e-05, lr0=0.035, lr1=0.0015, mu=0.85)
                Xt, pv = hard_polish(Xt, iters=250, step0=0.0015, K=400)
            else:
                rm = _rowmax(best)
                w2 = np.argpartition(-rm, 2)[:2]
                Xt = best.copy()
                for wi in w2:
                    Xt = void_replace(Xt, int(wi), n_cand=5000)
                Xt += 0.001 * rng.standard_normal(Xt.shape)
                Xt = _norm(Xt)
                Xt = smooth_opt(Xt, steps=1500, T0=0.001, T1=2.5e-05, lr0=0.03, lr1=0.001, mu=0.85)
                Xt, pv = hard_polish(Xt, iters=250, step0=0.001, K=500)
        except Exception:
            continue
        if pv < best_val and np.all(np.isfinite(Xt)):
            best_val = float(pv)
            best = Xt.copy()

    # final ultra-fine squeeze
    remain = DEADLINE - (time.time() - t0)
    if remain > 4:
        try:
            best = smooth_opt(best, steps=1200, T0=2e-04, T1=1.2e-05, lr0=0.008, lr1=0.0006, mu=0.9)
            best, bv = hard_polish(best, iters=500, step0=0.0006, K=600)
            if bv < best_val:
                best_val = bv
        except Exception:
            pass

    best = _norm(best)
    # strict feasibility
    for i in range(n):
        r = float(np.linalg.norm(best[i]))
        if not np.isfinite(r) or r < 1e-12:
            best[i] = _norm(rng.standard_normal((1, d)))[0]
    best = _norm(best)
    return best
