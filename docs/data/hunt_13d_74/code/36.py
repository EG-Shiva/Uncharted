import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(1)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0, J0 = IU[0].astype(np.int64), IU[1].astype(np.int64)
    NPAIR = len(I0)

    def lbfgs_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            # scaled exp
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=np.float64)
            W[I0, J0] = w
            W[J0, I0] = w
            Wx = W @ X
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 8e-7)
            mask = G > (gmax - delta)
            np.fill_diagonal(mask, False)
            # row sums
            Gr = mask.astype(np.float64) @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = (norms[:, 0] > 1e-14)
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur - 1e-15:
                X[:] = Xn
                cur = nv
                lr *= 1.08
                if lr > 8e-5:
                    lr = 8e-5
            else:
                lr *= 0.5
                if lr < 2e-11:
                    break
        return X, cur

    def degrees_topK(X, K=350):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        g = G[I0, J0]
        if K >= len(g):
            Ik, Jk = I0, J0
        else:
            part = np.argpartition(-g, K)[:K]
            Ik, Jk = I0[part], J0[part]
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, Ik, 1)
        np.add.at(deg, Jk, 1)
        return deg, float(g.max())

    def largest_hole(X, exclude=-1, n_cand=12000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ X.T
        if 0 <= exclude < n:
            S[:, exclude] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        return cand[j].copy(), float(sc[j])

    def polish_small(X):
        # two-stage smooth + greedy, time-aware
        remain = LIM - (time.time() - t0)
        if remain < 4:
            return X
        if remain > 25:
            X[:] = lbfgs_polish(X, 3.0e-4, 55)
            if time.time() - t0 > LIM - 4:
                return X
            X[:] = lbfgs_polish(X, 7.0e-5, 65)
            X[:], _ = greedy(X, 120, 1.2e-5)
        elif remain > 12:
            X[:] = lbfgs_polish(X, 1.5e-4, 45)
            X[:], _ = greedy(X, 80, 1.0e-5)
        else:
            X[:], _ = greedy(X, 60, 8e-6)
        return X

    # Phase 1: squeeze starting point a little
    if time.time() - t0 < LIM - 20:
        Xc = best.copy()
        Xc = lbfgs_polish(Xc, 3e-4, 60)
        v = _max_cos(Xc)
        if v < best_val:
            best_val = float(v); best = Xc.copy()
        else:
            # keep Xc for reference but stay on best
            pass
        if time.time() - t0 < LIM - 20:
            Xc2 = best.copy()
            Xc2 = lbfgs_polish(Xc2, 7e-5, 60)
            Xc2, vc2 = greedy(Xc2, 100, 1e-5)
            if vc2 < best_val:
                best_val = float(vc2); best = Xc2.copy()

    hop = 0
    # main topology-change loop
    while time.time() - t0 < LIM - 9:
        hop += 1
        remain = LIM - (time.time() - t0)
        if remain < 9:
            break
        mode = hop % 6
        Xtry = best.copy()
        try:
            deg, gmax = degrees_topK(Xtry, K=350)
        except Exception:
            deg = np.zeros(n, dtype=int); gmax = _max_cos(Xtry)
        order_low = np.argsort(deg)
        order_high = np.argsort(-deg)

        if mode == 0:
            # rattler recruitment: move all 0-degree points to holes
            rats = np.where(deg == 0)[0]
            if len(rats) == 0:
                rats = order_low[:3]
            else:
                rats = rats[:6]
            for r in rats:
                _, _ = degrees_topK(Xtry, K=350)
                hole_vec, _ = largest_hole(Xtry, exclude=int(r), n_cand=10000 if remain > 40 else 6000)
                Xtry[int(r)] = hole_vec
            # small noise to neighbours to allow re-fit
            Xtry = _normalize(Xtry + 0.0012 * rng.standard_normal(Xtry.shape))
        elif mode == 1:
            # hub offloading: worst hub to largest hole
            h = int(order_high[0])
            hole_vec, _ = largest_hole(Xtry, exclude=h, n_cand=12000 if remain > 40 else 6000)
            Xtry[h] = hole_vec
            # jiggle its former neighbours (top contacts of h)
            G = Xtry @ Xtry.T
            # recompute old neighbours from best
            Gb = best @ best.T
            np.fill_diagonal(Gb, -2.0)
            nbrs = np.argsort(-Gb[h])[:6]
            for nb in nbrs:
                Xtry[int(nb)] = Xtry[int(nb)] + 0.004 * rng.standard_normal(d)
            Xtry = _normalize(Xtry + 0.0008 * rng.standard_normal(Xtry.shape))
        elif mode == 2:
            # worst-pair split: relocate one endpoint of worst 2 pairs
            Gb = best @ best.T
            np.fill_diagonal(Gb, -2.0)
            # find worst pairs
            flat = Gb[I0, J0]
            idx = np.argpartition(-flat, 4)[:4]
            # pick one pair deterministically cycling
            pick = idx[hop % len(idx)]
            i = int(I0[pick]); j = int(J0[pick])
            victim = j if (hop % 2 == 0) else i
            hole_vec, _ = largest_hole(Xtry, exclude=victim, n_cand=12000 if remain > 40 else 6000)
            Xtry[victim] = hole_vec
            # push the other endpoint away from old position
            other = i if victim == j else j
            Xtry[other] = _normalize((Xtry[other] - 0.01 * best[victim]).reshape(1, -1)).ravel()
            Xtry = _normalize(Xtry + 0.001 * rng.standard_normal(Xtry.shape))
        elif mode == 3:
            # tight-frame isotropization kick
            try:
                Smat = Xtry.T @ Xtry
                wv, Vv = np.linalg.eigh(Smat)
                wv = np.maximum(wv, 1e-8)
                Sinvh = (Vv * (1.0 / np.sqrt(wv))) @ Vv.T
                Y = Xtry @ Sinvh
                Y = _normalize(Y)
                alpha = 0.55 if (hop % 12 < 6) else 0.35
                sig = 0.004 if remain > 50 else 0.002
                Xt = (1 - alpha) * Xtry + alpha * Y
                Xt = Xt + sig * rng.standard_normal(Xt.shape)
                Xtry = _normalize(Xt)
            except Exception:
                Xtry = _normalize(Xtry + 0.005 * rng.standard_normal(Xtry.shape))
        elif mode == 4:
            # anisotropic cluster split: project on random dir, move top slab
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = Xtry @ u
            k = 10 + (hop % 6)
            idx = np.argsort(-proj)[:k]
            # random small rotation in plane (u, v)
            v = rng.standard_normal(d); v -= (v @ u) * u; v /= (np.linalg.norm(v) + 1e-12)
            ang = 0.03 + 0.01 * (hop % 3)
            # rotate selected points by ang in u-v plane approx via tangent move
            for ii in idx:
                x = Xtry[int(ii)]
                # tangent components
                cu = float(x @ u); cv = float(x @ v)
                # push along combination to spread
                step_vec = ang * (cu * v - cv * u)
                Xtry[int(ii)] = x + step_vec + 0.002 * rng.standard_normal(d)
            Xtry = _normalize(Xtry)
        else:
            # global basin kick focused on hubs
            sig_base = 0.0025
            noise = sig_base * rng.standard_normal(Xtry.shape)
            # hubs get larger kick
            hubs = order_high[:8]
            noise[hubs] += 0.007 * rng.standard_normal((len(hubs), d))
            # rattlers get medium kick to explore
            lows = order_low[:8]
            noise[lows] += 0.004 * rng.standard_normal((len(lows), d))
            Xtry = _normalize(Xtry + noise)

        # polish attempt
        Xtry = polish_small(Xtry)
        vtry = _max_cos(Xtry)
        if vtry < best_val - 1e-14:
            best = Xtry.copy()
            best_val = float(vtry)
        # opportunistic extra squeeze if very close (within 5e-9) with different K
        if hop > 60:
            break
        # safety: if loop too many iterations
        if hop > 80:
            break

    # final squeeze on best with remaining time
    if time.time() - t0 < LIM - 1.5:
        try:
            Xf = best.copy()
            # last anneal
            remain = LIM - (time.time() - t0)
            if remain > 12:
                Xf = lbfgs_polish(Xf, 1.2e-4, 70)
                Xf = lbfgs_polish(Xf, 4e-5, 80)
                Xf, vf = greedy(Xf, 250, 6e-6)
            elif remain > 5:
                Xf = lbfgs_polish(Xf, 6e-5, 50)
                Xf, vf = greedy(Xf, 120, 5e-6)
            else:
                Xf, vf = greedy(Xf, 60, 5e-6)
            if vf < best_val:
                best = Xf.copy()
                best_val = float(vf)
        except Exception:
            pass

    best = _normalize(best)
    return best
