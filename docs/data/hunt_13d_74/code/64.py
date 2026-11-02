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
    rng = np.random.default_rng(7)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
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
    cur = best.copy()
    cur_val = float(best_val)

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def lse_polish(X0, tau, maxiter):
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
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def gd_squeeze(X, steps, lr0, cur_init=None):
        cur_v = cur_init if cur_init is not None else _max_cos(X)
        lr = float(lr0)
        for it in range(int(steps)):
            if (it & 15) == 0 and time.time() - t0 > LIM - 4:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur_v:
                X[:] = Xn
                cur_v = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur_v

    def micro_split(X, cur_v, budget=0.4):
        start = time.time()
        step = 4e-6
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni; gj /= nj
            improved = False
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                G2 = Y @ Y.T
                np.fill_diagonal(G2, -2.0)
                v = float(G2.max())
                if v < cur_v:
                    X[:] = Y
                    cur_v = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, cur_v

    def refine_hole(c, X, forbid, steps=40, lr0=0.06):
        c = c / (np.linalg.norm(c) + 1e-12)
        lr = lr0
        for _ in range(int(steps)):
            dots = X @ c
            if 0 <= forbid < n:
                dots[forbid] = -10.0
            j = int(np.argmax(dots))
            dj = float(dots[j])
            g = X[j] - dj * c
            ng = float(np.linalg.norm(g))
            if ng < 1e-12:
                break
            g /= ng
            c = c - lr * g
            nn = float(np.linalg.norm(c))
            if nn < 1e-12 or not np.isfinite(nn):
                break
            c /= nn
            lr *= 0.97
        return c

    def hole_replace(X, idx_list, n_cand=4000):
        for iw in idx_list:
            iw = int(iw)
            cand = _normalize(rng.standard_normal((int(n_cand), d)))
            S = cand @ X.T
            if 0 <= iw < n:
                S[:, iw] = -10.0
            sc = S.max(axis=1)
            order = np.argsort(sc)[:3]
            best_c = cand[order[0]]
            best_s = float(sc[order[0]])
            for o in order:
                cc = refine_hole(cand[int(o)], X, iw)
                if 0 <= iw < n:
                    s = float(np.delete(X @ cc, iw).max())
                else:
                    s = float((X @ cc).max())
                if s < best_s:
                    best_s = s
                    best_c = cc
            X[iw] = best_c
        return X

    def rowmax_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx), float(G.max()), G

    # initial light squeeze of best
    try:
        if time.time() - t0 < LIM - 30:
            Xb = lse_polish(best, 0.0012, 120)
            vb = _max_cos(Xb)
            if vb < best_val:
                best, best_val = Xb, float(vb)
                cur, cur_val = Xb.copy(), float(vb)
    except Exception:
        pass

    basin = 0
    T0 = 8e-05
    while time.time() - t0 < LIM - 9:
        remain = LIM - (time.time() - t0)
        if remain < 14:
            it1, it2, mm = 50, 70, 100
            nc = 1500
        elif remain < 40:
            it1, it2, mm = 70, 100, 140
            nc = 2500
        else:
            it1, it2, mm = 80, 130, 160
            nc = 4000
        Xt = cur.copy()
        mode = basin % 6
        try:
            Gc = Xt @ Xt.T
            np.fill_diagonal(Gc, -2.0)
            gmax_c = float(Gc.max())
            cnt_tight = (Gc > gmax_c - 1e-7).sum(axis=1)
            cnt_loose = (Gc > gmax_c - 5e-04).sum(axis=1)
            rowmax = Gc.max(axis=1)
            worst = np.argsort(-rowmax)
            hubs = np.argsort(-cnt_loose)
            rattlers = np.where(cnt_tight == 0)[0]
        except Exception:
            worst = np.arange(n)
            hubs = np.arange(n)
            rattlers = np.array([], dtype=int)

        try:
            if mode == 0:
                # worst + one random transplant
                w1 = int(worst[0])
                r1 = int(rng.integers(n))
                if r1 == w1:
                    r1 = (r1 + 7) % n
                hole_replace(Xt, [w1, r1], n_cand=nc)
                Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
            elif mode == 1:
                # 4 random transplants (strong topology change)
                idx = rng.choice(n, size=4, replace=False)
                hole_replace(Xt, [int(a) for a in idx], n_cand=nc)
                Xt = _normalize(Xt + 0.005 * rng.standard_normal(Xt.shape))
            elif mode == 2:
                # hub offload: transplant top hub to hole, jitter its neighbours
                hub = int(hubs[0])
                hole_replace(Xt, [hub], n_cand=nc + 1000)
                # neighbours of hub (largest dots)
                dots = Xt @ Xt[hub]
                # jitter top 6 neighbours harder
                nbr = np.argsort(-dots)[:7]
                noise = 0.003 * rng.standard_normal(Xt.shape)
                for nb in nbr:
                    if int(nb) != hub:
                        noise[int(nb)] += 0.012 * rng.standard_normal(d)
                # extra kick on hub row itself
                noise[hub] += 0.004 * rng.standard_normal(d)
                Xt = _normalize(Xt + noise)
            elif mode == 3:
                # pure large kick + anneal (no transplant)
                Xt = _normalize(Xt + 0.075 * rng.standard_normal(Xt.shape))
            elif mode == 4:
                # fresh random block + crossover with best
                idx = rng.choice(n, size=6, replace=False)
                for a in idx:
                    Xt[int(a)] = _normalize(rng.standard_normal((1, d)))[0]
                # refine those holes lightly
                for a in idx[:3]:
                    cc = refine_hole(Xt[int(a)], Xt, int(a), steps=25)
                    Xt[int(a)] = cc
                Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
            else:
                # rattler recruitment: worst + up to 2 rattlers
                lst = [int(worst[0])]
                if len(rattlers) > 0:
                    rr = rng.choice(rattlers, size=min(2, len(rattlers)), replace=False)
                    for a in rr:
                        if int(a) not in lst:
                            lst.append(int(a))
                else:
                    lst.append(int(worst[1]))
                    lst.append(int(worst[2]))
                hole_replace(Xt, lst, n_cand=nc)
                Xt = _normalize(Xt + 0.0035 * rng.standard_normal(Xt.shape))
        except Exception:
            Xt = _normalize(cur + 0.01 * rng.standard_normal(cur.shape))

        try:
            Xt = lse_polish(Xt, 0.007, it1)
            if time.time() - t0 > LIM - 6:
                vcur = _max_cos(Xt)
            else:
                Xt = lse_polish(Xt, 0.0012, it2)
                Xt, vcur = gd_squeeze(Xt, mm, 0.002)
                if vcur < best_val + 8e-08:
                    Xt, vcur = micro_split(Xt, vcur, budget=0.25)
            # Metropolis wandering: allow uphill vs cur
            T = max(T0 * (0.985 ** basin), 4e-06)
            if vcur < best_val - 1e-12:
                best = Xt.copy()
                best_val = float(vcur)
                cur = Xt.copy()
                cur_val = float(vcur)
            elif vcur < cur_val or rng.random() < np.exp(-(vcur - cur_val) / T):
                cur = Xt.copy()
                cur_val = float(vcur)
            # occasional pull cur back to best if drifted too far
            if cur_val > best_val + 0.015:
                cur = best.copy()
                cur_val = best_val
        except Exception:
            pass
        basin += 1
        if basin > 300:
            break

    # final exploitation of best
    if time.time() - t0 < LIM - 2:
        try:
            Xf = best.copy()
            Xf = lse_polish(Xf, 3e-04, 120)
            Xf, vf = gd_squeeze(Xf, 250, 0.0008, cur_init=_max_cos(Xf))
            Xf, vf = micro_split(Xf, vf, budget=1.0)
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
