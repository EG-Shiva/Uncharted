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
    rng = np.random.default_rng(0)
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
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

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

    def squeeze(X, cur, deadline):
        lr = 3e-5
        it = 0
        while time.time() < deadline and it < 800:
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(4.0 * lr, 8e-7)
            mask = G > (gmax - delta)
            # active matrix
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            nrm = np.linalg.norm(Gt, axis=1, keepdims=True)
            # normalize active rows to avoid hub dominance
            cnt = mask.sum(axis=1, keepdims=True)
            cnt = np.maximum(cnt, 1.0)
            Gt = Gt / cnt
            if not np.all(np.isfinite(Gt)):
                break
            improved = False
            for _ in range(10):
                if time.time() >= deadline:
                    break
                Xt = _normalize(X - lr * Gt)
                nv = _max_cos(Xt)
                if nv < cur - 1e-14:
                    X[:] = Xt
                    cur = nv
                    lr *= 1.15
                    if lr > 2e-4:
                        lr = 2e-4
                    improved = True
                    break
                lr *= 0.5
                if lr < 1e-12:
                    break
            if not improved:
                if lr < 2e-9:
                    break
        return X, cur

    def hole_point(Y, exclude_set, n_cand=15000):
        C = _normalize(rng.standard_normal((n_cand, d)))
        S = C @ Y.T
        # ignore excluded indices
        for e_ in exclude_set:
            S[:, e_] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        x = C[j].copy()
        bs = float(sc[j])
        # quick local refinement: walk away from nearest neighbours
        step = 0.04
        excl = np.zeros(n, dtype=bool)
        for e_ in exclude_set:
            excl[e_] = True
        for _ in range(30):
            dots = Y @ x
            dots[excl] = -10.0
            m = float(dots.max())
            # top-4 neighbours weighted
            k = 4
            idx = np.argpartition(-dots, k)[:k]
            g = np.zeros(d)
            for ii in idx:
                c = float(dots[ii])
                w = np.exp((c - m) / 0.01)
                g += w * (Y[ii] - c * x)
            ng = float(np.linalg.norm(g))
            if ng < 1e-12 or not np.isfinite(ng):
                break
            g /= ng
            ok = False
            for s in (step, step * 0.5, step * 0.25):
                xt = x - s * g
                nn = float(np.linalg.norm(xt))
                if nn < 1e-12:
                    continue
                xt /= nn
                d2 = Y @ xt
                d2[excl] = -10.0
                m2 = float(d2.max())
                if m2 < m:
                    x = xt
                    bs = m2
                    ok = True
                    break
            if not ok:
                step *= 0.5
                if step < 5e-4:
                    break
        return x

    # current walker for basin hopping
    cur = best.copy()
    cur_val = float(best_val)

    def analyse(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        rowmax = G.max(axis=1)
        cnt = (G > (gmax - 3e-4)).sum(axis=1)
        return G, gmax, rowmax, cnt

    trial = 0
    fails = 0
    # optional: one quick random-basin probe early is done inside loop as mode 5
    while time.time() - t0 < LIM - 30:
        trial += 1
        remain = (t0 + LIM - 30) - time.time()
        if remain < 8:
            break
        mode = trial % 6
        # occasionally reset walker to best if stuck
        if fails >= 6:
            cur = best.copy()
            cur_val = float(best_val)
            fails = 0

        if mode == 5 and trial < 12:
            # fresh random basin probe (coarse) to seek isotropic optimum
            Xt = _normalize(rng.standard_normal((n, d)))
            Xt = lse_polish(Xt, 0.02, 40)
            if time.time() - t0 > LIM - 32:
                pass
            else:
                Xt = lse_polish(Xt, 0.005, 50)
        else:
            Gc, gmaxc, rowmaxc, cntc = analyse(cur)
            # hubs: largest contact count, rattlers: small rowmax
            order_hub = np.argsort(-cntc, kind="stable")
            rattler_mask = rowmaxc < (gmaxc - 1e-4)
            ratt_idx = np.where(rattler_mask)[0]
            if len(ratt_idx) == 0:
                ratt_idx = np.argsort(rowmaxc, kind="stable")[:3]
            worst_row = int(np.argmax(rowmaxc))
            hub0 = int(order_hub[0])
            Xt = cur.copy()
            if mode == 0:
                # single hub to hole
                excl = {int(hub0)}
                hp = hole_point(Xt, excl, n_cand=15000)
                Xt[hub0] = hp
                Xt = _normalize(Xt + 0.003 * rng.standard_normal(Xt.shape))
            elif mode == 1:
                # hub + worst rattler to holes
                r0 = int(ratt_idx[0]) if int(ratt_idx[0]) != hub0 else int(ratt_idx[-1] if len(ratt_idx) > 1 else (hub0 + 1) % n)
                excl = {int(hub0), int(r0)}
                # sequential
                hp1 = hole_point(Xt, excl, n_cand=12000)
                Xt[hub0] = hp1
                hp2 = hole_point(Xt, excl, n_cand=12000)
                Xt[r0] = hp2
                Xt = _normalize(Xt + 0.004 * rng.standard_normal(Xt.shape))
            elif mode == 2:
                # move up to 3 rattlers to holes
                mv = list(ratt_idx[:3])
                if len(mv) == 0:
                    mv = [worst_row]
                excl = set(int(v) for v in mv)
                for v in mv:
                    hp = hole_point(Xt, excl, n_cand=10000)
                    Xt[int(v)] = hp
                Xt = _normalize(Xt + 0.003 * rng.standard_normal(Xt.shape))
            elif mode == 3:
                # global kick + one hole
                Xt = _normalize(Xt + 0.014 * rng.standard_normal(Xt.shape))
                # recruit worst rattler toward hole
                r0 = int(ratt_idx[0])
                excl = {r0}
                hp = hole_point(Xt, excl, n_cand=10000)
                # mix: halfway to hole to keep some memory
                Xt[r0] = _normalize((0.5 * Xt[r0] + 0.5 * hp).reshape(1, -1))[0]
                Xt = _normalize(Xt + 0.002 * rng.standard_normal(Xt.shape))
            else:  # mode 4
                sig = 0.007 if (trial % 2 == 0) else 0.011
                Xt = _normalize(Xt + sig * rng.standard_normal(Xt.shape))
                # hub offload: push hub neighbours apart slightly
                G2 = Xt @ Xt.T
                np.fill_diagonal(G2, -2.0)
                gmx = float(G2.max())
                cc = (G2 > (gmx - 3e-4)).sum(axis=1)
                h = int(np.argmax(cc))
                # small random tangent kick to hub
                kick = rng.standard_normal(d)
                kick -= (kick @ Xt[h]) * Xt[h]
                nk = float(np.linalg.norm(kick))
                if nk > 1e-12:
                    Xt[h] = _normalize((Xt[h] + 0.02 * kick / nk).reshape(1, -1))[0]

        if time.time() - t0 > LIM - 29:
            break
        # anneal: coarse then fine
        Xt = lse_polish(Xt, 0.008, 45)
        if time.time() - t0 > LIM - 28:
            vq = _max_cos(Xt)
            if vq < best_val:
                best_val = float(vq); best = Xt.copy()
            continue
        Xt = lse_polish(Xt, 0.002, 55)
        Xt = lse_polish(Xt, 5e-4, 60)
        vq = _max_cos(Xt)
        # short active-set squeeze
        Xt, vq = squeeze(Xt, vq, min(time.time() + 2.5, t0 + LIM - 24))
        if vq < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vq)
            cur = Xt.copy(); cur_val = float(vq)
            fails = 0
        elif vq < cur_val:
            cur = Xt.copy(); cur_val = float(vq)
            fails = 0
        elif vq < cur_val + 5e-4 and rng.random() < 0.30:
            cur = Xt.copy(); cur_val = float(vq)
            fails += 1
        else:
            fails += 1
        if trial > 22:
            break

    # final intensive squeeze of best
    if time.time() - t0 < LIM - 1.0:
        Xf = best.copy()
        Xf = lse_polish(Xf, 3e-4, 90)
        vf = _max_cos(Xf)
        if vf < best_val:
            best_val, best = float(vf), Xf.copy()
        else:
            Xf = best.copy()
        Xf, vf = squeeze(Xf, min(_max_cos(Xf), best_val), t0 + LIM - 0.6)
        if vf < best_val:
            best = Xf.copy(); best_val = float(vf)
        # one more ultra-fine LSE
        if time.time() - t0 < LIM - 0.9:
            Xg = lse_polish(best, 1e-4, 100)
            vg = _max_cos(Xg)
            if vg < best_val:
                best = Xg.copy(); best_val = float(vg)
    best = _normalize(best)
    return best
