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
    LIM = 170.0
    rng = np.random.default_rng(0)
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

    IU0 = np.triu_indices(n, 1)
    I0 = IU0[0].astype(np.int64)
    J0 = IU0[1].astype(np.int64)

    def lse_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        if not np.all(np.isfinite(y0)):
            return X0
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            if not np.isfinite(mmax):
                return 1e6, np.zeros_like(yf)
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
            if not np.all(np.isfinite(gradY)):
                return float(F), np.zeros_like(yf)
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-6, "maxfun": int(maxiter * 10)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def micro_split(X, cur, steps=220):
        step = 5e-6
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            if c < cur:
                cur = c
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
                ni2 = float(np.linalg.norm(Y[i])); nj2 = float(np.linalg.norm(Y[j]))
                if ni2 < 1e-12 or nj2 < 1e-12:
                    continue
                Y[i] /= ni2; Y[j] /= nj2
                v = _max_cos(Y)
                if v < cur - 1e-15:
                    X[:] = Y; cur = v; improved = True
                    break
            if improved:
                step = min(step * 1.2, 4e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, cur

    def multi_holes(Y, exclude_idx, k, n_cand=10000):
        # well-separated holes: sequentially add picked holes to Y
        Yext = Y.copy()
        # mask excluded rows by ignoring them: we will overwrite them anyway,
        # so exclude them from max computation by temp copy trick: use full Yext but
        # after each pick append hole
        excl = set(int(v) for v in exclude_idx)
        holes = []
        Cl = _normalize(rng.standard_normal((n_cand, d)))
        for t in range(k):
            S = Cl @ Yext.T
            # ignore excluded original points (they will be moved)
            for e_ in excl:
                S[:, e_] = -10.0
            # repel from already picked holes
            if holes:
                H = np.stack(holes, axis=0)
                Sh = Cl @ H.T
                Sm = Sh.max(axis=1)
                # combined score: max of both
                sc = np.maximum(S.max(axis=1), Sm)
            else:
                sc = S.max(axis=1)
            j = int(np.argmin(sc))
            x = Cl[j].copy()
            # tiny local walk away from neighbours
            for _ in range(12):
                dd = Yext @ x
                for e_ in excl:
                    dd[e_] = -10.0
                if holes:
                    hd = np.array([float(np.dot(h, x)) for h in holes])
                    m2 = float(hd.max()) if len(hd) else -10.0
                else:
                    m2 = -10.0
                m1 = float(dd.max())
                m = m1 if m1 > m2 else m2
                idx = int(np.argmax(dd))
                # gradient from top neighbour + hole repulsion
                g = Yext[idx] - float(dd[idx]) * x
                if holes:
                    for h in holes:
                        c = float(np.dot(h, x))
                        if c > m - 0.05:
                            g += 0.5 * (h - c * x)
                ng = float(np.linalg.norm(g))
                if ng < 1e-12 or not np.isfinite(ng):
                    break
                g /= ng
                improved = False
                for s in (0.02, 0.01, 0.005):
                    xt = x - s * g
                    nn = float(np.linalg.norm(xt))
                    if nn < 1e-12:
                        continue
                    xt /= nn
                    d2 = Yext @ xt
                    for e_ in excl:
                        d2[e_] = -10.0
                    v1 = float(d2.max())
                    if holes:
                        v2 = float(max(float(np.dot(h, xt)) for h in holes))
                    else:
                        v2 = -10.0
                    vv = v1 if v1 > v2 else v2
                    cur0_d = Yext @ x
                    for e_ in excl:
                        cur0_d[e_] = -10.0
                    c1 = float(cur0_d.max())
                    if holes:
                        c2 = float(max(float(np.dot(h, x)) for h in holes))
                    else:
                        c2 = -10.0
                    cc = c1 if c1 > c2 else c2
                    if vv < cc:
                        x = xt
                        improved = True
                        break
                if not improved:
                    break
            holes.append(x)
            # append to Yext for separation (replace one excluded slot progressively)
            # extend matrix by one row
            Yext = np.vstack([Yext, x.reshape(1, -1)])
        return holes

    # quick polish of baseline
    try:
        if time.time() - t0 < LIM - 30:
            Xb = lse_polish(best, 5e-4, 60)
            vb = _max_cos(Xb)
            if vb < best_val:
                best_val, best = float(vb), Xb.copy()
    except Exception:
        pass

    # Phase A: broad random basin sampling (cheap coarse)
    pool = [(best_val, best.copy())]
    n_rand = 10
    for r in range(n_rand):
        if time.time() - t0 > LIM - 95:
            break
        try:
            Xt = _normalize(rng.standard_normal((n, d)))
            Xt = lse_polish(Xt, 0.015, 35)
            Xt = lse_polish(Xt, 0.004, 45)
            v = _max_cos(Xt)
            pool.append((float(v), Xt.copy()))
            if v < best_val - 1e-13:
                best_val, best = float(v), Xt.copy()
        except Exception:
            continue
    pool.sort(key=lambda z: z[0])
    pool = pool[:4]

    cur = best.copy()
    cur_val = float(best_val)
    # pick diverse cur start: second best if not too bad (different basin)
    if len(pool) > 1 and pool[1][0] < cur_val + 0.01:
        # start hopping sometimes from pool[1] to enforce diversity
        cur_alt = pool[1][1].copy()
        cur_alt_val = float(pool[1][0])
    else:
        cur_alt = cur.copy()
        cur_alt_val = float(cur_val)

    trial = 0
    fails = 0
    while True:
        if time.time() - t0 > LIM - 28:
            break
        if trial > 30:
            break
        trial += 1
        remain = (t0 + LIM - 28) - time.time()
        if remain < 9:
            break
        # choose starter: alternate best / cur / alt
        if trial % 7 == 0:
            startX = cur_alt.copy()
            startV = float(cur_alt_val)
        elif trial % 3 == 0:
            startX = best.copy()
            startV = float(best_val)
        else:
            if fails >= 5:
                startX = best.copy()
                startV = float(best_val)
                fails = 0
                cur = best.copy(); cur_val = float(best_val)
            else:
                startX = cur.copy()
                startV = float(cur_val)
        try:
            Gc = startX @ startX.T
            np.fill_diagonal(Gc, -2.0)
            gmaxc = float(Gc.max())
            rowmax = Gc.max(axis=1)
            cnt = (Gc > (gmaxc - 3e-4)).sum(axis=1)
        except Exception:
            continue
        order_hub = np.argsort(-cnt, kind="stable")
        order_row_lo = np.argsort(rowmax, kind="stable")
        order_row_hi = np.argsort(-rowmax, kind="stable")
        # block size 4-6
        k = 4 + (trial % 3)  # 4,5,6
        # select indices: mix hubs, rattlers, worst, random
        sel = []
        # 1 hub
        sel.append(int(order_hub[0]))
        # worst
        w0 = int(order_row_hi[0])
        if w0 not in sel:
            sel.append(w0)
        # two rattlers / lowest rowmax
        for idx in order_row_lo:
            if len(sel) >= k - 1:
                break
            ii = int(idx)
            if ii not in sel:
                sel.append(ii)
        # random fill
        while len(sel) < k:
            rr = int(rng.integers(0, n))
            if rr not in sel:
                sel.append(rr)
        sel = sel[:k]
        Xt = startX.copy()
        try:
            holes = multi_holes(Xt, sel, k, n_cand=9000)
            for ii, h in zip(sel, holes):
                Xt[int(ii)] = h
            # small global noise to break symmetry
            sig = 0.002 + 0.001 * (trial % 3)
            Xt = _normalize(Xt + sig * rng.standard_normal(Xt.shape))
        except Exception:
            continue
        if time.time() - t0 > LIM - 27:
            break
        try:
            Xt = lse_polish(Xt, 0.006, 50)
            if time.time() - t0 > LIM - 26:
                vq = _max_cos(Xt)
                if vq < best_val:
                    best_val, best = float(vq), Xt.copy()
                continue
            Xt = lse_polish(Xt, 0.0015, 60)
            Xt = lse_polish(Xt, 3e-4, 60)
            vq = _max_cos(Xt)
            Xt, vq = micro_split(Xt, vq, steps=200)
        except Exception:
            continue
        if vq < best_val - 1e-13:
            best = Xt.copy(); best_val = float(vq)
            cur = Xt.copy(); cur_val = float(vq)
            fails = 0
        elif vq < cur_val - 1e-14:
            cur = Xt.copy(); cur_val = float(vq)
            fails = 0
        elif vq < cur_alt_val - 1e-14:
            cur_alt = Xt.copy(); cur_alt_val = float(vq)
            fails += 1
        elif vq < cur_val + 0.002 and rng.random() < 0.35:
            cur = Xt.copy(); cur_val = float(vq)
            fails += 1
        else:
            fails += 1

    # final intensive polish of best
    try:
        if time.time() - t0 < LIM - 1.0:
            Xf = lse_polish(best, 2e-4, 110)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            Xf2, vf2 = micro_split(best.copy(), float(best_val), steps=400)
            if vf2 < best_val:
                best_val, best = float(vf2), Xf2.copy()
            if time.time() - t0 < LIM - 0.8:
                Xg = lse_polish(best, 8e-5, 110)
                vg = _max_cos(Xg)
                if vg < best_val:
                    best_val, best = float(vg), Xg.copy()
                Xg2, vg2 = micro_split(best.copy(), float(best_val), steps=300)
                if vg2 < best_val:
                    best_val, best = float(vg2), Xg2.copy()
    except Exception:
        pass
    best = _normalize(best)
    return best
