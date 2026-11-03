import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    n, d = 93, 13

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    iu0, iu1 = np.triu_indices(n, 1)

    # load best
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
    x_best = best.copy()

    def adam_lse(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        # log interp
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for s in range(steps):
            if time.time() - t0 > BUDGET - 8:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            mmax = float(G.max())
            if mmax < bv:
                bv = mmax
                bX = X.copy()
            # weights
            # exp((G-mmax)/tau)
            W = np.exp((G - mmax) / tau)
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                continue
            grad = (W @ X) / S
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            # tangent project step
            d2 = np.einsum('ij,ij->i', step, X)[:, None]
            step = step - d2 * X
            X = X - step
            X = _norm(X)
        return bX, bv, X

    def hard_topk(X0, iters, K0, K1, step0):
        X = _norm(X0.copy())
        bv = _maxcos(X)
        bX = X.copy()
        step = float(step0)
        P = len(iu0)
        for it in range(iters):
            if time.time() - t0 > BUDGET - 6:
                break
            f = it / max(1, iters - 1)
            K = int(round(K0 + (K1 - K0) * f))
            K = max(1, min(K, P))
            G = X @ X.T
            c = G[iu0, iu1]
            if K < P:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(P)
            I = iu0[idx]
            J = iu1[idx]
            grad = np.zeros_like(X)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-15:
                break
            # global normalized direction
            direc = gt / ng
            # scale to keep per-point move ~ step
            Xt = _norm(X - step * direc * np.sqrt(n))
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv
                bX = Xt.copy()
                X = Xt
                step = min(step * 1.2, 0.02)
            else:
                step *= 0.5
                if step < 1e-12:
                    step = float(step0) * 0.3
                    # small random tangent kick to escape
                    rr = rng.standard_normal((n, d)) * 1e-4
                    rr -= np.einsum('ij,ij->i', rr, X)[:, None] * X
                    X = _norm(X + rr)
                    if it > iters - 20:
                        break
        return bX, bv

    def refine_void(vv, Y, steps=120, tau=0.008):
        vv = vv / (np.linalg.norm(vv) + 1e-18)
        lr = 0.05
        for _ in range(steps):
            dots = Y @ vv
            m0 = float(dots.max())
            w = np.exp((dots - m0) / tau)
            s = float(w.sum())
            if s <= 0 or not np.isfinite(s):
                break
            w /= s
            g = w @ Y
            gt = g - float(g @ vv) * vv
            nr = float(np.linalg.norm(gt))
            if nr < 1e-12 or not np.isfinite(nr):
                break
            gt /= nr
            v2 = vv - lr * gt
            n2 = float(np.linalg.norm(v2))
            if n2 < 1e-12:
                lr *= 0.5
                continue
            v2 /= n2
            m2 = float((Y @ v2).max())
            if m2 < m0:
                vv = v2
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return vv / (np.linalg.norm(vv) + 1e-18)

    def find_void(Y):
        # random search then refine
        db = 1e9
        vb = None
        CH = 4000
        for _ in range(3):
            C = rng.standard_normal((CH, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-18)
            D = C @ Y.T
            worst = D.max(axis=1)
            idx = int(np.argmin(worst))
            if float(worst[idx]) < db:
                db = float(worst[idx])
                vb = C[idx].copy()
        return refine_void(vb, Y, steps=100)

    def surgery(X, n_move):
        X = _norm(X.copy())
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        rm = G.max(axis=1)
        # contacts within 1e-3
        thr = 1e-3
        contacts = (G > mx - thr).sum(axis=1)
        # rattlers: zero contacts (excluding diag since diag=-2)
        ratt = np.where(contacts == 0)[0]
        if len(ratt) == 0:
            # pick lowest rowmax
            ratt = np.argsort(rm)[:max(1, n_move)]
        # choose n_move among rattlers: lowest rowmax first (most isolated) + random
        order = np.argsort(rm[ratt])
        ratt_sorted = ratt[order]
        # mix: take some isolated and some random rattlers
        chosen = []
        for k in range(n_move):
            if k < len(ratt_sorted):
                # alternate isolated / random
                if k % 2 == 0:
                    chosen.append(int(ratt_sorted[k % len(ratt_sorted)]))
                else:
                    chosen.append(int(ratt[rng.integers(len(ratt))]))
            else:
                chosen.append(int(rng.integers(n)))
        chosen = list(dict.fromkeys(chosen))
        # iteratively reinsert
        keep_mask = np.ones(n, dtype=bool)
        # we will place one by one using voids of the rest
        Y_fixed = X[~np.isin(np.arange(n), np.array(chosen))]
        new_pts = []
        for _ in chosen:
            vv = find_void(Y_fixed if len(new_pts) == 0 else np.vstack([Y_fixed] + new_pts))
            new_pts.append(vv[None, :])
        Xn = X.copy()
        for idx, pt in zip(chosen, new_pts):
            Xn[idx] = pt[0]
        return _norm(Xn)

    def perturb(X, sigma):
        Y = X + sigma * rng.standard_normal(size=X.shape)
        return _norm(Y)

    # quick initial anneal on best to tie plateau
    try:
        bX, bv, _ = adam_lse(x_best, 600, 0.012, 0.0009, 0.004, 0.0012)
        if bv < best_val:
            best_val = bv
            x_best = bX.copy()
    except Exception:
        pass

    pool = [(best_val, x_best.copy())]

    it = 0
    # main basin-hopping loop
    while time.time() - t0 < BUDGET - 22:
        it += 1
        # choose base
        if it % 4 == 1:
            base = x_best
            sigma = [0.08, 0.18, 0.32, 0.5][(it // 4) % 4]
        elif it % 4 == 2 and len(pool) > 1:
            # random pool element
            bi = int(rng.integers(len(pool)))
            base = pool[bi][1]
            sigma = float(rng.uniform(0.05, 0.4))
        elif it % 4 == 3:
            base = x_best
            # surgery start
            try:
                nm = int(rng.integers(1, 4))
                Xs = surgery(base, nm)
                # short anneal after surgery
                bX2, bv2, _ = adam_lse(Xs, 500, 0.008, 0.0006, 0.004, 0.001)
                if bv2 < best_val - 1e-9:
                    best_val = bv2
                    x_best = bX2.copy()
                pool.append((float(bv2), bX2.copy()))
                if len(pool) > 12:
                    pool = sorted(pool, key=lambda t: t[0])[:12]
                continue
            except Exception:
                base = x_best
                sigma = 0.2
        else:
            base = x_best
            sigma = float(rng.uniform(0.02, 0.12))

        try:
            Xs = perturb(base, sigma)
        except Exception:
            continue
        # vary tau schedule randomly
        tau0 = float(rng.choice([0.012, 0.008, 0.015, 0.005]))
        tau1 = float(rng.choice([0.0007, 0.0004, 0.001, 0.00025]))
        lr0 = 0.004
        lr1 = 0.0008
        steps = 450 if time.time() - t0 < BUDGET - 60 else 300
        try:
            bX, bv, Xlast = adam_lse(Xs, steps, tau0, tau1, lr0, lr1)
        except Exception:
            continue
        # hard polish occasionally
        if bv < best_val + 0.002:
            try:
                K0 = int(rng.choice([300, 500, 200]))
                bXh, bvh = hard_topk(bX, 120, K0, 30, 0.008)
                if bvh < bv:
                    bv = bvh
                    bX = bXh
            except Exception:
                pass
        pool.append((float(bv), bX.copy()))
        if len(pool) > 12:
            pool = sorted(pool, key=lambda t: t[0])[:12]
        if bv < best_val and np.all(np.isfinite(bX)):
            best_val = float(bv)
            x_best = bX.copy()

    # final intensive polish on top candidates
    pool = sorted(pool, key=lambda t: t[0])[:5]
    for v, Xc in list(pool):
        if time.time() - t0 > BUDGET - 7:
            break
        try:
            bX, bv, _ = adam_lse(Xc, 350, 0.004, 0.00018, 0.0025, 0.0005)
            bXh, bvh = hard_topk(bX if bv < v else Xc, 200, 400, 25, 0.006)
            if bvh < best_val:
                best_val = bvh
                x_best = bXh.copy()
            elif bv < best_val:
                best_val = bv
                x_best = bX.copy()
        except Exception:
            continue

    # micro jitter hill-climb on worst pair
    try:
        X = _norm(x_best.copy())
        bv = _maxcos(X)
        t_end = t0 + BUDGET - 1.0
        step = 0.002
        while time.time() < t_end:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.max(axis=1)))
            # find partner
            j = int(np.argmax(G[ij]))
            # push them apart slightly along tangent
            # move i away from j
            d = X[ij] - float(X[ij] @ X[j]) * X[j]
            nn = float(np.linalg.norm(d))
            if nn < 1e-12:
                rr = rng.standard_normal(d)
                rr -= float(rr @ X[ij]) * X[ij]
                nn = float(np.linalg.norm(rr))
                if nn < 1e-12:
                    break
                d = rr / nn
            else:
                d /= nn
            Xt = X.copy()
            Xt[ij] = _norm((X[ij] + step * d)[None, :])[0]
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv
                X = Xt
                best_val = bv
                x_best = X.copy()
                step = min(step * 1.15, 0.004)
            else:
                step *= 0.6
                if step < 1e-7:
                    # try another random point
                    step = 0.0015
                    # random tangent kick on worst point
                    rr = rng.standard_normal(d)
                    rr -= float(rr @ X[ij]) * X[ij]
                    nn2 = float(np.linalg.norm(rr))
                    if nn2 > 1e-12:
                        Xt2 = X.copy()
                        Xt2[ij] = _norm((X[ij] + 1e-4 * rr / nn2)[None, :])[0]
                    # give up if no progress after many fails
                    # count fails
                    pass
            # occasional full eval break
            if step < 5e-7:
                break
    except Exception:
        pass

    x_best = _norm(np.asarray(x_best, dtype=np.float64))
    # guard NaN
    if not np.all(np.isfinite(x_best)):
        try:
            x_best = _norm(np.asarray(np.load("best.npy"), dtype=float))
        except Exception:
            x_best = _norm(rng.standard_normal((n, d)))
    return x_best
