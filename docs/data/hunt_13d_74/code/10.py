import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    import time as _tm
    t0 = _tm.time()
    LIM = 171.0
    rng = np.random.default_rng(0)

    def remain():
        return LIM - (_tm.time() - t0)

    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    n, d = 74, 13

    # ---------- helpers ----------
    def adam_lse(X, sched, m, v, cnt):
        for tau, steps, lr0 in sched:
            for s in range(steps):
                if (s & 31) == 0 and remain() < 3:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                # stable softmax
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt[0] += 1
                c1 = 1.0 - 0.9 ** cnt[0]
                c2 = 1.0 - 0.999 ** cnt[0]
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if remain() < 2:
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
            nm = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = nm[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (nm[act] + 1e-12) * lr
            Xn = _norm(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def micro(X):
        # hill climb on true max with tiny steps, good for single-pair bottleneck
        cur = _max_cos(X)
        eps = 2e-4
        for _ in range(400):
            if remain() < 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            # directions away from each other projected
            di = X[ii] - G[ii, jj] * X[jj]
            dj = X[jj] - G[ii, jj] * X[ii]
            ni = np.linalg.norm(di); nj = np.linalg.norm(dj)
            if ni < 1e-14 or nj < 1e-14:
                break
            di /= ni; dj /= nj
            improved = False
            e = eps
            for _t in range(12):
                Xt = X.copy()
                Xt[ii] = _norm((X[ii] - e * di).reshape(1, -1))[0]
                Xt[jj] = _norm((X[jj] - e * dj).reshape(1, -1))[0]
                nv = _max_cos(Xt)
                if nv < cur:
                    X[:] = Xt
                    cur = nv
                    eps = min(e * 1.5, 2e-4)
                    improved = True
                    break
                e *= 0.5
            if not improved:
                eps *= 0.5
                if eps < 1e-12:
                    break
        return X, cur

    def farthest_reinsert(X, k=1):
        for _ in range(k):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            iw = int(np.argmax(rowmax))
            cand = _norm(rng.standard_normal((3000, d)))
            S = cand @ X.T
            S[:, iw] = -10.0
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X[iw] = cand[j]
        return X

    def lp_polish(X, budget):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X
        t_start = _tm.time()
        X = _norm(X)
        cur = _max_cos(X)
        radius = 8e-4
        iu_r = np.triu_indices(n, 1)
        while _tm.time() - t_start < budget and remain() > 3 and radius > 1e-13:
            cfull = (X @ X.T)[iu_r]
            m = float(cfull.max())
            sel = cfull > m - max(25 * radius, 2e-9)
            if np.count_nonzero(sel) > 2200:
                thr = np.sort(cfull)[-2200]
                sel = cfull >= thr
            I = iu_r[0][sel]; J = iu_r[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _norm(Y)
            v = _max_cos(Y)
            if v < cur:
                X, cur = Y, v
                if cur < best_val - 1e-12:
                    pass
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X

    # schedules
    FULL = [(0.03, 250, 0.012), (0.01, 500, 0.008), (0.003, 600, 0.005), (0.0012, 400, 0.003)]
    FAST = [(0.01, 300, 0.008), (0.003, 400, 0.005), (0.001, 300, 0.003)]
    TINY = [(0.003, 200, 0.004), (0.001, 200, 0.002)]

    # phase A: squeeze starting best
    try:
        X0 = best.copy()
        m = np.zeros_like(X0); v = np.zeros_like(X0); cnt = [0]
        X0 = adam_lse(X0, FAST, m, v, cnt)
        X0, v0 = greedy(X0, 250, 0.002)
        X0, v0 = micro(X0)
        if v0 < best_val:
            best, best_val = X0, v0
    except Exception:
        pass

    basin = 0
    while remain() > 28:
        basin += 1
        r = remain()
        if r < 40:
            sched = TINY
        elif r < 80:
            sched = FAST
        else:
            sched = FULL if (basin % 3 != 0) else FAST
        Xcur = best.copy()
        mode = basin % 8
        try:
            if mode == 0:
                Xcur = _norm(Xcur + 0.012 * rng.standard_normal(Xcur.shape))
            elif mode == 1:
                Xcur = _norm(Xcur + 0.05 * rng.standard_normal(Xcur.shape))
            elif mode == 2:
                Xcur = _norm(Xcur + 0.15 * rng.standard_normal(Xcur.shape))
            elif mode == 3:
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                cm = (G > gmax - 0.02).sum(axis=1)
                w = np.where(cm > 0)[0]
                noise = rng.standard_normal(Xcur.shape)
                sc = np.full((n, 1), 0.008)
                if len(w) > 0:
                    sc[w] = 0.12
                # also worst single gets big kick
                iw = int(np.argmax(G.max(axis=1)))
                sc[iw] = 0.25
                Xcur = _norm(Xcur + sc * noise)
            elif mode == 4:
                # reinsert worst 1-2
                Xcur = farthest_reinsert(Xcur, k=1 if (basin % 2 == 0) else 2)
                Xcur = _norm(Xcur + 0.005 * rng.standard_normal(Xcur.shape))
            elif mode == 5:
                # random rotation of random subset
                k = int(rng.integers(8, 25))
                idx = rng.choice(n, size=k, replace=False)
                Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                # small rotation: mix with identity
                ang = float(rng.uniform(0.05, 0.4))
                R = np.eye(d) * np.cos(ang) + Q * np.sin(ang)
                # orthonormalize R via QR to keep rotation
                R, _ = np.linalg.qr(R)
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1
                Xcur[idx] = _norm(Xcur[idx] @ R.T)
                Xcur = _norm(Xcur + 0.01 * rng.standard_normal(Xcur.shape))
            elif mode == 6:
                Xcur = _norm(rng.standard_normal(Xcur.shape)) * 0.3 + Xcur * 0.7
                Xcur = _norm(Xcur)
            else:
                # heavy shake of top-contact cluster
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                gmax = float(G.max())
                thr = gmax - 0.005
                mask = G > thr
                deg = mask.sum(axis=1)
                order = np.argsort(-deg)
                k = min(n, 12)
                top = order[:k]
                noise = rng.standard_normal(Xcur.shape) * 0.01
                noise[top] += rng.standard_normal((k, d)) * 0.15
                Xcur = _norm(Xcur + noise)

            m = np.zeros_like(Xcur); vv = np.zeros_like(Xcur); cnt = [0]
            Xcur = adam_lse(Xcur, sched, m, vv, cnt)
            Xcur, vc = greedy(Xcur, 200, 0.002)
            # opportunistic LP only if promising
            if vc < best_val + 0.002 and remain() > 30 and (basin % 4 == 0 or vc < best_val):
                Xcur = lp_polish(Xcur, 2.5)
                vc = _max_cos(Xcur)
            else:
                # cheap micro if very close
                if vc < best_val + 0.0005:
                    Xcur, vc = micro(Xcur)
            if vc < best_val:
                best, best_val = Xcur.copy(), float(vc)
        except Exception:
            continue

    # final intensive polish
    try:
        m = np.zeros_like(best); vv = np.zeros_like(best); cnt = [0]
        # allocate ~40% remaining to smooth, rest to LP/greedy
        if remain() > 6:
            best = adam_lse(best, [(0.003, 400, 0.004), (0.001, 500, 0.002)], m, vv, cnt)
        if remain() > 4:
            best, best_val2 = greedy(best, 400, 0.0015)
            best_val = min(best_val, best_val2)
        if remain() > 4:
            best, best_val2 = micro(best)
            best_val = min(best_val, best_val2)
        if remain() > 3:
            best = lp_polish(best, remain() - 1.0)
            best_val = _max_cos(best)
        # last tiny greedy to squeeze LP rounding
        if remain() > 1.5:
            best, bv = greedy(best, 150, 0.0005)
            best_val = min(best_val, bv)
    except Exception:
        pass

    best = _norm(best)
    # shrink guard not needed (we minimize max); just ensure finite
    best[~np.isfinite(best)] = 0.0
    # if any bad row, replace random
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            vv = rng.standard_normal(d); vv /= np.linalg.norm(vv)
            best[i] = vv
    best = _norm(best)
    return best
