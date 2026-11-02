import time
import numpy as np

def _nc(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_nc(a))
        except Exception:
            pass
    if not cands:
        cands.append(_nc(rng.standard_normal((74, 13))))
    vals = [_mc(a) for a in cands]
    bi = int(np.argmin(vals))
    gbest = cands[bi].copy()
    gval = float(vals[bi])
    n0, d = 74, 13

    # ---------- helpers ----------
    def adam_lse(X, steps, tau0, tau1, lr0):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        # geometric tau schedule
        lt0 = np.log(tau0); lt1 = np.log(tau1)
        for s in range(steps):
            if (s & 63) == 0 and time.time() - t0 > LIM - 8:
                break
            tau = float(np.exp(lt0 + (lt1 - lt0) * s / max(1, steps - 1)))
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gmax = float(G.max())
            E = np.exp((G - gmax) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            m *= 0.9; m += 0.1 * Gt
            v *= 0.999; v += 0.001 * (Gt * Gt)
            cnt += 1
            c1 = 1.0 - 0.9 ** cnt
            c2 = 1.0 - 0.999 ** cnt
            mh = m / c1; vh = v / c2
            lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
            X -= lr * mh / (np.sqrt(vh) + 1e-8)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def minimax(X, steps, lr0):
        cur = _mc(X)
        lr = lr0
        nn = X.shape[0]
        for it in range(steps):
            if (it & 31) == 0 and time.time() - t0 > LIM - 6:
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
            nr = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = nr[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (nr[act] + 1e-12) * lr
            Xn = _nc(X - step)
            nv = _mc(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def add_caps(X, k, n_cand=3000):
        X = X.copy()
        for _ in range(k):
            C = _nc(rng.standard_normal((n_cand, d)))
            S = C @ X.T
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X = np.vstack([X, C[j:j+1]])
        return X

    def prune_to(X, target):
        X = X.copy()
        while X.shape[0] > target:
            nn = X.shape[0]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            mx = G.max(axis=1)
            deg = (G > gmax - 1e-7).sum(axis=1).astype(float)
            # sum of top-5
            k5 = min(5, nn - 1)
            part = np.sort(G, axis=1)[:, -k5:].sum(axis=1)
            # lexicographic: mx primary, deg, part
            order = np.lexsort((part, deg, mx))
            worst = int(order[-1])
            X = np.delete(X, worst, axis=0)
        return X

    def polish74(X):
        # short repolish for 74
        X = adam_lse(X, 250, 0.004, 0.001, 0.008)
        X, _ = minimax(X, 120, 2e-4)
        X = adam_lse(X, 150, 0.0012, 0.0004, 0.004)
        X, c = minimax(X, 120, 5e-5)
        return X, c

    # ---------- N-homotopy cycles ----------
    cyc = 0
    # quick initial polish of incumbent to set baseline
    try:
        gbest, gval2 = polish74(gbest.copy())
        if gval2 < gval:
            gval = gval2
        else:
            # keep original if worse (adam noise)
            pass
    except Exception:
        pass

    while time.time() - t0 < LIM - 14:
        cyc += 1
        if cyc % 2 == 1:
            # OVERSHOOT: 76-78 then prune
            add = [3, 2, 4, 3][(cyc // 2) % 4]
            try:
                Xe = add_caps(gbest, add, n_cand=3000)
                # jitter existing to allow reorganisation under pressure
                sig = 0.008 + 0.004 * (cyc % 3)
                Xe[:-add] += sig * rng.standard_normal((74, d))
                Xe = _nc(Xe)
                # anneal enlarged (harder) problem
                # vary schedule by cycle for diversity
                if cyc % 4 == 1:
                    Xe = adam_lse(Xe, 500, 0.012, 0.002, 0.02)
                    Xe = adam_lse(Xe, 400, 0.004, 0.0008, 0.01)
                else:
                    Xe = adam_lse(Xe, 600, 0.008, 0.001, 0.015)
                    Xe = adam_lse(Xe, 300, 0.002, 0.0005, 0.008)
                Xe, _ = minimax(Xe, 150, 1e-4)
                # prune back with combined score
                Xp = prune_to(Xe, 74)
                Xp, vp = polish74(Xp)
                if vp < gval:
                    gval = vp
                    gbest = Xp.copy()
            except Exception:
                pass
        else:
            # UNDERSHOOT: drop to 70, anneal, staged reinsertion
            try:
                Xd = prune_to(gbest, 70)
                # small jitter + anneal easy problem
                Xd = _nc(Xd + 0.005 * rng.standard_normal(Xd.shape))
                Xd = adam_lse(Xd, 500, 0.010, 0.001, 0.018)
                Xd, _ = minimax(Xd, 120, 1e-4)
                # staged insertion one at a time with polishing
                Xg = Xd
                for _k in range(4):
                    if time.time() - t0 > LIM - 16:
                        break
                    Xg = add_caps(Xg, 1, n_cand=4000)
                    # polish between insertions
                    Xg = adam_lse(Xg, 220, 0.005, 0.001, 0.010)
                    Xg, _ = minimax(Xg, 80, 1e-4)
                # final repolish at 74
                if Xg.shape[0] == 74:
                    Xg, vg = polish74(Xg)
                    if vg < gval:
                        gval = vg
                        gbest = Xg.copy()
            except Exception:
                pass
        # occasional random-restart injection to avoid stagnation
        if cyc % 12 == 0 and time.time() - t0 < LIM - 30:
            try:
                Xr = _nc(gbest + 0.05 * rng.standard_normal(gbest.shape))
                Xr = adam_lse(Xr, 400, 0.010, 0.001, 0.015)
                Xr, vr = minimax(Xr, 100, 1e-4)
                if vr < gval:
                    gval = vr
                    gbest = Xr.copy()
            except Exception:
                pass

    # ---------- final polish ----------
    try:
        gbest = adam_lse(gbest, 400, 0.002, 0.0004, 0.006)
        gbest, gval = minimax(gbest, 300, 5e-5)
    except Exception:
        pass
    # trust-region LP polish if time remains
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        remain = (t0 + LIM) - time.time()
        if remain > 4:
            X = _nc(gbest)
            best = _mc(X)
            nn, dd = X.shape
            radius = 1e-3
            start = time.time()
            iu = np.triu_indices(nn, 1)
            budget = min(remain - 1.0, 6.0)
            while time.time() - start < budget and radius > 1e-14:
                Cfull = (X @ X.T)[iu]
                m = Cfull.max()
                sel = Cfull > m - max(20 * radius, 1e-9)
                if np.count_nonzero(sel) > 2500:
                    thr = np.sort(Cfull)[-2500]
                    sel = Cfull >= thr
                I = iu[0][sel]; J = iu[1][sel]
                c = Cfull[sel]
                p = len(I)
                if p == 0:
                    break
                nv = nn * dd + 1
                rows = np.repeat(np.arange(p), 2 * dd + 1)
                cols = np.concatenate([I[:, None] * dd + np.arange(dd), J[:, None] * dd + np.arange(dd), np.full((p, 1), nn * dd)], 1).ravel()
                valsA = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))), shape=(nn, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs")
                if res.status != 0:
                    radius *= 0.5
                    continue
                Y = X + res.x[:-1].reshape(nn, dd)
                Y = _nc(Y)
                v = _mc(Y)
                if v < best:
                    X, best = Y, v
                    gbest, gval = X.copy(), best
                    radius = min(radius * 1.5, 0.02)
                else:
                    radius *= 0.5
    except Exception:
        pass

    gbest = _nc(gbest)
    return gbest
