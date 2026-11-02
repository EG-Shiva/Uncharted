import time
import numpy as np

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

    # load starts
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
        a = rng.standard_normal((74, 13))
        cands.append(_normalize(a))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    def refine_lse(X, schedule, m, v, cnt):
        n, d = X.shape
        for tau, steps, lr0 in schedule:
            for s in range(steps):
                if time.time() - t0 > LIM - 4:
                    return X, cnt
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
                # renorm
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X, cnt

    def fine_max(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        n = X.shape[0]
        for _ in range(steps):
            if time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            # gradient from active pairs
            A = mask.astype(np.float64)
            Gr = A @ X
            d = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - d * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            step = np.zeros_like(Gt)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = X - step
            Xn = _normalize(Xn)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
        return X, cur

    def worst_index(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return int(np.argmax(G.max(axis=1)))

    def farthest_reinsert(X):
        iw = worst_index(X)
        cand = rng.standard_normal((2000, 13))
        cand = _normalize(cand)
        # score = max dot to others (excluding iw)
        S = cand @ X.T  # 2000x74
        # ignore column iw by setting very low
        S[:, iw] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[iw] = cand[j]
        return X

    # schedules
    full_sched = [(0.03, 350, 0.012), (0.01, 700, 0.008), (0.003, 1000, 0.005), (0.0012, 700, 0.003)]
    fast_sched = [(0.01, 400, 0.008), (0.003, 600, 0.005), (0.0012, 400, 0.003)]
    tiny_sched = [(0.003, 300, 0.004), (0.001, 300, 0.002)]

    # basin 0: just squeeze starting point
    basin = 0
    # first fine squeeze
    X0 = best.copy()
    m = np.zeros_like(X0); v = np.zeros_like(X0); cnt = [0]
    X0, _ = refine_lse(X0, full_sched, m, v, cnt)
    X0, val0 = fine_max(X0, 400, 0.002)
    if val0 < best_val:
        best, best_val = X0, val0

    basin = 1
    # main hopping
    while time.time() - t0 < LIM - 8:
        remain = LIM - (time.time() - t0)
        if remain < 10:
            sched = tiny_sched
        elif remain < 30:
            sched = fast_sched
        else:
            sched = full_sched if (basin % 3 != 0) else fast_sched

        Xcur = best.copy()
        mode = basin % 6
        if mode == 0:
            sigma = 0.018
            Xcur = _normalize(Xcur + sigma * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            # shake worst subset harder
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            gmax = G.max()
            thr = gmax - 0.03
            cntact = (G > thr).sum(axis=1)
            # points with many near contacts are jammed, shake them
            order = np.argsort(-cntact)
            k = 8 + (basin % 5)
            idx = order[:k]
            noise = 0.004 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.025 * rng.standard_normal((len(idx), 13))
            Xcur = _normalize(Xcur + noise)
        elif mode == 2:
            Xcur = farthest_reinsert(Xcur)
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            sigma = 0.008
            Xcur = _normalize(Xcur + sigma * rng.standard_normal(Xcur.shape))
            # also kick single worst pair apart
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            ii, jj = divmod(int(ij), 74)
            Xcur[ii] -= 0.01 * Xcur[jj]
            Xcur[jj] -= 0.01 * Xcur[ii]
            Xcur = _normalize(Xcur)
        elif mode == 4:
            # reinsert second-worst too
            Xcur = farthest_reinsert(Xcur)
            Xcur = farthest_reinsert(Xcur)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        else:
            sigma = 0.012
            Xcur = _normalize(Xcur + sigma * rng.standard_normal(Xcur.shape))

        m = np.zeros_like(Xcur); v = np.zeros_like(Xcur); cnt = [0]
        Xcur, _ = refine_lse(Xcur, sched, m, v, cnt)
        Xcur, vcur = fine_max(Xcur, 300, 0.002)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        basin += 1
        if basin > 200:
            break

    # final tiny squeeze on best
    if time.time() - t0 < LIM - 2:
        Xf = best.copy()
        Xf, vf = fine_max(Xf, 300, 0.0008)
        if vf < best_val:
            best, best_val = Xf, vf

    best = _normalize(best)
    return best
