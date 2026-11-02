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
    rng = np.random.default_rng(42)

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
    n, d = 74, 13

    def adam_lse(X, schedule, t0):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        for tau, steps, lr0 in schedule:
            for s in range(steps):
                if (s & 63) == 0 and time.time() - t0 > LIM - 6:
                    return X
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
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt += 1
                c1 = 1.0 - 0.9 ** cnt
                c2 = 1.0 - 0.999 ** cnt
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def minimax(X, steps, lr0, t0, cur_init=None):
        cur = cur_init if cur_init is not None else _max_cos(X)
        lr = lr0
        for it in range(steps):
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
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def worst_order(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx)[:k], float(G.max())

    def hole_replace(X, idx_list, n_cand=3000):
        for iw in idx_list:
            cand = _normalize(rng.standard_normal((n_cand, d)))
            S = cand @ X.T
            S[:, iw] = -10.0
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X[iw] = cand[j]
        return X

    def split_worst_pair(X, alpha=0.015):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        ij = int(np.argmax(G))
        ii, jj = divmod(ij, n)
        X[ii] = X[ii] - alpha * X[jj]
        X[jj] = X[jj] - alpha * X[ii]
        # renormalize those two
        for k in (ii, jj):
            nr = np.linalg.norm(X[k])
            if nr == 0 or not np.isfinite(nr):
                X[k] = _normalize(rng.standard_normal((1, d)))[0]
            else:
                X[k] /= nr
        return X

    def micro_squeeze(X, cur, t0, budget=2.0):
        start = time.time()
        step = 3e-6
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
            # try both, i-only, j-only
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
                # fast max: only need pairs involving i or j plus old max?
                # full eval for safety (cheap)
                G2 = Y @ Y.T
                np.fill_diagonal(G2, -2.0)
                v = float(G2.max())
                if v < cur:
                    X[:] = Y
                    cur = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-12:
                    break
        return X, cur

    sched_full = [(0.02, 300, 0.012), (0.008, 500, 0.008), (0.003, 700, 0.005), (0.0012, 500, 0.003)]
    sched_fast = [(0.008, 350, 0.008), (0.003, 450, 0.005), (0.0012, 350, 0.003)]
    sched_tiny = [(0.003, 250, 0.004), (0.001, 250, 0.002)]

    # initial squeeze of incumbent
    X0 = best.copy()
    X0 = adam_lse(X0, sched_full, t0)
    X0, v0 = minimax(X0, 350, 0.002, t0)
    X0, v0 = micro_squeeze(X0, v0, t0, budget=1.0)
    if v0 < best_val:
        best, best_val = X0, v0

    basin = 0
    while time.time() - t0 < LIM - 8:
        remain = LIM - (time.time() - t0)
        if remain < 12:
            sched = sched_tiny
            n_cand = 1500
        elif remain < 35:
            sched = sched_fast
            n_cand = 2000
        else:
            sched = sched_full if (basin % 3 != 0) else sched_fast
            n_cand = 3000

        Xcur = best.copy()
        mode = basin % 8
        if mode == 0:
            Xcur = _normalize(Xcur + 0.010 * rng.standard_normal(Xcur.shape))
            Xcur = split_worst_pair(Xcur, 0.012)
        elif mode == 1:
            wo, _ = worst_order(Xcur, 1)
            Xcur = hole_replace(Xcur, [int(wo[0])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        elif mode == 2:
            wo, _ = worst_order(Xcur, 2)
            Xcur = hole_replace(Xcur, [int(wo[0]), int(wo[1])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cntact = (G > gmax - 0.03).sum(axis=1)
            order = np.argsort(-cntact)
            k = 6 + (basin % 5)
            idx = order[:k]
            noise = 0.004 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.022 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + noise)
            Xcur = split_worst_pair(Xcur, 0.008)
        elif mode == 4:
            Xcur = _normalize(Xcur + 0.020 * rng.standard_normal(Xcur.shape))
        elif mode == 5:
            wo, _ = worst_order(Xcur, 1)
            Xcur = hole_replace(Xcur, [int(wo[0])], n_cand=n_cand)
            # also kick its worst partner
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] = _normalize(Xcur[ii:ii+1] + 0.006 * rng.standard_normal((1, d)))[0]
            Xcur[jj] = _normalize(Xcur[jj:jj+1] + 0.006 * rng.standard_normal((1, d)))[0]
        elif mode == 6:
            Xcur = _normalize(Xcur + 0.006 * rng.standard_normal(Xcur.shape))
            Xcur = split_worst_pair(Xcur, 0.02)
        else:
            # large diversity kick every 8th
            wo, _ = worst_order(Xcur, 3)
            Xcur = hole_replace(Xcur, [int(wo[0]), int(wo[1])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.008 * rng.standard_normal(Xcur.shape))

        Xcur = adam_lse(Xcur, sched, t0)
        Xcur, vcur = minimax(Xcur, 300, 0.002, t0)
        # quick micro only if promising
        if vcur < best_val + 5e-08:
            Xcur, vcur = micro_squeeze(Xcur, vcur, t0, budget=0.4)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        basin += 1
        if basin > 300:
            break

    if time.time() - t0 < LIM - 2:
        Xf = best.copy()
        Xf, vf = minimax(Xf, 300, 0.0008, t0, cur_init=best_val)
        Xf, vf = micro_squeeze(Xf, vf, t0, budget=1.5)
        if vf < best_val:
            best, best_val = Xf, vf

    best = _normalize(best)
    # safety: remove NaN
    best[~np.isfinite(best)] = 0.0
    # if any zero row happened, randomize it
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
