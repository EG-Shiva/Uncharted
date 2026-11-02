import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 165.0
    rng = np.random.default_rng(0)
    n, d = 323, 7

    x_best = None
    v_best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                v = float(G.max())
                if v < v_best:
                    v_best = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal((n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        v_best = float(G.max())

    def soft_anneal(x, steps, T0, T1, lr0, lr1):
        for s in range(steps):
            if s % 100 == 0 and time.time() - t0 > BUDGET - 12:
                break
            frac = s / max(steps - 1, 1)
            T = T0 * (T1 / T0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            # weights, stabilized
            W = np.exp((G - m) / max(T, 1e-12))
            np.fill_diagonal(W, 0.0)
            F = W @ x
            dot = np.sum(F * x, axis=1, keepdims=True)
            Ft = F - dot * x
            nrm = np.sqrt(np.sum(Ft * Ft, axis=1, keepdims=True)) + 1e-12
            avg = float(np.mean(nrm))
            if not np.isfinite(avg) or avg == 0:
                continue
            x -= (lr / avg) * Ft
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[~np.isfinite(nr)] = 1.0
            nr[nr == 0] = 1.0
            x /= nr
        return x

    def greedy_polish(x, iters, s0, s1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur = float(np.max(G))
        for it in range(iters):
            if it % 1500 == 0 and time.time() - t0 > BUDGET - 10:
                break
            # worst pair
            k = int(np.argmax(G))
            a = k // n
            b = k % n
            i = a if rng.random() < 0.5 else b
            j = b if i == a else a
            dot = float(G[a, b])
            t = x[j] - dot * x[i]
            ntt = float(np.linalg.norm(t)) + 1e-12
            d0 = -t / ntt
            r = rng.standard_normal(d)
            r -= float(r.dot(x[i])) * x[i]
            nr_ = float(np.linalg.norm(r)) + 1e-12
            r /= nr_
            alpha = 0.5
            direc = (1.0 - alpha) * d0 + alpha * r
            nd = float(np.linalg.norm(direc)) + 1e-12
            direc /= nd
            frac = it / max(iters - 1, 1)
            sig = s0 * (s1 / s0) ** frac
            y = x[i] + sig * direc
            ny = float(np.linalg.norm(y))
            if not np.isfinite(ny) or ny == 0:
                continue
            y /= ny
            dots = y.dot(x.T)
            dots[i] = -2.0
            worst = float(np.max(dots))
            old_row = float(np.max(G[i]))
            if worst < old_row - 1e-12 and worst <= cur + 1e-12:
                x[i] = y
                G[i, :] = dots
                G[:, i] = dots
                if old_row >= cur - 1e-12:
                    cur = float(np.max(G))
        return x, cur

    # initial direct squeeze of incumbent (no topology change)
    try:
        x_best, v0 = greedy_polish(x_best.copy(), 12000, 0.004, 0.0003)
        if v0 < v_best:
            v_best = v0
    except Exception:
        pass
    if time.time() - t0 > BUDGET - 12:
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        return x_best

    trial = 0
    while time.time() - t0 < BUDGET - 12 and trial < 60:
        trial += 1
        x = x_best.copy()
        mode = trial % 3
        if mode == 1:
            sigma = 0.018 + 0.008 * rng.random()
            x += rng.standard_normal((n, d)) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 350, 0.015, 0.0025, 0.010, 0.0012
            g_iter, gs0, gs1 = 6000, 0.004, 0.0003
        elif mode == 2:
            try:
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                worst = np.max(G, axis=1)
                k = 6 + int(rng.integers(0, 5))
                idx = np.argpartition(worst, -k)[-k:]
                for ii in idx:
                    v = rng.standard_normal(d) * 0.35
                    y = x[ii] + v
                    y /= np.linalg.norm(y)
                    x[ii] = y
            except Exception:
                pass
            steps, T0, T1, lr0, lr1 = 450, 0.018, 0.002, 0.012, 0.001
            g_iter, gs0, gs1 = 7000, 0.005, 0.0003
        else:
            try:
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                worst = np.max(G, axis=1)
                k = 3 + int(rng.integers(0, 3))
                idx = np.argpartition(worst, -k)[-k:]
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                Xf = x[mask]
                Nc = 16000
                for ii in idx:
                    cand = rng.standard_normal((Nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    dots = cand @ Xf.T
                    w = np.max(dots, axis=1)
                    bi = int(np.argmin(w))
                    x[ii] = cand[bi]
                    Xf = np.vstack([Xf, cand[bi][None, :]])
            except Exception:
                pass
            steps, T0, T1, lr0, lr1 = 350, 0.012, 0.002, 0.010, 0.001
            g_iter, gs0, gs1 = 6000, 0.004, 0.0003

        try:
            x = soft_anneal(x, steps, T0, T1, lr0, lr1)
            x, cur = greedy_polish(x, g_iter, gs0, gs1)
        except Exception:
            continue
        # exact score
        try:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cur2 = float(G.max())
        except Exception:
            continue
        if cur2 < v_best - 1e-12:
            v_best = cur2
            x_best = x.copy()

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
