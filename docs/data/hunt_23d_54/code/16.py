import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos_from_G(G, iu):
    return float(G[iu].max())

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(12345)
    n, d = 54, 23
    iu = np.triu_indices(n, 1)
    diag_idx = np.diag_indices(n)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also try record
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        Gr = xr @ xr.T
        vr = float(Gr[iu].max())
        Gb0 = xb @ xb.T
        vb0 = float(Gb0[iu].max())
        if vr < vb0:
            xb = xr.copy()
            vb0 = vr
    except Exception:
        pass
    Gb = xb @ xb.T
    best = float(Gb[iu].max())
    x_best = xb.copy()

    # smoothing helper inline
    def smooth_anneal(x, steps, k0, k1, lr0, lr1, mu, t0_, deadline):
        v = np.zeros_like(x)
        # precompute log ratio
        lk0 = np.log(k0); lk1 = np.log(k1)
        for s in range(steps):
            if (s & 63) == 0 and time.time() - t0_ > deadline:
                break
            frac = s / max(steps - 1, 1)
            k = np.exp(lk0 + (lk1 - lk0) * frac)
            lr = lr0 * (lr1 / lr0) ** frac
            G = x @ x.T
            G[diag_idx] = -2.0
            m = float(G.max())
            # stable exp, G-m <=0
            E = np.exp(k * (G - m))
            E[diag_idx] = 0.0
            deg = E.sum(axis=1, keepdims=True)
            # avoid div by zero (should not happen)
            Gr = (E @ x) / np.maximum(deg, 1e-12)
            dot = np.sum(Gr * x, axis=1, keepdims=True)
            Gt = Gr - dot * x
            v = mu * v + Gt
            x = x - lr * v
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    it = 0
    # reserve ~12s for final LP
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        # time check
        remain = BUDGET - 12.0 - (time.time() - t0)
        if remain < 2.0:
            break
        # contact analysis on x_best
        Gb2 = x_best @ x_best.T
        Gb2[diag_idx] = -2.0
        # wide window for looseness ranking
        wide = (Gb2 > best - 0.02)
        cnt = wide.sum(axis=1)
        order_lo = np.argsort(cnt)
        order_hi = np.argsort(-cnt)
        # row max for worst point
        rowmax = Gb2.max(axis=1)
        i_worst = int(np.argmax(rowmax))
        # second worst
        tmp_rm = rowmax.copy()
        tmp_rm[i_worst] = -2
        i_worst2 = int(np.argmax(tmp_rm))

        x = x_best.copy()
        mode = it % 7
        if mode == 1:
            x += rng.randn(n, d) * 0.008
            steps, k0, k1, lr0, lr1, mu = 500, 80.0, 700.0, 0.06, 0.004, 0.85
        elif mode == 2:
            x += rng.randn(n, d) * 0.025
            steps, k0, k1, lr0, lr1, mu = 600, 60.0, 900.0, 0.07, 0.004, 0.85
        elif mode == 3:
            # shake loose points hard
            ksel = 5
            idx = order_lo[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.09
            steps, k0, k1, lr0, lr1, mu = 600, 60.0, 800.0, 0.07, 0.004, 0.88
        elif mode == 4:
            # shake jammed core
            ksel = 4
            idx = order_hi[:ksel]
            x += rng.randn(n, d) * 0.005
            x[idx] += rng.randn(ksel, d) * 0.06
            steps, k0, k1, lr0, lr1, mu = 600, 80.0, 1000.0, 0.06, 0.003, 0.85
        elif mode == 5:
            # hole re-insert 1 worst point, best-of-random
            keep = [i for i in range(n) if i != i_worst]
            Xr = x[keep]
            cand = rng.randn(300, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ Xr.T
            mx = dots.max(axis=1)
            best_c = int(np.argmin(mx))
            x[i_worst] = cand[best_c]
            x += rng.randn(n, d) * 0.004
            steps, k0, k1, lr0, lr1, mu = 700, 50.0, 900.0, 0.07, 0.004, 0.88
        elif mode == 6:
            # hole re-insert 2 worst
            for iw in (i_worst, i_worst2):
                keep = [i for i in range(n) if i != iw]
                Xr = x[keep]
                cand = rng.randn(200, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xr.T
                mx = dots.max(axis=1)
                best_c = int(np.argmin(mx))
                x[iw] = cand[best_c]
            x += rng.randn(n, d) * 0.004
            steps, k0, k1, lr0, lr1, mu = 700, 50.0, 900.0, 0.07, 0.004, 0.88
        else:  # mode 0
            x += rng.randn(n, d) * 0.012
            # random single point big kick occasionally
            if (it // 7) % 3 == 2:
                j = int(rng.randint(n))
                x[j] += rng.randn(d) * 0.12
            steps, k0, k1, lr0, lr1, mu = 500, 80.0, 700.0, 0.06, 0.004, 0.85

        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # adapt steps to remaining time
        if remain < 6:
            steps = min(steps, 250)
        deadline = BUDGET - 12.0 - (time.time() - t0)
        # smooth
        try:
            x = smooth_anneal(x, steps, k0, k1, lr0, lr1, mu, time.time(), deadline if deadline > 0 else 0)
        except Exception:
            continue
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        G = x @ x.T
        v = float(G[iu].max())
        if v < best and np.all(np.isfinite(x)):
            best = v
            x_best = x.copy()

    # final short trust-region LP polish with time guard
    def lp_polish_short(x, seconds, t_global, budget_global):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        Gf = x @ x.T
        best_l = float(Gf[iu].max())
        radius = 1e-3
        start = time.time()
        while time.time() - start < seconds and radius > 1e-14:
            if time.time() - t_global > budget_global:
                break
            Gfull = x @ x.T
            c_all = Gfull[iu]
            m = float(c_all.max())
            sel = c_all > m - max(20 * radius, 1e-9)
            if np.count_nonzero(sel) > 900:
                idx = np.where(sel)[0]
                order = np.argsort(c_all[idx])[::-1][:900]
                nsel = np.zeros_like(sel, dtype=bool)
                nsel[idx[order]] = True
                sel = nsel
            I, J = iu[0][sel], iu[1][sel]
            c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            nv = nn * dd + 1
            try:
                rows = np.repeat(np.arange(p), 2 * dd + 1)
                cols = np.concatenate([I[:, None] * dd + np.arange(dd), J[:, None] * dd + np.arange(dd), np.full((p, 1), nn * dd)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))), shape=(nn, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(nn, dd)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            Gy = y @ y.T
            vv = float(Gy[iu].max())
            if vv < best_l:
                x, best_l = y, vv
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return x

    try:
        # use at most remaining budget up to 11s
        rem_total = 177.0 - (time.time() - t0)
        sec = float(min(11.0, max(1.0, rem_total - 0.5)))
        if sec > 0.5:
            x_best = lp_polish_short(x_best, sec, t0, 177.0)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
