import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > 173:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    np.random.seed(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = np.random.randn(292, 4)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (292, 4):
        xb = np.random.randn(292, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    iuI, iuJ = np.triu_indices(n, 1)

    def make_global_fun(alpha):
        def fun(yf):
            y = yf.reshape(n, d)
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            x = y / nr
            g = (x @ x.T)[iuI, iuJ]
            m = g.max()
            a = alpha * (g - m)
            e = np.exp(a)
            S = e.sum()
            L = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n))
            W[iuI, iuJ] = w
            W = W + W.T
            Gx = W @ x
            dot = (Gx * x).sum(axis=1, keepdims=True)
            Gy = (Gx - dot * x) / nr
            return float(L), Gy.ravel().astype(np.float64)
        return fun

    def global_polish(x_init, alpha, maxiter):
        fun = make_global_fun(alpha)
        y0 = x_init.ravel().astype(np.float64)
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14,
                                    "gtol": 1e-7, "maxfun": 8000})
            y = res.x.reshape(n, d)
        except Exception:
            return x_init
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = y / nr
        if not np.all(np.isfinite(x)):
            return x_init
        return x

    def subset_polish(x_init_full, idx, alpha, maxiter):
        idx = np.asarray(idx, dtype=int)
        k = len(idx)
        if k == 0 or k >= n:
            return global_polish(x_init_full, alpha, maxiter)
        x_fixed = x_init_full.copy()
        # map global pair indices to subset handling: build full x each eval
        # precompute fixed part normalized
        def fun(yf):
            y = yf.reshape(k, d)
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            xs = y / nr
            x = x_fixed.copy()
            x[idx] = xs
            # x already unit except updated part
            g = (x @ x.T)[iuI, iuJ]
            m = g.max()
            a = alpha * (g - m)
            e = np.exp(a)
            S = e.sum()
            L = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n))
            W[iuI, iuJ] = w
            W = W + W.T
            Gx = W @ x
            # gradient only for subset
            Gxs = Gx[idx]
            dot = (Gxs * xs).sum(axis=1, keepdims=True)
            Gy = (Gxs - dot * xs) / nr
            return float(L), Gy.ravel().astype(np.float64)
        y0 = x_init_full[idx].ravel().astype(np.float64)
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14,
                                    "gtol": 1e-7, "maxfun": 5000})
            y = res.x.reshape(k, d)
        except Exception:
            return x_init_full
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            return x_init_full
        out = x_init_full.copy()
        out[idx] = y
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def find_holes(x_keep, k, nsamp=20000):
        chunk = 5000
        cands = []
        scores = []
        for s in range(0, nsamp, chunk):
            m = min(chunk, nsamp - s)
            c = np.random.randn(m, d)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            dots = c @ x_keep.T
            sc = dots.max(axis=1)
            cands.append(c)
            scores.append(sc)
        cands = np.vstack(cands)
        scores = np.concatenate(scores)
        order = np.argsort(scores)
        picked = []
        for ii in order:
            c = cands[ii]
            ok = True
            for p in picked:
                if float(p @ c) > 0.88:
                    ok = False
                    break
            if ok:
                picked.append(c)
                if len(picked) >= k:
                    break
        if len(picked) < k:
            for ii in order:
                if len(picked) >= k:
                    break
                picked.append(cands[ii])
        return np.array(picked)

    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best_val

    # fast initial sharpen (cheap, already near optimum)
    # skip heavy work if little time; go straight to hopping

    it = 0
    T0 = 0.0010
    while time.time() - t0 < 132:
        it += 1
        if it % 15 == 0:
            cur = x_best.copy()
            cur_val = best_val
        T = T0 * (1.0 - 0.5 * (time.time() - t0) / 132.0)
        T = max(T, 0.0003)
        r = np.random.rand()
        pert = None
        cap_idx = None
        if r < 0.35:
            sigma = float(np.random.choice([0.012, 0.03, 0.07, 0.12]))
            pert = cur + sigma * np.random.randn(n, d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        elif r < 0.70:
            # cap large-neighbourhood: k nearest to random center
            knear = int(np.random.choice([12, 20, 30, 45]))
            cen = np.random.randint(n)
            # random direction cap sometimes instead of point center
            if np.random.rand() < 0.3:
                u = np.random.randn(d)
                u /= np.linalg.norm(u)
                dots = cur @ u
                nn = np.argsort(-dots)[:knear]
            else:
                dots = cur @ cur[cen]
                nn = np.argsort(-dots)[:knear]
            cap_idx = nn
            pert = cur.copy()
            # strong re-scatter inside cap: mix of shake + fresh random in cap
            sig_big = float(np.random.choice([0.15, 0.30, 0.55]))
            if np.random.rand() < 0.4:
                # fresh random directions biased to cap mean
                mean_dir = pert[nn].mean(axis=0)
                mean_dir /= (np.linalg.norm(mean_dir) + 1e-12)
                fresh = mean_dir[None, :] + 0.9 * np.random.randn(knear, d)
                fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
                pert[nn] = fresh
            else:
                pert[nn] = pert[nn] + sig_big * np.random.randn(knear, d)
                pert[nn] /= np.linalg.norm(pert[nn], axis=1, keepdims=True)
            # tiny global jitter
            pert = pert + 0.003 * np.random.randn(n, d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        else:
            nmove = int(np.random.choice([1, 2, 3]))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            glo = G.max()
            slack = glo - mx
            order_loose = np.argsort(-slack)
            cand_idx = []
            if np.random.rand() < 0.5:
                cnt = (G > glo - 1e-4).sum(axis=1)
                order_jam = np.argsort(-cnt)
                cand_idx.append(int(order_jam[0]))
            for k2 in range(nmove):
                if np.random.rand() < 0.7:
                    pool = order_loose[:25]
                    cand_idx.append(int(pool[np.random.randint(len(pool))]))
                else:
                    cand_idx.append(int(np.random.randint(n)))
            cand_idx = cand_idx[:nmove]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[np.array(cand_idx)] = False
            x_keep = cur[keep_mask]
            holes = find_holes(x_keep, nmove, nsamp=20000)
            pert = cur.copy()
            pert[np.array(cand_idx)] = holes[:len(cand_idx)]
            pert[np.array(cand_idx)] += 0.008 * np.random.randn(len(cand_idx), d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
            cap_idx = np.array(cand_idx)

        rem = t0 + 132 - time.time()
        if rem < 2:
            break
        # polish: if cap move, subset first then global
        if cap_idx is not None and len(cap_idx) < n and r >= 0.35 and r < 0.70:
            x1 = subset_polish(pert, cap_idx, alpha=90.0, maxiter=60)
            x2 = global_polish(x1, alpha=220.0, maxiter=70)
        else:
            x1 = global_polish(pert, alpha=80.0, maxiter=70)
            x2 = global_polish(x1, alpha=250.0, maxiter=70)
        v2 = _max_cos(x2)
        if it % 6 == 0 and (t0 + 128 - time.time()) > 3:
            x3 = global_polish(x2, alpha=450.0, maxiter=40)
            v3 = _max_cos(x3)
            if v3 < v2:
                x2, v2 = x3, v3
        if v2 < best_val - 1e-13:
            best_val = v2
            x_best = x2.copy()
            cur, cur_val = x2.copy(), v2
        else:
            if v2 < cur_val or np.random.rand() < np.exp(-(v2 - cur_val) / T):
                cur, cur_val = x2.copy(), v2
        if time.time() - t0 > 133:
            break

    rem = 178 - (time.time() - t0)
    if rem > 6:
        lp_time = min(rem - 2, 32)
        try:
            x_best = _lp_polish(x_best, lp_time, t0=t0)
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: keep best ever
    try:
        if _max_cos(x_best) > _max_cos(xb) + 1e-12:
            # still return x_best if marginally worse? prefer xb
            # return the better one
            if _max_cos(xb) <= _max_cos(x_best):
                return xb
    except Exception:
        pass
    return x_best
