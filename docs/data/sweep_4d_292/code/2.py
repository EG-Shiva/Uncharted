import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normed(y):
    n = y / np.linalg.norm(y, axis=1, keepdims=True)
    return n


def _lp_polish(x, seconds, t0=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > 172:
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
    np.random.seed(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = np.random.randn(292, 4)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (292, 4):
        # try to fix shape if transposed etc.
        xb = np.random.randn(292, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape

    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]

    def make_fun(alpha):
        def fun(yf):
            y = yf.reshape(n, d)
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            x = y / nr
            G = x @ x.T
            g = G[iuI, iuJ]
            m = g.max()
            a = alpha * (g - m)
            # a <= 0 so exp safe
            e = np.exp(a)
            S = e.sum()
            L = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iuI, iuJ] = w
            W = W + W.T
            Gx = W @ x  # dL/dX
            dot = (Gx * x).sum(axis=1, keepdims=True)
            Gy = (Gx - dot * x) / nr
            return float(L), Gy.ravel().astype(np.float64)
        return fun

    def lbfgs_polish(x_init, alpha, maxiter):
        fun = make_fun(alpha)
        y0 = x_init.ravel().astype(np.float64)
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14,
                                    "gtol": 1e-7, "maxfun": 10000})
            y = res.x.reshape(n, d)
        except Exception:
            return x_init
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = y / nr
        # guard against NaN
        if not np.all(np.isfinite(x)):
            return x_init
        return x

    def find_holes(x_fixed, k, nsamp=25000):
        # return k hole directions far from x_fixed
        best_cands = []
        best_scores = []
        # chunked sampling
        chunk = 5000
        cands_all = []
        scores_all = []
        for s in range(0, nsamp, chunk):
            m = min(chunk, nsamp - s)
            c = np.random.randn(m, d)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            dots = c @ x_fixed.T
            sc = dots.max(axis=1)
            cands_all.append(c)
            scores_all.append(sc)
        cands_all = np.vstack(cands_all)
        scores_all = np.concatenate(scores_all)
        order = np.argsort(scores_all)
        picked = []
        for idx in order:
            c = cands_all[idx]
            ok = True
            for p in picked:
                if float(p @ c) > 0.90:
                    ok = False
                    break
            if ok:
                picked.append(c)
                if len(picked) >= k:
                    break
        if len(picked) < k:
            for idx in order:
                if len(picked) >= k:
                    break
                picked.append(cands_all[idx])
        return np.array(picked)

    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best_val

    # quick two-stage refine of starting point with smooth loss (cheap)
    # only if time permits -- do one short polish
    # main basin-hopping loop until ~135s
    it = 0
    T = 0.0012
    while time.time() - t0 < 132:
        it += 1
        # periodic restart to best
        if it % 12 == 0:
            cur = x_best.copy()
            cur_val = best_val
        r = np.random.rand()
        if r < 0.55:
            # global shake
            sigma = float(np.random.choice([0.015, 0.03, 0.06, 0.10]))
            pert = cur + sigma * np.random.randn(n, d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        elif r < 0.85:
            # local shake of a neighbourhood
            knear = int(np.random.choice([10, 15, 25, 40]))
            sig_big = float(np.random.choice([0.08, 0.15, 0.25]))
            sig_small = 0.005
            pert = cur.copy() + sig_small * np.random.randn(n, d)
            cen = np.random.randint(n)
            dots = cur @ cur[cen]
            nn = np.argsort(-dots)[:knear]
            pert[nn] += sig_big * np.random.randn(knear, d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        else:
            # hole re-insertion: move 1-3 points to holes
            nmove = int(np.random.choice([1, 2, 3]))
            # choose points to move: prefer loose (rattlers) or overloaded
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)  # per-point max
            glo = G.max()
            # loose = far from glo
            slack = glo - mx
            # pick with bias to loosest + random + most jammed
            order_loose = np.argsort(-slack)  # loosest first
            cand_idx = []
            # include one most jammed sometimes
            if np.random.rand() < 0.5:
                # count contacts within 1e-4
                cnt = (G > glo - 1e-4).sum(axis=1)
                order_jam = np.argsort(-cnt)
                cand_idx.append(int(order_jam[0]))
            for k2 in range(nmove):
                if np.random.rand() < 0.7 and k2 < len(order_loose):
                    # pick among top-20 loosest randomly
                    pool = order_loose[:20]
                    cand_idx.append(int(pool[np.random.randint(len(pool))]))
                else:
                    cand_idx.append(int(np.random.randint(n)))
            cand_idx = cand_idx[:nmove]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[cand_idx] = False
            x_keep = cur[keep_mask]
            holes = find_holes(x_keep, nmove, nsamp=25000)
            pert = cur.copy()
            pert[cand_idx] = holes[:len(cand_idx)]
            # small jitter around holes
            pert[cand_idx] += 0.01 * np.random.randn(len(cand_idx), d)
            pert /= np.linalg.norm(pert, axis=1, keepdims=True)

        # two-stage annealed polish: moderate then sharper
        # adapt maxiter to remaining time
        rem = t0 + 132 - time.time()
        if rem < 2:
            break
        # first stage
        x1 = lbfgs_polish(pert, alpha=70.0, maxiter=100)
        # second stage sharper
        x2 = lbfgs_polish(x1, alpha=200.0, maxiter=100)
        v2 = _max_cos(x2)
        # also try even sharper quick refine occasionally
        if it % 5 == 0 and time.time() - t0 < 128:
            x3 = lbfgs_polish(x2, alpha=400.0, maxiter=60)
            v3 = _max_cos(x3)
            if v3 < v2:
                x2, v2 = x3, v3
        if v2 < best_val - 1e-13:
            best_val = v2
            x_best = x2.copy()
            cur, cur_val = x2.copy(), v2
        else:
            # Metropolis move of current basin
            if v2 < cur_val or np.random.rand() < np.exp(-(v2 - cur_val) / T):
                cur, cur_val = x2.copy(), v2
        # tiny time check
        if time.time() - t0 > 133:
            break

    # final LP trust-region polish on best with remaining time
    rem = 178 - (time.time() - t0)
    if rem > 5:
        # use up to rem but leave 2s margin, cap 35s
        lp_time = min(rem - 2, 35)
        try:
            x_best = _lp_polish(x_best, lp_time, t0=t0)
        except Exception:
            pass
        best_val = _max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
