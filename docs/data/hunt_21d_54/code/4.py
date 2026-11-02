import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds, deadline=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    P = len(iu0[0])
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        # cap size for speed
        if cnt > 1200:
            K = 1200
            part = np.argpartition(-c, K)[:K]
            m2 = float(np.min(c[part]))
            sel = c >= m2
            # ensure not too many due to ties
            if int(np.sum(sel)) > 1500:
                sel = np.zeros(P, dtype=bool)
                sel[part] = True
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols_I = (I[:, None] * d + np.arange(d)).ravel()
            cols_J = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.concatenate([
                cols_I,
                cols_J,
                np.full(p, n * d, dtype=int)
            ])
            # need interleaved per row: [xJ (d), xI (d), -1]
            vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
            # rebuild in correct row order: rows were repeat, cols above are blocked; fix ordering
            # Build directly row-major:
            crow = np.empty(p * (2 * d + 1), dtype=int)
            ccol = np.empty(p * (2 * d + 1), dtype=int)
            cval = np.empty(p * (2 * d + 1), dtype=float)
            for k in range(p):
                s = k * (2 * d + 1)
                crow[s:s + 2 * d + 1] = k
                ccol[s:s + d] = I[k] * d + np.arange(d)
                cval[s:s + d] = x[J[k]]
                ccol[s + d:s + 2 * d] = J[k] * d + np.arange(d)
                cval[s + d:s + 2 * d] = x[I[k]]
                ccol[s + 2 * d] = n * d
                cval[s + 2 * d] = -1.0
            A_ub = coo_matrix((cval, (crow, ccol)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (np.tile(x.ravel(), 1),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -c[sel]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = x + dx
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            best = v
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_relax(x, steps, beta0, beta1, s0, s1, rng=None):
    n, d = x.shape
    y = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 and beta1 > 0 else beta0
        step = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0
        G = y @ y.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # weights focused near max
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        # symmetrize (already symmetric)
        grad = E @ y  # (n,d)
        # tangential projection
        dot = np.sum(grad * y, axis=1, keepdims=True)
        grad_t = grad - dot * y
        nrm = np.linalg.norm(grad_t, axis=1, keepdims=True) + 1e-12
        y = y - step * (grad_t / nrm)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
    return y


def best_hole_against(Xothers, n_samples, rng):
    d = Xothers.shape[1]
    cand = rng.standard_normal((n_samples, d))
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    dots = cand @ Xothers.T
    vals = np.max(dots, axis=1)
    i = int(np.argmin(vals))
    return cand[i], float(vals[i])


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    # load start
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        try:
            x0 = np.load("record.npy").astype(float)
        except Exception:
            x0 = rng.standard_normal((54, 21))
    if x0.shape != (54, 21):
        # try to reshape / truncate
        tmp = rng.standard_normal((54, 21))
        m = min(x0.shape[0], 54)
        mm = min(x0.shape[1], 21)
        tmp[:m, :mm] = x0[:m, :mm]
        x0 = tmp
    x0 = x0 / (np.linalg.norm(x0, axis=1, keepdims=True) + 1e-30)
    cur = x0.copy()
    cur_val = max_cos(cur)
    best = cur.copy()
    best_val = cur_val
    n, d = 54, 21

    it = 0
    no_improve = 0
    # reserve final polish
    while time.time() < deadline - 22.0:
        it += 1
        remaining = deadline - 22.0 - time.time()
        if remaining <= 0:
            break
        mode = it % 7
        y = best.copy()
        if mode == 0:
            sigma = 0.006 if no_improve < 12 else 0.015
            y = y + sigma * rng.standard_normal(y.shape)
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=350, beta0=120.0, beta1=700.0, s0=0.004, s1=0.0004)
        elif mode == 1:
            sigma = 0.016 if no_improve < 12 else 0.035
            y = y + sigma * rng.standard_normal(y.shape)
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=400, beta0=150.0, beta1=900.0, s0=0.005, s1=0.0004)
        elif mode == 2:
            # jitter worst points only
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            per_max = np.max(G, axis=1)
            k = 6
            idx = np.argsort(-per_max)[:k]
            sigma = 0.05 if no_improve < 15 else 0.10
            y[idx] = y[idx] + sigma * rng.standard_normal((k, d))
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=350, beta0=120.0, beta1=800.0, s0=0.004, s1=0.0004)
        elif mode == 3:
            # hole re-insertion of worst point
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            per_max = np.max(G, axis=1)
            w = int(np.argmax(per_max))
            others = np.delete(y, w, axis=0)
            hv, _ = best_hole_against(others, 30000, rng)
            # refine hole a bit: few gradient steps on hv alone
            h = hv.copy()
            for _ in range(120):
                dd = others @ h  # (53,)
                mm = float(np.max(dd))
                ww = np.exp(400.0 * (dd - mm))
                g = ww @ others  # (d,)
                g = g - float(g @ h) * h
                nn = float(np.linalg.norm(g)) + 1e-12
                h = h - 0.01 * g / nn
                h = h / (float(np.linalg.norm(h)) + 1e-30)
            y[w] = h
            # small jitter others
            y = y + 0.002 * rng.standard_normal(y.shape)
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=300, beta0=150.0, beta1=800.0, s0=0.003, s1=0.0003)
        elif mode == 4:
            # resample 2-3 random points
            k = 3 if (it % 14 != 0) else 5
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, d))
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=450, beta0=100.0, beta1=800.0, s0=0.006, s1=0.0005)
        elif mode == 5:
            # targeted pair separation for top pairs
            G = y @ y.T
            iu = np.triu_indices(n, 1)
            c = G[iu]
            K = 12
            top = np.argsort(-c)[:K]
            s = 0.012
            for t in top:
                i = int(iu[0][t]); j = int(iu[1][t])
                # tangent directions away
                vi = y[i] - float(y[i] @ y[j]) * y[j]
                vj = y[j] - float(y[j] @ y[i]) * y[i]
                ni = float(np.linalg.norm(vi)) + 1e-12
                nj = float(np.linalg.norm(vj)) + 1e-12
                y[i] = y[i] + s * vi / ni
                y[j] = y[j] + s * vj / nj
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=300, beta0=200.0, beta1=900.0, s0=0.003, s1=0.0003)
        else:
            # random orthogonal-ish kick: rotate random subset slightly
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = y[idx] + 0.03 * rng.standard_normal((k, d))
            y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            y = smooth_relax(y, steps=350, beta0=120.0, beta1=750.0, s0=0.004, s1=0.0004)

        # quick filter: if smooth result is terrible skip long LP
        vs = max_cos(y)
        # adaptive LP budget
        if vs > best_val + 0.01:
            # still try short polish but very short
            budget = 0.8
        else:
            budget = 1.8
        # ensure we don't exceed loop reserve
        budget = min(budget, max(0.5, deadline - 22.0 - time.time()))
        if budget <= 0.2:
            break
        y = lp_polish(y, seconds=budget, deadline=deadline - 20.0)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            no_improve = 0
        else:
            no_improve += 1

    # final long polish
    final_budget = max(0.5, deadline - time.time() - 1.0)
    final_budget = min(final_budget, 30.0)
    if final_budget > 0.5:
        best = lp_polish(best, seconds=final_budget, deadline=deadline)
        best_val = max_cos(best)
    # safety normalize, de-NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    best = best / nn
    # tiny shrink not needed; return
    return best
