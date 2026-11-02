import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x, iu=None):
    g = x @ x.T
    if iu is None:
        np.fill_diagonal(g, -2.0)
        return float(g.max())
    c = g[iu]
    return float(c.max())


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    phase1_end = t0 + 55.0
    phase2_end = t0 + 105.0
    lp_end = HARD_END - 2.0

    np.random.seed(0)
    rng = np.random.default_rng(0)

    # load start
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.normal(size=(453, 5))
    x0 = np.asarray(x0, dtype=np.float64)
    # ensure shape (453,5)
    if x0.shape != (453, 5):
        # try to reshape / truncate / pad deterministically
        flat = x0.ravel()
        need = 453 * 5
        if flat.size < need:
            extra = rng.normal(size=(need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(453, 5)
    n, d = x0.shape
    nr = np.linalg.norm(x0, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x0 = x0 / nr

    iu = np.triu_indices(n, 1)
    best = x0.copy()
    best_val = _max_cos(best, iu)

    # ---------- Phase 1: annealed soft-max GD with momentum ----------
    x = best.copy()
    vel = np.zeros_like(x)
    beta = 80.0
    beta_max = 4000.0
    growth = 1.0015
    mu = 0.9
    lr0 = 0.008
    it = 0
    while time.time() < phase1_end and time.time() < HARD_END:
        for _ in range(200):
            G = x @ x.T
            c = G[iu]
            m = float(c.max())
            if m < best_val:
                best_val = m
                best = x.copy()
            dd = beta * (c - m)
            # exp in place
            w = np.exp(dd)
            s = float(w.sum())
            if s == 0 or not np.isfinite(s):
                w = np.ones_like(w)
                s = float(w.sum())
            S = np.zeros((n, n), dtype=np.float64)
            S[iu] = w
            S = S + S.T
            grad = (S @ x) / s
            dot = np.einsum('ij,ij->i', grad, x)
            gt = grad - dot[:, None] * x
            rms = float(np.sqrt(np.mean(gt * gt))) + 1e-12
            gn = gt / rms
            vel = mu * vel + (1.0 - mu) * gn
            lr = lr0 / (1.0 + it * 0.0003)
            x = x - lr * vel
            nn = np.linalg.norm(x, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            x = x / nn
            beta = beta * growth
            if beta > beta_max:
                beta = beta_max
            it += 1
            if time.time() >= phase1_end:
                break
        # re-anchor if diverged
        # m is last max
        try:
            cur_m = m
        except Exception:
            cur_m = best_val
        if cur_m > best_val + 0.015:
            x = best + rng.normal(0, 0.005, size=x.shape)
            nn = np.linalg.norm(x, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            x = x / nn
            vel *= 0.0
            beta = 80.0

    # ---------- Phase 2: basin hopping (perturb + short GD) ----------
    trial = 0
    while time.time() < phase2_end and time.time() < HARD_END:
        trial += 1
        scale = 0.01 if (trial % 3 != 0) else 0.03
        # bias noise to worst points every other trial
        if trial % 2 == 0:
            G0 = best @ best.T
            G0[np.diag_indices(n)] = -2.0
            pm = G0.max(axis=1)
            order = np.argsort(-pm)
            noise = rng.normal(0, scale * 0.5, size=best.shape)
            # extra kick to worst 30
            worst = order[:30]
            noise[worst] += rng.normal(0, scale, size=(30, d))
            x = best + noise
        else:
            x = best + rng.normal(0, scale, size=best.shape)
        nn = np.linalg.norm(x, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        x = x / nn
        vel2 = np.zeros_like(x)
        beta2 = 300.0 if (trial % 2 == 0) else 1000.0
        lr0_t = 0.005
        t_trial_end = min(time.time() + 2.5, phase2_end, HARD_END)
        it2 = 0
        while time.time() < t_trial_end:
            for _ in range(100):
                G = x @ x.T
                c = G[iu]
                m = float(c.max())
                if m < best_val:
                    best_val = m
                    best = x.copy()
                dd = beta2 * (c - m)
                w = np.exp(dd)
                s = float(w.sum())
                if s == 0 or not np.isfinite(s):
                    w = np.ones_like(w)
                    s = float(w.sum())
                S = np.zeros((n, n), dtype=np.float64)
                S[iu] = w
                S = S + S.T
                grad = (S @ x) / s
                dot = np.einsum('ij,ij->i', grad, x)
                gt = grad - dot[:, None] * x
                rms = float(np.sqrt(np.mean(gt * gt))) + 1e-12
                gn = gt / rms
                vel2 = mu * vel2 + (1.0 - mu) * gn
                lr = lr0_t / (1.0 + it2 * 0.001)
                x = x - lr * vel2
                nn = np.linalg.norm(x, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                x = x / nn
                beta2 = beta2 * 1.002
                if beta2 > 5000.0:
                    beta2 = 5000.0
                it2 += 1
                if time.time() >= t_trial_end:
                    break

    # ---------- Phase 3: trust-region LP polish ----------
    x = best.copy()
    radius = 2e-3
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    beq = np.zeros(n)
    # precompute eq matrix pattern (depends on x each iter, so rebuild)
    while time.time() < lp_end and radius > 1e-13:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        if m < best_val:
            best_val = m
            best = x.copy()
        thresh = max(20.0 * radius, 1e-9)
        sel = c > (m - thresh)
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 6000:
            kth = np.partition(c, -6000)[-6000]
            sel = c >= kth
            p = int(sel.sum())
            if p > 7000:
                # take top 6000 by argpartition
                idx = np.argpartition(-c, 6000)[:6000]
                msel = np.zeros_like(sel)
                msel[idx] = True
                sel = msel
                p = int(sel.sum())
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols_I = (I[:, None] * d + np.arange(d)).ravel()
            cols_J = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::(2 * d + 1)] = cols_I[0::d]  # placeholder, rebuilt below
            # build cols correctly
            cols = np.concatenate([
                (I[:, None] * d + np.arange(d)).ravel(),
                (J[:, None] * d + np.arange(d)).ravel(),
                np.full(p, n * d, dtype=np.int64)
            ]) if False else None
            # simpler: build in blocks
            cI = (I[:, None] * d + np.arange(d))
            cJ = (J[:, None] * d + np.arange(d))
            # interleave per row: [cI_row, cJ_row, t]
            cols2 = np.empty((p, 2 * d + 1), dtype=np.int64)
            cols2[:, :d] = cI
            cols2[:, d:2 * d] = cJ
            cols2[:, 2 * d] = n * d
            cols = cols2.ravel()
            vals2 = np.empty((p, 2 * d + 1), dtype=np.float64)
            vals2[:, :d] = x[J]
            vals2[:, d:2 * d] = x[I]
            vals2[:, 2 * d] = -1.0
            vals = vals2.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            # A_eq: x_i . dx_i = 0
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            evals = x.ravel()
            A_eq = coo_matrix((evals, (erows, ecols)), shape=(n, nv)).tocsr()
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=beq,
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nn = np.linalg.norm(y, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        y = y / nn
        v = _max_cos(y, iu)
        if v < m - 1e-12:
            x = y
            if v < best_val:
                best_val = v
                best = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5

    nn = np.linalg.norm(best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    best = best / nn
    return best
