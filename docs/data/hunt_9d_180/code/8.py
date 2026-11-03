import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _contact_counts(x, m=None, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    if m is None:
        m = float(g.max())
    c = (g > m - tol).sum(axis=1)
    return c, m, g


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    t0 = time.time()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    # cache iu arrays
    iu_r, iu_c = iu0
    while time.time() - t0 < seconds and radius > 1e-14:
        # remaining time guard
        if time.time() - t0 > seconds - 0.05:
            break
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = float(c_full.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 5000:
            # keep top 5000
            kth = np.partition(c_full, -5000)[-5000]
            sel = c_full >= kth
            cnt = int(sel.sum())
            if cnt > 6000:
                # take hardest 5000 by argpartition
                idx = np.argpartition(-c_full, 5000)[:5000]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
        I = iu_r[sel]
        J = iu_c[sel]
        csel = c_full[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        # interleave per row: [I-block, J-block, t]
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = 0  # placeholder
        # build systematically
        # faster: construct with concatenate per block
        base = np.arange(p)[:, None]
        # we already have rows; build cols array in order
        cmat = np.concatenate(
            [ (I[:, None] * d + ar), (J[:, None] * d + ar), np.full((p, 1), n * d) ], axis=1
        ).ravel()
        vmat = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vmat, (rows, cmat)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv),
            ).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj,
                A_ub=A_ub,
                b_ub=-csel,
                A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_descent(x, steps, a0, a1, lr0, beta=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    # precompute log ratio
    la0 = np.log(a0)
    la1 = np.log(a1)
    for t in range(steps):
        frac = t / (steps - 1) if steps > 1 else 0.0
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ x
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        grad /= rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        tang = grad - dot * x
        vel = beta * vel + (1.0 - beta) * tang
        # bias correction
        bc = 1.0 - beta ** (t + 1)
        vhat = vel / bc
        lr = lr0 * (0.2 + 0.8 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        x = x - lr * vhat
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _void_dirs_against(Y, k, rng, n_cand=20000):
    d = Y.shape[1]
    R = rng.standard_normal((n_cand, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ Y.T
    vals = D.max(axis=1)
    idx = np.argpartition(vals, k)[:k]
    # order by val
    o = np.argsort(vals[idx])
    return R[idx[o]]


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t_start = time.time()
    BUDGET = 168.0

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((180, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # also check record
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vb = _max_cos(xb)
        vr = _max_cos(xr)
        if vr < vb:
            xb = xr
            vb = vr
    except Exception:
        pass
    best = _max_cos(xb)
    best_x = xb.copy()

    counts, _, _ = _contact_counts(best_x)

    trial = 0
    # parameter cycles
    sigmas = [0.12, 0.05, 0.25, 0.03, 0.09, 0.18]
    lrs = [0.06, 0.04, 0.08, 0.03, 0.05, 0.07]
    a0s = [25.0, 15.0, 30.0, 20.0, 25.0, 35.0]
    a1s = [280.0, 350.0, 200.0, 300.0, 250.0, 220.0]
    steps_list = [500, 700, 450, 800, 600, 550]

    while time.time() - t_start < BUDGET:
        trial += 1
        remain = BUDGET - (time.time() - t_start)
        if remain < 8:
            break
        ci = (trial - 1) % len(sigmas)
        sigma = sigmas[ci]
        lr0 = lrs[ci]
        a0 = a0s[ci]
        a1 = a1s[ci]
        steps = steps_list[ci]
        # adapt steps if little time
        if remain < 30:
            steps = min(steps, 350)

        k = 2 + (trial % 6)  # 2..7
        # pick victims biased to high contacts
        # weights (contacts+1)^2
        w = (counts.astype(float) + 1.0) ** 2.0
        # occasionally include low-contact too: mix uniform
        if trial % 5 == 0:
            w = w * 0.5 + np.mean(w) * 0.5
        w /= w.sum()
        # choose k distinct
        try:
            victims = rng.choice(180, size=k, replace=False, p=w)
        except Exception:
            victims = rng.choice(180, size=k, replace=False)

        x = best_x.copy()
        if trial % 4 == 0:
            # void reinsertion: replace victims by voids w.r.t. rest
            mask = np.ones(180, dtype=bool)
            mask[victims] = False
            Y = best_x[mask]
            try:
                V = _void_dirs_against(Y, k, rng, n_cand=16000)
                x[victims] = V + 0.01 * rng.standard_normal((k, 9))
                x[victims] /= np.linalg.norm(x[victims], axis=1, keepdims=True)
            except Exception:
                # fallback gaussian kick
                x[victims] += sigma * rng.standard_normal((k, 9))
                x[victims] /= np.linalg.norm(x[victims], axis=1, keepdims=True)
        else:
            x[victims] += sigma * rng.standard_normal((k, 9))
            # renormalize those rows (and all to be safe)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # smooth descent
        try:
            x = _smooth_descent(x, steps, a0, a1, lr0)
        except Exception:
            continue
        v = _max_cos(x)
        # promising threshold: allow LP if within 0.008 of best or better
        if v < best + 0.008:
            # allocate short LP time
            lp_secs = 2.5 if v > best else 4.0
            remain2 = BUDGET - (time.time() - t_start)
            if remain2 < 8:
                # no more LP, just check
                if v < best - 1e-12:
                    best = v
                    best_x = x / np.linalg.norm(x, axis=1, keepdims=True)
                    counts, _, _ = _contact_counts(best_x)
                continue
            lp_secs = min(lp_secs, max(0.5, remain2 - 6.0))
            try:
                y = _lp_polish(x, lp_secs)
            except Exception:
                y = x
            vy = _max_cos(y)
            if vy < best - 1e-12:
                best = vy
                best_x = y / np.linalg.norm(y, axis=1, keepdims=True)
                counts, _, _ = _contact_counts(best_x)
            elif v < best - 1e-12:
                best = v
                best_x = x / np.linalg.norm(x, axis=1, keepdims=True)
                counts, _, _ = _contact_counts(best_x)
        # else discard

    # final squeeze with remaining time
    remain = 179.0 - (time.time() - t_start)
    if remain > 2.0:
        try:
            secs = min(remain - 1.0, 12.0)
            y = _lp_polish(best_x, secs)
            vy = _max_cos(y)
            if vy < best:
                best_x = y
                best = vy
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: remove NaN
    best_x = np.nan_to_num(best_x)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
