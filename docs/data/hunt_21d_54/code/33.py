import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_target(Y0, t, steps, lr, lam=0.005, tau=0.006, T0=None, deadline=None, track=None):
    # projected Adam for P = sum_{i<j} relu(G-t)^2 + lam * tau*logsumexp(G/tau)
    n, d = Y0.shape
    Y = Y0.copy()
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu = np.triu_indices(n, 1)
    bestY = Y.copy()
    # evaluate initial
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    for s in range(1, steps + 1):
        G = Y @ Y.T
        # penalty weights (zero diag)
        # use G with diag 0 for exp; but for relu diag irrelevant
        R = G - t
        np.fill_diagonal(R, 0.0)
        np.maximum(R, 0.0, out=R)
        # Wpen = 2*relu
        W = 2.0 * R
        np.fill_diagonal(W, 0.0)
        # lse weights
        # for stability subtract row? use global max of G off-diag
        Gd = G.copy()
        np.fill_diagonal(Gd, -2.0)
        gmax = float(Gd.max())
        E = np.exp(np.clip((Gd - gmax) / tau, -20, 0.0))
        np.fill_diagonal(E, 0.0)
        S = float(E[iu].sum()) + 1e-30
        W += (lam / S) * E
        grad = W @ Y
        # tangent projection
        dot = np.einsum('ij,ij->i', grad, Y)
        grad -= dot[:, None] * Y
        # Adam
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh = m / (1 - b1 ** s)
        vh = v / (1 - b2 ** s)
        # decay lr slightly
        lrs = lr * (1.0 - 0.4 * s / steps)
        step = lrs * mh / (np.sqrt(vh) + eps)
        # keep tangent (remove radial part of step)
        sdot = np.einsum('ij,ij->i', step, Y)
        step -= sdot[:, None] * Y
        # clip extreme step
        Y = Y - step
        Y = _proj(Y)
        if (s % 50 == 0) or (s == steps):
            Gc = Y @ Y.T
            np.fill_diagonal(Gc, -2.0)
            cm = float(Gc.max())
            if cm < best_m:
                best_m = cm
                bestY = Y.copy()
            if track is not None:
                if cm < track[0]:
                    track[0] = cm
                    track[1][:] = Y
            # early exit if comfortably feasible for this t
            if cm < t - 0.001:
                break
        if deadline is not None and (s % 200 == 0):
            if time.time() > deadline:
                break
    return bestY, best_m

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2500
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # hard stop if time exceeded (linprog may overshoot)
        if time.time() - start >= seconds:
            break
    return x

def solve():
    T0 = time.time()
    LIMIT = 150.0  # hard internal deadline well below 180
    deadline_homotopy = T0 + 105.0
    deadline_total = T0 + LIMIT
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        try:
            b0 = np.load("record.npy")
        except Exception:
            b0 = rng.normal(size=(54, 21))
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape != (54, 21):
            r0 = b0
    except Exception:
        r0 = b0
    bv = _max_cos(b0)
    rv = _max_cos(r0)
    if rv < bv:
        best, best_val = r0.copy(), rv
    else:
        best, best_val = b0.copy(), bv
    thresh = 0.0666634
    # track = [best_val_global, best_X_global]
    track = [best_val, best.copy()]
    n, d = 54, 21
    targets = [0.066, 0.064, 0.062, 0.060]
    # build starts: perturbed copies + random
    starts = []
    # small tangent perturbations of best
    for scale in [0.03, 0.10, 0.25]:
        noise = rng.normal(size=(n, d))
        # tangentize w.r.t best
        noise -= (noise * best).sum(axis=1, keepdims=True) * best
        Y0 = _proj(best + scale * noise)
        starts.append(Y0)
    # random starts
    for _ in range(2):
        starts.append(_proj(rng.normal(size=(n, d))))
    # also include best itself
    starts.insert(0, best.copy())
    # Adam hyperparams per start
    lrs = [0.006, 0.008, 0.008, 0.010, 0.010, 0.012]
    steps_per_stage = [900, 900, 900, 1100]
    for si, Y0 in enumerate(starts):
        if time.time() > deadline_homotopy:
            break
        Y = Y0.copy()
        lr = lrs[si % len(lrs)]
        for ti, t in enumerate(targets):
            if time.time() > deadline_homotopy:
                break
            steps = steps_per_stage[ti]
            # shrink steps if little time
            tl = deadline_homotopy - time.time()
            if tl < 8:
                break
            # scale steps to remaining time roughly (avoid overrun)
            # each ~900 steps ~0.3-0.5s, so keep as is
            Y, mloc = _adam_target(Y, t, steps, lr, lam=0.005, tau=0.006,
                                   T0=T0, deadline=deadline_homotopy, track=track)
            # anneal only if penalty near zero: max < t + 2e-4
            # check current max of Y
            cm = _max_cos(Y)
            if cm > t + 0.0005:
                # not feasible: try next start rather than pushing lower t
                # but still allow one more stage with same Y? break to next start
                break
            # feasible: continue to lower t (keep Y)
        # loop next start
    best_val_found = float(track[0])
    best_found = _proj(track[1])
    # decide polish
    # keep overall best (including original)
    if best_val_found < best_val:
        best, best_val = best_found, best_val_found
    beaten = best_val < thresh
    tl = deadline_total - time.time()
    if beaten:
        # spend remaining budget on LP polish
        budget = max(1.0, tl - 1.0)
        # cap to avoid linprog overrun
        budget = min(budget, 40.0)
        best = _lp_polish(best, budget)
        best_val = _max_cos(best)
    else:
        # no breakthrough: light polish only if time permits, to not lose record
        # short polish of the best-found (and of original best) to return best feasible
        if tl > 8:
            # polish current best briefly
            cand = _lp_polish(best_found if best_val_found < best_val + 1e-12 else best, min(6.0, tl - 2.0))
            cv = _max_cos(cand)
            if cv < best_val:
                best, best_val = cand, cv
    best = _proj(best)
    # tiny shrink guard not needed; ensure valid
    best = np.nan_to_num(best)
    best = _proj(best)
    return best
