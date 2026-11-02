import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.3:
            break
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
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
    return x

def _adam_run(y0, t, steps, lr, w, tau, t_deadline):
    n, d = y0.shape
    y = y0.copy()
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_y = y.copy()
    best_v = _max_cos(y)
    eye = np.eye(n)
    for s in range(1, steps + 1):
        if s % 200 == 0 and time.time() > t_deadline:
            break
        G = y @ y.T
        # current max ignoring diagonal
        mcur = float((G - 2.0 * eye).max())
        D = G - t
        np.maximum(D, 0.0, out=D)
        np.fill_diagonal(D, 0.0)
        # tie-breaker weights
        E = (G - mcur) / tau
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5) + 1e-12
        grad_pen = 2.0 * (D @ y)
        grad_tie = (W @ y) / S
        grad = grad_pen + w * grad_tie
        # tangent projection
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        m1h = m1 / (1 - b1 ** s)
        m2h = m2 / (1 - b2 ** s)
        upd = m1h / (np.sqrt(m2h) + eps)
        # project update to tangent
        dot2 = np.einsum('ij,ij->i', upd, y)
        upd -= dot2[:, None] * y
        # schedule
        lr_s = lr * (1.0 - 0.5 * s / steps)
        y = y - lr_s * upd
        y = _proj(y)
        if s % 25 == 0 or s == steps:
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_y = y.copy()
    return best_y, best_v

def solve():
    T0 = time.time()
    LIMIT = 168.0
    def tleft():
        return LIMIT - (time.time() - T0)
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    try:
        r0 = np.load("record.npy")
    except Exception:
        r0 = b0
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    r0 = _proj(np.nan_to_num(np.asarray(r0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    if r0.shape != (54, 21):
        r0 = b0.copy()
    bv = _max_cos(b0)
    rv = _max_cos(r0)
    if rv < bv:
        best, best_val = r0.copy(), rv
    else:
        best, best_val = b0.copy(), bv
    RECORD = 0.066663628471
    THRESH = 0.0666634
    # homotopy targets
    targets = [0.066, 0.064, 0.062, 0.060]
    cur = best.copy()
    cur_val = best_val
    # time allocation: leave >=30s for final LP
    adam_deadline = T0 + max(10.0, LIMIT - 38.0)
    w0 = 3e-4
    tau0 = 0.007
    t_idx = 0
    # restart sigmas: 0 means carryover, 'rand' means random start
    while t_idx < len(targets) and time.time() < adam_deadline:
        t = float(targets[t_idx])
        # configs for this target
        configs = [
            (0.0, 2500, 0.015),
            (0.02, 2000, 0.015),
            (0.08, 2000, 0.020),
            ('rand', 2500, 0.015),
            (0.20, 1500, 0.025),
            ('rand', 1500, 0.030),
        ]
        improved_feas = False
        for ci, (sig, steps, lr) in enumerate(configs):
            if time.time() > adam_deadline:
                break
            if tleft() < 34:
                break
            if sig == 'rand':
                y0 = _proj(rng.normal(size=(54, 21)))
            elif sig == 0.0:
                y0 = cur.copy()
            else:
                y0 = cur + float(sig) * rng.normal(size=(54, 21))
                # also try perturbing global best occasionally
                if ci % 2 == 1:
                    y0 = 0.5 * y0 + 0.5 * (best + float(sig) * 0.5 * rng.normal(size=(54, 21)))
                y0 = _proj(y0)
            # small adaptation of w/tau per restart for diversity
            w = w0 * (1.0 + 0.5 * ((ci % 3) - 1))
            tau = tau0 * (1.0 + 0.3 * ((ci % 2) * 2 - 1) * 0.5)
            steps = int(min(steps, max(800, steps * min(1.0, tleft() / 40.0))))
            yo, vo = _adam_run(y0, t, steps, lr, w, tau, adam_deadline)
            if vo < best_val - 1e-12:
                best, best_val = yo.copy(), vo
            # track best for this target
            if vo < cur_val:
                cur_val = vo
                cur = yo.copy()
            # early exit to LP if record beaten
            if best_val < THRESH - 1e-12:
                # spend remaining on LP polish immediately
                break
            # if feasible (penalty near zero), mark and go to next target
            if vo <= t + 1e-9:
                improved_feas = True
                break
        if best_val < THRESH - 1e-12:
            break
        # annealing rule: only go lower if near-feasible
        if cur_val <= t + 5e-5:
            # feasible enough -> advance, carry cur forward
            t_idx += 1
        else:
            # not feasible for this t, do not try harder lower targets
            # try one more round with larger steps? break to save time for polish
            break
    # final LP polish: if record beaten spend remaining, else short squeeze
    rem = tleft()
    if best_val < THRESH:
        budget = max(1.0, rem - 1.0)
        budget = min(budget, 35.0)
        best = _lp_polish(best, budget)
        best_val = _max_cos(best)
    else:
        # short squeeze attempt (could still beat)
        if rem > 8:
            budget = min(12.0, rem - 2.0)
            cand = _lp_polish(best, budget)
            v = _max_cos(cand)
            if v < best_val:
                best, best_val = cand, v
    best = _proj(best)
    return best
