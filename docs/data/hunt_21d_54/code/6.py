import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_anneal(x_init, steps, b0, b1, lr0, lr1, mu):
    x = x_init.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    lb = np.log(b0)
    le = np.log(b1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(steps):
        frac = t / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb + (le - lb) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        # weights
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S == 0.0:
            break
        Grad = (W @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        vel = mu * vel - lr * Gt
        # clip step
        vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True) + 1e-20)
        sc = np.minimum(1.0, 0.02 / vn)
        vel *= sc
        x = x + vel
        x /= np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x


def _single_refine(x, rng, samples=4000):
    """Try to reposition worst points. Returns (x_new, new_val)."""
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    gmax = rowmax.max()
    order = np.argsort(-rowmax)
    # try worst few
    for idx in order[:6]:
        others = np.delete(x, idx, axis=0)
        cur = float(rowmax[idx])
        cand = rng.standard_normal(size=(samples, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # scores
        s = cand @ others.T
        smax = s.max(axis=1)
        bi = int(np.argmin(smax))
        if smax[bi] >= cur - 1e-9:
            continue
        # local hill climb for this single point
        v = cand[bi:bi + 1, :].copy()
        for tt in range(350):
            beta = 2500.0
            dots = (v @ others.T).ravel()
            m0 = dots.max()
            w = np.exp(beta * (dots - m0))
            ws = w.sum()
            if ws == 0:
                break
            w /= ws
            grad = w @ others  # (d,)
            vv = v[0]
            dg = float(np.dot(grad, vv))
            gt = grad - dg * vv
            # decaying step
            step = 0.02 * (1.0 - tt / 350.0) + 0.002
            vv = vv - step * gt
            nrm = np.linalg.norm(vv)
            if nrm == 0:
                break
            v[0] = vv / nrm
        x_new = x.copy()
        x_new[idx] = v[0]
        Gn = x_new @ x_new.T
        np.fill_diagonal(Gn, -2.0)
        nm = float(Gn.max())
        if nm < gmax - 1e-12:
            return x_new, nm
    return x, gmax


def _lp_polish(x, seconds, t0=None):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    if t0 is None:
        t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
    t_start = time.time()
    rng = np.random.default_rng(0)
    n, d = 54, 21
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # exploration budget: leave ~25s for final LP
    EXPLORE_END = 148.0
    sigmas = [0.03, 0.05, 0.08, 0.12, 0.02, 0.01, 0.20, 0.04]
    trial = 0
    while time.time() - t_start < EXPLORE_END:
        trial += 1
        sigma = sigmas[trial % len(sigmas)] * (0.7 + 0.6 * rng.random())
        r = rng.random()
        if r < 0.15:
            k = 1
        elif r < 0.22:
            k = 2
        else:
            k = 0
        x_try = best.copy()
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.standard_normal(size=(k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
        else:
            noise = rng.standard_normal(size=(n, d))
            x_try = x_try + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        b0 = float(rng.uniform(80, 300))
        b1 = float(rng.uniform(1500, 8000))
        lr0 = float(rng.uniform(0.008, 0.025))
        lr1 = float(rng.uniform(0.001, 0.004))
        mu = float(rng.uniform(0.80, 0.92))
        steps = int(rng.integers(1200, 2600))
        try:
            x_opt = _smooth_anneal(x_try, steps, b0, b1, lr0, lr1, mu)
        except Exception:
            continue
        try:
            v = _max_cos(x_opt)
        except Exception:
            continue
        # single-point rescue if promising
        if v < best_val + 0.006:
            try:
                x_opt2, v2 = _single_refine(x_opt, rng)
                x_opt, v = x_opt2, v2
            except Exception:
                pass
        if v < best_val:
            best = x_opt
            best_val = v
            # extra squeeze with high-beta short anneal
            try:
                best = _smooth_anneal(best, 600, 500, 10000, 0.004, 0.0008, 0.85)
                best_val = _max_cos(best)
            except Exception:
                pass
        # timeout guard
        if time.time() - t_start >= EXPLORE_END:
            break
    # final LP polish with remaining time
    remain = 178.0 - (time.time() - t_start)
    if remain > 4:
        try:
            best = _lp_polish(best, seconds=remain - 1.0, t0=time.time())
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
