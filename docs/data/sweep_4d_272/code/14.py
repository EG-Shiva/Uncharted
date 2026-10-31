import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    x = np.where(np.isfinite(x), x, 0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1.0 - b1) * g
        v *= b2
        v += (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _normalize(x)
    return x


def _find_void(others, rng, n_cand=12000, tau=0.01, iters=120, lr=0.05):
    cand = rng.standard_normal((n_cand, 4))
    cand = _normalize(cand)
    sims = cand @ others.T
    scores = sims.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    # refine single vector with soft-max repulsion
    for _ in range(iters):
        dots = others @ b  # (m,)
        mx = float(dots.max())
        w = np.exp((dots - mx) / tau)
        s = float(w.sum())
        if not np.isfinite(s) or s == 0:
            break
        w /= s
        grad = w @ others  # (4,)
        grad = grad - float(grad @ b) * b
        b = b - lr * grad
        nb = float(np.linalg.norm(b))
        if not np.isfinite(nb) or nb < 1e-12:
            b = cand[bi].copy()
            break
        b = b / nb
    return b


def _lp_polish(x, seconds, deadline=1e30):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        y = _normalize(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _normalize(x0)
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    sigmas = [0.025, 0.06, 0.012, 0.09, 0.008, 0.04, 0.15, 0.02]
    taus = [(0.02, 0.002), (0.008, 0.0008), (0.03, 0.003), (0.015, 0.0012)]
    lrs = [(0.010, 0.001), (0.006, 0.0006), (0.015, 0.0015)]

    trial = 0
    # basin hopping phase, reserve ~40s for final polish
    while time.time() < t0 + 128.0 and time.time() < deadline - 42.0:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[trial % len(taus)]
        lr0, lr1 = lrs[trial % len(lrs)]
        # base: mostly from best, sometimes from cur to wander
        base = best if (trial % 5 != 0) else cur
        x = base + sigma * rng.standard_normal((272, 4))
        x = _normalize(x)
        nkill = trial % 4  # 0..3
        if trial % 11 == 0:
            nkill = 5
        for _ in range(nkill):
            if time.time() > deadline - 45.0:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            order = np.argsort(-pw)
            top = order[:8]
            wi = int(top[rng.integers(0, len(top))])
            others = np.delete(x, wi, axis=0)
            b = _find_void(others, rng, n_cand=10000, tau=0.008, iters=80, lr=0.05)
            x[wi] = b
            x = _normalize(x)
        # adaptive steps: shorter when little time
        remain = min(deadline - 42.0 - time.time(), t0 + 128.0 - time.time())
        if remain <= 2.0:
            break
        steps = 380
        if remain < 15:
            steps = 200
        elif sigma > 0.08:
            steps = 450
        x = _anneal(x, steps, tau0, tau1, lr0, lr1)
        vv = _max_cos(x)
        if vv < best_val:
            best = x.copy()
            best_val = vv
            cur = x.copy()
            cur_val = vv
        else:
            # simulated-annealing style acceptance to change basin
            Temp = 0.0015
            if vv < cur_val or rng.random() < np.exp(-(vv - cur_val) / Temp):
                cur = x.copy()
                cur_val = vv
        # early stop if clearly record-beating (keep polishing)
        if best_val < 0.8975:
            break

    # final snap: polish best
    time_left = deadline - time.time() - 1.0
    if time_left > 5.0:
        polished = _lp_polish(best, min(time_left, 38.0), deadline - 1.0)
        pv = _max_cos(polished)
        if pv < best_val:
            best = polished
            best_val = pv
        # also try polishing cur if different and time remains
        time_left2 = deadline - time.time() - 1.0
        if time_left2 > 8.0 and cur_val < best_val + 0.002:
            polished2 = _lp_polish(cur, min(10.0, time_left2), deadline - 1.0)
            pv2 = _max_cos(polished2)
            if pv2 < best_val:
                best = polished2
                best_val = pv2
    best = _normalize(best)
    # safety: return best seen; never return worse than input
    if _max_cos(best) > _max_cos(x0) + 1e-12:
        best = x0
    return best
