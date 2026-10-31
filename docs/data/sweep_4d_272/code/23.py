import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    x = x / n
    x = np.where(np.isfinite(x), x, 0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _adam_lse(x, steps, tau0, tau1, lr0, lr1):
    n = x.shape[0]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        G -= gm
        G /= tau
        # clip to avoid overflow, exp(-50)~2e-22 negligible
        np.clip(G, -50.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _adam_riesz(x, steps, s0, s1, lr0, lr1):
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        s = s0 + (s1 - s0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        r2min = max(2.0 - 2.0 * gm, 1e-12)
        R2 = 2.0 - 2.0 * G
        np.clip(R2, 1e-12, 4.0, out=R2)
        # W = (R2/r2min)^(-(s+2)/2), max =1
        lratio = np.log(R2) - np.log(r2min)
        W = np.exp(-0.5 * (s + 2.0) * lratio)
        # sparsify far pairs
        W[W < 1e-12] = 0.0
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _find_void(x_fixed, rng, n_samples=5000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    b = cand[idx].copy()
    bs = float(best[idx])
    step = 0.05
    for _ in range(150):
        prop = b + step * rng.standard_normal((24, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ x_fixed.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.05
        else:
            step *= 0.96
        if step < 1e-5:
            break
    return b

def _cap_rotate(x, rng, k, angle):
    n = x.shape[0]
    u = rng.standard_normal((4,))
    u /= np.linalg.norm(u)
    d = x @ u
    # k nearest to u
    idx = np.argpartition(-d, k)[:k]
    # random skew
    M = rng.standard_normal((4, 4))
    K = (M - M.T) * 0.5
    s = angle
    # Cayley: R=(I+sK/2)(I-sK/2)^{-1}
    A = np.eye(4) + 0.5 * s * K
    B = np.eye(4) - 0.5 * s * K
    try:
        R = np.linalg.solve(B.T, A.T).T
    except Exception:
        R = np.eye(4) + s * K
    y = x.copy()
    y[idx] = y[idx] @ R.T
    return _norm(y)

def _pair_split(x, rng, strength):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # top few pairs
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    k = min(6, c.size)
    top = np.argpartition(-c, k)[:k]
    y = x.copy()
    for t in top:
        i = int(iu[0][t]); j = int(iu[1][t])
        # push apart along tangent
        vi = y[i] - (float(y[i] @ y[j])) * y[j]
        vj = y[j] - (float(y[j] @ y[i])) * y[i]
        ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
        if ni > 1e-12:
            y[i] = y[i] + strength * (vi / ni) * (0.5 + rng.random())
        if nj > 1e-12:
            y[j] = y[j] + strength * (vj / nj) * (0.5 + rng.random())
    return _norm(y)

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(20261031)
    t0 = time.time()
    deadline = t0 + 172.0
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
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    cur = x0.copy()
    cur_val = best_val

    taus = [(0.02, 0.001), (0.008, 0.0003), (0.04, 0.002), (0.012, 0.0002), (0.06, 0.004)]
    lrs_lse = [(0.010, 0.0008), (0.005, 0.0004), (0.016, 0.0012), (0.007, 0.0005)]
    s_vals = [(20, 80), (60, 200), (10, 40), (100, 300), (40, 120)]
    lrs_r = [(0.010, 0.0008), (0.006, 0.0004), (0.014, 0.001)]
    sigmas = [0.015, 0.04, 0.008, 0.08, 0.025, 0.12, 0.005]

    trial = 0
    explore_until = t0 + 118.0
    while time.time() < explore_until and time.time() < deadline - 48:
        trial += 1
        elapsed = time.time() - t0
        # Metropolis temperature decay
        Temp = 0.0015 * (0.01 ** (elapsed / max(explore_until - t0, 1.0))) + 1e-6
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 7
        y = cur.copy()
        if mode == 0:
            # global kick
            y = y + sigma * rng.standard_normal((272, 4))
            y = _norm(y)
        elif mode == 1:
            # cap rigid rotation + jitter
            k = int(rng.integers(8, 55))
            ang = float(rng.uniform(0.05, 0.5))
            y = _cap_rotate(y, rng, k, ang)
            y = y + (sigma * 0.4) * rng.standard_normal((272, 4))
            y = _norm(y)
        elif mode == 2:
            # worst-point void reinsertion (1-3)
            nkill = 1 + (trial % 3)
            # jitter first
            y = y + (sigma * 0.5) * rng.standard_normal((272, 4))
            y = _norm(y)
            for _ in range(nkill):
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                order = np.argsort(-mx)
                K = min(10, 272)
                victim = int(order[rng.integers(0, K)])
                mask = np.ones(272, dtype=bool); mask[victim] = False
                y[victim] = _find_void(y[mask], rng, n_samples=4000)
        elif mode == 3:
            # pair split + small kick
            y = _pair_split(y, rng, sigma * 1.2)
            y = y + (sigma * 0.3) * rng.standard_normal((272, 4))
            y = _norm(y)
        elif mode == 4:
            # subset strong reshuffle
            sz = int(rng.integers(15, 55))
            idx = rng.choice(272, size=sz, replace=False)
            y[idx] += sigma * 1.8 * rng.standard_normal((sz, 4))
            y += (sigma * 0.15) * rng.standard_normal((272, 4))
            y = _norm(y)
        elif mode == 5:
            # rattler move + cap
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            victim = int(mx.argmin())
            mask = np.ones(272, dtype=bool); mask[victim] = False
            # also jitter others slightly
            y = y + (sigma * 0.3) * rng.standard_normal((272, 4))
            y = _norm(y)
            y[victim] = _find_void(y[mask], rng, n_samples=4000)
        else:
            # restart partly from best with big kick
            y = x_best.copy()
            y = y + sigma * 1.2 * rng.standard_normal((272, 4))
            y = _norm(y)
            # one void
            if rng.random() < 0.5:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                victim = int(np.argmax(mx))
                mask = np.ones(272, dtype=bool); mask[victim] = False
                y[victim] = _find_void(y[mask], rng, n_samples=4000)

        # choose annealer
        if rng.random() < 0.55:
            tau0, tau1 = taus[rng.integers(0, len(taus))]
            lr0, lr1 = lrs_lse[rng.integers(0, len(lrs_lse))]
            steps = 420
            if time.time() > t0 + 85:
                steps = 300
            y = _adam_lse(y, steps, tau0, tau1, lr0, lr1)
        else:
            s0, s1 = s_vals[rng.integers(0, len(s_vals))]
            lr0, lr1 = lrs_r[rng.integers(0, len(lrs_r))]
            steps = 420
            if time.time() > t0 + 85:
                steps = 300
            y = _adam_riesz(y, steps, float(s0), float(s1), lr0, lr1)

        vv = _max_cos(y)
        if vv < best_val:
            best_val = vv
            x_best = y.copy()
            cur = y.copy()
            cur_val = vv
        elif vv < cur_val or rng.random() < np.exp((cur_val - vv) / max(Temp, 1e-9)):
            cur = y.copy()
            cur_val = vv
        # occasional reset of walker to best if wandering too high
        if cur_val > best_val + 0.015 and rng.random() < 0.3:
            cur = x_best.copy()
            cur_val = best_val

    # intensify around best
    for _ in range(3):
        if time.time() > deadline - 42:
            break
        y = _adam_lse(x_best.copy(), 380, 0.006, 0.00025, 0.004, 0.00035)
        vv = _max_cos(y)
        if vv < best_val:
            best_val = vv
            x_best = y.copy()
    for _ in range(2):
        if time.time() > deadline - 40:
            break
        y = _adam_riesz(x_best.copy(), 300, 60.0, 250.0, 0.004, 0.0004)
        vv = _max_cos(y)
        if vv < best_val:
            best_val = vv
            x_best = y.copy()

    remain = deadline - time.time() - 2.0
    if remain > 5:
        xp = _lp_polish(x_best, min(remain, 36.0), deadline - 1.0)
        vp = _max_cos(xp)
        if vp <= best_val:
            x_best = xp
            best_val = vp
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best = _norm(x_best)
    return x_best
