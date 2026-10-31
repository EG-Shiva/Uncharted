import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _descend(x, T, steps, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        if mx <= T:
            break
        D = G - T
        np.maximum(D, 0.0, out=D)
        grad = D @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_point(x, victim, rng, n_cand=8000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ x.T
    S[:, victim] = -2.0
    scores = S.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(120):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        s = prop @ x.T
        s[:, victim] = -2.0
        sc = s.max(axis=1)
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

def _torus_init(rng):
    L = int(rng.integers(3, 7))
    base = 272 // L
    rem = 272 % L
    ns = [base + (1 if i < rem else 0) for i in range(L)]
    ts = np.sort(rng.uniform(0.18, 1.39, size=L))
    X = []
    for l in range(L):
        nl = ns[l]
        t = float(ts[l])
        ct, st = float(np.cos(t)), float(np.sin(t))
        au = float(rng.uniform(0.2, 0.8))
        av = float(rng.uniform(0.2, 0.8))
        o1 = float(rng.uniform(0, 1))
        o2 = float(rng.uniform(0, 1))
        idx = np.arange(nl, dtype=float)
        u = (idx * au + o1 * nl) % 1.0
        vv = (idx * av + o2 * nl) % 1.0
        phi = 2.0 * np.pi * u
        psi = 2.0 * np.pi * vv
        pts = np.stack([ct * np.cos(phi), ct * np.sin(phi),
                        st * np.cos(psi), st * np.sin(psi)], axis=1)
        X.append(pts)
    x = np.concatenate(X, axis=0)
    if x.shape[0] != 272:
        # adjust by truncate/pad
        if x.shape[0] > 272:
            x = x[:272]
        else:
            extra = rng.standard_normal((272 - x.shape[0], 4))
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            x = np.concatenate([x, extra], axis=0)
    x = x + 0.015 * rng.standard_normal(x.shape)
    return _norm(x)

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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _norm(tmp)
    best = x0.copy()
    best_val = _max_cos(best)
    FINAL = 22.0
    search_end = deadline - FINAL
    it = 0
    sigmas = [0.25, 0.12, 0.35, 0.18, 0.08, 0.45]
    deltas = [0.0012, 0.0006, 0.0020, 0.0003, 0.0009, 0.0015]
    while time.time() < search_end:
        it += 1
        if time.time() > search_end:
            break
        mode = it % 5
        if mode == 0 or mode == 1:
            sigma = sigmas[(it // 5) % len(sigmas)]
            delta = deltas[(it // 5) % len(deltas)]
            y = best + sigma * rng.standard_normal((272, 4))
            y = _norm(y)
            T = best_val - delta
            steps = 1300 if time.time() < search_end - 10 else 500
            lr0, lr1 = 0.03, 0.002
            if delta > 0.001:
                lr0 = 0.05
            y = _descend(y, T, steps, lr0, lr1)
            vv = _max_cos(y)
            if vv < best_val:
                best, best_val = y, vv
        elif mode == 2:
            k = int(rng.integers(10, 30))
            y = best.copy()
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            order = np.argsort(-pw)
            pool = order[:40]
            victims = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
            # add rattlers / low contact if any
            contacts = (G > best_val - 1e-9).sum(axis=1)
            ratt = np.where(contacts <= 1)[0].tolist()
            for r in ratt:
                if len(victims) < 32 and r not in victims:
                    victims.append(int(r))
            for vic in victims:
                if time.time() > search_end:
                    break
                b = _void_point(y, int(vic), rng, n_cand=6000)
                y[int(vic)] = b
            T = best_val - float(rng.uniform(0.0003, 0.0012))
            y = _descend(y, T, 1100, 0.03, 0.002)
            vv = _max_cos(y)
            if vv < best_val:
                best, best_val = y, vv
        elif mode == 3:
            y = _torus_init(rng)
            T = best_val - float(rng.uniform(0.0003, 0.0010))
            # staged compression from loose to target
            for Ts, st, a0, a1 in [(0.95, 300, 0.05, 0.01), (0.92, 300, 0.04, 0.005), (T, 1200, 0.03, 0.002)]:
                if time.time() > search_end:
                    break
                y = _descend(y, Ts, st, a0, a1)
            vv = _max_cos(y)
            if vv < best_val:
                best, best_val = y, vv
        else:
            y = rng.standard_normal((272, 4))
            y = _norm(y)
            T = best_val - float(rng.uniform(0.0002, 0.0008))
            for Ts, st in [(0.95, 250), (0.92, 250), (0.905, 400), (T, 900)]:
                if time.time() > search_end:
                    break
                y = _descend(y, Ts, st, 0.04, 0.003)
            vv = _max_cos(y)
            if vv < best_val:
                best, best_val = y, vv
        # opportunistic tight polish if close
        if best_val < 0.89816031515 and (it % 10 == 0):
            # quick light polish to lock in
            best = _lp_polish(best, 2.0, search_end)
            best_val = _max_cos(best)
    best = _lp_polish(best, FINAL - 1.0, deadline)
    best_val = _max_cos(best)
    # safety: if search somehow worsened, return original polished
    try:
        orig = _norm(np.asarray(np.load("best.npy"), dtype=float))
        ov = _max_cos(orig)
        if ov < best_val:
            # polish orig briefly if time remains
            if time.time() < deadline - 2:
                orig = _lp_polish(orig, min(5.0, deadline - time.time() - 0.5), deadline)
                ov = _max_cos(orig)
            if ov < best_val:
                return orig
    except Exception:
        pass
    return best
