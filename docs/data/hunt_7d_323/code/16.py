import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _anneal(x, rng, steps, T0, T1, lr0, lr1, mu, noise0, t_end):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        if s % 400 == 0 and time.time() > t_end - 8.0:
            break
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        invT = 1.0 / max(T, 1e-12)
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # weights exp((G-m)/T)
        W = np.exp((G - m) * invT)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        Ft /= rs
        v = mu * v - lr * Ft
        # clip per-point step
        max_step = 0.02 * (1.0 - frac) + 0.002
        vn = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, max_step / vn)
        v = v * scale
        x = x + v
        if noise0 > 0:
            ns = noise0 * (T / T0)
            if ns > 1e-12:
                nz = rng.standard_normal(size=(n, d)) * ns
                nzd = np.sum(nz * x, axis=1, keepdims=True)
                nz -= nzd * x
                x = x + nz
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x

def _hill_worst(x, rng, iters, n_cand, sigma, t_end):
    n, d = x.shape
    for it in range(iters):
        if time.time() > t_end - 3.0:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        w = G.max(axis=1)
        k = int(np.argmax(w))
        cur = float(w[k])
        cand = x[k][None, :] + sigma * rng.standard_normal(size=(n_cand, d))
        cn = np.linalg.norm(cand, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        cand = cand / cn
        dots = cand @ x.T
        dots[:, k] = -2.0
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        if worst[bi] < cur - 1e-13:
            x[k] = cand[bi]
        else:
            # try smaller sigma occasionally
            if it % 20 == 0:
                sigma *= 0.9
                if sigma < 5e-4:
                    sigma = 5e-4
    return x

def _uniform_try(x, rng, n_try, n_cand, t_end):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    glob = float(G.max())
    w = G.max(axis=1)
    order = np.argsort(-w)
    for ii in range(min(n_try, n)):
        if time.time() > t_end - 3.0:
            break
        k = int(order[ii])
        cur_w = float(w[k])
        # only try bad points
        if cur_w < glob - 5e-4 and ii > 12:
            continue
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x, k, axis=0)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        # global after replacement
        # max among others themselves
        Go = others @ others.T
        np.fill_diagonal(Go, -2.0)
        rest = float(Go.max())
        newglob = max(float(worst[bi]), rest)
        if newglob < glob - 1e-12:
            x[k] = cand[bi]
            glob = newglob
            # update w quickly
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            order = np.argsort(-w)
    return x

def solve():
    t0 = time.time()
    t_end = t0 + 165.0
    rng = np.random.default_rng(12345)
    # load start
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best = a
        best_val = _max_cos(a)
    overall = best.copy()
    overall_val = best_val

    # quick hill on start (cheap polish of rattlers neighbourhood)
    try:
        tmp = overall.copy()
        tmp = _hill_worst(tmp, rng, iters=120, n_cand=300, sigma=0.003, t_end=t_end)
        v = _max_cos(tmp)
        if v < overall_val:
            overall = tmp
            overall_val = v
    except Exception:
        pass

    # basin-hop trials
    # configs: (n_vict, sigma_kick, steps, T0, T1, lr0, lr1, mu, noise0)
    configs = [
        (10, 0.10, 2600, 0.015, 0.0007, 0.012, 0.0025, 0.80, 0.0015),
        (14, 0.13, 2600, 0.020, 0.0007, 0.014, 0.0025, 0.82, 0.0020),
        (8,  0.07, 2200, 0.010, 0.0005, 0.010, 0.0020, 0.78, 0.0010),
        (20, 0.15, 2800, 0.022, 0.0008, 0.015, 0.0030, 0.83, 0.0022),
        (12, 0.09, 2400, 0.012, 0.0006, 0.011, 0.0022, 0.80, 0.0012),
        (6,  0.06, 2000, 0.008, 0.0005, 0.009, 0.0020, 0.75, 0.0008),
        (16, 0.11, 2600, 0.018, 0.0006, 0.013, 0.0025, 0.81, 0.0018),
        (10, 0.12, 2400, 0.016, 0.0007, 0.012, 0.0025, 0.80, 0.0015),
        (24, 0.16, 2800, 0.025, 0.0009, 0.016, 0.0030, 0.84, 0.0025),
        (10, 0.08, 2200, 0.011, 0.0005, 0.010, 0.0020, 0.79, 0.0010),
    ]
    for ci, cfg in enumerate(configs):
        if time.time() > t_end - 12.0:
            break
        n_vict, sk, steps, T0, T1, lr0, lr1, mu, noise0 = cfg
        x = overall.copy()
        n, d = x.shape
        # victim selection: worst + random
        try:
            w = _per_point_worst(x)
            order = np.argsort(-w)
            vict = set()
            # worst 60%
            nw = max(1, int(n_vict * 0.6))
            for i in range(nw):
                vict.add(int(order[i]))
            # random rest
            while len(vict) < n_vict:
                vict.add(int(rng.integers(0, n)))
            vict = np.array(sorted(vict), dtype=int)
            # kick victims
            x[vict] += sk * rng.standard_normal(size=(len(vict), d))
            # tiny global jitter
            x += 0.002 * rng.standard_normal(size=(n, d))
            xn = np.linalg.norm(x, axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x = x / xn
            # anneal
            x = _anneal(x, rng, steps, T0, T1, lr0, lr1, mu, noise0, t_end)
            # greedy uniform reinsert for worst few
            x = _uniform_try(x, rng, n_try=10, n_cand=2500, t_end=t_end)
            # hill polish
            x = _hill_worst(x, rng, iters=250, n_cand=250, sigma=0.004, t_end=t_end)
            v = _max_cos(x)
            if v < overall_val - 1e-12:
                overall = x.copy()
                overall_val = v
        except Exception:
            continue

    overall = overall / np.linalg.norm(overall, axis=1, keepdims=True)
    return overall
