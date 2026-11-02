import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_phase(x, N, tau0, tau1, lr0, lr1, mu, t0, time_limit, best_val, best_x):
    n, d = x.shape
    v = np.zeros_like(x)
    ltau = np.log(tau1 / tau0)
    llr = np.log(lr1 / lr0)
    for it in range(N):
        if time.time() - t0 > time_limit:
            break
        frac = it / max(N - 1, 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_x = x.copy()
        # weights focused near max
        # G-m <= 0
        P = np.exp((G - m) / tau)
        # kill diagonal residual
        # (already tiny) but zero for cleanliness
        # diag of G is -2 -> exp ~0, ignore
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + (1.0 - mu) * Gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # safety revert if exploded
        if m > best_val + 0.02:
            x = best_x.copy()
            v *= 0.5
    return x, best_val, best_x

def greedy_shave(x, best_val, best_x, t0, time_limit, steps=9000):
    n, d = x.shape
    eps = 1e-3
    # ensure x is current best region
    for it in range(steps):
        if time.time() - t0 > time_limit:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_x = x.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(x[i] @ x[j])
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            # random tangent kick
            ti = np.random.randn(d)
            ti -= float(ti @ x[i]) * x[i]
            ni = float(np.linalg.norm(ti)) + 1e-12
        ti /= ni
        tj = x[i] - cij * x[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = np.random.randn(d)
            tj -= float(tj @ x[j]) * x[j]
            nj = float(np.linalg.norm(tj)) + 1e-12
        tj /= nj
        improved = False
        # try a few scales
        for scale in (eps, eps * 0.5, eps * 0.25):
            xp = x.copy()
            vi = x[i] - scale * ti
            vi /= np.linalg.norm(vi)
            vj = x[j] - scale * tj
            vj /= np.linalg.norm(vj)
            xp[i] = vi
            xp[j] = vj
            g2 = xp @ xp.T
            np.fill_diagonal(g2, -2.0)
            m2 = float(g2.max())
            if m2 < m - 1e-12 and m2 < best_val + 1e-12:
                x = xp
                if m2 < best_val:
                    best_val = m2
                    best_x = xp.copy()
                eps = min(scale * 1.08, 3e-3)
                improved = True
                break
        if not improved:
            eps *= 0.96
            if eps < 5e-07:
                # tiny random tangent nudge to escape exact tie
                k = it % n
                rnd = np.random.randn(d)
                rnd -= float(rnd @ x[k]) * x[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    xn = x[k] + 1e-06 * rnd
                    xn /= np.linalg.norm(xn)
                    x[k] = xn
                eps = 5e-07
    return best_x, best_val

def solve():
    t0 = time.time()
    np.random.seed(0)
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(453, 5))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    else:
        vals = []
        for c in cands:
            c = np.asarray(c, dtype=float)
            if c.shape != (453, 5):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append((max_cos(c), c))
        vals.sort(key=lambda t: t[0])
        x = vals[0][1].copy()
        best0 = vals[0][0]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    best_val = max_cos(x)
    best_x = x.copy()
    # tiny deterministic break of symmetry
    rng = np.random.default_rng(0)
    x = x + 2e-04 * rng.standard_normal(x.shape)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # budget: keep total < 170s, reserve final margin
    # phases
    x, best_val, best_x = smooth_phase(x, 6000, 3e-03, 3e-04, 4e-03, 1e-03, 0.9, t0, 70, best_val, best_x)
    # continue from best
    x = best_x.copy()
    x, best_val, best_x = smooth_phase(x, 6000, 3e-04, 3e-05, 1e-03, 3e-04, 0.9, t0, 115, best_val, best_x)
    x = best_x.copy()
    x, best_val, best_x = smooth_phase(x, 8000, 3e-05, 3e-06, 3e-04, 5e-05, 0.92, t0, 150, best_val, best_x)

    # greedy shave starting from best
    xg = best_x.copy()
    best_x2, best_val2 = greedy_shave(xg, best_val, best_x.copy(), t0, 168, steps=12000)
    if best_val2 < best_val:
        best_val = best_val2
        best_x = best_x2
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
