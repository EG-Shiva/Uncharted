import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
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

def _adam(x, steps, tau0, tau1, lr0, lr1):
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
        gm = G.max()
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _lbfgs(x_init, tau, maxiter=60):
    n, d = x_init.shape
    y0 = x_init.ravel().copy()
    def fg(y_flat):
        y = y_flat.reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = y / nrm
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = float(G.max())
        E = np.exp((G - mval) / tau)
        np.fill_diagonal(E, 0.0)
        Sup = float(E.sum() / 2.0)
        if not np.finite(Sup) or Sup <= 0:
            Sup = 1.0
        f = mval + tau * np.log(Sup)
        W = E / Sup
        grad_x = W @ x
        dot = (grad_x * x).sum(axis=1, keepdims=True)
        grad_y = (grad_x - dot * x) / nrm
        return float(f), grad_y.ravel().astype(float)
    try:
        res = minimize(fg, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
        y = res.x.reshape(n, d)
    except Exception:
        return _norm(x_init)
    return _norm(y)

def _find_void(x_fixed, rng, n_cand=8000):
    cand = rng.standard_normal((n_cand, 4))
    cand = _norm(cand)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    b = cand[idx].copy()
    bs = float(best[idx])
    step = 0.05
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop = _norm(prop)
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
    rng = np.random.default_rng(12345)
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
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    x_best = x0.copy()
    best_val = _max_cos(x_best)
    cur = x_best.copy()
    cur_val = best_val

    sigmas = [0.03, 0.06, 0.015, 0.10, 0.025, 0.15, 0.05, 0.008]
    tau_pairs = [(0.015, 0.0012), (0.025, 0.002), (0.008, 0.0006), (0.04, 0.003)]
    lr_pairs = [(0.008, 0.0008), (0.014, 0.001), (0.005, 0.0004)]

    it = 0
    # Phase A: basin hopping ~ up to 125s
    while time.time() < t0 + 122 and time.time() < deadline - 48:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        tau0, tau1 = tau_pairs[it % len(tau_pairs)]
        lr0, lr1 = lr_pairs[it % len(lr_pairs)]
        y = cur + sigma * rng.standard_normal((272, 4))
        y = _norm(y)
        # topology change: kill overloaded points
        if it % 2 == 0:
            nkill = 2 + (it % 4)  # 2..5
        else:
            nkill = 1 + (it % 2)  # 1..2
        # personal worst ranking
        Gtmp = y @ y.T
        np.fill_diagonal(Gtmp, -2.0)
        pw = Gtmp.max(axis=1)
        order = np.argsort(-pw)
        for kk in range(nkill):
            if time.time() > deadline - 48:
                break
            # pick victim among top-10 randomly
            top = order[:10]
            # bias to worst but randomize
            vi = int(top[rng.integers(0, len(top))])
            mask = np.ones(272, dtype=bool)
            mask[vi] = False
            vv = _find_void(y[mask], rng, n_cand=6000)
            y[vi] = vv
            # update ranking cheaply every kill for next pick
            # mark chosen as used by moving it to end
            order = order[order != vi]
            if order.size < 10:
                Gtmp2 = y @ y.T
                np.fill_diagonal(Gtmp2, -2.0)
                pw2 = Gtmp2.max(axis=1)
                order = np.argsort(-pw2)
        steps = 380 if (it % 3) else 550
        y = _adam(y, steps, tau0, tau1, lr0, lr1)
        # periodic second-order refinement
        if it % 4 == 0 and time.time() < deadline - 55:
            y = _lbfgs(y, tau=0.0022, maxiter=50)
            # short cool adam after lbfgs
            y = _adam(y, 120, 0.004, 0.0006, 0.003, 0.0004)
        v = _max_cos(y)
        if v < best_val - 1e-14:
            best_val = v
            x_best = y.copy()
            cur = y.copy()
            cur_val = v
        elif v < cur_val:
            cur = y.copy()
            cur_val = v
        else:
            T = 0.0025
            if rng.random() < np.exp((cur_val - v) / T):
                cur = y.copy()
                cur_val = v
            # occasional restart from best if wandering too high
            if cur_val > best_val + 0.015 and (it % 7 == 0):
                cur = x_best.copy()
                cur_val = best_val
    # Phase B: intensify best with low-temp runs
    for rep in range(4):
        if time.time() > deadline - 42:
            break
        tau0, tau1 = (0.006, 0.0005) if rep % 2 == 0 else (0.010, 0.0008)
        y = _adam(x_best.copy(), 450, tau0, tau1, 0.004, 0.0004)
        if time.time() < deadline - 44:
            y = _lbfgs(y, tau=0.0012, maxiter=80)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            x_best = y.copy()
    # Phase C: LP snap
    remain = deadline - time.time() - 2.0
    if remain > 6:
        x_pol = _lp_polish(x_best, min(remain, 38.0), deadline - 1.0)
        vpol = _max_cos(x_pol)
        if vpol <= best_val:
            x_best = x_pol
            best_val = vpol
    if not np.all(np.isfinite(x_best)):
        x_best = _norm(x0)
    x_best = _norm(x_best)
    return x_best
