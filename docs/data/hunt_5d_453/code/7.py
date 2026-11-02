import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def anneal_run(X, N, tau0, tau1, lr0, lr1, mu, rng, t0, deadline, bestV, bestX, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for it in range(N):
        if time.time() - t0 > deadline:
            break
        frac = it / max(1, N - 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
            bestX = X.copy()
        # softmax weights near max
        # exp((G-m)/tau), G-m<=0
        P = np.exp((G - m) / tau)
        # diagonal is exp((-2-m)/tau) ~ 0, zero it
        # zero diag via flat indexing to avoid extra alloc
        # P flat diag indices
        # quick: set diag 0
        np.fill_diagonal(P, 0.0)
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # global normalisation
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if mn < 1e-14:
            continue
        Gt /= mn
        V = mu * V + (1.0 - mu) * Gt
        if noise > 0:
            # tiny random tangent noise to avoid exact ties
            Rn = rng.standard_normal(X.shape)
            Rn -= np.sum(Rn * X, axis=1, keepdims=True) * X
            rmn = float(np.max(np.sqrt(np.sum(Rn * Rn, axis=1)))) + 1e-12
            Rn /= rmn
            step = X - lr * V + noise * lr * Rn
        else:
            step = X - lr * V
        step /= np.linalg.norm(step, axis=1, keepdims=True)
        X = step
    return X, bestV, bestX

def greedy_pair_shave(x, bestV, bestX, t0, deadline, steps=2000, eps0=1e-3):
    n, d = x.shape
    eps = eps0
    for it in range(steps):
        if time.time() - t0 > deadline:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
            bestX = x.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(x[i] @ x[j])
        # tangent directions pushing apart
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
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
            m2 = float(np.max(g2))
            if m2 < m - 1e-12:
                x = xp
                if m2 < bestV:
                    bestV = m2
                    bestX = xp.copy()
                eps = min(scale * 1.1, 3e-3)
                improved = True
                break
        if not improved:
            eps *= 0.97
            if eps < 5e-07:
                eps = 5e-07
                # tiny kick to worst point
                k = i
                rnd = np.random.randn(d)
                rnd -= float(rnd @ x[k]) * x[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    xn = x[k] + 1e-06 * rnd
                    xn /= np.linalg.norm(xn)
                    x[k] = xn
    return x, bestV, bestX

def single_point_reinsert(X, bestV, bestX, t0, deadline, rng, tries_per_point=6, tau=2e-4, steps=250, lr=0.02):
    n, d = X.shape
    # find worst points by pointwise max
    if time.time() - t0 > deadline:
        return bestV, bestX
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmax = np.max(G, axis=1)
    order = np.argsort(-pmax)  # descending
    # try worst few
    for idx in order[:4]:
        if time.time() - t0 > deadline:
            break
        # current value already in best; try replacements
        for t in range(tries_per_point):
            if time.time() - t0 > deadline:
                break
            # random init for point idx
            cand = rng.standard_normal(d)
            cand /= np.linalg.norm(cand)
            # if first try, start from current + large kick
            if t == 0:
                cand = X[idx] + 0.05 * rng.standard_normal(d)
                cand -= float(cand @ X[idx]) * X[idx] if False else 0
                # just renormalize a perturbed version
                cand = X[idx] + 0.02 * rng.standard_normal(d)
                cand /= np.linalg.norm(cand)
            xi = cand.copy()
            Others = X.copy()  # fixed others (use bestX others?)
            # use bestX as background for stability
            B = bestX.copy()
            # optimize xi alone against B (excluding idx)
            v = np.zeros(d)
            mu = 0.8
            for it in range(steps):
                g = B @ xi  # n,
                g[idx] = -2.0  # ignore self
                m = float(np.max(g))
                # softmax
                w = np.exp((g - m) / tau)
                w[idx] = 0.0
                # gradient = sum w_j * b_j
                grad = w @ B  # (d,)
                grad_t = grad - float(grad @ xi) * xi
                gn = float(np.linalg.norm(grad_t))
                if gn < 1e-14:
                    break
                grad_t /= gn
                v = mu * v + (1.0 - mu) * grad_t
                frac = it / max(1, steps - 1)
                lri = lr * (0.2 + 0.8 * (1 - frac))
                xi = xi - lri * v
                xi /= np.linalg.norm(xi)
            # evaluate full config
            Xt = B.copy()
            Xt[idx] = xi
            vv = max_cos(Xt)
            if vv < bestV - 1e-12:
                bestV = vv
                bestX = Xt.copy()
                B = bestX.copy()
                # adopt immediately for further tries on next point?
                # update X reference too
        # end tries
    return bestV, bestX

def lp_polish(x, t0, deadline, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bestX = x.copy()
    radius = float(radius0)
    iu_r, iu_c = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < deadline and radius > 1e-14:
        if time.time() - t0 > deadline - 0.8:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        tol = 20.0 * radius
        if tol < 1e-9:
            tol = 1e-9
        sel = c > m - tol
        I = iu_r[sel]
        J = iu_c[sel]
        p = int(I.size)
        if p == 0:
            radius *= 0.5
            continue
        if p > 5000:
            cs = c[sel]
            kp = np.argpartition(-cs, 5000)[:5000]
            I = I[kp]; J = J[kp]
            cssel = cs[kp]
            p = 5000
        else:
            cssel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cssel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            # keep global best too
            x, best = y, v
            if v < max_cos(bestX):
                bestX = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            # if stuck small, revert to bestX
            if radius < 1e-7:
                x = bestX.copy()
                best = max_cos(x)
                radius = 1e-4
                # break to avoid loop
                if time.time() - t0 > deadline - 5:
                    break
    return bestX

def solve():
    t0 = time.time()
    TIME_END = 172.0  # seconds from start
    rng = np.random.default_rng(0)
    np.random.seed(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((453, 5))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    bestX = None
    bestV = float("inf")
    for cc in cands:
        vv = max_cos(cc)
        if vv < bestV:
            bestV = vv
            bestX = cc.copy()
    cur = bestX.copy()
    # Phase A: coarse anneal from incumbent
    # deadline allocations
    # A: until 45s
    cur, bestV, bestX = anneal_run(cur, 5000, 3e-3, 3e-4, 0.004, 0.001, 0.9, rng, t0, min(45.0, TIME_END-60), bestV, bestX)
    cur = bestX.copy()
    cur, bestV, bestX = anneal_run(cur, 5000, 4e-4, 5e-5, 0.001, 0.0003, 0.9, rng, t0, min(80.0, TIME_END-50), bestV, bestX)
    # greedy shave to attack single outlier
    cur = bestX.copy()
    cur, bestV, bestX = greedy_pair_shave(cur, bestV, bestX, t0, min(95.0, TIME_END-45), steps=3000, eps0=1e-3)
    # single-point reinsertion jumps
    bestV, bestX = single_point_reinsert(cur, bestV, bestX, t0, min(115.0, TIME_END-40), rng, tries_per_point=5, tau=2e-4, steps=250, lr=0.02)
    cur = bestX.copy()
    # iterated shake + short anneal cycles
    sigmas = [0.005, 0.003, 0.002, 0.001, 0.0007, 0.0004]
    si = 0
    while time.time() - t0 < min(150.0, TIME_END - 25):
        sig = sigmas[si % len(sigmas)]
        si += 1
        # perturb: stronger on worst points
        G = cur @ cur.T if False else bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        pmax = np.max(G, axis=1)
        # top 20 worst get 3x sigma
        order = np.argsort(-pmax)
        noise = rng.standard_normal(bestX.shape) * sig
        # boost worst
        noise[order[:20]] *= 3.0
        # tangent projection
        noise -= np.sum(noise * bestX, axis=1, keepdims=True) * bestX
        trial = bestX + noise
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # short anneal
        trial, bestV, bestX = anneal_run(trial, 1200, 2e-4, 3e-5, 0.0008, 0.0002, 0.9, rng, t0, min(150.0, TIME_END-25), bestV, bestX, noise=0.0)
        cur = bestX.copy()
        if time.time() - t0 > min(150.0, TIME_END-25):
            break
        # occasional greedy
        if si % 2 == 0:
            cur, bestV, bestX = greedy_pair_shave(cur, bestV, bestX, t0, min(150.0, TIME_END-25), steps=400, eps0=5e-4)
            cur = bestX.copy()
    # final polish
    bestX = lp_polish(bestX, t0, TIME_END - 0.5, radius0=8e-4)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
