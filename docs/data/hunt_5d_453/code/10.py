import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def soft_relax(X, beta, steps, lr0, lr1, mu, t_end):
    n, d = X.shape
    v = np.zeros_like(X)
    with np.errstate(over='ignore', invalid='ignore'):
        for it in range(steps):
            if time.time() > t_end:
                break
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            m = float(np.max(G))
            if not np.isfinite(m):
                break
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            Z = float(np.sum(W) * 0.5)
            if not np.isfinite(Z) or Z <= 0:
                break
            grad = (W @ X) / Z
            dot = np.sum(grad * X, axis=1, keepdims=True)
            Gt = grad - dot * X
            rn = np.sqrt(np.sum(Gt * Gt, axis=1))
            mn = float(np.max(rn))
            if not np.isfinite(mn) or mn < 1e-14:
                break
            Gt /= mn
            v = mu * v + (1.0 - mu) * Gt
            frac = it / max(1, steps - 1)
            lr = lr0 + (lr1 - lr0) * frac
            Y = X - lr * v
            yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
            yn[yn == 0] = 1.0
            X = Y / yn
    return X


def strict_descent(x, cur, beta, lr_init, max_steps, t_end):
    lr = lr_init
    for _ in range(max_steps):
        if time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        with np.errstate(over='ignore'):
            W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Z = float(np.sum(W) * 0.5)
        if not np.isfinite(Z) or Z == 0:
            break
        grad = (W @ x) / Z
        dot = np.sum(grad * x, axis=1, keepdims=True)
        Gt = grad - dot * x
        nrm = float(np.linalg.norm(Gt))
        if not np.isfinite(nrm) or nrm < 1e-16:
            break
        y = x - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < cur:
            x = y
            cur = v
            lr = min(lr * 1.3, 2e-3)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    return x, cur


def find_hole(X, exclude, rng, n_samples=16000, refine=60):
    n, d = X.shape
    S = rng.normal(size=(n_samples, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
    mask = np.ones(n, dtype=bool)
    mask[exclude] = False
    Xe = X[mask]
    # farthest sample = minimal max-dot
    D = S @ Xe.T
    md = np.max(D, axis=1)
    bi = int(np.argmin(md))
    h = S[bi].copy()
    # refine hole by pushing away from closest point
    for _ in range(refine):
        dots = Xe @ h
        ci = int(np.argmax(dots))
        c = float(dots[ci])
        xc = Xe[ci]
        t = xc - c * h
        tn = float(np.linalg.norm(t))
        if tn < 1e-12:
            break
        t /= tn
        # move away from closest: h <- h - step*t
        step = 0.01
        h = h - step * t
        h /= np.linalg.norm(h) + 1e-30
    return h


def lp_polish_small(x, cur, t_end, r_init=2.5e-4):
    n, d = x.shape
    best = x.copy()
    best_val = cur
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]
    while time.time() < t_end and radius > 1e-13:
        if t_end - time.time() < 2.5:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        delta = max(20.0 * radius, 2e-7)
        sel = c > m - delta
        if np.count_nonzero(sel) > 4000:
            idx = np.argpartition(c, -4000)[-4000:]
            mk = np.zeros_like(sel)
            mk[idx] = True
            sel = mk
        I = iu_r[sel]
        J = iu_c[sel]
        p = int(I.size)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best_val:
            x = y
            best = y.copy()
            best_val = v
            radius = min(radius * 1.4, 5e-4)
        else:
            radius *= 0.5
    return best, best_val


def solve():
    t0 = time.time()
    t_end_total = t0 + 172.0
    t_hop_end = t0 + 118.0
    t_strict_end = t0 + 145.0
    t_lp_end = t0 + 163.0
    rng = np.random.default_rng(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.normal(size=(453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    n, d = bestX.shape
    iu0 = np.triu_indices(n, 1)
    iu_r = iu0[0]
    iu_c = iu0[1]

    betas = [180.0, 450.0, 1000.0, 250.0, 700.0]
    sigmas = [0.003, 0.006, 0.010, 0.0015, 0.0045]
    it = 0
    # hopping loop
    while time.time() < t_hop_end:
        it += 1
        if t_hop_end - time.time() < 0.6:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        c = G[iu_r, iu_c]
        a = int(np.argmax(c))
        wi = int(iu_r[a])
        wj = int(iu_c[a])
        op = int(rng.integers(0, 3))
        # time for inner relax
        inner_end = min(time.time() + 3.5, t_hop_end)
        beta = float(betas[it % len(betas)])
        y = bestX.copy()
        if op == 0:
            # neighbourhood jitter
            row_i = G[wi].copy()
            row_j = G[wj].copy()
            row_i[wi] = -2
            row_j[wj] = -2
            ti = np.argpartition(row_i, -8)[-8:]
            tj = np.argpartition(row_j, -8)[-8:]
            sset = set([wi, wj])
            for t_ in list(ti) + list(tj):
                sset.add(int(t_))
            for _ in range(6):
                sset.add(int(rng.integers(0, n)))
            sidx = np.array(list(sset), dtype=int)
            sigma = float(sigmas[it % len(sigmas)])
            noise = rng.normal(scale=sigma, size=(len(sidx), d))
            for k2, idx in enumerate(sidx):
                vv = noise[k2] - float(np.dot(noise[k2], y[idx])) * y[idx]
                y[idx] = y[idx] + vv
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = soft_relax(y, beta, 260, 0.0025, 0.00025, 0.88, inner_end)
        elif op == 1:
            # void reinsertion of one worst endpoint
            # pick endpoint with larger sum of top dots
            si = float(np.sum(np.sort(G[wi])[-6:]))
            sj = float(np.sum(np.sort(G[wj])[-6:]))
            victim = wi if si > sj else wj
            try:
                h = find_hole(bestX, victim, rng)
            except Exception:
                continue
            y[victim] = h
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = soft_relax(y, beta, 300, 0.002, 0.0002, 0.88, inner_end)
        else:
            # direct pair split then relax
            cij = float(np.dot(y[wi], y[wj]))
            cij = max(-1.0, min(1.0, cij))
            ti = cij * y[wi] - y[wj]
            tn = float(np.linalg.norm(ti))
            if tn > 1e-12:
                ti /= tn
                s = float(rng.choice([0.0008, 0.0015, 0.0025]))
                y[wi] = y[wi] + s * ti
                y[wi] /= np.linalg.norm(y[wi])
            tj = cij * y[wj] - y[wi]
            # recompute with updated wi? use original for stability
            tj = cij * bestX[wj] - bestX[wi]
            tn2 = float(np.linalg.norm(tj))
            if tn2 > 1e-12:
                tj /= tn2
                s = 0.0015
                y[wj] = y[wj] + s * tj
                y[wj] /= np.linalg.norm(y[wj])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = soft_relax(y, beta, 220, 0.0015, 0.00015, 0.90, inner_end)
        if not np.all(np.isfinite(y)):
            continue
        v = max_cos(y)
        if v < bestV:
            bestX = y
            bestV = v

    # strict descent squeeze
    if time.time() < t_strict_end:
        x, v = strict_descent(bestX, bestV, beta=700.0, lr_init=8e-5,
                              max_steps=2500, t_end=t_strict_end)
        if v < bestV:
            bestX, bestV = x, v
    if time.time() < t_strict_end:
        x, v = strict_descent(bestX, bestV, beta=2200.0, lr_init=2.5e-5,
                              max_steps=2500, t_end=t_strict_end)
        if v < bestV:
            bestX, bestV = x, v

    # LP polish
    if time.time() < t_lp_end:
        try:
            xlp, vlp = lp_polish_small(bestX, bestV, t_end=t_lp_end, r_init=2.0e-4)
            if vlp < bestV:
                bestX, bestV = xlp, vlp
        except Exception:
            pass

    # final micro hill on worst pair
    sigmas2 = [3e-5, 1e-5, 1e-4, 2e-4, 5e-6]
    k = 0
    while time.time() < t_end_total:
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        sigma = float(sigmas2[k % len(sigmas2)])
        k += 1
        y = bestX.copy()
        for kk in (i, j):
            nz = rng.normal(scale=sigma, size=5)
            nz = nz - float(np.dot(nz, y[kk])) * y[kk]
            y[kk] = y[kk] + nz
            nn = float(np.linalg.norm(y[kk]))
            if nn == 0 or not np.isfinite(nn):
                y[kk] = bestX[kk]
            else:
                y[kk] = y[kk] / nn
        if (k % 6) == 0:
            r = int(rng.integers(0, n))
            nz = rng.normal(scale=sigma * 2.0, size=5)
            nz = nz - float(np.dot(nz, y[r])) * y[r]
            y[r] = y[r] + nz
            nn = float(np.linalg.norm(y[r]))
            if nn != 0 and np.isfinite(nn):
                y[r] = y[r] / nn
        v = max_cos(y)
        if v < bestV:
            bestX = y
            bestV = v
        if k > 20000:
            break

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
