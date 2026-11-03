import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def build_e8():
    pts = []
    d = 8
    for i in range(d):
        for j in range(i+1, d):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = s1
                    v[j] = s2
                    pts.append(v)
    # half-integer vectors with even parity
    for mask in range(256):
        bits = [(1.0 if ((mask >> k) & 1) else -1.0) * 0.5 for k in range(8)]
        # even number of minus => sum? count minus
        nminus = sum(1 for b in bits if b < 0)
        if nminus % 2 == 0:
            pts.append(np.array(bits))
    E = np.array(pts, dtype=np.float64)  # 240 x 8
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E

def smooth_run(X0, steps, beta0, beta1, lr0, lr1, mu=0.75):
    X = X0.copy()
    V = np.zeros_like(X)
    # init best
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bestV = float(G0.max())
    bestX = X.copy()
    lbeta0 = np.log(beta0)
    lbeta1 = np.log(beta1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    for k in range(steps):
        f = k / max(steps - 1, 1)
        beta = float(np.exp(lbeta0 + (lbeta1 - lbeta0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bestV:
            bestV = m
            bestX = X.copy()
        D = G - m  # <=0, diag <<0
        cut = 8.0 / beta + 0.015
        # mask near-max pairs
        # D> -cut
        # compute weights
        # to avoid overflow exp(beta*D) with D<=0 is <=1 safe
        mask = D > -cut
        W = np.exp(beta * D, out=np.zeros_like(D), where=mask)
        # zero diag already (mask false)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rms = float(np.sqrt(np.mean(Gt ** 2))) + 1e-12
        Gt = Gt / rms
        V = mu * V + (1.0 - mu) * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bestV:
        bestV = m
        bestX = X.copy()
    return bestX, bestV

def surgery_fill(Xdrop, k_new, rng, n_sample=12000):
    # Xdrop: (n-k, d) remaining points
    Y = Xdrop.copy()
    for _ in range(k_new):
        S = rng.standard_normal(size=(n_sample, Y.shape[1]))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ Y.T
        worst = dots.max(axis=1)
        idx = int(np.argmin(worst))
        Y = np.vstack([Y, S[idx:idx+1]])
    return Y

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        xxT = x @ x.T
        c_all = xxT[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        # cap number of constraints for speed
        # if too many, keep tightest 4000
        idx = np.where(sel)[0]
        if len(idx) > 5000:
            order = np.argsort(-c_all[idx])[:5000]
            idx = idx[order]
            I = iu[0][idx]
            J = iu[1][idx]
            csel = c_all[idx]
        else:
            I, J = iu[0][sel], iu[1][sel]
            csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(12345)
    # load best
    try:
        bestX = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            bestX = np.load("record.npy").astype(np.float64)
        except Exception:
            bestX = rng.standard_normal(size=(168, 8))
    if bestX.shape != (168, 8):
        # try to reshape / truncate
        tmp = rng.standard_normal(size=(168, 8))
        m = min(len(bestX), 168)
        tmp[:m] = bestX[:m, :8] if bestX.shape[1] >= 8 else bestX
        bestX = tmp
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestV = max_cos(bestX)
    E8 = build_e8()  # 240 x 8
    n, d = 168, 8
    # quick E8 subset screening + basin hopping
    trial = 0
    # reserve ~35s for final polish
    explore_end = DEADLINE - 38.0
    # adaptive: short runs first
    while time.time() < explore_end and trial < 1000:
        trial += 1
        mode = trial % 4
        X0 = None
        if mode == 0:
            # E8 subset start
            idx = rng.choice(len(E8), size=n, replace=False)
            X0 = E8[idx].copy()
            sig = 0.025
            X0 = X0 + sig * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 800
            b0, b1 = 30.0, 400.0
            l0, l1 = 0.02, 0.002
        elif mode == 1:
            # kick best
            lsigma = rng.uniform(np.log(0.015), np.log(0.35))
            sig = float(np.exp(lsigma))
            X0 = bestX + sig * rng.standard_normal(size=bestX.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 800
            b0, b1 = 30.0, 500.0
            l0, l1 = 0.02, 0.002
        elif mode == 2:
            # surgery on best: remove worst k, reinsert voids
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            k = int(rng.integers(2, 9))  # 2..8
            worst_idx = np.argsort(-per)[:k]
            keep = np.ones(n, dtype=bool)
            keep[worst_idx] = False
            Yrem = bestX[keep]
            # small jitter remaining to break symmetry
            Yrem = Yrem + 0.005 * rng.standard_normal(size=Yrem.shape)
            Yrem /= np.linalg.norm(Yrem, axis=1, keepdims=True)
            X0 = surgery_fill(Yrem, k, rng, n_sample=10000)
            # shuffle order randomly to avoid bias
            perm = rng.permutation(n)
            X0 = X0[perm]
            steps = 900
            b0, b1 = 30.0, 500.0
            l0, l1 = 0.02, 0.002
        else:
            # random uniform start (diversity, rarely)
            X0 = rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 700
            b0, b1 = 20.0, 300.0
            l0, l1 = 0.03, 0.003
        Xopt, Vopt = smooth_run(X0, steps, b0, b1, l0, l1)
        if Vopt < bestV - 1e-12:
            bestV = Vopt
            bestX = Xopt
            # intensify immediately around new best with a longer run if time allows
            if time.time() < explore_end - 5:
                Xref, Vref = smooth_run(bestX, 1200, 50.0, 800.0, 0.012, 0.0012)
                if Vref < bestV:
                    bestV = Vref
                    bestX = Xref
        # occasional time check abort
        if time.time() > explore_end:
            break
    # refine best with longer smooth run
    if time.time() < DEADLINE - 32.0:
        Xref, Vref = smooth_run(bestX, 2500, 50.0, 900.0, 0.010, 0.001)
        if Vref < bestV:
            bestV = Vref
            bestX = Xref
    # final LP polish with remaining time
    rem = DEADLINE - time.time() - 1.0
    if rem > 5.0:
        bestX = lp_polish(bestX, seconds=min(rem, 34.0), t_deadline=DEADLINE - 0.5)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
