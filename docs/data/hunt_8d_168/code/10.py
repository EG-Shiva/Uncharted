import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_e8():
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
    for mask in range(256):
        bits = np.empty(8)
        nminus = 0
        for k in range(8):
            if (mask >> k) & 1:
                bits[k] = 0.5
            else:
                bits[k] = -0.5
                nminus += 1
        if nminus % 2 == 0:
            pts.append(bits.copy())
    E = np.array(pts, dtype=np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E

def _anneal(X0, steps, b0, b1, l0, l1, noise=0.0, mu=0.78, rng=None):
    X = X0.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bestV = float(G0.max())
    bestX = X.copy()
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(l0); ll1 = np.log(l1)
    for k in range(steps):
        f = k / max(steps-1, 1)
        beta = float(np.exp(lb0 + (lb1-lb0)*f))
        lr = float(np.exp(ll0 + (ll1-ll0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bestV:
            bestV = m
            bestX = X.copy()
        D = G - m
        cut = 9.0/beta + 0.012
        mask = D > -cut
        # exp safe (D<=0)
        W = np.zeros_like(D)
        # only where mask to save exp
        W[mask] = np.exp(beta*D[mask])
        Grad = W @ X
        dot = np.sum(Grad*X, axis=1, keepdims=True)
        Gt = Grad - dot*X
        rms = float(np.sqrt(np.mean(Gt*Gt))) + 1e-12
        Gt /= rms
        V = mu*V + (1.0-mu)*Gt
        X = X - lr*V
        if noise > 0 and rng is not None:
            # decaying tangent noise
            ns = noise*(1.0-f)
            if ns > 1e-12:
                Z = rng.standard_normal(size=X.shape)
                zd = np.sum(Z*X, axis=1, keepdims=True)
                Zt = Z - zd*X
                X = X + ns*lr*Zt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bestV:
        bestV = m
        bestX = X.copy()
    return bestX, bestV

def _void_fill(Y, k_new, rng, n_sample=10000):
    Y = Y.copy()
    d = Y.shape[1]
    for _ in range(k_new):
        S = rng.standard_normal(size=(n_sample, d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ Y.T
        worst = dots.max(axis=1)
        idx = int(np.argmin(worst))
        Y = np.vstack([Y, S[idx:idx+1]])
    return Y

def _greedy_single_relocate(X, rng, tries=3, n_sample=6000):
    # try to relocate worst points to best void; hill climb on max
    n, d = X.shape
    cur = _max_cos(X)
    for _ in range(tries):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        order = np.argsort(-per)
        improved = False
        for ii in order[:8]:
            # candidate voids
            S = rng.standard_normal(size=(n_sample, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            # dots to all except ii
            mask = np.ones(n, dtype=bool); mask[ii] = False
            Y = X[mask]
            dots = S @ Y.T
            worst = dots.max(axis=1)
            bidx = int(np.argmin(worst))
            if worst[bidx] < per[ii] - 1e-12:
                X2 = X.copy()
                X2[ii] = S[bidx]
                v2 = _max_cos(X2)
                if v2 < cur:
                    X = X2
                    cur = v2
                    improved = True
                    break
        if not improved:
            break
    return X, cur

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        xxT = x @ x.T
        c_all = xxT[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20*radius, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) > 5000:
            order = np.argsort(-c_all[idx])[:5000]
            idx = idx[order]
        if len(idx) == 0:
            radius *= 0.5
            continue
        I = iu[0][idx]; J = iu[1][idx]
        csel = c_all[idx]
        p = len(I)
        nv = n*d + 1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(7)
    try:
        bestX = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            bestX = np.load("record.npy").astype(np.float64)
        except Exception:
            bestX = rng.standard_normal(size=(168, 8))
    if bestX.shape != (168, 8):
        tmp = rng.standard_normal(size=(168, 8))
        try:
            m = min(len(bestX), 168)
            dd = min(bestX.shape[1], 8)
            tmp[:m, :dd] = bestX[:m, :dd]
        except Exception:
            pass
        bestX = tmp
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestV = _max_cos(bestX)
    initV = bestV
    E8 = _build_e8()
    n, d = 168, 8
    explore_end = DEADLINE - 36.0
    trial = 0
    # contact counts for guided surgery (computed on best)
    while time.time() < explore_end and trial < 2000:
        trial += 1
        mode = trial % 5
        X0 = None
        steps = 800
        b0, b1 = 30.0, 500.0
        l0, l1 = 0.022, 0.0022
        noise = 0.6
        if mode == 0:
            idx = rng.choice(len(E8), size=n, replace=False)
            X0 = E8[idx].copy()
            X0 = X0 + 0.03*rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 850; b0, b1 = 28.0, 420.0; noise = 0.7
        elif mode == 1:
            sig = float(np.exp(rng.uniform(np.log(0.012), np.log(0.38))))
            X0 = bestX + sig*rng.standard_normal(size=bestX.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 800; b0, b1 = 32.0, 550.0; noise = 0.5
        elif mode == 2:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            # contacts within 1e-7 of max for overload measure
            m = float(per.max())
            cnt = np.sum(G > m - 1e-7, axis=1)
            score = per + 0.002*cnt  # bias to overloaded hubs
            k = int(rng.integers(3, 10))
            worst_idx = np.argsort(-score)[:k]
            keep = np.ones(n, dtype=bool); keep[worst_idx] = False
            Yrem = bestX[keep] + 0.006*rng.standard_normal(size=(n-k, d))
            Yrem /= np.linalg.norm(Yrem, axis=1, keepdims=True)
            X0 = _void_fill(Yrem, k, rng, n_sample=9000)
            perm = rng.permutation(n)
            X0 = X0[perm]
            steps = 950; b0, b1 = 30.0, 600.0; noise = 0.6
        elif mode == 3:
            # contact-guided anisotropic kick: kick hubs harder
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = np.sum(G > m - 2e-4, axis=1).astype(float)
            amp = 0.02 + 0.30*(cnt/(cnt.max()+1e-9))
            X0 = bestX + (amp[:, None])*rng.standard_normal(size=bestX.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 750; b0, b1 = 35.0, 650.0; noise = 0.4
        else:
            X0 = rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            steps = 650; b0, b1 = 20.0, 300.0; l0, l1 = 0.03, 0.003; noise = 0.8
        Xopt, Vopt = _anneal(X0, steps, b0, b1, l0, l1, noise=noise, rng=rng)
        # greedy single-point void improvement (topology change)
        if Vopt < bestV + 0.005 and time.time() < explore_end - 2:
            Xopt2, Vopt2 = _greedy_single_relocate(Xopt, rng, tries=2, n_sample=4000)
            if Vopt2 < Vopt:
                # short re-anneal after surgery
                Xopt2, Vopt2b = _anneal(Xopt2, 400, 60.0, 700.0, 0.008, 0.0012, noise=0.15, rng=rng)
                Xopt, Vopt = Xopt2, Vopt2b
        if Vopt < bestV - 1e-12:
            bestV = Vopt
            bestX = Xopt
            if time.time() < explore_end - 6:
                Xref, Vref = _anneal(bestX, 1300, 55.0, 900.0, 0.011, 0.0011, noise=0.1, rng=rng)
                if Vref < bestV:
                    bestV = Vref
                    bestX = Xref
        if time.time() > explore_end:
            break
    if time.time() < DEADLINE - 32.0:
        Xref, Vref = _anneal(bestX, 2600, 55.0, 1000.0, 0.009, 0.0009, noise=0.05, rng=rng)
        if Vref < bestV:
            bestV = Vref
            bestX = Xref
    rem = DEADLINE - time.time() - 1.0
    if rem > 5.0:
        bestX = _lp_polish(bestX, seconds=min(rem, 33.0), deadline=DEADLINE-0.5)
        bestV = _max_cos(bestX)
    # safety: if we failed to improve, return incumbent exactly
    try:
        ref = np.load("best.npy").astype(np.float64)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        rv = _max_cos(ref)
        if bestV > rv:
            bestX = ref
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
