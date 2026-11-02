import time
import numpy as np

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

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
    if not cands:
        x0 = rng.standard_normal((453, 5))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    bestX = None
    bestV = float("inf")
    for cc in cands:
        v = max_cos(cc)
        if v < bestV:
            bestV = v
            bestX = cc.copy()
    n, d = bestX.shape

    def anneal(X, N, tau0, tau1, lr0, lr1, mu, allow_revert=0.05):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        if tau0 == tau1:
            ltau = 0.0
        else:
            ltau = np.log(tau1 / tau0)
        if lr0 == lr1:
            llr = 0.0
        else:
            llr = np.log(lr1 / lr0)
        for it in range(N):
            if (it & 63) == 0 and time.time() > DEAD - 12:
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
            if m > bestV + allow_revert:
                X = bestX.copy()
                V *= 0.5
                continue
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            nrm = np.sqrt(np.sum(Gt * Gt, axis=1))
            mn = float(np.max(nrm))
            if mn < 1e-14 or not np.isfinite(mn):
                continue
            Gt /= mn
            V = mu * V + (1.0 - mu) * Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def best_hole(Xref, exclude, n_cand=8000):
        if isinstance(exclude, int):
            excl = [exclude]
        else:
            excl = list(exclude)
        mask = np.ones(n, dtype=bool)
        mask[excl] = False
        B = Xref[mask]
        best_c = None
        best_s = float("inf")
        chunk = 4000
        for s in range(0, n_cand, chunk):
            cc = min(chunk, n_cand - s)
            C = rng.standard_normal((cc, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = np.max(C @ B.T, axis=1)
            k = int(np.argmin(S))
            sk = float(S[k])
            if sk < best_s:
                best_s = sk
                best_c = C[k].copy()
        # quick single-point polish (repulsion from B)
        xi = best_c
        v = np.zeros(d)
        mu = 0.8
        tau = 2e-4
        for it in range(120):
            g = B @ xi
            m = float(np.max(g))
            w = np.exp((g - m) / tau)
            grad = w @ B
            gt = grad - float(grad @ xi) * xi
            gn = float(np.linalg.norm(gt))
            if gn < 1e-14:
                break
            gt /= gn
            v = mu * v + (1.0 - mu) * gt
            lr = 0.02 * (1.0 - 0.8 * it / 119)
            xi = xi - lr * v
            xi /= np.linalg.norm(xi)
        return xi

    # Stage 1: diversified basin hopping with collective annealing
    sigmas = [0.035, 0.015, 0.055, 0.022, 0.04, 0.012, 0.07, 0.018]
    n_cycles = 0
    # short warm collective anneal to get gradient scale
    cur = bestX.copy()
    cur = anneal(cur, 1200, 1.5e-3, 2e-4, 0.002, 0.0006, 0.9)

    cycle = 0
    while time.time() < DEAD - 30:
        cycle += 1
        mode = cycle % 5
        if mode == 0 or mode == 3:
            # multi-hole surgery: replace 4 worst points
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)
            k = 4 if mode == 0 else 2
            victims = order[:k].tolist()
            trial = bestX.copy()
            # sequential reinsert using evolving background
            for idx in victims:
                hc = best_hole(trial, idx, n_cand=8000)
                trial[idx] = hc
            trial = anneal(trial, 1600, 8e-4, 6e-5, 0.0012, 0.00025, 0.88)
        else:
            sig = sigmas[(cycle - 1) % len(sigmas)]
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)
            noise = rng.standard_normal((n, d)) * sig
            # boost worst 40 to break shell coherence
            noise[order[:40]] *= 2.5
            noise -= np.sum(noise * bestX, axis=1, keepdims=True) * bestX
            trial = bestX + noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            if mode % 2 == 1:
                trial = anneal(trial, 1600, 1.2e-3, 8e-5, 0.0015, 0.0003, 0.9)
            else:
                trial = anneal(trial, 1600, 5e-4, 4e-5, 0.001, 0.00022, 0.88)
        n_cycles += 1
        if n_cycles > 40:
            break
        if time.time() > DEAD - 30:
            break

    # Stage 2: fine collective squeeze from elite
    cur = bestX.copy()
    cur = anneal(cur, 3500, 1.4e-4, 1.8e-5, 4.5e-4, 9e-5, 0.9, allow_revert=0.02)

    # Stage 3: limited greedy worst-pair shave (cheap, few hundred steps)
    x = bestX.copy()
    eps = 4e-4
    for _ in range(300):
        if time.time() > DEAD - 1.5:
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
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            break
        ti /= ni
        tj = x[i] - cij * x[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            break
        tj /= nj
        improved = False
        for sc in (eps, eps * 0.5):
            vi = x[i] - sc * ti
            vi /= np.linalg.norm(vi)
            vj = x[j] - sc * tj
            vj /= np.linalg.norm(vj)
            oi = x[i].copy(); oj = x[j].copy()
            x[i] = vi; x[j] = vj
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            if m2 < m - 1e-13:
                if m2 < bestV:
                    bestV = m2
                    bestX = x.copy()
                eps = min(sc * 1.1, 2e-3)
                improved = True
                break
            else:
                x[i] = oi; x[j] = oj
        if not improved:
            eps *= 0.94
            if eps < 8e-08:
                break

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
