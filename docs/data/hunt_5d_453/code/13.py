import time
import numpy as np

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(1)
    np.random.seed(1)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load starts
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

    def anneal(X, N, tau0, tau1, lr0, lr1, mu):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(N):
            if (it & 31) == 0 and time.time() > DEAD - 10:
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
            if m > bestV + 0.015:
                X = bestX.copy()
                V *= 0.5
                continue
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if mn < 1e-14:
                continue
            Gt /= mn
            V = mu * V + (1.0 - mu) * Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def best_hole(Xref, exclude, n_cand=20000):
        # exclude: int or list
        if isinstance(exclude, int):
            excl = [exclude]
        else:
            excl = list(exclude)
        mask = np.ones(Xref.shape[0], dtype=bool)
        mask[excl] = False
        B = Xref[mask]
        best_c = None
        best_s = float("inf")
        chunk = 5000
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
        return best_c

    def refine_single(xi, B, excl, tau=1.5e-4, lr=0.02, steps=200):
        v = np.zeros(d)
        mu = 0.8
        for it in range(steps):
            g = B @ xi
            if isinstance(excl, int):
                g[excl] = -2.0
            else:
                for e in excl:
                    g[e] = -2.0
            m = float(np.max(g))
            w = np.exp((g - m) / tau)
            if isinstance(excl, int):
                w[excl] = 0.0
            else:
                for e in excl:
                    w[e] = 0.0
            grad = w @ B
            gt = grad - float(grad @ xi) * xi
            gn = float(np.linalg.norm(gt))
            if gn < 1e-14:
                break
            gt /= gn
            v = mu * v + (1.0 - mu) * gt
            lri = lr * (1.0 - 0.8 * it / max(1, steps - 1))
            xi = xi - lri * v
            nn = float(np.linalg.norm(xi))
            if nn == 0 or not np.isfinite(nn):
                break
            xi = xi / nn
        return xi

    # Phase A: coarse anneal from incumbent
    cur = bestX.copy()
    cur = anneal(cur, 3500, 3e-3, 2e-4, 0.003, 0.0006, 0.9)
    cur = bestX.copy()
    cur = anneal(cur, 2500, 4e-4, 5e-5, 0.001, 0.0003, 0.9)

    sigmas = [0.004, 0.0015, 0.008, 0.0025, 0.006, 0.001]
    cycle = 0
    while time.time() < DEAD - 28:
        cycle += 1
        mode = cycle % 6
        trial = None
        if mode == 0 or mode == 3:
            # single worst hole reinsert
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            idx = int(np.argmax(pmax))
            hc = best_hole(bestX, idx, n_cand=20000)
            hc = refine_single(hc, bestX, idx, tau=1.5e-4, lr=0.02, steps=200)
            trial = bestX.copy()
            trial[idx] = hc
        elif mode == 1:
            # worst pair double reinsert
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii = flat // n
            jj = flat % n
            hi = best_hole(bestX, [ii, jj], n_cand=15000)
            tmp = bestX.copy()
            tmp[ii] = hi
            # refine ii against background without ii,jj
            tmp[ii] = refine_single(tmp[ii], bestX, [ii, jj], tau=1.5e-4, lr=0.02, steps=150)
            hj = best_hole(tmp, jj, n_cand=15000)
            hj = refine_single(hj, tmp, jj, tau=1.5e-4, lr=0.02, steps=150)
            trial = tmp
            trial[jj] = hj
        else:
            sig = sigmas[cycle % len(sigmas)]
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)
            noise = rng.standard_normal(bestX.shape) * sig
            noise[order[:30]] *= 3.0
            noise -= np.sum(noise * bestX, axis=1, keepdims=True) * bestX
            trial = bestX + noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # short re-anneal with alternating focus
        if mode % 2 == 0:
            trial = anneal(trial, 1100, 2.5e-4, 3e-5, 8e-4, 2e-4, 0.88)
        else:
            trial = anneal(trial, 1100, 4e-4, 4e-5, 9e-4, 2e-4, 0.88)
        if time.time() > DEAD - 28:
            break

    # final fine squeeze
    cur = bestX.copy()
    cur = anneal(cur, 4000, 1.2e-4, 2e-5, 4e-4, 1e-4, 0.9)
    # tiny greedy shave on worst pair to squeeze last 1e-8
    x = bestX.copy()
    eps = 5e-4
    for _ in range(400):
        if time.time() > DEAD - 1.0:
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
            if m2 < m - 1e-12:
                if m2 < bestV:
                    bestV = m2
                    bestX = x.copy()
                eps = min(sc * 1.1, 2e-3)
                improved = True
                break
            else:
                x[i] = oi; x[j] = oj
        if not improved:
            eps *= 0.95
            if eps < 1e-07:
                break

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
