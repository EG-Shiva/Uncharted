import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(1)
    np.random.seed(1)

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
    iu_r, iu_c = np.triu_indices(n, 1)

    def anneal(X, N, tau0, tau1, lr0, lr1, mu):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(N):
            if (it & 31) == 0 and time.time() > DEAD - 12:
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
            if mn < 1e-14 or not np.isfinite(mn):
                continue
            Gt /= mn
            V = mu * V + (1.0 - mu) * Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def hole_vs(B, n_cand=18000):
        best_c = None
        best_s = float("inf")
        chunk = 4500
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

    def refine_vs(xi, B, tau=1.5e-4, lr=0.02, steps=160):
        v = np.zeros(d)
        mu = 0.8
        for it in range(steps):
            g = B @ xi
            m = float(np.max(g))
            w = np.exp((g - m) / tau)
            grad = w @ B
            gt = grad - float(grad @ xi) * xi
            gn = float(np.linalg.norm(gt))
            if gn < 1e-14 or not np.isfinite(gn):
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

    def lp_polish(x, cur, t_end, r_init=3e-4):
        nonlocal bestV, bestX
        radius = r_init
        nv = n * d + 1
        while time.time() < t_end and radius > 1e-12:
            if t_end - time.time() < 2.0:
                break
            Gf = x @ x.T
            cfull = Gf[iu_r, iu_c]
            m = float(np.max(cfull))
            if m < bestV:
                bestV = m
                bestX = x.copy()
                cur = m
            delta = max(30.0 * radius, 4e-7)
            sel = cfull > m - delta
            cnt = int(np.count_nonzero(sel))
            if cnt > 4000:
                idx = np.argpartition(cfull, -4000)[-4000:]
                mk = np.zeros_like(sel)
                mk[idx] = True
                sel = mk
                cnt = 4000
            if cnt == 0:
                break
            I = iu_r[sel]
            J = iu_c[sel]
            csel = cfull[sel]
            p = cnt
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                b_ub = (m - csel).astype(np.float64)
                er = np.repeat(np.arange(n), d)
                ec = np.arange(n * d)
                ev = x.ravel()
                A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            tp = float(res.x[-1])
            if not np.isfinite(tp) or tp >= -1e-12:
                radius *= 0.5
                continue
            dx = res.x[:n * d].reshape(n, d)
            y = x + dx
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            mn = max_cos(y)
            if mn < m - 1e-12:
                x = y
                cur = mn
                if mn < bestV:
                    bestV = mn
                    bestX = x.copy()
                radius = min(radius * 1.25, 5e-4)
            else:
                radius *= 0.5
        return x, cur

    # warm collective anneal
    cur = bestX.copy()
    cur = anneal(cur, 2200, 3e-3, 2e-4, 0.003, 0.0006, 0.9)
    cur = bestX.copy()
    cur = anneal(cur, 1800, 4e-4, 5e-5, 0.001, 0.0003, 0.9)

    sigmas = [0.004, 0.0015, 0.008, 0.0025, 0.006, 0.001]
    cycle = 0
    while time.time() < DEAD - 38:
        cycle += 1
        mode = cycle % 6
        trial = None
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        pmax = np.max(G, axis=1)
        order = np.argsort(-pmax)
        if mode == 0:
            idx = int(order[0])
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            B = bestX[mask]
            hc = hole_vs(B, n_cand=18000)
            hc = refine_vs(hc, B, steps=180)
            trial = bestX.copy()
            trial[idx] = hc
        elif mode == 1 or mode == 3:
            K = 4 if mode == 1 else 3
            idxs = [int(v) for v in order[:K]]
            mask = np.ones(n, dtype=bool)
            mask[idxs] = False
            curB = bestX[mask].copy()
            vecs = []
            for _id in idxs:
                hc = hole_vs(curB, n_cand=14000)
                hc = refine_vs(hc, curB, steps=130)
                vecs.append(hc)
                curB = np.vstack([curB, hc[None, :]])
            trial = bestX.copy()
            for _id, hc in zip(idxs, vecs):
                trial[_id] = hc
        elif mode == 2:
            # worst-pair double reinsert diverse
            flat = int(np.argmax(G))
            ii = flat // n
            jj = flat % n
            mask = np.ones(n, dtype=bool)
            mask[ii] = False
            mask[jj] = False
            curB = bestX[mask].copy()
            h1 = hole_vs(curB, n_cand=14000)
            h1 = refine_vs(h1, curB, steps=130)
            curB = np.vstack([curB, h1[None, :]])
            h2 = hole_vs(curB, n_cand=14000)
            h2 = refine_vs(h2, curB, steps=130)
            trial = bestX.copy()
            trial[ii] = h1
            trial[jj] = h2
        else:
            sig = sigmas[cycle % len(sigmas)]
            noise = rng.standard_normal(bestX.shape) * sig
            noise[order[:30]] *= 3.0
            noise -= np.sum(noise * bestX, axis=1, keepdims=True) * bestX
            trial = bestX + noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        if mode == 5:
            # occasional direct LP on best before annealing trial
            if time.time() < DEAD - 50:
                _x, _v = lp_polish(bestX.copy(), bestV, time.time() + 3.0, r_init=2.0e-4)
                if _v < bestV:
                    bestX = _x.copy()
                    bestV = _v
        if trial is None:
            continue
        if mode % 2 == 0:
            trial = anneal(trial, 950, 2.5e-4, 3e-5, 8e-4, 2e-4, 0.88)
        else:
            trial = anneal(trial, 950, 4e-4, 4e-5, 9e-4, 2e-4, 0.88)
        if time.time() > DEAD - 38:
            break

    # intensive LP phase
    if time.time() < DEAD - 22:
        _x, _v = lp_polish(bestX.copy(), bestV, DEAD - 14.0, r_init=3e-4)
        if _v < bestV:
            bestX = _x.copy()
            bestV = _v
        else:
            if max_cos(_x) < max_cos(bestX):
                bestX = _x.copy()
                bestV = max_cos(bestX)

    # final fine squeeze
    cur = bestX.copy()
    cur = anneal(cur, 3000, 1.2e-4, 2e-5, 4e-4, 1e-4, 0.9)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
