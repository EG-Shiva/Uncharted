import time
import numpy as np

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(2)
    np.random.seed(2)

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

    def anneal(X, N, tau0, tau1, lr0, lr1, mu):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(N):
            if (it & 31) == 0 and time.time() > DEAD - 14:
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
            if m > bestV + 0.018:
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

    def refine_single(xi, Bmat, excl, tau=1.5e-4, lr=0.02, steps=180):
        v = np.zeros(d)
        mu = 0.8
        if isinstance(excl, int):
            is_int = True
        else:
            is_int = False
            eset = list(excl)
        for it in range(steps):
            g = Bmat @ xi
            if is_int:
                g[excl] = -2.0
            else:
                for e in eset:
                    g[e] = -2.0
            m = float(np.max(g))
            w = np.exp((g - m) / tau)
            if is_int:
                w[excl] = 0.0
            else:
                for e in eset:
                    w[e] = 0.0
            grad = w @ Bmat
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

    # initial coarse polish
    cur = bestX.copy()
    cur = anneal(cur, 3000, 3e-3, 2e-4, 0.003, 0.0006, 0.9)
    cur = bestX.copy()
    cur = anneal(cur, 2200, 4e-4, 5e-5, 0.001, 0.0003, 0.9)

    sigmas = [0.004, 0.0015, 0.009, 0.0025, 0.02, 0.001, 0.006]
    cycle = 0
    while time.time() < DEAD - 30:
        cycle += 1
        mode = cycle % 7
        trial = None
        if mode == 0 or mode == 3:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            idx = int(np.argmax(pmax))
            hc = best_hole(bestX, idx, n_cand=20000)
            hc = refine_single(hc, bestX, idx, tau=1.5e-4, lr=0.02, steps=180)
            trial = bestX.copy()
            trial[idx] = hc
        elif mode == 1:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii = flat // n
            jj = flat % n
            hi = best_hole(bestX, [ii, jj], n_cand=14000)
            tmp = bestX.copy()
            tmp[ii] = hi
            tmp[ii] = refine_single(tmp[ii], bestX, [ii, jj], tau=1.5e-4, lr=0.02, steps=130)
            hj = best_hole(tmp, jj, n_cand=14000)
            hj = refine_single(hj, tmp, jj, tau=1.5e-4, lr=0.02, steps=130)
            trial = tmp
            trial[jj] = hj
        elif mode == 5:
            # triple surgery on worst 3
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)[:3]
            trial = bestX.copy()
            done = []
            for k, idx in enumerate(list(order)):
                excl = list(order) + done
                # for already moved, use current trial as background
                hc = best_hole(trial, list(order), n_cand=12000)
                hc = refine_single(hc, trial, list(order), tau=2e-4, lr=0.02, steps=120)
                trial[int(idx)] = hc
                done.append(int(idx))
        else:
            sig = sigmas[cycle % len(sigmas)]
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)
            noise = rng.standard_normal(bestX.shape) * sig
            # boost worst points
            nb = 40 if sig > 0.008 else 30
            noise[order[:nb]] *= 3.0
            noise -= np.sum(noise * bestX, axis=1, keepdims=True) * bestX
            trial = bestX + noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        if mode % 2 == 0:
            trial = anneal(trial, 950, 2.5e-4, 3e-5, 8e-4, 2e-4, 0.88)
        elif mode % 3 == 0:
            trial = anneal(trial, 950, 1.2e-3, 8e-5, 1.2e-3, 2.5e-4, 0.88)
        else:
            trial = anneal(trial, 950, 4e-4, 4e-5, 9e-4, 2e-4, 0.88)
        if time.time() > DEAD - 30:
            break

    # final fine squeeze
    cur = bestX.copy()
    cur = anneal(cur, 3500, 1.2e-4, 2e-5, 4e-4, 1e-4, 0.9)

    # greedy worst-pair shave
    x = bestX.copy()
    eps = 5e-4
    for _ in range(500):
        if time.time() > DEAD - 6.0:
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

    # tiny trust-region LP polish
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        xp = bestX.copy()
        curv = bestV
        radius = 2.0e-4
        iu_r, iu_c = np.triu_indices(n, 1)
        while time.time() < DEAD - 1.2 and radius > 1e-12:
            Gfull = xp @ xp.T
            c = Gfull[iu_r, iu_c]
            m = float(np.max(c))
            delta = max(25.0 * radius, 2.5e-7)
            sel = c > m - delta
            cnt = int(np.count_nonzero(sel))
            if cnt > 3500:
                idx = np.argpartition(c, -3500)[-3500:]
                mk = np.zeros_like(sel)
                mk[idx] = True
                sel = mk
                cnt = 3500
            if cnt == 0:
                break
            I = iu_r[sel]; J = iu_c[sel]
            csel = c[sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            colI = (I[:, None] * d + np.arange(d)).ravel()
            colJ = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::(2 * d + 1)] = 0  # placeholder
            # build cols vectorized
            base = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            cols = base
            vals = np.concatenate([xp[J], xp[I], -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                er = np.repeat(np.arange(n), d)
                ec = np.arange(n * d)
                A_eq = coo_matrix((xp.ravel(), (er, ec)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
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
            sol = res.x
            dx = sol[:n * d].reshape(n, d)
            # remove normal component exactly
            dx = dx - np.sum(dx * xp, axis=1, keepdims=True) * xp
            # clip
            mx = float(np.max(np.abs(dx)))
            if mx > radius + 1e-12:
                dx *= (radius / mx)
            y = xp + dx
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < curv - 1e-12:
                xp = y
                curv = v
                bestX = xp.copy()
                bestV = curv
                radius *= 1.2
            else:
                radius *= 0.5
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
