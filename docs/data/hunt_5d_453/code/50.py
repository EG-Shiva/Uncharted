import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    HOP_END = t0 + 158.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5

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
            cc = cc / (np.linalg.norm(cc, axis=1, keepdims=True) + 1e-30)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
        bestV = _max_cos(bestX)
    baseX = bestX.copy()

    def anneal(x, N, tau0, tau1, lr0, lr1, mu, deadline):
        try:
            ltau = np.log(tau1 / tau0)
        except Exception:
            ltau = 0.0
        try:
            llr = np.log(lr1 / lr0)
        except Exception:
            llr = 0.0
        v = np.zeros_like(x)
        for it in range(N):
            if (it & 31) == 0 and time.time() > deadline:
                break
            frac = it / max(1, N - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            with np.errstate(over='ignore', invalid='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            P = np.where(np.isfinite(P), P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            xn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            xn[xn == 0] = 1.0
            x = x / xn
        return x

    def greedy_shave(x, steps):
        for it in range(steps):
            if (it & 15) == 0 and time.time() > HOP_END:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            cij = float(np.dot(x[i], x[j]))
            if cij < bestV - 0.002 and it > 5:
                break
            ti = x[j] - cij * x[i]
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                continue
            ti /= ni
            tj = x[i] - cij * x[j]
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                continue
            tj /= nj
            step = 0.0012 / (1.0 + it * 0.08)
            vi = x[i] - step * ti
            vi /= (np.linalg.norm(vi) + 1e-30)
            vj = x[j] - step * tj
            vj /= (np.linalg.norm(vj) + 1e-30)
            oi = x[i].copy(); oj = x[j].copy()
            x[i] = vi; x[j] = vj
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            # revert if worsened a lot (allow small uphill for topology)
            if m2 > cij + 0.0008:
                x[i] = oi; x[j] = oj
        return x

    def largest_holes(Xkeep, k, n_cand=12000):
        holes = []
        H = None
        for t in range(k):
            C = rng.standard_normal((n_cand, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
            D = C @ Xkeep.T
            md = np.max(D, axis=1)
            if H is not None and len(holes) > 0:
                Dh = C @ H.T
                mdh = np.max(Dh, axis=1)
                # require new hole far from previous holes
                score = np.maximum(md, mdh * 0.98)
            else:
                score = md
            bi = int(np.argmin(score))
            h = C[bi].copy()
            # quick local push away from nearest neighbours
            for _ in range(8):
                dots = Xkeep @ h
                ci = int(np.argmax(dots))
                c = float(dots[ci])
                tt = Xkeep[ci] - c * h
                tn = float(np.linalg.norm(tt))
                if tn < 1e-12:
                    break
                tt /= tn
                h = h - 0.05 * tt
                if H is not None and len(holes) > 0:
                    dh = H @ h
                    cj = int(np.argmax(dh))
                    cc2 = float(dh[cj])
                    if cc2 > 0.9:
                        tt2 = H[cj] - cc2 * h
                        tn2 = float(np.linalg.norm(tt2))
                        if tn2 > 1e-12:
                            h = h - 0.05 * (tt2 / tn2)
                nh = float(np.linalg.norm(h))
                if not np.isfinite(nh) or nh == 0:
                    break
                h /= nh
            holes.append(h)
            if H is None:
                H = h[None, :]
            else:
                H = np.vstack([H, h[None, :]])
        return holes

    def row_max_idx(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        idx = np.argsort(-rm)[:k]
        return idx, rm, G

    it = 0
    # short warm polish of incumbent (very cheap)
    try:
        if time.time() < HOP_END - 2:
            y0 = baseX.copy()
            y0 = anneal(y0, 200, 0.003, 0.0006, 0.003, 0.0004, 0.86, HOP_END)
            v0 = _max_cos(y0)
            if v0 < bestV:
                bestV = v0
                bestX = y0.copy()
                baseX = y0.copy()
    except Exception:
        pass

    taus = [(0.0045, 0.0006), (0.008, 0.0009), (0.0025, 0.0004), (0.006, 0.0005)]
    lrs = [(0.004, 0.0004), (0.007, 0.0006), (0.0025, 0.0003), (0.005, 0.0005)]
    sigmas = [0.003, 0.006, 0.002, 0.009, 0.0045, 0.012]

    while time.time() < HOP_END - 0.6:
        it += 1
        if HOP_END - time.time() < 0.6:
            break
        ti = it % len(taus)
        tau0, tau1 = taus[ti]
        lr0, lr1 = lrs[ti % len(lrs)]
        sigma = float(sigmas[it % len(sigmas)])
        mode = it % 6
        try:
            if mode in (0, 1, 3):
                # surgery: eject worst k into holes
                k = 1 if mode == 0 else (2 if mode == 1 else 3)
                # vary k occasionally
                if it % 17 == 0:
                    k = 4
                idx, rm, _ = row_max_idx(baseX, k + 4)
                # pick k among top k+4 with randomization for diversity
                perm = rng.permutation(k + 4)[:k]
                worst = idx[perm]
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                Xkeep = baseX[mask]
                holes = largest_holes(Xkeep, k, n_cand=10000 if k < 3 else 7000)
                y = baseX.copy()
                for a2, wi in enumerate(worst):
                    y[wi] = holes[a2]
                # tiny noise to break symmetry
                nz = rng.standard_normal((n, d)) * 0.0008
                nz -= np.sum(nz * y, axis=1, keepdims=True) * y
                y = y + nz
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                N = 320 if k <= 2 else 260
                y = anneal(y, N, tau0, tau1, lr0, lr1, 0.86, HOP_END)
                y = greedy_shave(y, 12)
                vv = _max_cos(y)
                if vv < bestV:
                    bestV = vv
                    bestX = y.copy()
                    baseX = y.copy()
                elif vv < bestV + 0.0015 and (it % 4 == 0):
                    # side basin: anneal a bit more from this topology
                    y2 = anneal(y, 150, 0.002, 0.0004, 0.002, 0.0003, 0.86, HOP_END)
                    v2 = _max_cos(y2)
                    if v2 < bestV:
                        bestV = v2
                        bestX = y2.copy()
                        baseX = y2.copy()
            else:
                # pure shake with hub-focused amplitude
                _, rm, G = row_max_idx(baseX, n)
                m = float(np.max(G))
                # hubs: within 2e-7 of max get bigger shake; rattlers small shake
                tight = (m - rm) < 2e-7
                y = baseX.copy()
                nz = rng.standard_normal((n, d)) * sigma
                # boost hubs, suppress deep rattlers
                boost = np.ones((n, 1), dtype=np.float64)
                boost[tight] *= 1.6
                # rattlers: row max far -> smaller random walk so they can jam gradually
                ratt = (m - rm) > 5e-7
                boost[ratt[:, 0] if boost.ndim == 2 and False else ratt] = 0.6
                nz = nz * boost
                nz -= np.sum(nz * y, axis=1, keepdims=True) * y
                y = y + nz
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                N = 300 if sigma < 0.008 else 220
                y = anneal(y, N, tau0, tau1, lr0, lr1, 0.85, HOP_END)
                if it % 3 == 0:
                    y = greedy_shave(y, 10)
                vv = _max_cos(y)
                if vv < bestV:
                    bestV = vv
                    bestX = y.copy()
                    baseX = y.copy()
        except Exception:
            continue

    # final polish of best
    try:
        deadline = T_END - 0.8
        y = bestX.copy()
        # a couple of low-temp passes
        rem = deadline - time.time()
        if rem > 3:
            y = anneal(y, 500, 0.0022, 0.00025, 0.0022, 0.00022, 0.88, deadline)
            v = _max_cos(y)
            if v < bestV:
                bestV = v
                bestX = y.copy()
        rem = deadline - time.time()
        if rem > 2:
            y = greedy_shave(bestX.copy(), 40)
            v = _max_cos(y)
            if v < bestV:
                bestV = v
                bestX = y.copy()
        rem = deadline - time.time()
        if rem > 2:
            y2 = anneal(bestX.copy(), 400, 0.0012, 0.00015, 0.0012, 0.00015, 0.9, deadline)
            v2 = _max_cos(y2)
            if v2 < bestV:
                bestV = v2
                bestX = y2.copy()
    except Exception:
        pass

    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    # fix any degenerate row
    nr = np.linalg.norm(bestX, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-9:
            v = rng.standard_normal(d)
            v /= (np.linalg.norm(v) + 1e-30)
            bestX[i] = v
    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    return bestX
