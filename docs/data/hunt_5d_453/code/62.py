import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_EXPLORE = t0 + 152.0
    rng = np.random.default_rng(1)
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
            nn = np.linalg.norm(cc, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cc = cc / nn
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
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    def anneal_inplace(X, N, tau0, tau1, lr0, lr1, mu):
        nonlocal_best = [bestV]
        # use closure lists to update best
        v = np.zeros_like(X)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if (it & 31) == 0 and time.time() > T_END - 6.0:
                break
            frac = it / max(N - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # track in-loop best (cheap, G already done)
            inv_tau = 1.0 / tau
            # softmax weights; G-m <= 0 so exp stable
            P = np.exp((G - m) * inv_tau)
            # gradient
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            # normalize step by mean row norm to keep lr stable
            gn = float(np.mean(np.linalg.norm(Gt, axis=1))) + 1e-12
            # scale so typical step ~ lr
            Gt = Gt / gn * 2.0
            v = mu * v + (1.0 - mu) * Gt
            X -= lr * v
            nn = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            X /= nn
            if m > bestV + 0.02:
                # blow-up: revert to best and damp velocity
                X[:] = bestX
                v *= 0.3
        return X

    cycle = 0
    Kseq = [6, 10, 4, 12, 7, 3, 9, 5]
    while time.time() < T_EXPLORE:
        cycle += 1
        if time.time() > T_END - 20.0:
            break
        X = bestX.copy()
        # row max to pick victims
        try:
            G0 = X @ X.T
            np.fill_diagonal(G0, -2.0)
            rmax = np.max(G0, axis=1)
            m0 = float(np.max(rmax))
        except Exception:
            rmax = np.zeros(n)
            m0 = bestV
        K = Kseq[cycle % len(Kseq)]
        victims = np.argsort(-rmax)[:K]
        # occasionally include a random low-contact point to move rattlers
        if (cycle % 4) == 0:
            try:
                contact = (m0 - rmax) < 1e-9  # approx; better count contacts
                # count contacts per point
                cnt = np.sum(G0 > m0 - 1e-9, axis=1)
                low = np.where(cnt <= 1)[0]
                if low.size > 0:
                    victims[-1] = int(low[rng.integers(0, low.size)])
            except Exception:
                pass
        victims = np.asarray(victims, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[victims] = False
        Xref = X[keep_mask]

        Ncand = 6000
        S = rng.standard_normal((Ncand, d))
        S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
        # largest holes w.r.t. Xref
        D = S @ Xref.T
        dmax = np.max(D, axis=1)
        order = np.argsort(dmax)  # smallest first
        picked = []
        # diversity threshold varies per cycle
        div_thr = 0.88 if (cycle % 2 == 0) else 0.80
        for idx in order:
            if len(picked) >= K:
                break
            s = S[idx]
            ok = True
            for q in picked:
                if float(s @ q) > div_thr:
                    ok = False
                    break
            if ok:
                picked.append(s)
        if len(picked) < K:
            for idx in order:
                if len(picked) >= K:
                    break
                s = S[idx]
                # allow duplicates if needed
                dup = False
                for q in picked:
                    if float(s @ q) > 0.999:
                        dup = True
                        break
                if not dup:
                    picked.append(s)
        for vi, s in zip(victims, picked):
            X[int(vi)] = s
        # small global tangent jitter to bulk to break symmetry
        sig = 0.002 if (cycle % 3 != 0) else 0.006
        R = rng.standard_normal((n, d)) * sig
        # make tangent
        R -= np.sum(R * X, axis=1, keepdims=True) * X
        X += R
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)

        # choose anneal schedule cyclically
        if cycle % 3 == 0:
            tau0, tau1, lr0, lr1, mu, N = 0.008, 0.0007, 0.012, 0.0012, 0.80, 320
        elif cycle % 3 == 1:
            tau0, tau1, lr0, lr1, mu, N = 0.004, 0.0005, 0.008, 0.0008, 0.85, 380
        else:
            tau0, tau1, lr0, lr1, mu, N = 0.012, 0.0010, 0.015, 0.0015, 0.75, 300
        # shorten if little time
        if time.time() > T_EXPLORE - 20.0:
            N = min(N, 220)
        X = anneal_inplace(X, N, tau0, tau1, lr0, lr1, mu)
        # evaluate
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
        except Exception:
            m = float("inf")
        if np.isfinite(m) and m < bestV - 1e-12:
            bestV = m
            bestX = X.copy()
        # else discard (elitist)

        # quick greedy worst-pair shave on current best occasionally
        if (cycle % 5) == 0 and time.time() < T_END - 15.0:
            try:
                Xg = bestX.copy()
                for _ in range(30):
                    Gg = Xg @ Xg.T
                    np.fill_diagonal(Gg, -2.0)
                    mg = float(np.max(Gg))
                    flat = int(np.argmax(Gg))
                    i = flat // n; j = flat % n
                    xi = Xg[i].copy(); xj = Xg[j].copy()
                    cij = float(xi @ xj)
                    cij = max(-1.0, min(1.0, cij))
                    ti = xj - cij * xi
                    ni = float(np.linalg.norm(ti))
                    if ni < 1e-12:
                        break
                    ti /= ni
                    improved = False
                    for sc in (7e-4, 3e-4, 1.2e-4):
                        vi = xi - sc * ti
                        vi /= (np.linalg.norm(vi) + 1e-30)
                        oi = Xg[i].copy()
                        Xg[i] = vi
                        # fast check: max involving i
                        di = float(np.max(Xg[i] @ Xg.T[np.arange(n) != i].T))
                        if di < mg - 1e-13:
                            Gf = Xg @ Xg.T
                            np.fill_diagonal(Gf, -2.0)
                            mf = float(np.max(Gf))
                            if mf < mg - 1e-13:
                                if mf < bestV:
                                    bestV = mf
                                    bestX = Xg.copy()
                                improved = True
                                break
                        Xg[i] = oi
                    if not improved:
                        break
            except Exception:
                pass

    # final intensive polish of best
    try:
        X = bestX.copy()
        # two-stage fine anneal
        t_rem = T_END - time.time() - 1.5
        if t_rem > 3.0:
            X = anneal_inplace(X, 500, 0.003, 0.00035, 0.006, 0.0006, 0.85)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if np.isfinite(m) and m < bestV:
                bestV = m
                bestX = X.copy()
            else:
                X = bestX.copy()
        t_rem = T_END - time.time() - 1.0
        if t_rem > 2.0:
            X = anneal_inplace(X, 400, 0.0015, 0.00025, 0.003, 0.0004, 0.85)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if np.isfinite(m) and m < bestV:
                bestV = m
                bestX = X.copy()
    except Exception:
        pass

    # normalize, guard
    try:
        nn = np.linalg.norm(bestX, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        bestX = bestX / nn
        bestX[~np.isfinite(bestX)] = 0.0
        # fix any zero rows
        for i in range(n):
            if not np.all(np.isfinite(bestX[i])) or float(np.linalg.norm(bestX[i])) < 0.5:
                v = rng.standard_normal(d)
                v /= (np.linalg.norm(v) + 1e-30)
                bestX[i] = v
        nn = np.linalg.norm(bestX, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        bestX = bestX / nn
    except Exception:
        pass
    return np.asarray(bestX, dtype=np.float64)
