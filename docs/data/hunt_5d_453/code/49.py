import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    MAIN_END = t0 + 152.0
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
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
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

    # ---------- helpers ----------
    def rowmax_deg(X, m, thr=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        deg = np.sum(G >= (m - thr), axis=1)
        return rm, deg, G

    def find_holes(Xref, k, n_cand=8000):
        # Xref: (m,5) points to avoid
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xref.T
        md = np.max(D, axis=1)
        # pick k well-separated holes: greedy on smallest md
        order = np.argpartition(md, min(k * 20, n_cand - 1))[:min(k * 20, n_cand)]
        # sort candidate pool by md
        order = order[np.argsort(md[order])]
        chosen = []
        for idx in order:
            c = C[idx]
            ok = True
            for h in chosen:
                if float(c @ h) > 0.9:
                    ok = False
                    break
            if ok:
                chosen.append(c.copy())
            if len(chosen) >= k:
                break
        if len(chosen) < k:
            # fill randomly
            while len(chosen) < k:
                chosen.append(C[order[len(chosen) % len(order)]].copy())
        # refine each hole: push away from nearest ref point
        out = []
        for h in chosen:
            h = h.copy()
            for _ in range(15):
                dots = Xref @ h
                j = int(np.argmax(dots))
                c = float(dots[j])
                xj = Xref[j]
                tt = xj - c * h
                tn = float(np.linalg.norm(tt))
                if tn < 1e-12:
                    break
                tt /= tn
                h = h - 0.05 * tt
                nh = float(np.linalg.norm(h))
                if not np.finite(nh) or nh == 0:
                    break
                h /= nh
            out.append(h)
        return out

    def smooth_run(x, N, tau0, tau1, lr0, lr1, mu, deadline):
        nonlocal bestV, bestX
        v = np.zeros_like(x)
        try:
            ltau = np.log(tau1 / tau0)
        except Exception:
            ltau = 0.0
        try:
            llr = np.log(lr1 / lr0)
        except Exception:
            llr = 0.0
        for it in range(N):
            if (it % 50 == 0) and time.time() > deadline:
                break
            frac = it / max(1, N - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = x.copy()
            # softmax weights
            with np.errstate(over='ignore', invalid='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            P = np.where(np.isfinite(P), P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            # degree-balancing boost: worst rows step more
            try:
                rm = np.max(G, axis=1)
                with np.errstate(over='ignore'):
                    w = 1.0 + 1.5 * np.exp((rm - m) / max(tau, 1e-12))
                w = np.where(np.isfinite(w), w, 1.0)
                w = np.clip(w, 1.0, 2.6)[:, None]
                Gt = Gt * w
            except Exception:
                pass
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            xn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            xn[xn == 0] = 1.0
            x = x / xn
            # safety: if diverged far, reset to best
            if m > bestV + 0.03:
                x = bestX.copy()
                v *= 0.3
        if time.time() <= deadline + 0.5:
            try:
                m2 = _max_cos(x)
                if m2 < bestV:
                    bestV = m2
                    bestX = x.copy()
            except Exception:
                pass
        return x

    # initial quick polish of incumbent
    try:
        y = bestX.copy()
        y = smooth_run(y, 600, 0.003, 0.0005, 0.003, 0.0003, 0.86, MAIN_END)
        # keep best tracked inside
    except Exception:
        pass

    taus = [(0.004, 0.0006), (0.007, 0.0008), (0.0025, 0.00035), (0.005, 0.0005)]
    lrs = [(0.004, 0.0004), (0.006, 0.0005), (0.0025, 0.0003), (0.005, 0.00035)]
    sigmas = [0.003, 0.006, 0.002, 0.009, 0.0045]
    mus = [0.86, 0.88, 0.84]

    it = 0
    while time.time() < MAIN_END - 0.6:
        it += 1
        if time.time() > MAIN_END - 0.5:
            break
        ti = it % len(taus)
        tau0, tau1 = taus[ti]
        lr0, lr1 = lrs[ti % len(lrs)]
        mu = mus[it % len(mus)]
        sigma = sigmas[it % len(sigmas)]
        mode = it % 6

        # current stats on best
        try:
            Gb = bestX @ bestX.T
            np.fill_diagonal(Gb, -2.0)
            mb = float(np.max(Gb))
            rmb = np.max(Gb, axis=1)
            degb = np.sum(Gb >= (mb - 1e-9), axis=1)
            nratt = int(np.sum(degb == 0))
        except Exception:
            mb = bestV
            rmb = None
            degb = None
            nratt = 0

        x = bestX.copy()

        try:
            if mode == 0 and nratt > 0:
                # rattler surgery: replace all rattlers with holes
                idx = np.where(degb == 0)[0]
                keep = np.ones(n, dtype=bool)
                keep[idx] = False
                Xkeep = x[keep]
                holes = find_holes(Xkeep, len(idx), n_cand=8000)
                for ii, h in zip(idx, holes):
                    x[ii] = h
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 350, tau0, tau1, lr0, lr1, mu, MAIN_END)
            elif mode == 1 or (mode == 0 and nratt == 0):
                # worst-point hole surgery (1-3 points)
                k = 1 + (it % 3)  # 1..3
                worst = np.argsort(rmb)[-k:]
                keep = np.ones(n, dtype=bool)
                keep[worst] = False
                Xkeep = x[keep]
                holes = find_holes(Xkeep, k, n_cand=8000)
                for ii, h in zip(worst, holes):
                    x[ii] = h
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 350, tau0, tau1, lr0, lr1, mu, MAIN_END)
            elif mode == 2:
                # hub shake: top-degree points + neighbours get big kick
                k = min(8, n)
                hubs = np.argsort(degb)[-k:]
                nz = rng.standard_normal((n, d)) * (sigma * 0.6)
                # extra to hubs
                nz[hubs] *= 3.0
                nz -= np.sum(nz * x, axis=1, keepdims=True) * x
                x = x + nz
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 300, tau0, tau1, lr0, lr1, mu, MAIN_END)
            elif mode == 3:
                # worst-pair separation + global shake
                flat = int(np.argmax(Gb))
                i = flat // n
                j = flat % n
                nz = rng.standard_normal((n, d)) * sigma
                nz -= np.sum(nz * x, axis=1, keepdims=True) * x
                x = x + nz
                # push i,j apart tangentially a bit more
                cij = float(x[i] @ x[j])
                ti_ = x[j] - cij * x[i]
                nti = float(np.linalg.norm(ti_))
                if nti > 1e-12:
                    ti_ /= nti
                    x[i] = x[i] - 0.004 * ti_
                tj_ = x[i] - cij * x[j]
                ntj = float(np.linalg.norm(tj_))
                if ntj > 1e-12:
                    tj_ /= ntj
                    # recompute after i moved? approximate
                    x[j] = x[j] - 0.004 * tj_
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 300, tau0, tau1, lr0, lr1, mu, MAIN_END)
            elif mode == 4:
                # double-hole: remove worst + random high point
                w1 = int(np.argmax(rmb))
                # second: random among top-30 worst
                top30 = np.argsort(rmb)[-30:]
                w2 = int(top30[rng.integers(0, len(top30))])
                if w2 == w1:
                    w2 = int(top30[rng.integers(0, len(top30))])
                worst = np.array([w1, w2], dtype=int)
                keep = np.ones(n, dtype=bool)
                keep[worst] = False
                Xkeep = x[keep]
                holes = find_holes(Xkeep, 2, n_cand=10000)
                for ii, h in zip(worst, holes):
                    x[ii] = h
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 350, tau0, tau1, lr0, lr1, mu, MAIN_END)
            else:
                # plain global shake
                nz = rng.standard_normal((n, d)) * sigma
                nz -= np.sum(nz * x, axis=1, keepdims=True) * x
                x = x + nz
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = smooth_run(x, 280, tau0, tau1, lr0, lr1, mu, MAIN_END)
        except Exception:
            continue

    # ---------- final fine polish ----------
    try:
        z = bestX.copy()
        # tiny shake to escape exact tie without leaving basin
        if time.time() < T_END - 3.0:
            nz = rng.standard_normal((n, d)) * 0.0008
            nz -= np.sum(nz * z, axis=1, keepdims=True) * z
            z = z + nz
            z /= np.linalg.norm(z, axis=1, keepdims=True)
        z = smooth_run(z, 1200, 0.0015, 0.00018, 0.0015, 0.00012, 0.90, T_END - 0.6)
        # greedy worst-pair micro-shave if time remains
        rem = T_END - time.time()
        if rem > 4.0:
            steps = int(min(400, rem * 60))
            eps = 8e-4
            for s in range(steps):
                if s % 40 == 0 and time.time() > T_END - 0.6:
                    break
                G = z @ z.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                if m < bestV:
                    bestV = m
                    bestX = z.copy()
                flat = int(np.argmax(G))
                i = flat // n
                j = flat % n
                cij = float(z[i] @ z[j])
                ti_ = z[j] - cij * z[i]
                ni = float(np.linalg.norm(ti_))
                if ni < 1e-12:
                    break
                ti_ /= ni
                tj_ = z[i] - cij * z[j]
                nj = float(np.linalg.norm(tj_))
                if nj < 1e-12:
                    break
                tj_ /= nj
                improved = False
                for sc in (eps, eps * 0.5):
                    vi = z[i] - sc * ti_
                    vi /= np.linalg.norm(vi)
                    vj = z[j] - sc * tj_
                    vj /= np.linalg.norm(vj)
                    oi = z[i].copy(); oj = z[j].copy()
                    z[i] = vi; z[j] = vj
                    G2 = z @ z.T
                    np.fill_diagonal(G2, -2.0)
                    m2 = float(np.max(G2))
                    if m2 < m - 1e-13:
                        if m2 < bestV:
                            bestV = m2
                            bestX = z.copy()
                        improved = True
                        break
                    else:
                        z[i] = oi; z[j] = oj
                if improved:
                    eps = min(eps * 1.05, 1.5e-3)
                else:
                    eps *= 0.97
                    if eps < 1e-7:
                        eps = 1e-7
    except Exception:
        pass

    X = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    X = np.where(np.isfinite(X), X, 0.0)
    nn = np.linalg.norm(X, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    X = X / nn
    return X
