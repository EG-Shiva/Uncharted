import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_HOP = t0 + 132.0
    np.random.seed(0)
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
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    n, d = bestX.shape

    def anneal_track(X, steps, tau0, tau1, lr0, lr1, mu, deadline):
        # returns (X_last, bestX_in_run, bestV_in_run)
        v = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 else 0.0
        llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        bv = float(np.max(G0))
        bX = X.copy()
        for it in range(steps):
            if it % 25 == 0 and time.time() > deadline:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bv:
                bv = m
                bX = X.copy()
            # softmax weights
            with np.errstate(over='ignore', invalid='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if not np.isfinite(mn) or mn < 1e-14:
                break
            Gt /= mn
            v = mu * v + (1.0 - mu) * Gt
            Y = X - lr * v
            yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
            yn[yn == 0] = 1.0
            X = Y / yn
        # final check
        try:
            Gf = X @ X.T
            np.fill_diagonal(Gf, -2.0)
            mf = float(np.max(Gf))
            if mf < bv:
                bv = mf
                bX = X.copy()
        except Exception:
            pass
        return X, bX, bv

    def micro_shave(X, deadline, eps_init=6e-7):
        n0, d0 = X.shape
        eps = float(eps_init)
        # estimate best
        while time.time() < deadline:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            flat = int(np.argmax(G))
            i = flat // n0
            j = flat % n0
            xi = X[i].copy()
            xj = X[j].copy()
            cij = float(xi @ xj)
            if cij > 1.0:
                cij = 1.0
            elif cij < -1.0:
                cij = -1.0
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = np.random.randn(d0)
                ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = np.random.randn(d0)
                tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj
            Ri = G[i, :].copy()
            Rj = G[j, :].copy()
            G[i, :] = -2.0; G[:, i] = -2.0
            G[j, :] = -2.0; G[:, j] = -2.0
            m_rest = float(np.max(G))
            best_m2 = m
            best_vi = None
            best_vj = None
            mode = -1
            # both moves
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti
                nvi = float(np.linalg.norm(vi))
                if nvi == 0 or not np.isfinite(nvi):
                    continue
                vi /= nvi
                vj = xj - sc * tj
                nvj = float(np.linalg.norm(vj))
                if nvj == 0 or not np.isfinite(nvj):
                    continue
                vj /= nvj
                di = X @ vi
                dj = X @ vj
                di[i] = -2.0; di[j] = -2.0
                dj[i] = -2.0; dj[j] = -2.0
                mc = m_rest
                a = float(np.max(di))
                if a > mc:
                    mc = a
                b = float(np.max(dj))
                if b > mc:
                    mc = b
                c2 = float(vi @ vj)
                if c2 > mc:
                    mc = c2
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_vi = vi
                    best_vj = vj
                    mode = 0
                    break
            if mode == -1:
                # single i
                for sc in (eps, eps * 0.5):
                    vi = xi - sc * ti
                    nvi = float(np.linalg.norm(vi))
                    if nvi == 0 or not np.isfinite(nvi):
                        continue
                    vi /= nvi
                    di = X @ vi
                    di[i] = -2.0
                    a = float(np.max(di))
                    mc = m_rest if m_rest > a else a
                    if mc < best_m2 - 1e-15:
                        best_m2 = mc
                        best_vi = vi
                        best_vj = None
                        mode = 1
                        break
                # single j (only if i failed, else keep both)
                if mode == -1:
                    for sc in (eps, eps * 0.5):
                        vj = xj - sc * tj
                        nvj = float(np.linalg.norm(vj))
                        if nvj == 0 or not np.isfinite(nvj):
                            continue
                        vj /= nvj
                        dj = X @ vj
                        dj[j] = -2.0
                        a = float(np.max(dj))
                        mc = m_rest if m_rest > a else a
                        if mc < best_m2 - 1e-15:
                            best_m2 = mc
                            best_vi = None
                            best_vj = vj
                            mode = 2
                            break
            if mode != -1:
                if mode == 0:
                    X[i] = best_vi
                    X[j] = best_vj
                elif mode == 1:
                    X[i] = best_vi
                else:
                    X[j] = best_vj
                eps *= 1.03
                if eps > 2e-6:
                    eps = 2e-6
            else:
                eps *= 0.84
                if eps < 1.5e-10:
                    eps = 1.5e-10
                    # tiny kick to worst point to change tie
                    k = i
                    rnd = np.random.randn(d0)
                    rnd -= float(rnd @ X[k]) * X[k]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        rnd /= rn
                        nk = X[k] + 4e-10 * rnd
                        nn = float(np.linalg.norm(nk))
                        if nn != 0 and np.isfinite(nn):
                            X[k] = nk / nn
        return X

    def find_hole(Xkeep, n_cand=12000):
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xkeep.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        h = C[bi].copy()
        # quick projected repulsion refine
        for _ in range(18):
            dots = Xkeep @ h
            ci = int(np.argmax(dots))
            c = float(dots[ci])
            xc = Xkeep[ci]
            tt = xc - c * h
            tn = float(np.linalg.norm(tt))
            if tn < 1e-12:
                break
            tt /= tn
            h = h - 0.03 * tt
            nh = float(np.linalg.norm(h))
            if nh == 0 or not np.isfinite(nh):
                break
            h /= nh
        return h

    it = 0
    while time.time() < T_HOP:
        it += 1
        if T_HOP - time.time() < 1.0:
            break
        X = bestX.copy()
        cyc = it % 4
        try:
            if cyc == 0 or cyc == 2:
                # density-balancing multi-ejection
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                delta = 0.003 if cyc == 0 else 0.0015
                cnt = np.sum(G > (m - delta), axis=1)
                rm = np.max(G, axis=1)
                k = 3 if cyc == 0 else 2
                # pick top-k overloaded (stable order)
                order = np.argsort(-(cnt.astype(float) * 10.0 + rm))
                victims = []
                for o in order[:k * 3]:
                    if len(victims) >= k:
                        break
                    if o not in victims:
                        victims.append(int(o))
                # remove victims from keep set progressively
                mask = np.ones(n, dtype=bool)
                # we will replace one by one using holes w.r.t. remaining
                for v in victims:
                    mask[v] = False
                    Xkeep = X[mask]
                    h = find_hole(Xkeep, n_cand=10000)
                    # small jitter
                    h = h + 1e-9 * rng.standard_normal(d)
                    h /= np.linalg.norm(h)
                    X[v] = h
                    mask[v] = True  # include for next hole search
                dl = min(time.time() + 1.6, T_HOP)
                # wide then narrow
                X, _, _ = anneal_track(X, 220, 0.0032, 0.00045, 0.0022, 0.00022, 0.86, dl)
                if time.time() > T_HOP:
                    V = _max_cos(X)
                    if V < bestV - 1e-15:
                        bestV = V
                        bestX = X.copy()
                    break
                X, bXa, bVa = anneal_track(X, 120, 0.0007, 0.00012, 0.0009, 0.00012, 0.88, dl)
                # take best of anneal
                if bVa < _max_cos(X) - 1e-15:
                    X = bXa.copy()
                dl2 = min(time.time() + 1.1, T_HOP)
                X = micro_shave(X, dl2, 5e-7)
                V = _max_cos(X)
                if V < bestV - 1e-15 and np.all(np.isfinite(X)):
                    bestV = V
                    bestX = X.copy()
            elif cyc == 1:
                # single worst reinsertion + narrow anneal
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                w = int(np.argmax(rm))
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                Xkeep = X[mask]
                h = find_hole(Xkeep, n_cand=16000)
                X[w] = h
                dl = min(time.time() + 1.4, T_HOP)
                X, _, _ = anneal_track(X, 180, 0.0012, 0.0002, 0.0016, 0.00016, 0.87, dl)
                dl2 = min(time.time() + 1.2, T_HOP)
                X = micro_shave(X, dl2, 5e-7)
                V = _max_cos(X)
                if V < bestV - 1e-15 and np.all(np.isfinite(X)):
                    bestV = V
                    bestX = X.copy()
            else:
                # shake + shell equalization
                sigma = 0.0035 if (it % 8 == 3) else 0.0012
                X = X + sigma * rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                dl = min(time.time() + 1.5, T_HOP)
                X, _, _ = anneal_track(X, 200, 0.005, 0.0006, 0.0026, 0.00026, 0.85, dl)
                dl2 = min(time.time() + 1.0, T_HOP)
                X = micro_shave(X, dl2, 4e-7)
                V = _max_cos(X)
                if V < bestV - 1e-15 and np.all(np.isfinite(X)):
                    bestV = V
                    bestX = X.copy()
        except Exception:
            continue

    # final polish
    try:
        remain = T_END - time.time() - 0.8
        if remain > 1.0:
            if remain > 36.0:
                remain = 36.0
            bestX = micro_shave(bestX.copy(), time.time() + remain, 5e-7)
            bestV = _max_cos(bestX)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
