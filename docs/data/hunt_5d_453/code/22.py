import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _softmax_relax(X, beta, steps, lr, mu, t_end):
    n, d = X.shape
    v = np.zeros_like(X)
    for it in range(steps):
        if time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if not np.isfinite(m):
            break
        with np.errstate(over='ignore', invalid='ignore'):
            W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        # normalize to keep scale stable
        s = float(np.sum(W))
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-16:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        Y = X - lr * v
        yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        yn[yn == 0] = 1.0
        X = Y / yn
    return X

def _micro_shave(X, eps_init, t_end, bestV=None):
    n, d = X.shape
    if bestV is None:
        bestV = _max_cos(X)
    eps = float(eps_init)
    it = 0
    # local refs for speed
    while time.time() < t_end:
        it += 1
        # full Gram
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = X[i].copy()
        xj = X[j].copy()
        cij = float(xi @ xj)
        if cij > 1.0: cij = 1.0
        if cij < -1.0: cij = -1.0
        # tangent directions separating i,j
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            # random tangent
            ti = np.random.randn(d)
            ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = np.random.randn(d)
            tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        # row maxima before masking
        # copy rows for baseline
        Ri = G[i, :].copy()
        Rj = G[j, :].copy()
        # mask to get m_rest_both
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        # max of rows excluding i,j
        # Ri[j] is m, Ri[i]==-2
        r_imax = float(np.max(np.delete(Ri, [i, j])) ) if n > 2 else -2.0
        r_jmax = float(np.max(np.delete(Rj, [i, j])) ) if n > 2 else -2.0
        base_i_only = m_rest if m_rest > r_jmax else r_jmax
        base_j_only = m_rest if m_rest > r_imax else r_imax

        best_m2 = m
        best_vi = None
        best_vj = None
        mode = 0  # 0 both,1 i,2 j

        # try both moves at several scales
        for sc in (eps, eps * 0.5, eps * 0.25):
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
            # ignore i,j entries (stale)
            # max over k != i,j
            # manual max to avoid alloc heavy? use np.max on masked copy
            di[i] = -2.0; di[j] = -2.0
            dj[i] = -2.0; dj[j] = -2.0
            mc = m_rest
            a = float(np.max(di))
            if a > mc: mc = a
            b = float(np.max(dj))
            if b > mc: mc = b
            c2 = float(vi @ vj)
            if c2 > mc: mc = c2
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_vi = vi
                best_vj = vj
                mode = 0
                break  # take largest successful scale
        # single moves if both not much better, still try for finer
        if best_m2 >= m - 1e-15:
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti
                nvi = float(np.linalg.norm(vi))
                if nvi == 0 or not np.isfinite(nvi):
                    continue
                vi /= nvi
                di = X @ vi
                di[i] = -2.0
                # di[j] valid (vj not moved)
                mc = base_i_only
                a = float(np.max(di))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_vi = vi
                    best_vj = None
                    mode = 1
                    break
            for sc in (eps, eps * 0.5):
                vj = xj - sc * tj
                nvj = float(np.linalg.norm(vj))
                if nvj == 0 or not np.isfinite(nvj):
                    continue
                vj /= nvj
                dj = X @ vj
                dj[j] = -2.0
                mc = base_j_only
                a = float(np.max(dj))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_vi = None
                    best_vj = vj
                    mode = 2
                    break
        else:
            # both succeeded; optionally check single for even better (rare)
            pass

        if best_vi is not None or best_vj is not None:
            if mode == 0:
                X[i] = best_vi
                X[j] = best_vj
            elif mode == 1:
                X[i] = best_vi
            else:
                X[j] = best_vj
            if best_m2 < bestV:
                bestV = best_m2
            # slightly grow eps on success
            eps *= 1.02
            if eps > 2e-6:
                eps = 2e-6
        else:
            # no improvement: shrink
            eps *= 0.82
            if eps < 2e-10:
                # tiny random tangent kick to a rattler to change tie
                # kick worst point by eps-scale noise
                k = i if (it % 2 == 0) else j
                rnd = np.random.randn(d)
                rnd -= float(rnd @ X[k]) * X[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    X[k] = X[k] + 5e-10 * rnd
                    X[k] /= float(np.linalg.norm(X[k]))
                eps = 2e-10
                # occasional check time
            # if eps got too small without progress, keep grinding;
            # bulk may need relaxation -> return to caller periodically
        # periodic time check is in while; also adapt if stuck long
        if it % 200 == 0:
            if time.time() > t_end:
                break
            # if current max far above best, reset to best region? caller handles
    return X, bestV, eps

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    t_end_all = t0 + TIME_LIMIT
    np.random.seed(0)
    rng = np.random.default_rng(0)
    cands = []
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
    except Exception:
        pass
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
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

    # Stage 1: intensive squeeze of loaded best (25 s)
    try:
        X1, V1, _ = _micro_shave(bestX.copy(), 5e-7, t0 + 28.0, bestV)
        if V1 < bestV and np.all(np.isfinite(X1)):
            bestV = V1
            bestX = X1.copy()
    except Exception:
        pass

    # Stage 2: basin hopping with alternating large/small moves
    it = 0
    # reserve 35 s for final polish
    hop_end = t0 + 135.0
    while time.time() < hop_end and time.time() < t_end_all - 40.0:
        it += 1
        X = bestX.copy()
        mode = it % 5
        if mode == 0:
            # larger shake + anneal to change topology
            sigma = 0.004 if (it % 10 == 0) else 0.0015
            X += sigma * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # short anneal: moderate beta
            try:
                X = _softmax_relax(X, beta=300.0, steps=120, lr=0.002, mu=0.85, t_end=hop_end)
                X = _softmax_relax(X, beta=1500.0, steps=80, lr=0.0006, mu=0.9, t_end=hop_end)
            except Exception:
                pass
            # shave burst 4 s
            try:
                X, V, _ = _micro_shave(X, 4e-7, min(time.time() + 4.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
        elif mode == 1 or mode == 2:
            # micro hop: tiny global jitter + tiny relax
            sigma = 2.5e-7 if mode == 1 else 8e-7
            X += sigma * rng.standard_normal((n, d))
            # also kick worst pair slightly more
            try:
                Gtmp = X @ X.T
                np.fill_diagonal(Gtmp, -2.0)
                flat = int(np.argmax(Gtmp))
                wi = flat // n; wj = flat % n
                X[wi] += sigma * 3.0 * rng.standard_normal(d)
                X[wj] += sigma * 3.0 * rng.standard_normal(d)
            except Exception:
                pass
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=3000000.0, steps=40, lr=2.5e-7, mu=0.8, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 3.5e-7, min(time.time() + 5.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
        else:
            # worst-point reinsertion into hole + small anneal
            try:
                Gtmp = X @ X.T
                np.fill_diagonal(Gtmp, -2.0)
                per = np.max(Gtmp, axis=1)
                w = int(np.argmax(per))
            except Exception:
                w = int(rng.integers(0, n))
            M = 8000
            cand = rng.standard_normal((M, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ X.T
            dots[:, w] = -2.0
            scores = np.max(dots, axis=1)
            bi = int(np.argmin(scores))
            if it % 6 == 0:
                part = np.argpartition(scores, 4)[:4]
                bi = int(rng.choice(part))
            X[w] = cand[bi]
            X[w] += 1e-7 * rng.standard_normal(d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            try:
                X = _softmax_relax(X, beta=800.0, steps=100, lr=0.001, mu=0.85, t_end=hop_end)
            except Exception:
                pass
            try:
                X, V, _ = _micro_shave(X, 4e-7, min(time.time() + 4.0, hop_end), float("inf"))
            except Exception:
                continue
            V = _max_cos(X)
        try:
            if np.all(np.isfinite(X)) and V < bestV - 1e-14:
                bestV = V
                bestX = X.copy()
        except Exception:
            pass

    # Stage 3: final long polish on best
    remain = t_end_all - time.time() - 1.0
    if remain < 5.0:
        remain = 5.0
    if remain > 38.0:
        remain = 38.0
    try:
        XP, VP, _ = _micro_shave(bestX.copy(), 3e-7, time.time() + remain, bestV)
        if VP < bestV and np.all(np.isfinite(XP)):
            bestV = VP
            bestX = XP.copy()
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
