import time
import numpy as np

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 453, 5

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
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    bestX = None
    bestV = float("inf")
    for cc in cands:
        v = max_cos(cc)
        if v < bestV:
            bestV = v
            bestX = cc.copy()

    # ---------- core kernels ----------
    def riesz_anneal(X, steps, s0, s1, lr0, lr1, mu):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        ls = np.log(s1 / s0) if s0 != s1 else 0.0
        ll = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(steps):
            if (it & 63) == 0 and time.time() > DEAD - 14:
                break
            frac = it / max(1, steps - 1)
            s = s0 * np.exp(ls * frac)
            lr = lr0 * np.exp(ll * frac)
            G = X @ X.T
            # need G for max tracking occasionally
            if (it & 127) == 0:
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                if m < bestV:
                    bestV = m
                    bestX = X.copy()
                # restore diag for D2 (diag D2=0, we zero weight anyway)
                np.fill_diagonal(G, 1.0)
            else:
                np.fill_diagonal(G, 1.0)
            D2 = 2.0 - 2.0 * G
            np.clip(D2, 1e-4, 4.0, out=D2)
            # W = D2^(-s/2-1)
            Lg = np.log(D2)
            e = (-s * 0.5 - 1.0) * Lg
            # cap exponent to avoid inf
            np.clip(e, -60.0, 230.0, out=e)
            W = np.exp(e, out=e)
            np.fill_diagonal(W, 0.0)
            Grad = W @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if not np.isfinite(mn) or mn < 1e-14:
                continue
            Gt /= mn
            V = mu * V + (1.0 - mu) * Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def softmax_anneal(X, steps, tau0, tau1, lr0, lr1, mu):
        nonlocal bestV, bestX
        V = np.zeros_like(X)
        lt = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        ll = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(steps):
            if (it & 63) == 0 and time.time() > DEAD - 12:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * np.exp(lt * frac)
            lr = lr0 * np.exp(ll * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = X.copy()
            P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if not np.isfinite(mn) or mn < 1e-14:
                continue
            Gt /= mn
            V = mu * V + (1.0 - mu) * Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def micro_shave(X, t_end, eps_init):
        nonlocal bestV, bestX
        eps = float(eps_init)
        bv = max_cos(X)
        # keep local best to allow elitist update
        while time.time() < t_end:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = X.copy()
            if m < bv:
                bv = m
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            xi = X[i].copy(); xj = X[j].copy()
            cij = float(xi @ xj)
            if cij > 1.0: cij = 1.0
            elif cij < -1.0: cij = -1.0
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d); ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d); tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj
            Ri = G[i, :].copy(); Rj = G[j, :].copy()
            # m_rest excluding i,j rows/cols
            # compute quickly: max of G with rows i,j masked
            # use precomputed per-row max excluding i,j?
            # brute: temporarily mask
            G[i, :] = -2.0; G[:, i] = -2.0
            G[j, :] = -2.0; G[:, j] = -2.0
            m_rest = float(np.max(G))
            # row maxima excluding i,j and self
            # Ri[i]=-2, Ri[j]=m ; similarly
            r_imax = -2.0
            r_jmax = -2.0
            # manual max loop in numpy (fast)
            # delete is alloc-heavy; do masked max
            # set entries to -2 then max
            Ri[i] = -2.0; Ri[j] = -2.0
            Rj[i] = -2.0; Rj[j] = -2.0
            r_imax = float(np.max(Ri)); r_jmax = float(np.max(Rj))
            base_i_only = r_jmax if r_jmax > m_rest else m_rest
            base_j_only = r_imax if r_imax > m_rest else m_rest
            best_m2 = m; bvi = None; bvj = None
            improved = False
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti; vi /= float(np.linalg.norm(vi))
                vj = xj - sc * tj; vj /= float(np.linalg.norm(vj))
                di = X @ vi; dj = X @ vj
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
                    best_m2 = mc; bvi = vi; bvj = vj
                    improved = True
                    break
            if not improved:
                # try single-sided with eps
                for sc in (eps,):
                    vi = xi - sc * ti; vi /= float(np.linalg.norm(vi))
                    di = X @ vi; di[i] = -2.0
                    mc = base_i_only
                    a = float(np.max(di))
                    if a > mc: mc = a
                    if mc < best_m2 - 1e-15:
                        best_m2 = mc; bvi = vi; bvj = None
                        improved = True
                        break
                if not improved:
                    for sc in (eps,):
                        vj = xj - sc * tj; vj /= float(np.linalg.norm(vj))
                        dj = X @ vj; dj[j] = -2.0
                        mc = base_j_only
                        a = float(np.max(dj))
                        if a > mc: mc = a
                        if mc < best_m2 - 1e-15:
                            best_m2 = mc; bvi = None; bvj = vj
                            improved = True
                            break
            if improved:
                if bvi is not None: X[i] = bvi
                if bvj is not None: X[j] = bvj
                if best_m2 < bv: bv = best_m2
                eps *= 1.03
                if eps > 3e-6: eps = 3e-6
            else:
                eps *= 0.85
                if eps < 2e-10:
                    k = i if (rng.integers(0, 2) == 0) else j
                    rnd = rng.standard_normal(d)
                    rnd -= float(rnd @ X[k]) * X[k]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        rnd /= rn
                        X[k] = X[k] + 5e-10 * rnd
                        X[k] /= float(np.linalg.norm(X[k]))
                    eps = 2e-10
        return X

    # ---------- stage 1: diverse Riesz basin hopping ----------
    sigmas = [0.06, 0.03, 0.10, 0.045, 0.15, 0.02, 0.08, 0.12, 0.035, 0.055]
    trial = 0
    hop_end = t0 + 118.0
    # one quick shell squeeze of start to calibrate
    # do not spend too long
    n_trials_done = 0
    while time.time() < hop_end:
        trial += 1
        if trial == 3:
            # one fully random restart basin
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = riesz_anneal(X, 1400, 8.0, 50.0, 0.004, 0.0008, 0.88)
            if time.time() > DEAD - 14:
                break
            X = softmax_anneal(X, 500, 8e-4, 5e-5, 0.0012, 0.00025, 0.88)
            n_trials_done += 1
            continue
        sig = sigmas[(trial - 1) % len(sigmas)]
        X = bestX.copy()
        # boost worst points to break shell coherence
        try:
            Gtmp = X @ X.T
            np.fill_diagonal(Gtmp, -2.0)
            pmax = np.max(Gtmp, axis=1)
            order = np.argsort(-pmax)
            noise = rng.standard_normal((n, d)) * sig
            nb = min(40, n)
            noise[order[:nb]] *= 2.0
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        except Exception:
            X = X + sig * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # alternate Riesz schedules to diversify
        if trial % 3 == 1:
            X = riesz_anneal(X, 1100, 6.0, 40.0, 0.0035, 0.0007, 0.9)
        elif trial % 3 == 2:
            X = riesz_anneal(X, 1100, 12.0, 70.0, 0.0030, 0.0006, 0.85)
        else:
            X = riesz_anneal(X, 900, 4.0, 25.0, 0.004, 0.001, 0.9)
        if time.time() > hop_end or time.time() > DEAD - 14:
            break
        # shell squeeze for this basin
        if trial % 2 == 1:
            X = softmax_anneal(X, 450, 5e-4, 4e-5, 0.001, 0.0002, 0.88)
        else:
            X = softmax_anneal(X, 450, 1.2e-3, 8e-5, 0.0015, 0.0003, 0.9)
        n_trials_done += 1
        # occasional micro burst
        if n_trials_done % 4 == 0 and time.time() < hop_end - 5:
            X = micro_shave(X, min(time.time() + 3.0, hop_end), 5e-7)
        if time.time() > DEAD - 14:
            break

    # ---------- stage 2: final polish on global best ----------
    remain = DEAD - time.time() - 1.5
    if remain < 4:
        return bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # alternate short softmax + shave
    t_final = time.time() + remain
    # first a medium shell anneal
    try:
        if remain > 12:
            bestX = softmax_anneal(bestX.copy(), 700, 3e-4, 2e-5, 0.0008, 0.00012, 0.9)
    except Exception:
        pass
    try:
        t_shave_end = t_final - 1.0
        if t_shave_end > time.time():
            bestX = micro_shave(bestX.copy(), t_shave_end, 6e-7)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
