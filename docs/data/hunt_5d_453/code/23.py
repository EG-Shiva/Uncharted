import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)

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

    bestV = float("inf")
    bestX = None
    for cc in cands:
        v = _max_cos(cc)
        if v < bestV:
            bestV = v
            bestX = cc.copy()
    n, d = bestX.shape

    # ---------- helpers (closure over bestV/bestX via lists) ----------
    state_V = [bestV]
    state_X = [bestX.copy()]
    cur = bestX.copy()

    def anneal_from(X, steps, tau0, tau1, lr0, lr1, mu, deadline):
        bv = state_V[0]
        bx = state_X[0]
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for it in range(steps):
            if it % 50 == 0 and time.time() > deadline:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bv:
                bv = m
                bx = X.copy()
                state_V[0] = bv
                state_X[0] = bx
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

    def largest_hole(Xref, exclude=-1, n_cand=15000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Xref.T
        if exclude >= 0:
            S[:, exclude] = -2.0
        vals = np.max(S, axis=1)
        k = int(np.argmin(vals))
        return C[k]

    def refine_single(xi, B, idx, steps=120, tau=3e-4, lr=0.02, mu=0.8):
        v = np.zeros(d)
        for it in range(steps):
            g = B @ xi
            g[idx] = -2.0
            m = float(np.max(g))
            w = np.exp((g - m) / tau)
            w[idx] = 0.0
            grad = w @ B
            grad = grad - float(grad @ xi) * xi
            gn = float(np.linalg.norm(grad))
            if gn < 1e-14:
                break
            grad /= gn
            v = mu * v + (1.0 - mu) * grad
            frac = it / max(1, steps - 1)
            lri = lr * (0.3 + 0.7 * (1.0 - frac))
            xi = xi - lri * v
            xi /= np.linalg.norm(xi)
        return xi

    def greedy_shave(X, deadline, max_iters=4000, eps_init=8e-4):
        bv = state_V[0]
        bx = state_X[0]
        eps = float(eps_init)
        it = 0
        while it < max_iters and time.time() < deadline:
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bv:
                bv = m
                bx = X.copy()
                state_V[0] = bv
                state_X[0] = bx
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            xi = X[i]
            xj = X[j]
            cij = float(xi @ xj)
            if cij > 1.0:
                cij = 1.0
            elif cij < -1.0:
                cij = -1.0
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d)
                ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d)
                tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj
            # m_rest among untouched pairs
            G[i, :] = -2.0
            G[:, i] = -2.0
            G[j, :] = -2.0
            G[:, j] = -2.0
            m_rest = float(np.max(G))
            improved = False
            for scale in (eps, eps * 0.5, eps * 0.22):
                vi = xi - scale * ti
                nvi = float(np.linalg.norm(vi))
                if nvi == 0 or not np.isfinite(nvi):
                    continue
                vi /= nvi
                vj = xj - scale * tj
                nvj = float(np.linalg.norm(vj))
                if nvj == 0 or not np.isfinite(nvj):
                    continue
                vj /= nvj
                di = X @ vi
                di[i] = -2.0
                di[j] = -2.0
                dj = X @ vj
                dj[i] = -2.0
                dj[j] = -2.0
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
                if mc < m - 1e-13:
                    X[i] = vi
                    X[j] = vj
                    if mc < bv:
                        bv = mc
                        bx = X.copy()
                        state_V[0] = bv
                        state_X[0] = bx
                    eps = min(scale * 1.08, 3e-3)
                    improved = True
                    break
            if not improved:
                # try single-sided moves
                for scale in (eps,):
                    vi = xi - scale * ti
                    nvi = float(np.linalg.norm(vi))
                    if nvi == 0 or not np.isfinite(nvi):
                        continue
                    vi /= nvi
                    di = X @ vi
                    di[i] = -2.0
                    mc = m_rest
                    # j row still valid: vi vs old xj
                    cj = float(vi @ xj)
                    if cj > mc:
                        mc = cj
                    a = float(np.max(di))
                    if a > mc:
                        mc = a
                    if mc < m - 1e-13:
                        X[i] = vi
                        if mc < bv:
                            bv = mc
                            bx = X.copy()
                            state_V[0] = bv
                            state_X[0] = bx
                        eps = min(scale * 1.05, 3e-3)
                        improved = True
                        break
                if not improved:
                    for scale in (eps,):
                        vj = xj - scale * tj
                        nvj = float(np.linalg.norm(vj))
                        if nvj == 0 or not np.isfinite(nvj):
                            continue
                        vj /= nvj
                        dj = X @ vj
                        dj[j] = -2.0
                        mc = m_rest
                        cj = float(vj @ xi)
                        if cj > mc:
                            mc = cj
                        a = float(np.max(dj))
                        if a > mc:
                            mc = a
                        if mc < m - 1e-13:
                            X[j] = vj
                            if mc < bv:
                                bv = mc
                                bx = X.copy()
                                state_V[0] = bv
                                state_X[0] = bx
                            eps = min(scale * 1.05, 3e-3)
                            improved = True
                            break
                if not improved:
                    eps *= 0.94
                    if eps < 2e-07:
                        eps = 2e-07
        return X

    # ---------- Phase 1: bulk-aware hot anneal (raise then lower bulk) ----------
    cur = state_X[0].copy()
    cur = anneal_from(cur, 3500, 6e-3, 6e-4, 0.008, 0.002, 0.90, min(DEAD - 130, t0 + 38))
    # quick cool to recover max
    cur = state_X[0].copy()
    cur = anneal_from(cur, 2000, 6e-4, 8e-5, 0.002, 0.0005, 0.90, min(DEAD - 120, t0 + 55))

    # ---------- Phase 2: basin hopping ----------
    sigmas = [0.016, 0.010, 0.006, 0.0035, 0.002, 0.012, 0.007, 0.004]
    hop = 0
    HOP_END = min(DEAD - 22, t0 + 148)
    while time.time() < HOP_END:
        hop += 1
        base = state_X[0]
        # pointwise max to boost worst points
        if hop % 3 == 0:
            # hole reinsertion jump on worst point
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            idx = int(np.argmax(pmax))
            hole = largest_hole(base, exclude=idx, n_cand=15000)
            # light single-point refine against base
            hole = refine_single(hole, base, idx, steps=120, tau=3e-4, lr=0.03, mu=0.8)
            trial = base.copy()
            trial[idx] = hole
            # occasionally also kick second worst slightly
            if hop % 6 == 0:
                G2 = trial @ trial.T
                np.fill_diagonal(G2, -2.0)
                p2 = np.max(G2, axis=1)
                idx2 = int(np.argmax(p2)) if int(np.argmax(p2)) != idx else int(np.argsort(-p2)[1])
                nz = rng.standard_normal(d) * 0.01
                nz -= float(nz @ trial[idx2]) * trial[idx2]
                tt = trial[idx2] + nz
                tt /= np.linalg.norm(tt)
                trial[idx2] = tt
            cur = anneal_from(trial, 900, 7e-4, 7e-5, 0.0018, 0.0004, 0.90, HOP_END)
        else:
            sig = sigmas[(hop - 1) % len(sigmas)]
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            pmax = np.max(G, axis=1)
            order = np.argsort(-pmax)
            noise = rng.standard_normal(base.shape) * sig
            if sig > 0.005:
                noise[order[:25]] *= 2.5
            else:
                noise[order[:15]] *= 2.0
            noise -= np.sum(noise * base, axis=1, keepdims=True) * base
            trial = base + noise
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            # alternate temperature: hot bulk vs cold minimax
            if hop % 2 == 0:
                cur = anneal_from(trial, 1100, 1.2e-3, 1e-4, 0.0025, 0.0005, 0.90, HOP_END)
            else:
                cur = anneal_from(trial, 1100, 3e-4, 4e-5, 0.0012, 0.0003, 0.90, HOP_END)
        if time.time() > HOP_END:
            break
        # occasional fast shave to keep worst pair tight (cheap, 250 iters)
        if hop % 4 == 0 and time.time() < HOP_END - 3:
            cur = state_X[0].copy()
            cur = greedy_shave(cur, min(HOP_END, time.time() + 2.5), max_iters=250, eps_init=5e-4)

    # ---------- Phase 3: fine minimax cool ----------
    cur = state_X[0].copy()
    cur = anneal_from(cur, 2500, 2.5e-4, 2.5e-5, 0.001, 0.0002, 0.90, DEAD - 6)
    cur = state_X[0].copy()
    cur = greedy_shave(cur, DEAD - 0.6, max_iters=4000, eps_init=6e-4)

    out = state_X[0].copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    # strict finite guard
    out = np.where(np.isfinite(out), out, 0.0)
    # renormalize after guard
    nn = np.linalg.norm(out, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    out /= nn
    return out
