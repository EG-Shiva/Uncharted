import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _softmax_relax(X, beta, steps, lr0, lr1, mu, t_end):
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
        Z = float(np.sum(W) * 0.5)
        if not np.isfinite(Z) or Z <= 0:
            break
        grad = (W @ X) / Z
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-14:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        Y = X - lr * v
        yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        yn[yn == 0] = 1.0
        X = Y / yn
    return X

def _fine_greedy(x, bestV, t_end, eps_init=1.2e-7, max_iters=1000000):
    n, d = x.shape
    eps = float(eps_init)
    it = 0
    # preallocate
    while it < max_iters and time.time() < t_end:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = x[i].copy()
        xj = x[j].copy()
        cij = float(xi @ xj)
        cij = max(-1.0, min(1.0, cij))
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
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
        # m_rest excluding i,j
        # mask via temp copy of two rows: compute max without i,j
        # fast: use G with rows/cols masked
        # G is n x n, set rows/cols to -2
        G[i, :] = -2.0
        G[:, i] = -2.0
        G[j, :] = -2.0
        G[:, j] = -2.0
        m_rest = float(np.max(G))
        # row dots for single-move m_rest
        # ri = x @ xi, rj = x @ xj
        ri = x @ xi
        rj = x @ xj
        ri[i] = -2.0; ri[j] = -2.0
        rj[i] = -2.0; rj[j] = -2.0
        m_ri_max = float(np.max(ri))
        m_rj_max = float(np.max(rj))
        m_rest_i = m_rest if m_rest > m_rj_max else m_rj_max
        m_rest_j = m_rest if m_rest > m_ri_max else m_ri_max

        best_m2 = m
        best_vi = None
        best_vj = None
        best_single = 0  # 0 both, 1 i-only, 2 j-only
        best_single_v = None

        # both moves
        for scale in (eps, eps * 0.5, eps * 0.22):
            vi = xi - scale * ti
            nv = float(np.linalg.norm(vi))
            if nv == 0 or not np.isfinite(nv):
                continue
            vi /= nv
            vj = xj - scale * tj
            nv2 = float(np.linalg.norm(vj))
            if nv2 == 0 or not np.isfinite(nv2):
                continue
            vj /= nv2
            di = x @ vi
            dj = x @ vj
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
                best_single = 0
        # single moves (only if both failed to improve much, still try)
        # try i-only
        for scale in (eps, eps * 0.5):
            vi = xi - scale * ti
            nv = float(np.linalg.norm(vi))
            if nv == 0 or not np.isfinite(nv):
                continue
            vi /= nv
            di = x @ vi
            di[i] = -2.0
            # di[j] is vi·old_xj valid (j not moved)
            mc = m_rest_i
            a = float(np.max(di))
            if a > mc:
                mc = a
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_single = 1
                best_single_v = vi
                best_vi = vi
                best_vj = None
        for scale in (eps, eps * 0.5):
            vj = xj - scale * tj
            nv2 = float(np.linalg.norm(vj))
            if nv2 == 0 or not np.isfinite(nv2):
                continue
            vj /= nv2
            dj = x @ vj
            dj[j] = -2.0
            mc = m_rest_j
            a = float(np.max(dj))
            if a > mc:
                mc = a
            if mc < best_m2 - 1e-15:
                best_m2 = mc
                best_single = 2
                best_single_v = vj
                best_vi = None
                best_vj = vj

        if best_m2 < m - 1e-15:
            if best_single == 0:
                x[i] = best_vi
                x[j] = best_vj
            elif best_single == 1:
                x[i] = best_single_v
            else:
                x[j] = best_single_v
            bestV = best_m2
            # keep eps (slightly shrink to follow gap)
            eps *= 0.98
            if eps < 1e-13:
                eps = 1e-13
        else:
            eps *= 0.55
            if eps < 5e-14:
                break
            # tiny random kick when very stuck to change active set
            if (it % 400) == 0:
                k = i if (it % 800) == 0 else j
                rnd = np.random.randn(d)
                rnd -= float(rnd @ x[k]) * x[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    xn = x[k] + 1e-12 * rnd
                    xn /= np.linalg.norm(xn)
                    # only accept if not worse
                    # quick check: max dot of xn vs others
                    dd = float(np.max(x @ xn)) if False else None
                    x[k] = xn
    # final exact value
    fv = _max_cos(x)
    if fv < bestV:
        bestV = fv
    return x, bestV

def _find_hole(X, exclude, rng, n_samples=20000):
    n, d = X.shape
    S = rng.standard_normal((n_samples, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-30
    mask = np.ones(n, dtype=bool)
    mask[exclude] = False
    Xe = X[mask]
    D = S @ Xe.T
    md = np.max(D, axis=1)
    bi = int(np.argmin(md))
    h = S[bi].copy()
    for _ in range(50):
        dots = Xe @ h
        ci = int(np.argmax(dots))
        xc = Xe[ci]
        c = float(dots[ci])
        t = xc - c * h
        tn = float(np.linalg.norm(t))
        if tn < 1e-12:
            break
        t /= tn
        h = h - 0.01 * t
        h /= np.linalg.norm(h) + 1e-30
    return h

def solve():
    t0 = time.time()
    t_end_total = t0 + 172.0
    t_hop_end = t0 + 122.0
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

    # Stage 1: immediate squeeze of isolated pair (guaranteed gain)
    try:
        bestX, bestV = _fine_greedy(bestX, bestV, min(time.time() + 9.0, t_hop_end), eps_init=1.2e-7)
    except Exception:
        pass

    betas = [250.0, 600.0, 1200.0, 400.0, 900.0]
    sigmas = [0.004, 0.008, 0.002, 0.006, 0.0012]
    it = 0
    iu_r = None
    # hopping to change topology
    while time.time() < t_hop_end:
        it += 1
        if t_hop_end - time.time() < 1.0:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        flat = int(np.argmax(G))
        wi = flat // n
        wj = flat % n
        op = int(rng.integers(0, 3))
        inner_end = min(time.time() + 3.0, t_hop_end)
        beta = float(betas[it % len(betas)])
        sigma = float(sigmas[it % len(sigmas)])
        y = bestX.copy()
        try:
            if op == 0:
                row_i = G[wi].copy(); row_i[wi] = -2
                row_j = G[wj].copy(); row_j[wj] = -2
                ti = np.argpartition(row_i, -8)[-8:]
                tj = np.argpartition(row_j, -8)[-8:]
                sset = set([int(wi), int(wj)])
                for t_ in list(ti) + list(tj):
                    sset.add(int(t_))
                for _ in range(6):
                    sset.add(int(rng.integers(0, n)))
                sidx = np.array(list(sset), dtype=int)
                noise = rng.standard_normal((len(sidx), d)) * sigma
                for k2, idx in enumerate(sidx):
                    vv = noise[k2] - float(np.dot(noise[k2], y[idx])) * y[idx]
                    y[idx] = y[idx] + vv
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _softmax_relax(y, beta, 240, 0.0022, 0.00022, 0.88, inner_end)
            elif op == 1:
                si = float(np.sum(np.sort(G[wi])[-6:]))
                sj = float(np.sum(np.sort(G[wj])[-6:]))
                victim = int(wi) if si > sj else int(wj)
                h = _find_hole(bestX, victim, rng)
                y[victim] = h
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _softmax_relax(y, beta, 280, 0.0018, 0.00018, 0.88, inner_end)
            else:
                cij = float(np.dot(y[wi], y[wj]))
                cij = max(-1.0, min(1.0, cij))
                den = max(1e-12, np.sqrt(max(0.0, 1.0 - cij * cij)))
                # unit tangent directions
                dix = (bestX[wj] - cij * bestX[wi]) / den
                djx = (bestX[wi] - cij * bestX[wj]) / den
                s = float(rng.choice(np.array([0.0009, 0.0018, 0.003])))
                y[wi] = y[wi] - s * dix
                y[wi] /= np.linalg.norm(y[wi])
                y[wj] = y[wj] - s * djx
                y[wj] /= np.linalg.norm(y[wj])
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _softmax_relax(y, beta, 200, 0.0014, 0.00014, 0.90, inner_end)
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
            continue
        # short micro-shave to project to local optimum
        try:
            y, vy = _fine_greedy(y, _max_cos(y), min(time.time() + 0.9, t_hop_end), eps_init=5e-8, max_iters=2500)
        except Exception:
            vy = _max_cos(y)
        if vy < bestV:
            bestX = y
            bestV = vy

    # final long squeeze
    try:
        bestX, bestV = _fine_greedy(bestX, bestV, t_end_total - 0.5, eps_init=8e-8, max_iters=1000000)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
