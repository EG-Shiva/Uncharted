import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_sym(theta, a):
    # theta length 9: for j=0,1,2: alpha, beta, delta
    N = 151
    k = np.arange(N)
    w = 2.0 * np.pi / 151.0
    base1 = k * w  # (151,)
    base2a = (a * k * w)  # + delta later
    X = np.empty((3 * N, 5), dtype=np.float64)
    for j in range(3):
        alpha = float(theta[j * 3 + 0])
        beta = float(theta[j * 3 + 1])
        delta = float(theta[j * 3 + 2])
        ca = np.cos(alpha)
        r1 = ca * np.cos(beta)
        r2 = ca * np.sin(beta)
        z = np.sin(alpha)
        a1 = base1  # p1 fixed 0
        a2 = base2a + delta
        c1 = np.cos(a1); s1 = np.sin(a1)
        c2 = np.cos(a2); s2 = np.sin(a2)
        blk = np.empty((N, 5))
        blk[:, 0] = r1 * c1
        blk[:, 1] = r1 * s1
        blk[:, 2] = r2 * c2
        blk[:, 3] = r2 * s2
        blk[:, 4] = z
        X[j * N:(j + 1) * N, :] = blk
    # normalize (should already be unit)
    nrm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    nrm[nrm == 0] = 1.0
    X /= nrm
    return X

def _sym_max(theta, a):
    try:
        X = _build_sym(theta, a)
    except Exception:
        return 2.0
    return _max_cos(X)

def _sym_lse(theta, a, beta):
    X = _build_sym(theta, a)
    G = X @ X.T
    # upper triangle without diag
    iu = np.triu_indices(X.shape[0], 1)
    g = G[iu]
    m = float(np.max(g))
    # logsumexp
    e = np.exp(beta * (g - m))
    s = float(np.sum(e))
    return m + np.log(s + 1e-300) / beta

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

def _micro_shave(X, eps_init, t_end, max_iters=20000):
    n, d = X.shape
    bestV = _max_cos(X)
    eps = float(eps_init)
    it = 0
    while it < max_iters and time.time() < t_end:
        it += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = X[i].copy(); xj = X[j].copy()
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
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        ri = X @ xi; rj = X @ xj
        ri[i] = -2.0; ri[j] = -2.0
        rj[i] = -2.0; rj[j] = -2.0
        m_ri = float(np.max(ri)); m_rj = float(np.max(rj))
        m_rest_i = m_rest if m_rest > m_rj else m_rj
        m_rest_j = m_rest if m_rest > m_ri else m_ri
        best_m2 = m
        best_vi = None; best_vj = None
        mode = 0
        best_single_v = None
        for sc in (eps, eps * 0.5, eps * 0.22):
            vi = xi - sc * ti; vi /= (np.linalg.norm(vi) + 1e-30)
            vj = xj - sc * tj; vj /= (np.linalg.norm(vj) + 1e-30)
            di = X @ vi; dj = X @ vj
            di[i] = -2.0; di[j] = -2.0
            dj[i] = -2.0; dj[j] = -2.0
            mc = m_rest
            a1 = float(np.max(di)); 
            if a1 > mc: mc = a1
            b1 = float(np.max(dj));
            if b1 > mc: mc = b1
            c2 = float(vi @ vj)
            if c2 > mc: mc = c2
            if mc < best_m2 - 1e-15:
                best_m2 = mc; best_vi = vi; best_vj = vj; mode = 0
                break
        if best_m2 >= m - 1e-15:
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti; vi /= (np.linalg.norm(vi) + 1e-30)
                di = X @ vi; di[i] = -2.0
                mc = m_rest_i
                a1 = float(np.max(di))
                if a1 > mc: mc = a1
                if mc < best_m2 - 1e-15:
                    best_m2 = mc; mode = 1; best_single_v = vi; best_vi = vi
                    break
            for sc in (eps, eps * 0.5):
                vj = xj - sc * tj; vj /= (np.linalg.norm(vj) + 1e-30)
                dj = X @ vj; dj[j] = -2.0
                mc = m_rest_j
                a1 = float(np.max(dj))
                if a1 > mc: mc = a1
                if mc < best_m2 - 1e-15:
                    best_m2 = mc; mode = 2; best_single_v = vj; best_vj = vj
                    break
        if best_m2 < m - 1e-15:
            if mode == 0:
                X[i] = best_vi; X[j] = best_vj
            elif mode == 1:
                X[i] = best_single_v
            else:
                X[j] = best_single_v
            bestV = best_m2
            eps *= 0.98
            if eps < 1e-13: eps = 1e-13
        else:
            eps *= 0.55
            if eps < 5e-14:
                break
    return X, _max_cos(X)

def solve():
    import time as _tm
    t0 = _tm.time()
    DEAD = t0 + 172.0
    t_sym_end = t0 + 38.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.standard_normal((453, 5))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_inc = cands[bi].copy()
    best_inc_val = float(vals[bi])

    # ---- symmetric search over a ----
    a_list = [1, 10, 20, 37, 50]
    bounds = [(-1.3, 1.3), (0.05, 1.52), (0.0, 6.283185307179586)] * 3
    best_sym_theta = None
    best_sym_a = None
    best_sym_val = float("inf")
    # contact stats placeholder
    try:
        from scipy.optimize import differential_evolution, minimize
        has_scipy = True
    except Exception:
        has_scipy = False
    if has_scipy:
        for ai, aa in enumerate(a_list):
            if _tm.time() > t_sym_end - 2.0:
                break
            # quick random sampling first (cheap, diverse)
            try:
                n_rand = 120
                R = rng.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_rand, 9))
                for r in R:
                    if _tm.time() > t_sym_end - 1.0:
                        break
                    v = _sym_max(r, aa)
                    if v < best_sym_val:
                        best_sym_val = v
                        best_sym_theta = r.copy()
                        best_sym_a = aa
            except Exception:
                pass
            # DE with small budget
            try:
                if _tm.time() > t_sym_end - 1.0:
                    break
                def _f(th):
                    return _sym_max(th, aa)
                # time-box: use maxiter 12, popsize 10
                res = differential_evolution(_f, bounds, maxiter=12, popsize=10,
                                             tol=0.001, mutation=(0.5, 1.0),
                                             recombination=0.7, seed=ai,
                                             polish=False, updating='deferred',
                                             workers=1, disp=False)
                v = float(res.fun)
                if v < best_sym_val:
                    best_sym_val = v
                    best_sym_theta = np.asarray(res.x, dtype=float).copy()
                    best_sym_a = aa
            except Exception:
                continue
            if _tm.time() > t_sym_end:
                break
        # LSE refinement of best symmetric
        if best_sym_theta is not None and _tm.time() < t_sym_end:
            try:
                from scipy.optimize import minimize
                for beta in (400.0, 1500.0):
                    if _tm.time() > t_sym_end:
                        break
                    def _flse(th):
                        return _sym_lse(th, best_sym_a, beta)
                    r2 = minimize(_flse, best_sym_theta, method='L-BFGS-B',
                                  bounds=bounds, options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8})
                    th2 = np.asarray(r2.x, dtype=float)
                    v2 = _sym_max(th2, best_sym_a)
                    if v2 < best_sym_val:
                        best_sym_val = v2
                        best_sym_theta = th2.copy()
            except Exception:
                pass
    # build symmetric code
    Xsym = None
    Xsym_val = float("inf")
    if best_sym_theta is not None:
        try:
            Xsym = _build_sym(best_sym_theta, best_sym_a)
            Xsym_val = _max_cos(Xsym)
        except Exception:
            Xsym = None
    # diagnostics: contact/rattler comparison (for basin choice, cheap)
    def _stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - 1e-9
        # contacts per point
        C = (G >= thr)
        # each pair counted twice; per-point counts
        cnt = np.sum(C, axis=1)
        ratt = int(np.sum(cnt == 0))
        return m, ratt, float(np.mean(cnt))
    try:
        m_inc, ratt_inc, mean_inc = _stats(best_inc)
    except Exception:
        m_inc, ratt_inc, mean_inc = best_inc_val, 0, 0.0
    try:
        if Xsym is not None:
            m_s, ratt_s, mean_s = _stats(Xsym)
        else:
            m_s, ratt_s, mean_s = float("inf"), 0, 0.0
    except Exception:
        m_s, ratt_s, mean_s = float("inf"), 0, 0.0

    # build candidate basins: incumbent, broken-symmetry sym, hybrid
    basins = []
    basins.append(best_inc.copy())
    if Xsym is not None and np.all(np.isfinite(Xsym)):
        # break symmetry with small noise (two levels)
        for sig in (0.004, 0.001):
            Y = Xsym.copy() + rng.standard_normal(Xsym.shape) * sig
            Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-30)
            basins.append(Y)
        basins.append(Xsym.copy())
        # partially symmetrised incumbent: match each sym point to nearest incumbent
        try:
            D = Xsym @ best_inc.T  # 453x453
            match = np.argmax(D, axis=1)  # 453
            Xmatch = best_inc[match]
            Hyb = 0.5 * Xsym + 0.5 * Xmatch
            Hyb /= (np.linalg.norm(Hyb, axis=1, keepdims=True) + 1e-30)
            # add tiny jitter to avoid duplicates collapsing
            Hyb = Hyb + rng.standard_normal(Hyb.shape) * 1e-4
            Hyb /= (np.linalg.norm(Hyb, axis=1, keepdims=True) + 1e-30)
            basins.append(Hyb)
        except Exception:
            pass

    # evaluate basins, keep order by value, limit count to save time
    bvals = []
    for b in basins:
        try:
            bvals.append(_max_cos(b))
        except Exception:
            bvals.append(float("inf"))
    order = np.argsort(np.asarray(bvals))
    # keep at most 4 most promising: always include incumbent + best sym-related
    keep = []
    keep.append(basins[order[0]])
    for idx in order[1:]:
        if len(keep) >= 4:
            break
        keep.append(basins[idx])
    # ensure incumbent included
    # polish each basin briefly, track global best
    global_best_X = best_inc.copy()
    global_best_V = best_inc_val
    if Xsym_val < global_best_V:
        global_best_X = Xsym.copy()
        global_best_V = Xsym_val
    # time allocation for basin polishing
    t_polish_end = DEAD - 14.0
    # quick softmax + micro for each kept basin (round-robin)
    betas = [600.0, 1200.0, 350.0]
    for bi2, bX in enumerate(keep):
        if _tm.time() > t_polish_end:
            break
        X = bX.copy()
        # skip heavy work on clearly bad symmetric (> record+0.02) except one light try
        bv = _max_cos(X)
        light = bv > global_best_V + 0.02
        steps_list = [120] if light else [220, 220]
        for si, st in enumerate(steps_list):
            if _tm.time() > t_polish_end:
                break
            beta = betas[(bi2 + si) % len(betas)]
            lr0 = 0.0022 if si == 0 else 0.0012
            X = _softmax_relax(X, beta, st, lr0, lr0 * 0.1, 0.88, min(_tm.time() + 2.5, t_polish_end))
            # micro shave short
            X, vx = _micro_shave(X, 5e-8, min(_tm.time() + 1.2, t_polish_end), max_iters=2500)
            vx = _max_cos(X)
            if vx < global_best_V:
                global_best_V = vx
                global_best_X = X.copy()
        # one more micro if promising
        if _tm.time() < t_polish_end and _max_cos(X) < global_best_V + 0.001:
            X2, _ = _micro_shave(X, 8e-8, min(_tm.time() + 1.0, t_polish_end), max_iters=3000)
            v2 = _max_cos(X2)
            if v2 < global_best_V:
                global_best_V = v2
                global_best_X = X2.copy()
                X = X2
    # elitist hopping around global best to change topology (remaining time before final squeeze)
    t_hop_end = DEAD - 9.0
    it = 0
    while _tm.time() < t_hop_end:
        it += 1
        if t_hop_end - _tm.time() < 1.5:
            break
        G = global_best_X @ global_best_X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        flat = int(np.argmax(G))
        n = 453
        wi = flat // n; wj = flat % n
        y = global_best_X.copy()
        op = int(rng.integers(0, 3))
        inner_end = min(_tm.time() + 2.2, t_hop_end)
        try:
            if op == 0:
                # shake neighbourhood of worst pair
                row_i = G[wi].copy(); row_j = G[wj].copy()
                ti_idx = np.argpartition(row_i, -6)[-6:]
                tj_idx = np.argpartition(row_j, -6)[-6:]
                sset = set([int(wi), int(wj)] + list(map(int, list(ti_idx) + list(tj_idx))))
                for _ in range(4):
                    sset.add(int(rng.integers(0, n)))
                sidx = np.array(list(sset), dtype=int)
                sigma = float(rng.choice(np.array([0.002, 0.005, 0.0012])))
                noise = rng.standard_normal((len(sidx), 5)) * sigma
                for k2, idx in enumerate(sidx):
                    vv = noise[k2] - float(np.dot(noise[k2], y[idx])) * y[idx]
                    y[idx] = y[idx] + vv
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                y = _softmax_relax(y, 700.0, 200, 0.002, 0.0002, 0.88, inner_end)
            elif op == 1:
                # hole reinsertion of worse endpoint
                si = float(np.sum(np.sort(G[wi])[-5:]))
                sj = float(np.sum(np.sort(G[wj])[-5:]))
                victim = int(wi) if si > sj else int(wj)
                # find hole: random samples minimizing max dot
                S = rng.standard_normal((12000, 5))
                S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                Xe = y[mask]
                Dd = S @ Xe.T
                md = np.max(Dd, axis=1)
                h = S[int(np.argmin(md))].copy()
                for _ in range(30):
                    dots = Xe @ h
                    ci = int(np.argmax(dots))
                    xc = Xe[ci]; c = float(dots[ci])
                    tt = xc - c * h
                    tn = float(np.linalg.norm(tt))
                    if tn < 1e-12:
                        break
                    tt /= tn
                    h = h - 0.01 * tt
                    h /= (np.linalg.norm(h) + 1e-30)
                y[victim] = h
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                y = _softmax_relax(y, 500.0, 220, 0.0018, 0.00018, 0.88, inner_end)
            else:
                # separate worst pair directly then relax
                cij = float(np.dot(y[wi], y[wj]))
                cij = max(-1.0, min(1.0, cij))
                den = max(1e-12, np.sqrt(max(0.0, 1.0 - cij * cij)))
                dix = (y[wj] - cij * y[wi]) / den
                djx = (y[wi] - cij * y[wj]) / den
                s = float(rng.choice(np.array([0.001, 0.002, 0.004])))
                y[wi] = y[wi] - s * dix; y[wi] /= (np.linalg.norm(y[wi]) + 1e-30)
                y[wj] = y[wj] - s * djx; y[wj] /= (np.linalg.norm(y[wj]) + 1e-30)
                y = _softmax_relax(y, 900.0, 180, 0.0014, 0.00014, 0.90, inner_end)
        except Exception:
            continue
        if not np.all(np.isfinite(y)):
            continue
        try:
            y, vy = _micro_shave(y, 5e-8, min(_tm.time() + 0.8, t_hop_end), max_iters=2000)
        except Exception:
            continue
        vy = _max_cos(y)
        if vy < global_best_V:
            global_best_X = y
            global_best_V = vy
    # final long squeeze
    try:
        global_best_X, global_best_V = _micro_shave(global_best_X, 8e-8, DEAD - 0.5, max_iters=1000000)
    except Exception:
        pass
    global_best_X = global_best_X / (np.linalg.norm(global_best_X, axis=1, keepdims=True) + 1e-30)
    return global_best_X
