import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _edges_and_max(X, thr=1e-7):
    n = X.shape[0]
    G = X @ X.T
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    m = float(c.max())
    sel = c > (m - thr)
    return m, sel, G

def _jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return 1.0 - float(inter) / float(union)

def _smooth_polish(X, steps, a0, a1, lr0, deadline=None):
    n, d = X.shape
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    bestX = X.copy()
    for t in range(1, steps + 1):
        if deadline is not None and (t % 64 == 0) and time.time() > deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi * frac))
        lr = max(lr, lr0 * 0.05)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        if gmax < best:
            best = gmax
            bestX = X.copy()
        E = alpha * (G - gmax)
        W = np.exp(np.clip(E, -30, 0.0))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1 - b1) * Gt
        V = b2 * V + (1 - b2) * (Gt * Gt)
        Mh = M / (1 - b1 ** t)
        Vh = V / (1 - b2 ** t)
        step = Mh / (np.sqrt(Vh) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = _max_cos(X)
    if v < best:
        best = v
        bestX = X.copy()
    else:
        # ensure bestX is returned but also consider final X
        pass
    return bestX, best

def _lp_polish(X, budget, t_deadline=1e18):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - t0 < budget and time.time() < t_deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_full = G[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_full > thresh
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c_full[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[idx], J[idx], cc[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best

def solve():
    t_start = time.time()
    T_END = t_start + 172.0
    T_EXPL = t_start + 146.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (131, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((131, 9))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    best_overall = 1e9
    bestX_overall = None
    for a in cands:
        v = _max_cos(a)
        if v < best_overall:
            best_overall = v
            bestX_overall = a.copy()
    n, d = bestX_overall.shape
    # archive of 3 elites: lists
    arch_X = [bestX_overall.copy()]
    arch_s = [best_overall]
    _, sel0, _ = _edges_and_max(bestX_overall, thr=1e-7)
    arch_e = [sel0]
    trial = 0
    ks = [3, 5, 8]
    while time.time() < T_EXPL:
        # choose base from archive round-robin
        base_idx = trial % len(arch_X)
        baseX = arch_X[base_idx]
        k = ks[trial % len(ks)]
        # contact degrees at 1e-7
        m_base, sel_base, G_base = _edges_and_max(baseX, thr=1e-7)
        iu0 = np.triu_indices(n, 1)
        # degree from full graph threshold
        Gtmp = baseX @ baseX.T
        np.fill_diagonal(Gtmp, -2.0)
        deg = np.sum(Gtmp > (m_base - 1e-7), axis=1)
        # biased sampling toward degrees > 20 : weight (deg+1)^3
        w = (deg.astype(float) + 1.0) ** 3.0
        # extra boost for >20 to follow direction explicitly
        w = np.where(deg > 20, w * 3.0, w)
        ws = w.sum()
        if ws <= 0 or not np.isfinite(ws):
            subset = rng.choice(n, size=k, replace=False)
        else:
            w = w / ws
            try:
                subset = rng.choice(n, size=k, replace=False, p=w)
            except Exception:
                subset = np.argsort(-deg)[:k]
        order = rng.permutation(subset)
        Xnew = baseX.copy()
        active_mask = np.ones(n, dtype=bool)
        active_mask[subset] = False
        # sequential reinsertion into jointly recomputed largest holes
        for idx in order:
            if time.time() > T_EXPL:
                break
            act = np.where(active_mask)[0]
            Xa = Xnew[act]
            Ncand = 6000
            C = rng.standard_normal((Ncand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            dots = C @ Xa.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            z = C[bi].copy()
            # single-point softmax refinement against retained+reinserted
            for it in range(220):
                dvec = Xa @ z
                mx = float(dvec.max())
                wv = np.exp(50.0 * (dvec - mx))
                s = float(wv.sum()) + 1e-300
                g = (wv[:, None] * Xa).sum(axis=0) / s
                g = g - float(g @ z) * z
                gn = float(np.linalg.norm(g))
                if gn < 1e-12:
                    break
                step = 0.06 / (1.0 + it * 0.02)
                z = z - step * g / (gn + 1e-12)
                nz = float(np.linalg.norm(z))
                if nz < 1e-12:
                    break
                z = z / nz
            Xnew[idx] = z
            active_mask[idx] = True
        # moderate (not fully converged) weighted tangential polish
        if k == 3:
            steps, a0, a1, lr0 = 1100, 35.0, 260.0, 0.0030
        elif k == 5:
            steps, a0, a1, lr0 = 1400, 28.0, 240.0, 0.0032
        else:
            steps, a0, a1, lr0 = 1700, 22.0, 220.0, 0.0035
        Xs, vs = _smooth_polish(Xnew, steps, a0, a1, lr0, deadline=T_EXPL)
        m_new, sel_new, _ = _edges_and_max(Xs, thr=1e-7)
        # Jaccard vs old (base) graph
        jd_base = _jaccard(sel_base, sel_new)
        # min Jaccard to archive
        min_jd = min(_jaccard(e, sel_new) for e in arch_e) if len(arch_e) > 0 else 1.0
        # track overall best regardless
        if vs < best_overall - 1e-15:
            best_overall = vs
            bestX_overall = Xs.copy()
        # archive maintenance: keep 3 elites mutually dissimilar + low scoring
        # accept if competitive (within 8e-3 of overall best) to allow novel basins
        if vs < best_overall + 0.008:
            if len(arch_X) < 3:
                if min_jd > 0.02 or vs < min(arch_s) - 1e-12:
                    arch_X.append(Xs.copy()); arch_s.append(vs); arch_e.append(sel_new)
                else:
                    # similar but better than its nearest neighbour -> replace it
                    dists = [_jaccard(e, sel_new) for e in arch_e]
                    j = int(np.argmin(dists))
                    if vs < arch_s[j] - 1e-12:
                        arch_X[j] = Xs.copy(); arch_s[j] = vs; arch_e[j] = sel_new
            else:
                worst = int(np.argmax(arch_s))
                if vs < arch_s[worst] - 1e-9 and min_jd > 0.01:
                    arch_X[worst] = Xs.copy(); arch_s[worst] = vs; arch_e[worst] = sel_new
                else:
                    # if much better than a similar member, replace that member
                    dists = [_jaccard(e, sel_new) for e in arch_e]
                    j = int(np.argmin(dists))
                    if dists[j] < 0.05 and vs < arch_s[j] - 5e-10:
                        arch_X[j] = Xs.copy(); arch_s[j] = vs; arch_e[j] = sel_new
        trial += 1
        if time.time() >= T_EXPL:
            break
    # final LP polish on overall best
    rem = T_END - time.time()
    if rem > 1.2 and bestX_overall is not None:
        b = min(rem - 0.8, 24.0)
        if b > 0.8:
            Xp, vp = _lp_polish(bestX_overall, b, T_END)
            if vp < best_overall:
                best_overall = vp
                bestX_overall = Xp.copy()
    bestX_overall = bestX_overall / np.linalg.norm(bestX_overall, axis=1, keepdims=True)
    return bestX_overall
