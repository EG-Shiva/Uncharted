import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _softmax_adam(X0, steps, tau0, tau1, lr0, lr1):
    X = _norm(X0.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    best = _mc(X)
    bestX = X.copy()
    ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 and tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 and lr0 != lr1 else 0.0
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 5e-4:
            tau = 5e-4
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        if not np.all(np.isfinite(gt)):
            break
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X - step)
        if (s % 25 == 0) or (s == steps - 1):
            vv = _mc(X)
            if vv < best:
                best = vv
                bestX = X.copy()
    return bestX, best

def _hard_topK(X0, iters, step0, K0, K1):
    X = _norm(X0.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _mc(X)
    bestX = X.copy()
    step = step0
    for it in range(iters):
        f = it / max(1, iters - 1)
        K = int(round(K0 + (K1 - K0) * f))
        K = max(1, min(K, len(iu0)))
        G = X @ X.T
        c = G[iu0, iu1]
        if K < len(c):
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(len(c))
        I = iu0[idx]; J = iu1[idx]
        grad = np.zeros_like(X)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = rn[:, 0] > 1e-14
        direc = np.zeros_like(gt)
        if np.any(nz):
            direc[nz] = gt[nz] / rn[nz]
        else:
            step *= 0.5
            continue
        Xt = _norm(X - step * direc)
        vv = _mc(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.25, 0.006)
        else:
            step *= 0.55
            if step < 1e-12:
                X = bestX.copy()
                step = step0 * 0.25
                if it > iters * 0.7:
                    break
    return bestX, best

def _refine_void(others, vec, iters=120):
    v = vec / np.linalg.norm(vec)
    lr = 0.05
    tau = 0.006
    for _ in range(iters):
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ v) * v
        nr = float(np.linalg.norm(gt))
        if nr < 1e-12 or not np.isfinite(nr):
            break
        gt /= nr
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.06, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return v / np.linalg.norm(v)

def _void_insert(Xt, rng, krep, n_cand=2048):
    n, d = Xt.shape
    X = Xt.copy()
    for _ in range(krep):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        iw = int(np.argmax(per))
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(X, iw, axis=0)
        mdots = (cand @ others.T).max(axis=1)
        ib = int(np.argmin(mdots))
        vv = _refine_void(others, cand[ib], iters=100)
        X[iw] = vv
    return X

def _contact_set(X, tol=1e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(X.shape[0], 1)
    c = G[iu0, iu1]
    sel = np.where(c > m - tol)[0]
    # map to pair flat id i*n+j
    n = X.shape[0]
    s = set((iu0[k] * n + iu1[k]) for k in sel)
    return s, m

def _align_parents(X, Y):
    # alternating Hungarian + Procrustes, returns Y_aligned (n,d)
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return Y.copy()
    n, d = X.shape
    Y0 = _norm(Y.copy())
    Xa = _norm(X.copy())
    Q = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Ya = Y0 @ Q
        C = 1.0 - Xa @ Ya.T
        # clip for stability
        C = np.clip(C, 0.0, 3.0)
        r, c = linear_sum_assignment(C)
        # build perm: perm[row]=col
        p = np.empty(n, dtype=int)
        p[r] = c
        perm = p
        Yp = Y0[perm]
        M = Yp.T @ Xa
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        except Exception:
            break
        Q = U @ Vt
    Yp = Y0[perm]
    Ya = Yp @ Q
    return _norm(Ya)

def _child_slerp(X, Ya, alpha, rng, jitter=1e-3):
    Z = (1.0 - alpha) * X + alpha * Ya
    if jitter > 0:
        N = rng.standard_normal(Z.shape)
        # tangent jitter
        Zn = _norm(Z)
        dot = np.einsum('ij,ij->i', N, Zn)[:, None]
        Nt = N - dot * Zn
        Z = Zn + jitter * Nt
    return _norm(Z)

def _lp_polish(x, seconds):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _mc(x)
    x = _norm(x.copy())
    n, d = x.shape
    best = _mc(x)
    start = time.time()
    iu = np.triu_indices(n, 1)
    radius = 0.0015
    while time.time() - start < seconds and radius > 1e-13:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thresh = max(20.0 * radius, 5e-8)
        sel = cfull > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.argpartition(-cfull, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        csub = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _mc(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.3:
            break
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best_val = 1e9
    best_x = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                va = _mc(a)
                if va < best_val:
                    best_val = va
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_val = _mc(best_x)
    try:
        rec = _norm(np.asarray(np.load("record.npy"), dtype=np.float64))
    except Exception:
        rec = best_x.copy()
    base_best = best_x.copy()

    # ---- build starts ----
    starts = []
    starts.append(base_best.copy())
    starts.append(rec.copy())
    for sig in (0.03, 0.08, 0.15, 0.30):
        starts.append(_norm(base_best + sig * rng.standard_normal((n, d))))
    starts.append(_norm(rec + 0.20 * rng.standard_normal((n, d))))
    starts.append(_norm(rng.standard_normal((n, d))))
    starts.append(_norm(rng.standard_normal((n, d))))
    # hole-reinserted variants (applied just before polish)
    hole_flags = [0]*len(starts)
    # add two more hole starts
    starts.append(base_best.copy()); hole_flags.append(2)
    starts.append(base_best.copy()); hole_flags.append(5)

    archive = []  # list of (val, X, cset)
    # seed archive with base
    cs0, _ = _contact_set(base_best)
    archive.append((best_val, base_best.copy(), cs0))

    for si, s0 in enumerate(starts):
        if time.time() - t0 > 70.0:
            break
        try:
            X = s0.copy()
            if hole_flags[si] > 0:
                X = _void_insert(_norm(X), rng, krep=int(hole_flags[si]))
            # p-continuation-lite: two softmax stages
            X, _ = _softmax_adam(X, steps=300, tau0=0.02, tau1=0.005, lr0=0.005, lr1=0.002)
            if time.time() - t0 > 75.0:
                # skip heavy hard polish, quick eval
                vv = _mc(X)
                cs, _ = _contact_set(X)
                archive.append((vv, X.copy(), cs))
                if vv < best_val:
                    best_val = vv; best_x = X.copy()
                continue
            X, vp = _hard_topK(X, iters=250, step0=0.0025, K0=250, K1=60)
            # second short softmax at low temp to equalize cloud
            X2, _ = _softmax_adam(X, steps=150, tau0=0.004, tau1=0.001, lr0=0.002, lr1=0.0008)
            X2, vp2 = _hard_topK(X2, iters=150, step0=0.0015, K0=150, K1=40)
            if vp2 < vp:
                X, vp = X2, vp2
            cs, _ = _contact_set(X)
            archive.append((vp, X.copy(), cs))
            if vp < best_val:
                best_val = vp; best_x = X.copy()
        except Exception:
            continue

    # sort archive by val
    archive.sort(key=lambda t: t[0])
    if archive[0][0] < best_val:
        best_val = archive[0][0]; best_x = archive[0][1].copy()
    incumb_X = archive[0][1].copy()
    incumb_cs = archive[0][2]

    # ---- diversity selection: most different contact fingerprints ----
    def jacc_dist(a, b):
        if len(a) == 0 and len(b) == 0:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - inter / union

    # rank by diversity (break ties by value)
    scored = []
    for i, (vv, XX, cs) in enumerate(archive):
        if i == 0:
            continue
        jd = jacc_dist(incumb_cs, cs)
        scored.append((jd, vv, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    # keep top-3 diverse (need at least 1)
    div_idx = [s[2] for s in scored[:3]]
    if len(div_idx) == 0:
        # fallback: perturb incumbent
        div_idx = []
        for _ in range(2):
            Y = _norm(incumb_X + 0.25 * rng.standard_normal((n, d)))
            cs, _ = _contact_set(Y)
            archive.append((_mc(Y), Y, cs))
            div_idx.append(len(archive) - 1)

    alphas = (0.25, 0.5, 0.75)
    # ---- crossover generations ----
    for rnd in range(2):
        if time.time() - t0 > 130.0:
            break
        # refresh incumbent = global best
        archive.sort(key=lambda t: t[0])
        incumb_X = archive[0][1].copy()
        incumb_cs = archive[0][2]
        # recompute diverse partners (exclude incumbent)
        scored = []
        for i, (vv, XX, cs) in enumerate(archive):
            if i == 0:
                continue
            jd = jacc_dist(incumb_cs, cs)
            scored.append((jd, vv, i))
        scored.sort(key=lambda t: (-t[0], t[1]))
        partners = [s[2] for s in scored[:3]]
        if len(partners) == 0:
            break
        for pi in partners:
            if time.time() - t0 > 135.0:
                break
            Y = archive[pi][1]
            try:
                Ya = _align_parents(incumb_X, Y)
            except Exception:
                Ya = Y.copy()
            for alpha in alphas:
                if time.time() - t0 > 136.0:
                    break
                try:
                    Z = _child_slerp(incumb_X, Ya, alpha, rng, jitter=1e-3)
                    Zr, _ = _softmax_adam(Z, steps=220, tau0=0.012, tau1=0.0015, lr0=0.003, lr1=0.001)
                    Zr, vvr = _hard_topK(Zr, iters=220, step0=0.002, K0=200, K1=50)
                    cs, _ = _contact_set(Zr)
                    archive.append((vvr, Zr.copy(), cs))
                    if vvr < best_val:
                        best_val = vvr; best_x = Zr.copy()
                except Exception:
                    continue
        # keep archive bounded: best per fingerprint bucket + global bests
        # deduplicate by keeping best 16 by value plus most diverse
        archive.sort(key=lambda t: t[0])
        # retain top 12 + 4 most diverse from rest
        if len(archive) > 16:
            top = archive[:12]
            rest = archive[12:]
            # diversity vs incumbent
            inc = top[0][2]
            rest_sc = sorted(rest, key=lambda t: (-jacc_dist(inc, t[2]), t[0]))
            archive = top + rest_sc[:4]

    # ---- final repair: LP polish best few ----
    archive.sort(key=lambda t: t[0])
    for k in range(min(3, len(archive))):
        if time.time() - t0 > BUDGET - 6.0:
            break
        try:
            Xk = archive[k][1].copy()
            # short softmax touch before LP
            Xk, _ = _softmax_adam(Xk, steps=120, tau0=0.003, tau1=0.001, lr0=0.0015, lr1=0.0007)
            secs = max(1.0, min(6.0, (t0 + BUDGET - 2.0 - time.time()) / (3 - k)))
            Xl, vl = _lp_polish(Xk, seconds=secs)
            Xl, vl2 = _hard_topK(Xl, iters=250, step0=0.001, K0=120, K1=30)
            vl = vl2; Xl = Xl
            cs, _ = _contact_set(Xl)
            archive.append((vl, Xl.copy(), cs))
            if vl < best_val:
                best_val = vl; best_x = Xl.copy()
        except Exception:
            continue

    archive.sort(key=lambda t: t[0])
    best_x = archive[0][1].copy()
    best_x = _norm(best_x)
    return best_x
