import time
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if cnt > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return x

def _diagnose(X, tol=3e-4):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gmax = float(G.max())
    # adapt tol to keep pairs manageable
    for _ in range(5):
        mask = G > gmax - tol
        # count upper
        cnt = int((np.triu(mask, 1)).sum())
        if cnt <= 4500 or tol <= 5e-5:
            break
        tol *= 0.5
    iu = np.triu_indices(n, 1)
    # neighbor lists
    res = np.zeros(n)
    tot = np.zeros(n)
    mcnt = np.zeros(n, dtype=int)
    # pair stress dict
    pair_w = {}
    for i in range(n):
        nbrs = np.where(G[i] > gmax - tol)[0]
        nbrs = nbrs[nbrs != i]
        mcnt[i] = len(nbrs)
        if len(nbrs) == 0:
            res[i] = 0.0
            tot[i] = 0.0
            continue
        Gij = G[i, nbrs]  # (m,)
        # V columns = xj - Gij*xi
        V = (X[nbrs] - Gij[:, None] * X[i]).T  # d x m
        r = V.sum(axis=1)  # d
        b = -r
        try:
            u, rnorm = nnls(V, b)
        except Exception:
            u = np.zeros(len(nbrs))
            rnorm = float(np.linalg.norm(r))
        res[i] = float(rnorm)
        w = 1.0 + u
        tot[i] = float(w.sum())
        for k, j in enumerate(nbrs):
            a, bb = (i, int(j)) if i < j else (int(j), i)
            # accumulate both sides, will average later
            if (a, bb) not in pair_w:
                pair_w[(a, bb)] = []
            pair_w[(a, bb)].append(float(w[k]))
    # average two sides
    pairs = []
    wvals = []
    for (a, bb), lst in pair_w.items():
        wv = float(np.mean(lst))
        pairs.append((a, bb))
        wvals.append(wv)
    pairs = np.array(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)
    wvals = np.array(wvals, dtype=float) if wvals else np.zeros(0)
    return {
        "gmax": gmax, "tol": tol, "res": res, "tot": tot,
        "mcnt": mcnt, "pairs": pairs, "wvals": wvals, "G": G
    }

def _adam_stage(Xinit, T, steps, lr0, alpha_mat, lam_var, lam_iso,
                k_soft, use_hinge, use_lse, deadline, seed=0):
    n, d = Xinit.shape
    X = Xinit.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    c0 = n / d
    for it in range(1, steps + 1):
        if time.time() > deadline:
            break
        # linear decay of lr
        frac = it / steps
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        # hinge part
        Grad = np.zeros_like(X)
        if use_hinge:
            D = G - T
            D[D < 0] = 0.0
            np.fill_diagonal(D, 0.0)
            # weight by alpha
            Hh = 2.0 * alpha_mat * D
            Grad += Hh @ X
        if use_lse:
            # LSE with k_soft, weighted by alpha
            E = np.exp(k_soft * (G - gmax)) * alpha_mat
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S > 0 and np.isfinite(S):
                Grad += (E @ X) / S
        if lam_var > 0:
            kv = 80.0
            Ev = np.exp(kv * (G - gmax))
            np.fill_diagonal(Ev, 0.0)
            deg = Ev.sum(axis=1)
            mu = float(deg.mean())
            c = 2.0 * (deg - mu) / n  # (n,)
            # H = lam*k*E*(c_i + c_j)
            Csum = c[:, None] + c[None, :]
            Hv = lam_var * kv * Ev * Csum
            # symmetrize (already symmetric)
            Grad += Hv @ X
        if lam_iso > 0:
            XtX = X.T @ X
            F = XtX - c0 * np.eye(d)
            Grad += lam_iso * 4.0 * (X @ F)
        # project to tangent
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # adam
        b1t *= b1; b2t *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1t); vhat = v1 / (1.0 - b2t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -0.05, 0.05)
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
    FINAL_POLISH = 26.0
    n, d = 121, 8
    rng = np.random.default_rng(11)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))
    # stress-dual diagnostics on incumbent
    try:
        diag = _diagnose(Xbest, tol=3e-4)
    except Exception:
        diag = None
    if diag is not None:
        res = diag["res"]; tot = diag["tot"]; mcnt = diag["mcnt"]
        pairs = diag["pairs"]; wvals = diag["wvals"]
        # build alpha inverse-stress weights
        alpha = np.ones((n, n))
        if len(wvals) > 0:
            meanw = float(wvals.mean()) + 1e-12
            wnorm = wvals / meanw
            # inverse: high stress -> low alpha
            beta = 1.2
            avals = 1.0 / (1.0 + beta * np.maximum(wnorm - 1.0, 0.0) + 0.3 * np.maximum(1.0 - wnorm, 0.0) * 0.0)
            # also boost low-stress slightly? keep >=0.25
            avals = np.clip(avals, 0.25, 1.5)
            # very high stress clip lower
            for (a, b), av in zip(pairs, avals):
                alpha[a, b] = av
                alpha[b, a] = av
        np.fill_diagonal(alpha, 0.0)
        # neediness score
        rn = res / (res.max() + 1e-12)
        # low contacts -> needy
        need = rn + 0.5 * (1.0 - (mcnt - mcnt.min()) / (mcnt.max() - mcnt.min() + 1e-12))
        # low total stress -> needy
        if tot.max() > 0:
            need += 0.3 * (1.0 - tot / (tot.max() + 1e-12))
        order_needy = np.argsort(-need)
        # high stress pairs
        if len(wvals) > 0:
            o = np.argsort(-wvals)
            top_pairs = pairs[o[:60]]
        else:
            top_pairs = np.zeros((0, 2), dtype=int)
    else:
        alpha = np.ones((n, n)); np.fill_diagonal(alpha, 0.0)
        order_needy = np.argsort(rng.random(n))
        top_pairs = np.zeros((0, 2), dtype=int)

    # build diverse rebalanced starts
    starts = []
    starts.append(Xbest.copy())
    # start 1: jitter needy strongly, balanced weakly
    try:
        r1 = np.random.default_rng(101)
        X1 = Xbest.copy()
        sig = np.ones(n) * 0.01
        K = min(35, n)
        sig[order_needy[:K]] = 0.12
        sig[order_needy[K:2 * K]] = 0.05
        X1 = X1 + sig[:, None] * r1.standard_normal((n, d))
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
        # repulsion on top stressed pairs
        for (a, b) in top_pairs[:30]:
            gab = float(X1[a] @ X1[b])
            # push apart along tangent
            pa = X1[b] - gab * X1[a]
            pb = X1[a] - gab * X1[b]
            X1[a] -= 0.02 * pa
            X1[b] -= 0.02 * pb
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
        starts.append(X1)
    except Exception:
        pass
    # start 2: different jitter scale
    try:
        r2 = np.random.default_rng(102)
        X2 = Xbest.copy()
        sig2 = np.ones(n) * 0.02
        sig2[order_needy[:25]] = 0.18
        X2 = X2 + sig2[:, None] * r2.standard_normal((n, d))
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
        starts.append(X2)
    except Exception:
        pass
    # start 3: fresh random (basin hop)
    try:
        r3 = np.random.default_rng(203)
        X3 = r3.standard_normal((n, d))
        X3 /= np.linalg.norm(X3, axis=1, keepdims=True)
        starts.append(X3)
    except Exception:
        pass
    # start 4: moderate uniform perturb
    try:
        r4 = np.random.default_rng(104)
        X4 = Xbest + 0.07 * r4.standard_normal((n, d))
        X4 /= np.linalg.norm(X4, axis=1, keepdims=True)
        starts.append(X4)
    except Exception:
        pass

    # staged rebalancing: relaxed -> tight, remove penalties
    # (T, steps, lr, lam_var, lam_iso, k, hinge?, lse?)
    stages_relaxed = [
        (0.472, 900, 0.012, 2.0e-3, 5.0e-4, 60.0, True, False),
        (0.465, 900, 0.010, 1.0e-3, 2.5e-4, 80.0, True, False),
        (0.462, 800, 0.008, 4.0e-4, 1.0e-4, 120.0, True, True),
        (0.461, 700, 0.006, 1.0e-4, 2.0e-5, 200.0, True, True),
        (0.460, 700, 0.005, 0.0, 0.0, 350.0, True, True),
    ]
    alpha_ones = np.ones((n, n)); np.fill_diagonal(alpha_ones, 0.0)
    results = []
    # keep incumbent as candidate
    results.append((Vbest, Xbest.copy(), 1e9, 1e9))
    time_for_search = DEADLINE - FINAL_POLISH - 4.0
    for si, s0 in enumerate(starts):
        if time.time() > time_for_search - 5.0:
            break
        # use inverse-stress alpha only for first 2 stages, then fade
        Xcur = s0.copy()
        for sti, (T, steps, lr, lv, li, kk, uh, ul) in enumerate(stages_relaxed):
            if time.time() > time_for_search:
                break
            # fade alpha to ones after stage 2
            if sti <= 1:
                Ause = alpha
            elif sti == 2:
                Ause = 0.5 * alpha + 0.5 * alpha_ones
            else:
                Ause = alpha_ones
            # fewer steps for later starts
            st = steps if si < 2 else max(400, steps - 300)
            Xcur = _adam_stage(Xcur, T, st, lr, Ause, lv, li, kk, uh, ul, time_for_search)
            v = _max_cos(Xcur)
            # stress/imbalance metric (cheap: contact std at 1e-4)
            try:
                Gc = Xcur @ Xcur.T
                np.fill_diagonal(Gc, -2.0)
                gm = float(Gc.max())
                cntp = (Gc > gm - 1e-4).sum(axis=1)  # includes? diag excluded
                imb = float(cntp.std())
                # residual approx: norm of sum of near-max tangent vectors
                # quick approx without nnls
                resap = 0.0
            except Exception:
                imb = 1e9
            results.append((v, Xcur.copy(), imb, sti))
            # early exit if much worse and time tight? keep going for diversity
        # end stages
    # rank: prefer lower cosine, but retain improved-balance configs slightly worse
    results.sort(key=lambda z: z[0])
    # pick top distinct for final tightening + polish
    # procrustes distinctness
    def _Q(Y, R):
        M = Y.T @ R
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            return Q
        except Exception:
            return np.eye(d)
    kept = []
    for v, X, imb, sti in results:
        if len(kept) >= 4:
            break
        if v > Vbest + 0.008 and len(kept) >= 2:
            continue
        dup = False
        for _, Y in kept:
            try:
                Q = _Q(X, Y)
                rmsd = float(np.linalg.norm(X @ Q - Y)) / np.sqrt(n)
                if rmsd < 0.02:
                    dup = True
                    break
            except Exception:
                pass
        if not dup:
            kept.append((v, X))
    # final tightening with pure LSE/high-k hinge (no penalties) before LP
    final_cands = []
    for v, X in kept:
        if time.time() > DEADLINE - FINAL_POLISH - 2.0:
            break
        # short pure anneal: hinge at 0.460 + LSE k 400->800
        Xc = _adam_stage(X.copy(), 0.4595, 600, 0.004, alpha_ones, 0.0, 0.0, 500.0, True, True, DEADLINE - FINAL_POLISH)
        vc = _max_cos(Xc)
        final_cands.append((vc, Xc))
    for v, X in final_cands:
        if v < Vbest:
            Vbest = v
            Xbest = X.copy()
    # also consider raw results best
    for v, X, _, _ in results:
        if v < Vbest:
            Vbest = v
            Xbest = X.copy()
    # final LP polish on best 2
    pool = [(Vbest, Xbest)] + sorted(final_cands, key=lambda z: z[0])[:2]
    # dedup pool
    for v, X in pool:
        if time.time() > DEADLINE - 1.0:
            break
        rem = DEADLINE - time.time() - 0.5
        if rem < 1.0:
            break
        secs = min(rem * 0.5, 9.0)
        if secs < 0.8:
            continue
        try:
            Xp = _lp_polish(X.copy(), seconds=secs, deadline=DEADLINE - 0.3)
            vp = _max_cos(Xp)
            if vp < Vbest:
                Vbest = vp
                Xbest = Xp.copy()
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard (no-op, ensure finite)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xbest = Xbest / nn
    return Xbest
