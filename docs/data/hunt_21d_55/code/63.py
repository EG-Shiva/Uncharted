import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _active_count(x, thr=None, m=None):
    g = x @ x.T
    iu = np.triu_indices(g.shape[0], 1)
    c = g[iu]
    if m is None:
        m = float(c.max())
    if thr is None:
        thr = m - 1e-6
    return int(np.count_nonzero(c > thr)), m

def _lp_polish(x, seconds, radius_init=1e-3, cap=1500):
    x = _norm_rows(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:cap]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_quench(x_init, steps, lr0, beta0, beta1, rng, noise=0.0):
    x = x_init.copy()
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 * (1.0 - 0.6 * f)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: beta*(G-m) <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        dot = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dot * x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nrm
        if noise > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz = nz / nn
            Gt = (1.0 - noise) * Gt + noise * nz
            # renormalize direction roughly
            n2 = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            Gt = Gt / n2
        x = x - lr * Gt
        x = _norm_rows(x)
    return x

def _build_escape_basis(x, pairs_I, pairs_J, cmax, rng, n_small=20, n_null=20):
    n, d = x.shape
    N = n * d
    p = len(pairs_I)
    A = np.empty((p, N), dtype=np.float64)
    s = 1.0 / np.sqrt(2.0)
    for k in range(p):
        i = pairs_I[k]; j = pairs_J[k]
        c = float(x[i] @ x[j])
        vi = (x[j] - c * x[i]) * s
        vj = (x[i] - c * x[j]) * s
        row = np.zeros(N, dtype=np.float64)
        row[i*d:(i+1)*d] = vi
        row[j*d:(j+1)*d] = vj
        A[k] = row
    # thin SVD
    try:
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
    except Exception:
        # fallback: random tangent basis
        Vh = None
        S = None
    vecs = []
    if Vh is not None:
        ns = min(n_small, Vh.shape[0])
        for t in range(ns):
            v = Vh[Vh.shape[0]-ns+t].copy()
            # Vh rows already tangent; enforce tangent numerically
            vr = v.reshape(n, d)
            corr = np.sum(vr * x, axis=1, keepdims=True)
            vr -= corr * x
            nv = np.linalg.norm(vr)
            if nv > 1e-12:
                vr /= nv
                vecs.append(vr.reshape(-1))
    # nullspace via random tangent orthogonal to row space
    K = n_null
    R = rng.standard_normal((N, K))
    # tangent projection
    Rm = R.reshape(n, d, K) if False else None
    # do per-point projection
    Rr = R.reshape(n, d, K)
    # Rr[k,:,:] is d x K ; subtract component along x[k]
    for k in range(n):
        xk = x[k]  # d
        # coeffs: xk @ block  -> K
        coeff = xk @ Rr[k]  # K
        Rr[k] -= xk[:, None] * coeff[None, :]
    R = Rr.reshape(N, K)
    if Vh is not None:
        # remove row-space component
        # Vh is (p,N); Vh @ R is (p,K)
        coeff2 = Vh @ R
        R = R - Vh.T @ coeff2
        # re-project tangent (numerical)
        Rr = R.reshape(n, d, K)
        for k in range(n):
            xk = x[k]
            coeff = xk @ Rr[k]
            Rr[k] -= xk[:, None] * coeff[None, :]
        R = Rr.reshape(N, K)
    # orthonormalize
    try:
        Q, _ = np.linalg.qr(R)
    except Exception:
        Q = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    for k in range(min(K, Q.shape[1])):
        v = Q[:, k].copy()
        vr = v.reshape(n, d)
        # ensure tangent
        corr = np.sum(vr * x, axis=1, keepdims=True)
        vr -= corr * x
        nv = np.linalg.norm(vr)
        if nv > 1e-12:
            vr /= nv
            vecs.append(vr.reshape(-1))
    return vecs

def solve():
    t0 = time.time()
    LIMIT = 172.0
    SEARCH_END = t0 + 148.0
    rng = np.random.default_rng(20261101)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((55, 21))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (55, 21):
        tmp = rng.standard_normal((55, 21))
        xb = _norm_rows(tmp)
    xb = _norm_rows(xb)
    n, d = 55, 21
    best_score = _max_cos(xb)
    x_best = xb.copy()
    # base point for escapes (starts at best, may drift to sparse basin)
    x_base = xb.copy()
    base_score = best_score

    amps = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    iu = np.triu_indices(n, 1)
    cyc = 0
    while time.time() < SEARCH_END:
        cyc += 1
        remain = LIMIT - (time.time() - t0)
        if remain < 8:
            break
        # build active set for base: pairs with cos > max-1e-6
        G0 = x_base @ x_base.T
        c0full = G0[iu]
        m0 = float(c0full.max())
        sel = c0full > m0 - 1e-6
        I0 = iu[0][sel]; J0 = iu[1][sel]
        p0 = len(I0)
        if p0 < 5 or p0 > 2000:
            # degenerate: randomize base slightly and continue
            x_base = _norm_rows(x_best + 0.02 * rng.standard_normal((n, d)))
            base_score = _max_cos(x_base)
            continue
        # escape basis: smallest 20 + 20 null (no LP inside)
        try:
            vecs = _build_escape_basis(x_base, I0, J0, m0, rng, n_small=20, n_null=20)
        except Exception:
            continue
        if len(vecs) == 0:
            continue
        # line search: score by (active, max), allow max up to 0.075
        best_cand = None
        best_key = None
        # to save time, evaluate all amps/signs; each eval is just Gram
        for v in vecs:
            vr = v.reshape(n, d)
            # v already unit + tangent
            for amp in amps:
                # quick time check
                if time.time() > SEARCH_END:
                    break
                for sgn in (1.0, -1.0):
                    y = x_base + (sgn * amp) * vr
                    y = _norm_rows(y)
                    Gy = y @ y.T
                    cy = Gy[iu]
                    my = float(cy.max())
                    if my > 0.075 or not np.isfinite(my):
                        continue
                    act = int(np.count_nonzero(cy > my - 1e-6))
                    key = (act, my)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_cand = y.copy()
            if time.time() > SEARCH_END:
                break
        if best_cand is None or best_key is None:
            # no admissible move: kick base randomly in nullspace
            v = vecs[int(rng.integers(0, len(vecs)))]
            x_base = _norm_rows(x_base + 0.04 * v.reshape(n, d))
            base_score = _max_cos(x_base)
            continue
        act_best, max_best = best_key
        # material reduction? require active drop
        # active of base w.r.t its own top:
        act0 = int(np.count_nonzero(c0full > m0 - 1e-6))  # == p0
        improved_sparse = (act_best <= act0 - 15) or (act_best <= int(0.75 * act0))
        # also consider max improvement directly
        direct_better = max_best < best_score - 2e-6
        if not (improved_sparse or direct_better):
            # try new random null sample next cycle; occasionally drift base
            if cyc % 4 == 0:
                # small random tangent drift to get new Jacobian sample
                nz = rng.standard_normal((n, d))
                nz -= np.sum(nz * x_base, axis=1, keepdims=True) * x_base
                nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
                nz /= nn
                x_base = _norm_rows(x_base + 0.01 * nz)
                base_score = _max_cos(x_base)
            else:
                # stay at best to try different random null vectors
                x_base = x_best.copy()
                base_score = best_score
            continue
        # quench with soft-max descent (no LP)
        # parameters depend on how much max rose
        rise = max_best - best_score
        if rise > 0.004:
            steps, lr0, b0, b1 = 900, 0.012, 120.0, 900.0
        elif rise > 0.001:
            steps, lr0, b0, b1 = 800, 0.009, 150.0, 850.0
        else:
            steps, lr0, b0, b1 = 700, 0.007, 200.0, 800.0
        # time guard: quench ~ fast
        yq = _smooth_quench(best_cand, steps=steps, lr0=lr0, beta0=b0, beta1=b1, rng=rng, noise=0.02)
        vq = _max_cos(yq)
        if vq < best_score - 1e-12:
            # short LP polish only now
            budget = 2.0
            rem2 = LIMIT - (time.time() - t0) - 6.0
            if rem2 < budget:
                budget = max(0.4, rem2)
            yp = _lp_polish(yq, seconds=budget, radius_init=1.2e-3)
            vp = _max_cos(yp)
            if vp < best_score:
                x_best, best_score = yp, vp
                x_base = yp.copy(); base_score = vp
            elif vq < best_score:
                x_best, best_score = yq, vq
                x_base = yq.copy(); base_score = vq
            continue
        else:
            # quench did not beat best; if it sparsified, keep exploring from it
            aq, _ = _active_count(yq)
            # short LP polish sparingly (only if sparse and close)
            if aq < int(0.6 * act0) and vq < best_score + 0.004 and remain > 20:
                budget = 1.2
                yp = _lp_polish(yq, seconds=budget, radius_init=8e-4)
                vp = _max_cos(yp)
                if vp < best_score - 1e-12:
                    x_best, best_score = yp, vp
                    x_base = yp.copy(); base_score = vp
                    continue
                # move base to sparse basin to continue escape chain
                if vq < best_score + 0.002:
                    x_base = yq.copy(); base_score = vq
                    continue
            # otherwise reset to best for fresh direction sample
            x_base = x_best.copy(); base_score = best_score
            continue
    # final short polish of best (strictly limited)
    remain = LIMIT - (time.time() - t0)
    if remain > 2.5:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain - 1.0, 8.0), radius_init=8e-4)
        except Exception:
            pass
    x_best = _norm_rows(x_best)
    return x_best
