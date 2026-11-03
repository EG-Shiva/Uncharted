import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _lp_polish(x, deadline, r0=1e-2, width=1e-3, maxpairs=4000, drop=None, drop_target=None):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    # precompute full c index for drop
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > (m - width)
        # drop pair handling: exclude drop from t-constraints, add fixed cap
        drop_idx = -1
        if drop is not None:
            a, b = int(drop[0]), int(drop[1])
            if a > b:
                a, b = b, a
            # linear index in iu: need to find
            # compute via search (n=138 small, do mask)
            pass
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > maxpairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:maxpairs]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        # if drop pair inside selected, remove it from t-set (will add fixed row)
        extra = 0
        drop_row = None
        if drop is not None:
            a, b = int(drop[0]), int(drop[1])
            # find positions where (I,J)==(a,b) or (b,a)
            dm = ((I == a) & (J == b)) | ((I == b) & (J == a))
            if np.any(dm):
                # remove from t set
                keep = ~dm
                I = I[keep]; J = J[keep]
                # c_sel aligned: need c values for kept
                csel_full = c[sel]
                csel = csel_full[keep]
                p = len(I)
                extra = 1
                # coefficients for fixed row
                drop_c = float(x[a] @ x[b])
                drop_row = (a, b, drop_c)
            else:
                csel = c[sel]
                # still need fixed row if drop dot is large? add it anyway
                drop_c = float(x[a] @ x[b])
                if drop_target is not None and drop_c > drop_target - 5e-3:
                    extra = 1
                    drop_row = (a, b, drop_c)
                else:
                    csel = c[sel]
        else:
            csel = c[sel]
        if p == 0 and extra == 0:
            break
        p2 = p + extra
        nv = n * d + 1
        # build A_ub: p rows with t, extra row without t
        # rows for t-part: [xJ, xI, -1]
        nrows = p2
        # estimate nnz: p*(2d+1)+2d
        rows_list = []
        cols_list = []
        vals_list = []
        # vectorized for p rows
        if p > 0:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1
            ).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_rows = rows; A_cols = cols; A_vals = vals
            b_ub = -csel
            if extra == 1:
                # fixed row index p
                a, b, dc = drop_row
                erows = np.full(2 * d, p)
                ecols = np.concatenate([np.arange(d) + a * d, np.arange(d) + b * d])
                evals = np.concatenate([x[b], x[a]])
                A_rows = np.concatenate([A_rows, erows])
                A_cols = np.concatenate([A_cols, ecols])
                A_vals = np.concatenate([A_vals, evals])
                b_ub = np.concatenate([b_ub, np.array([drop_target - dc])])
            A_ub = coo_matrix((A_vals, (A_rows, A_cols)), shape=(nrows, nv)).tocsr()
        else:
            # only fixed row
            a, b, dc = drop_row
            erows = np.full(2 * d, 0)
            ecols = np.concatenate([np.arange(d) + a * d, np.arange(d) + b * d])
            evals = np.concatenate([x[b], x[a]])
            A_ub = coo_matrix((evals, (erows, ecols)), shape=(nrows, nv)).tocsr()
            b_ub = np.array([drop_target - dc])
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        # for drop case, check feasibility of cap
        if drop is not None and drop_target is not None:
            a, b = int(drop[0]), int(drop[1])
            if float(y[a] @ y[b]) > drop_target + 1e-12:
                radius *= 0.5
                continue
        v = _max_cos(y)
        if v < best - 1e-14:
            x, best = y, v
            radius = min(radius * 1.3, 0.05)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    dm1 = d - 1
    B = np.empty((n, d, dm1))
    Eye = np.eye(d)
    for i in range(n):
        xi = x[i]
        k0 = int(np.argmax(np.abs(xi)))
        P = np.empty((d, dm1))
        col = 0
        for k in range(d):
            if k == k0:
                continue
            v = Eye[k].copy()
            v = v - float(v @ xi) * xi
            P[:, col] = v
            col += 1
        Q, _ = np.linalg.qr(P)
        B[i] = Q
    return B

def _floppy_dirs(x, tol, kkeep=8, cap=2500):
    n, d = x.shape
    dm1 = d - 1
    G = x @ x.T
    np.fill_diagonal(G, -1.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > (m - tol)
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return [], m, 0
    if len(idx) > cap:
        order = np.argsort(-c[idx])[:cap]
        idx = idx[order]
    I = iu[0][idx]; J = iu[1][idx]
    p = len(idx)
    dots = c[idx]
    B = _tangent_bases(x)
    N = n * dm1
    Jt = np.empty((p, N), dtype=float)
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        dij = float(dots[r])
        vi = x[j] - dij * x[i]
        vj = x[i] - dij * x[j]
        Jt[r, i * dm1:(i + 1) * dm1] = vi @ B[i]
        Jt[r, j * dm1:(j + 1) * dm1] = vj @ B[j]
    try:
        _, s, Vt = np.linalg.svd(Jt, full_matrices=False)
    except Exception:
        return [], m, p
    dirs = []
    # smallest singular values last
    tot = Vt.shape[0]
    take = min(kkeep, tot)
    for t in range(take):
        y = Vt[tot - 1 - t]
        sv = float(s[tot - 1 - t]) if tot <= len(s) else 0.0
        Y = y.reshape(n, dm1)
        D = np.empty((n, d))
        for i in range(n):
            D[i] = B[i] @ Y[i]
        # D already tangent; normalize
        fn = float(np.sqrt((D * D).sum()))
        if fn < 1e-12:
            continue
        D /= fn
        dirs.append((D, sv))
    return dirs, m, p

def _eval_line(x, D, alphas):
    outs = []
    for a in alphas:
        y = _norm(x + a * D)
        v = _max_cos(y)
        outs.append(v)
    return outs

def _lse_anneal(x, steps, beta0, beta1, lr0, lr1, drop=None, drop_target=None, seed=0):
    n, d = x.shape
    y = x.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        if drop is not None:
            a, b = int(drop[0]), int(drop[1])
            G[a, b] = -2.0; G[b, a] = -2.0
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        if drop is not None:
            E[a, b] = 0.0; E[b, a] = 0.0
        S = float(E.sum() / 2.0) + 1e-300
        W = E / S
        Grad = W @ y
        # drop penalty push-apart
        if drop is not None and drop_target is not None:
            a, b = int(drop[0]), int(drop[1])
            dab = float(y[a] @ y[b])
            if dab > drop_target:
                # gradient of (dab-target)^2 pushes apart
                coeff = 8.0 * (dab - drop_target)
                Grad[a] += coeff * y[b]
                Grad[b] += coeff * y[a]
        dot = (Grad * y).sum(axis=1, keepdims=True)
        Tan = Grad - dot * y
        tn = float(np.sqrt((Tan * Tan).sum())) + 1e-30
        scale = 1.0 / max(1.0, tn * 0.3)
        y = y - lr * scale * Tan
        y = _norm(y)
    return y

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    m0 = best_val
    # ---- floppy SVD experiments ----
    try:
        alphas = np.logspace(-4, -0.7, 14)  # 1e-4 .. ~0.2
        cands = []
        for tol in (1e-7, 1e-5, 1e-4):
            if time.time() > deadline - 70:
                break
            dirs, mcur, pnum = _floppy_dirs(best, tol, kkeep=8, cap=2500)
            if not dirs:
                continue
            # original argmax pair for topology detection
            G0 = best @ best.T; np.fill_diagonal(G0, -2.0)
            am0 = int(np.argmax(G0)); a0 = am0 // n; b0 = am0 % n
            for (D, sv) in dirs:
                if time.time() > deadline - 65:
                    break
                for sign in (1.0, -1.0):
                    Ds = sign * D
                    # scan alphas
                    best_a = 0; best_v = m0
                    topo_a = None
                    for a in alphas:
                        y = _norm(best + a * Ds)
                        G = y @ y.T; np.fill_diagonal(G, -2.0)
                        v = float(G.max())
                        if v < best_v:
                            best_v = v; best_a = a
                        if topo_a is None:
                            am = int(np.argmax(G)); ai = am // n; bi = am % n
                            pair_same = (ai == a0 and bi == b0) or (ai == b0 and bi == a0)
                            if not pair_same and v < m0 + 2e-3:
                                topo_a = a
                    if best_a > 0:
                        yb = _norm(best + best_a * Ds)
                        cands.append((best_v, yb))
                    if topo_a is not None and topo_a != best_a:
                        yt = _norm(best + topo_a * Ds)
                        vt = _max_cos(yt)
                        cands.append((vt, yt))
        # deduplicate-ish: sort by value, keep top few
        cands.sort(key=lambda t: t[0])
        seen = 0
        for vv, yy in cands[:12]:
            if time.time() > deadline - 55:
                break
            # only polish promising (within 3e-3 of incumbent)
            if vv > best_val + 3e-3:
                continue
            yp = _lp_polish(yy, min(deadline - 40, time.time() + 2.2), r0=1e-2, width=1e-3)
            vp = _max_cos(yp)
            if vp < best_val - 1e-12:
                best, best_val = yp, vp
            seen += 1
            if seen >= 8:
                # keep time for drop experiments
                if time.time() > deadline - 60:
                    break
    except Exception:
        pass
    # ---- drop-one-contact experiments ----
    try:
        G = best @ best.T; np.fill_diagonal(G, -2.0)
        iu = np.triu_indices(n, 1)
        c = G[iu]
        order = np.argsort(-c)
        K = 10
        tight_idx = order[:K]
        mcur = float(c.max())
        target = mcur - 1e-3
        for kk in range(K):
            if time.time() > deadline - 18:
                break
            pi = int(tight_idx[kk])
            a = int(iu[0][pi]); b = int(iu[1][pi])
            y = best.copy()
            # separate the pair
            y[a] = y[a] - 0.03 * y[b]
            y[b] = y[b] - 0.03 * y[a]
            y[a] /= (np.linalg.norm(y[a]) + 1e-30)
            y[b] /= (np.linalg.norm(y[b]) + 1e-30)
            y = _lse_anneal(y, steps=500, beta0=30.0, beta1=320.0, lr0=0.06, lr1=0.006,
                            drop=(a, b), drop_target=target)
            # check cap satisfied else skip LP
            if float(y[a] @ y[b]) > target + 5e-4:
                # still try but with stronger push: one more short anneal
                y = _lse_anneal(y, steps=250, beta0=60.0, beta1=350.0, lr0=0.05, lr1=0.005,
                                drop=(a, b), drop_target=target)
            yp = _lp_polish(y, min(deadline - 12, time.time() + 2.0), r0=1e-2, width=1e-3,
                            drop=(a, b), drop_target=target)
            vp = _max_cos(yp)
            # also try releasing the constraint: free polish
            if time.time() < deadline - 12:
                yf = _lp_polish(yp, min(deadline - 12, time.time() + 1.2), r0=1e-2, width=1e-3)
                vf = _max_cos(yf)
                if vf < vp:
                    yp, vp = yf, vf
            if vp < best_val - 1e-12:
                best, best_val = yp, vp
                # update target to new best
                target = best_val - 1e-3
    except Exception:
        pass
    # ---- final polish of incumbent ----
    try:
        if time.time() < deadline - 1.0:
            best = _lp_polish(best, min(deadline, time.time() + 8.0), r0=1e-2, width=1e-3)
    except Exception:
        pass
    best = _norm(best)
    return best
