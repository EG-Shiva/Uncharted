import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    T = np.empty((n, d, dm), dtype=np.float64)
    e1 = np.zeros(d)
    e1[0] = 1.0
    I = np.eye(d)
    for i in range(n):
        a = X[i]
        if 1.0 - a[0] < 1e-7 and np.linalg.norm(a - e1) < 1e-4:
            Ti = np.zeros((d, dm))
            Ti[1:, :] = np.eye(dm)
            T[i] = Ti
        else:
            v = e1 - a
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                Ti = np.zeros((d, dm))
                Ti[1:, :] = np.eye(dm)
                T[i] = Ti
            else:
                v /= nv
                H = I - 2.0 * np.outer(v, v)
                T[i] = H[:, 1:]
    return T

def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.sqrt((X * X).sum(axis=1, keepdims=True))[:, None]
    return X

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mm = c.max()
        sel = c > mm - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 2500
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds:
            break
    return x

def _edges_at(X, slack):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    sel = c > m - slack
    keys = (iu[0][sel].astype(np.int64) * 1000 + iu[1][sel].astype(np.int64))
    return set(keys.tolist()), m

def _jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n_dim = 8; n_pts = 168
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(7)
    n, d = 168, 8
    dm = 7
    N = n * dm
    KSMALL = 50
    parent = best.copy()
    parent_val = best_val
    parent_edges, _ = _edges_at(parent, 1e-5)
    slack_choices = [1e-7, 1e-5, 1e-4]
    it = 0
    # main rigidity loop
    while time.time() - t0 < BUDGET - 12.0:
        it += 1
        if time.time() + 3.0 > t0 + BUDGET:
            break
        slack = slack_choices[rng.randint(len(slack_choices))]
        # occasionally use larger slack for very floppy
        if rng.rand() < 0.12:
            slack = 3e-4
        G = parent @ parent.T
        np.fill_diagonal(G, -2.0)
        mpar = float(G.max())
        iu0 = np.triu_indices(n, 1)
        c0 = G[iu0]
        order = np.argsort(c0)
        sel_mask = c0 > mpar - slack
        I = iu0[0][sel_mask]; J = iu0[1][sel_mask]
        p = len(I)
        if p < 250:
            # enlarge slack to get enough rows
            take = 400
            idx = order[-take:]
            I = iu0[0][idx]; J = iu0[1][idx]
            p = take
        if p > 2800:
            idx = np.argsort(c0[sel_mask])[-2800:]
            I = I[idx]; J = J[idx]; p = 2800
        # build tangent bases
        try:
            T = _tangent_bases(parent)
        except Exception:
            # fallback isotropic kick
            X = parent + rng.randn(n, d) * 0.02
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            Xq = _quench(X, 1500, 50.0, 600.0, 0.004, 0.0004)
            vq = _max_cos(Xq)
            if vq < best_val:
                best = Xq; best_val = vq
            continue
        # build dense rigidity matrix R (p x N)
        R1 = np.einsum('pj,pjk->pk', parent[J], T[I])
        R2 = np.einsum('pj,pjk->pk', parent[I], T[J])
        R = np.zeros((p, N), dtype=np.float64)
        ar = np.arange(dm)
        # fill with loop over p (fast enough for p~2000)
        for k in range(p):
            R[k, I[k]*dm:(I[k]+1)*dm] = R1[k]
            R[k, J[k]*dm:(J[k]+1)*dm] = R2[k]
        # smallest singular vectors via eigh of R^T R
        try:
            M = R.T @ R
            # symmetrize
            M = 0.5 * (M + M.T)
            # add tiny diagonal for stability
            w, V = np.linalg.eigh(M)
            # V columns ascending
            Vsmall = V[:, :KSMALL]  # N x 50
        except Exception:
            continue
        # number of trials from this basis (amortize cost)
        ntrials = 2 if p > 600 else 1
        for tr in range(ntrials):
            if time.time() + 2.5 > t0 + BUDGET - 10.0:
                break
            coeffs = rng.randn(KSMALL)
            # optionally emphasize softer modes
            # weight by 1/sqrt(eig+eps)
            try:
                ev = w[:KSMALL]
                wt = 1.0 / np.sqrt(np.maximum(ev, 0.0) + 1e-3)
                wt /= np.linalg.norm(wt) / np.sqrt(float(KSMALL))
                coeffs = coeffs * wt
            except Exception:
                pass
            nc = np.linalg.norm(coeffs)
            if nc == 0:
                continue
            coeffs /= nc
            # rms per point 0.02..0.15 log-uniform
            rms = np.exp(np.log(0.02) + rng.rand() * (np.log(0.15) - np.log(0.02)))
            # occasionally larger jump
            if rng.rand() < 0.15:
                rms = 0.15 + rng.rand() * 0.10
            total = rms * np.sqrt(n)
            u = Vsmall @ coeffs * total  # N,
            U = u.reshape(n, dm)
            dX = np.einsum('nij,nj->ni', T, U)
            Xt = parent + dX
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # anneal
            Xq = _quench(Xt, 1500, 50.0, 600.0, 0.0045, 0.00045)
            # short LP polish
            try:
                Xq = _lp_polish(Xq, 0.5)
            except Exception:
                pass
            vq = _max_cos(Xq)
            if not np.all(np.isfinite(Xq)):
                continue
            trial_edges, _ = _edges_at(Xq, 1e-5)
            jac = _jaccard(parent_edges, trial_edges)
            if vq + 1e-12 < best_val:
                best = Xq.copy(); best_val = vq
            # move rule
            if jac < 0.9 or vq < parent_val:
                parent = Xq.copy(); parent_val = vq; parent_edges = trial_edges
                # keep best as parent if parent degraded too much
                if parent_val > best_val + 0.004:
                    parent = best.copy(); parent_val = best_val
                    parent_edges, _ = _edges_at(parent, 1e-5)
            else:
                # if not diverse and not better, try second trial with bigger step
                pass
        # occasional reset to best to avoid drift
        if it % 25 == 0 and parent_val > best_val + 0.001:
            parent = best.copy(); parent_val = best_val
            parent_edges, _ = _edges_at(parent, 1e-5)
    # final polish of best with leftover
    rem = (t0 + BUDGET) - time.time()
    if rem > 5.0:
        try:
            secs = min(rem - 1.5, 9.0)
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished; best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
