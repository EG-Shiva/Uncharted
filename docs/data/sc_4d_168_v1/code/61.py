import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _batch_max_cos(G, eye2):
    # G is (B,N,N) raw Gram, eye2 = 2*eye for masking
    Gm = G - eye2
    return Gm.max(axis=(1, 2))

def lp_polish_deadline(x, deadline, max_pairs=4000):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos_single(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    # preallocate nothing
    while time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            # keep tightest
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:max_pairs]] = True
            sel = sel & keep
        I, J = iu0[0][sel], iu0[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = _norm(x + dx)
        v = _max_cos_single(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def refine_single_adam(x0, steps, beta0, beta1, lr0, lr1, deadline, rng, mu1=0.9, mu2=0.999):
    x = _norm(x0.copy())
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    n = x.shape[0]
    for t in range(1, steps + 1):
        if time.time() >= deadline:
            break
        f = t / max(1, steps)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # scale to keep Adam stable
        sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        gt = gt / sc
        m = mu1 * m + (1 - mu1) * gt
        vv = mu2 * vv + (1 - mu2) * (gt * gt)
        mhat = m / (1.0 - mu1 ** t)
        vhat = vv / (1.0 - mu2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    SEARCH_END = t0 + 118.0
    REFINE_END = t0 + 158.0
    rng = np.random.default_rng(0)
    n, d = 168, 4
    B = 16
    # load incumbents only for final safety comparison
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 = _norm(xb0)
        if xb0.shape != (n, d):
            xb0 = None
    except Exception:
        xb0 = None
    try:
        xr0 = np.load("record.npy").astype(float)
        xr0 = _norm(xr0)
        if xr0.shape != (n, d):
            xr0 = None
    except Exception:
        xr0 = None
    if xb0 is not None:
        inc_best = xb0.copy()
        inc_val = _max_cos_single(inc_best)
    else:
        inc_best = None
        inc_val = 1e9
    if xr0 is not None:
        rv = _max_cos_single(xr0)
        if rv < inc_val:
            inc_best = xr0.copy()
            inc_val = rv
    if inc_best is None:
        inc_best = _norm(rng.normal(size=(n, d)))
        inc_val = _max_cos_single(inc_best)

    # ---- Phase 1: batched de novo homotopy ----
    X = _norm(rng.normal(size=(B, n, d)))
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    lr_lanes = np.array([0.030] * 4 + [0.015] * 4 + [0.008] * 4 + [0.004] * 4, dtype=float)
    noise_lanes = np.array([0.005] * 4 + [0.0025] * 4 + [0.001] * 4 + [0.0] * 4, dtype=float)
    eyeN = np.eye(n, dtype=float)
    eye2 = 2.0 * eyeN
    mask = 1.0 - eyeN  # (N,N)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    T_total = 6000
    step = 0
    lb_lo = np.log(5.0); lb_hi = np.log(1000.0)
    cull_every = 1000
    next_cull = cull_every
    # run in chunks
    while step < T_total and time.time() < SEARCH_END:
        # do up to next_cull
        target = min(next_cull, T_total)
        while step < target:
            if time.time() >= SEARCH_END:
                break
            step += 1
            f = step / T_total
            beta = float(np.exp(lb_lo + (lb_hi - lb_lo) * f))
            # batched Gram
            G = np.matmul(X, X.transpose(0, 2, 1))
            cmax = _batch_max_cos(G, eye2)  # (B,)
            # weights
            # to save memory compute in place-ish
            D = G - cmax[:, None, None]
            D *= beta
            np.clip(D, -25.0, 0.0, out=D)
            np.exp(D, out=D)
            D *= mask  # broadcast (N,N)
            W = D
            Grad = np.matmul(W, X)
            dot = np.sum(Grad * X, axis=2, keepdims=True)
            Gt = Grad - dot * X
            # Langevin tangential noise for exploratory lanes
            # only if any noise active and decaying amplitude
            amp_base = (1.0 - f)
            if amp_base > 0:
                Nz = rng.normal(size=(B, n, d))
                Ndot = np.sum(Nz * X, axis=2, keepdims=True)
                Nz -= Ndot * X
                Gt = Gt + Nz * (noise_lanes[:, None, None] * amp_base)
            # Adam update
            m = b1 * m + (1.0 - b1) * Gt
            v = b2 * v + (1.0 - b2) * (Gt * Gt)
            c1 = 1.0 - b1 ** step
            c2 = 1.0 - b2 ** step
            mhat = m / c1
            vhat = v / c2
            upd = (lr_lanes[:, None, None] * mhat / (np.sqrt(vhat) + eps))
            X = X - upd
            X = _norm(X)
        if time.time() >= SEARCH_END:
            break
        # cull / rank at next_cull
        if step >= next_cull:
            Gf = np.matmul(X, X.transpose(0, 2, 1))
            vals = _batch_max_cos(Gf, eye2)
            order = np.argsort(vals)  # best first
            # keep top half
            top = order[:B // 2]
            bot = order[B // 2:]
            # build replacements
            for k, bidx in enumerate(bot):
                if k < (B // 2 - 2):
                    src = top[k % len(top)]
                    sigma = 0.10 if (k % 2 == 0) else 0.22
                    X[bidx] = _norm(X[src] + sigma * rng.normal(size=(n, d)))
                else:
                    # fresh random
                    X[bidx] = _norm(rng.normal(size=(n, d)))
                # reset momentum for replaced
                m[bidx] = 0.0
                v[bidx] = 0.0
            # reset adam bias step? keep global step for bias correction (fine)
            next_cull += cull_every
            if next_cull > T_total:
                pass
    # rank final batch
    try:
        Gf = np.matmul(X, X.transpose(0, 2, 1))
        vals = _batch_max_cos(Gf, eye2)
        order = np.argsort(vals)
    except Exception:
        order = np.arange(B)
    best3_idx = order[:3]
    best3 = [_norm(X[i].copy()) for i in best3_idx]
    best3_vals = []
    for c in best3:
        try:
            best3_vals.append(_max_cos_single(c))
        except Exception:
            best3_vals.append(1e9)

    # ---- Phase 2: individual high-beta refinement ----
    refined = []
    for k, c0 in enumerate(best3):
        if time.time() >= REFINE_END:
            refined.append((c0, best3_vals[k]))
            continue
        # budget per config
        # first two get more steps
        steps = 900 if k < 2 else 600
        beta0, beta1 = (80.0, 900.0) if k == 0 else ((60.0, 800.0) if k == 1 else (40.0, 600.0))
        lr0, lr1 = (0.012, 0.0008) if k == 0 else (0.010, 0.0007)
        dl = min(REFINE_END, time.time() + 12.0)
        try:
            cr = refine_single_adam(c0, steps, beta0, beta1, lr0, lr1, dl, rng)
            vr = _max_cos_single(cr)
        except Exception:
            cr, vr = c0, best3_vals[k]
        # LP polish short
        if time.time() < REFINE_END:
            try:
                cp, vp = lp_polish_deadline(cr, min(REFINE_END, time.time() + 5.0))
                if vp < vr:
                    cr, vr = cp, vp
                else:
                    # keep refined without LP if LP worse (keep cr)
                    pass
            except Exception:
                pass
        refined.append((cr, vr))
        if time.time() >= REFINE_END:
            break
    # pick best refined
    if len(refined) > 0:
        refined.sort(key=lambda t: t[1])
        de_novo_best, de_novo_val = refined[0]
    else:
        de_novo_best = best3[0]
        de_novo_val = best3_vals[0]

    # ---- Final safety: LP-polish loaded incumbent ----
    try:
        rem = HARD_END - time.time() - 1.0
        if rem > 2.0:
            inc_pol, inc_pol_val = lp_polish_deadline(inc_best, time.time() + min(rem, 12.0))
        else:
            inc_pol_val = inc_val
            inc_pol = inc_best
    except Exception:
        inc_pol = inc_best
        inc_pol_val = inc_val

    if de_novo_val < inc_pol_val:
        final = _norm(de_novo_best)
    else:
        final = _norm(inc_pol)
    # final safety normalize
    final = _norm(final)
    return final
