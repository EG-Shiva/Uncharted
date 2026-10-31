import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        iu_c = G[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = iu_c[sel]
        p = len(I)
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _refine_balance(x_init, steps, rng, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008,
                    mu=0.88, balpha=0.6, deadline=None):
    x = x_init.copy()
    vel = np.zeros_like(x)
    n = x.shape[0]
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    per_mult = np.ones(n)
    for s in range(steps):
        if deadline is not None and s % 20 == 0 and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        gt = gt / sc
        if balpha > 0 and (s % 40 == 0):
            cnt = np.sum(G > cmax - 1e-3, axis=1).astype(float)
            mcnt = float(np.mean(cnt)) + 1e-9
            per_mult = 1.0 + balpha * (cnt - mcnt) / mcnt
            per_mult = np.clip(per_mult, 0.4, 2.5)
        vel = mu * vel + (1.0 - mu) * gt
        x = x - (lr * per_mult[:, None]) * vel
        x = _norm(x)
    return x

def _rand_rot4(rng, scale):
    Z = rng.randn(4, 4)
    S = (Z - Z.T) * 0.5 * scale
    R = np.eye(4) + S + 0.5 * S @ S
    try:
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    except Exception:
        pass
    return R

def _spectral_twist(base, rng, ang=0.06):
    n = base.shape[0]
    G = base @ base.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    A = (G > m - 2e-3).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    D = np.diag(deg)
    L = D - A
    try:
        w, V = np.linalg.eigh(L)
        fied = V[:, 1]
    except Exception:
        fied = rng.randn(n)
    med = float(np.median(fied))
    mask = fied > med
    if mask.sum() == 0 or mask.sum() == n:
        mask = rng.rand(n) > 0.5
    R = _rand_rot4(rng, ang * (0.5 + rng.rand()))
    R2 = _rand_rot4(rng, ang * (0.5 + rng.rand()))
    out = base.copy()
    out[mask] = out[mask] @ R.T
    out[~mask] = out[~mask] @ R2.T
    out = _norm(out)
    return out, mask

def _refine_single_pt(s0, Xref, rng, steps=70, beta0=40.0, beta1=250.0, lr0=0.09, lr1=0.006):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w /= (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            s = rng.randn(4); ns = np.linalg.norm(s)
        s = s / ns
    return s

def _best_void(Xref, rng, n_sample=8000):
    C = rng.randn(n_sample, 4)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    D = C @ Xref.T
    m = D.max(axis=1)
    return C[int(np.argmin(m))]

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 150.0
    rng = np.random.RandomState(168777)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        xb = _norm(xb)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb = _norm(xb)
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val

    # one-shot antipodal symmetrization attempt (symmetry basin)
    try:
        if time.time() < SEARCH_END - 30:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, 2.0)
            used = np.zeros(168, dtype=bool)
            pairs = []
            for i in range(168):
                if used[i]:
                    continue
                j = int(np.argmin(G0[i]))
                if used[j]:
                    # find unused best partner
                    order = np.argsort(G0[i])
                    for jj in order:
                        if not used[jj] and jj != i:
                            j = int(jj)
                            break
                used[i] = True
                used[j] = True
                pairs.append((i, j))
            Y = np.zeros((84, 4))
            for k, (i, j) in enumerate(pairs):
                v = x_best[i] - x_best[j]
                nv = np.linalg.norm(v)
                if nv < 1e-9:
                    v = rng.randn(4); nv = np.linalg.norm(v)
                Y[k] = v / nv
            # refine antipodal base with paired gradient
            Ya = Y.copy()
            vel = np.zeros_like(Ya)
            for s in range(220):
                Xf = np.vstack([Ya, -Ya])
                Xf = _norm(Xf)
                G = Xf @ Xf.T
                np.fill_diagonal(G, -2.0)
                cmax = float(np.max(G))
                beta = 30.0 * (600.0 / 30.0) ** (s / 219.0)
                E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
                np.fill_diagonal(E, 0.0)
                rs = E.sum(axis=1, keepdims=True) + 1e-12
                grad = (E @ Xf) / rs
                dot = np.sum(grad * Xf, axis=1, keepdims=True)
                gt = grad - dot * Xf
                # combine antipodal pairs
                gt_top = gt[:84] - gt[84:]
                # average scale
                sc = np.mean(np.linalg.norm(gt_top, axis=1)) + 1e-12
                gt_top = gt_top / sc
                mu = 0.88
                lr = 0.016 * (1 - s / 219.0) + 0.0008 * (s / 219.0)
                vel = mu * vel + (1.0 - mu) * gt_top
                Ya = _norm(Ya - lr * vel)
                if time.time() > SEARCH_END - 20:
                    break
            Xsym = _norm(np.vstack([Ya, -Ya]))
            Xsym = _refine_balance(Xsym, 200, rng, 30.0, 700.0, 0.012, 0.0006, 0.88, 0.0, SEARCH_END)
            vsym = _max_cos(Xsym)
            if vsym < best_val and time.time() < SEARCH_END - 2:
                Xs, vss = _lp_polish_deadline(Xsym, min(time.time() + 1.0, SEARCH_END))
                if vss < best_val:
                    x_best, best_val = Xs.copy(), vss
                    x_cur, cur_val = Xs.copy(), vss
    except Exception:
        pass

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        # pick base: mostly best, sometimes walker
        if rng.rand() < 0.72 or cur_val > best_val + 0.004:
            base = x_best
        else:
            base = x_cur
        op = rng.randint(0, 4)
        Xt = None
        try:
            if op == 0:
                ang = float(rng.choice([0.03, 0.05, 0.08, 0.12]))
                Xt, mask = _spectral_twist(base, rng, ang)
                # add small tangent noise, larger on overloaded side
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m0 = float(np.max(G))
                cnt = np.sum(G > m0 - 1e-3, axis=1)
                sig = 0.004 + 0.003 * (cnt / (cnt.max() + 1e-9))
                Xt = Xt + rng.randn(168, 4) * sig[:, None]
                Xt = _norm(Xt)
                Xt = _refine_balance(Xt, 420, rng, 28.0, 750.0, 0.015, 0.0006, 0.88, 0.7, SEARCH_END)
            elif op == 1:
                # degree-targeted kick: hammer hubs, free rattlers
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m0 = float(np.max(G))
                cnt = np.sum(G > m0 - 1e-3, axis=1)
                order = np.argsort(-cnt, kind="stable")
                Xt = base.copy()
                nh = int(rng.choice([6, 8, 10, 12]))
                hubs = order[:nh]
                # strong kick to hubs
                Xt[hubs] = Xt[hubs] + rng.randn(len(hubs), 4) * 0.06
                # small kick elsewhere
                rest = np.ones(168, dtype=bool); rest[hubs] = False
                Xt[rest] = Xt[rest] + rng.randn(168 - len(hubs), 4) * 0.008
                Xt = _norm(Xt)
                Xt = _refine_balance(Xt, 420, rng, 25.0, 650.0, 0.016, 0.0006, 0.88, 0.5, SEARCH_END)
            elif op == 2:
                # delete high-load cluster + void regrow
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m0 = float(np.max(G))
                cnt = np.sum(G > m0 - 1e-3, axis=1).astype(float)
                exc = np.sum(np.clip(G - 0.80, 0.0, None), axis=1)
                load = cnt + 2.0 * exc
                kdel = int(rng.choice([5, 7, 9, 12]))
                order = np.argsort(-load, kind="stable")
                victims = order[:kdel].copy()
                if rng.rand() < 0.5:
                    victims[-1] = order[rng.randint(0, min(30, 168))]
                keep = np.ones(168, dtype=bool)
                keep[victims] = False
                Xsmall = base[keep].copy()
                Xsmall = _refine_balance(Xsmall, 220, rng, 25.0, 550.0, 0.015, 0.0007, 0.88, 0.0, SEARCH_END)
                Xcur2 = Xsmall.copy()
                for ii in range(kdel):
                    c0 = _best_void(Xcur2, rng, n_sample=7000)
                    co = _refine_single_pt(c0, Xcur2, rng, steps=70)
                    Xcur2 = np.vstack([Xcur2, co[None, :]])
                    if (ii + 1) % 3 == 0 or ii == kdel - 1:
                        Xcur2 = _refine_balance(Xcur2, 140, rng, 30.0, 320.0, 0.012, 0.001, 0.88, 0.0, SEARCH_END)
                Xt = _refine_balance(Xcur2, 380, rng, 30.0, 700.0, 0.014, 0.0006, 0.88, 0.4, SEARCH_END)
            else:
                # global small-noise ridge walk with balanced LR
                Xt = base + rng.randn(168, 4) * float(rng.choice([0.008, 0.012, 0.02]))
                Xt = _norm(Xt)
                b0 = float(rng.choice([22.0, 30.0, 40.0]))
                b1 = float(rng.choice([500.0, 700.0, 1000.0]))
                Xt = _refine_balance(Xt, 380, rng, b0, b1, 0.014, 0.0006, 0.88, 0.8, SEARCH_END)
        except Exception:
            continue
        if Xt is None:
            continue
        v = _max_cos(Xt)
        # Metropolis acceptance for walker
        temp = 0.0006 * (1.0 - (time.time() - t0) / 160.0)
        temp = max(temp, 0.00008)
        if v < cur_val or rng.rand() < np.exp(-(v - cur_val) / temp):
            x_cur, cur_val = Xt.copy(), v
        if v < best_val - 1e-12:
            # promising: LP squeeze
            if time.time() < SEARCH_END - 1.0:
                try:
                    Xs, vs = _lp_polish_deadline(Xt, min(time.time() + 0.9, SEARCH_END))
                except Exception:
                    Xs, vs = Xt, v
                if vs < best_val:
                    x_best, best_val = Xs.copy(), vs
                    x_cur, cur_val = Xs.copy(), vs
                elif v < best_val:
                    x_best, best_val = Xt.copy(), v
                    x_cur, cur_val = Xt.copy(), v
            else:
                x_best, best_val = Xt.copy(), v
        elif v < best_val + 0.00015 and time.time() < SEARCH_END - 2.0 and (trial % 7 == 0):
            # opportunistic squeeze of near-best to find different vertex
            try:
                Xs, vs = _lp_polish_deadline(Xt, min(time.time() + 0.7, SEARCH_END))
            except Exception:
                continue
            if vs < best_val:
                x_best, best_val = Xs.copy(), vs
                x_cur, cur_val = Xs.copy(), vs
    try:
        x_best, best_val = _lp_polish_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
