import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _nm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(12345)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = _nm(xb.astype(float))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _nm(xr)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (n, d) or not np.all(np.isfinite(xb)):
        xb = _nm(rng.randn(n, d))
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def refine_full(x_init, steps, mu=0.9, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _nm(x)
            if time.time() > SEARCH_END and steps > 300 and s > 200:
                break
        return x

    def relax_mask(x_init, mask, steps=300, mu=0.9, beta0=25.0, beta1=400.0, lr0=0.012, lr1=0.001):
        x = x_init.copy()
        vel = np.zeros_like(x)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return x
        for s in range(steps):
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel[idx] = mu * vel[idx] + (1.0 - mu) * gt[idx]
            x[idx] = x[idx] - lr * vel[idx]
            x[idx] = _nm(x[idx])
        return x

    def deepest_hole_dir(Xk):
        ns = 20000
        S = rng.randn(ns, 4)
        S = _nm(S)
        D = S @ Xk.T
        m = D.max(axis=1)
        return S[int(np.argmin(m))].copy()

    def best_insertion(cur, nsample=16384, ntop=8, iters=90):
        # sample candidates
        S = rng.randn(nsample, 4)
        S = _nm(S)
        D = S @ cur.T
        m = D.max(axis=1)
        part = np.argpartition(m, ntop)[:ntop]
        cand = S[part].copy()  # (ntop,4)
        # annealed softmax local optimisation of each candidate vs cur
        b0, b1 = 30.0, 260.0
        lr0, lr1 = 0.08, 0.005
        for it in range(iters):
            f = it / max(1, iters - 1)
            beta = b0 * (b1 / b0) ** f
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = cand @ cur.T  # (ntop, m)
            cm = np.max(dots, axis=1, keepdims=True)
            W = np.exp(np.clip(beta * (dots - cm), -20.0, 0.0))
            W /= (np.sum(W, axis=1, keepdims=True) + 1e-30)
            g = W @ cur  # (ntop,4)
            dg = np.sum(g * cand, axis=1, keepdims=True)
            gt = g - dg * cand
            cand = cand - lr * gt
            cand = _nm(cand)
        Df = cand @ cur.T
        mf = Df.max(axis=1)
        bi = int(np.argmin(mf))
        return cand[bi].copy()

    cap_cycle = [24, 32, 40]
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2.0:
            break
        cap_size = cap_cycle[(trial - 1) % len(cap_cycle)]
        mode = (trial - 1) % 3  # 0 random, 1 hole, 2 degree
        X = x_best.copy()
        # choose cap centre
        try:
            G0 = X @ X.T
            np.fill_diagonal(G0, -2.0)
            cmax0 = float(np.max(G0))
            deg0 = np.sum(G0 > cmax0 - 1e-7, axis=1)
        except Exception:
            deg0 = np.zeros(n)
            cmax0 = 0.85
        if mode == 0:
            centre = rng.randn(4)
            centre /= (np.linalg.norm(centre) + 1e-30)
        elif mode == 1:
            centre = deepest_hole_dir(X)
        else:
            i0 = int(np.argmax(deg0))
            # tie-break randomly among top
            top = np.argsort(-deg0, kind="stable")[:8]
            i0 = int(top[rng.randint(0, len(top))])
            centre = X[i0].copy()
        dots_c = X @ centre
        cap_idx = np.argsort(-dots_c, kind="stable")[:cap_size]
        # optionally also delete contact neighbours (every 3rd of mode 2 and periodic)
        if trial % 2 == 0:
            try:
                Gc = X @ X.T
                np.fill_diagonal(Gc, -2.0)
                cm = float(np.max(Gc))
                in_cap = np.zeros(n, dtype=bool)
                in_cap[cap_idx] = True
                # neighbours in contact with cap
                contact = Gc > cm - 1e-7
                add = []
                for j in range(n):
                    if not in_cap[j] and np.any(contact[j, cap_idx]):
                        add.append(j)
                # add up to 8 most aligned neighbours to keep size bounded
                if len(add) > 0:
                    add = np.array(add)
                    al = X[add] @ centre
                    order = np.argsort(-al, kind="stable")[:8]
                    add = add[order]
                    cap_idx = np.concatenate([cap_idx, add])
            except Exception:
                pass
        kdel = len(cap_idx)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cap_idx] = False
        cur_keep = X[keep_mask]  # (n-kdel,4)
        # reinsert one at a time in random order
        perm = rng.permutation(kdel)
        del _perm_dummy if False else None
        # build cur incrementally
        cur_list = [cur_keep]
        # use array that grows
        cur = cur_keep.copy()
        for _kk in range(kdel):
            if time.time() > SEARCH_END - 1.0:
                break
            h = best_insertion(cur, nsample=16384, ntop=8, iters=85)
            cur = np.vstack([cur, h[None, :]])
        else:
            _completed = True
        if cur.shape[0] != n:
            # aborted by time; skip
            continue
        Xnew = cur.copy()
        # jointly relax only points within cos>0.75 of a reinserted point
        try:
            new_pts = Xnew[-kdel:]
            Dprox = Xnew @ new_pts.T
            prox = Dprox.max(axis=1) > 0.75
        except Exception:
            prox = np.zeros(n, dtype=bool)
            prox[-kdel:] = True
        # ensure at least new points
        if np.sum(prox) < kdel:
            prox[-kdel:] = True
        Xnew = relax_mask(Xnew, prox, steps=300, beta0=25.0, beta1=400.0, lr0=0.012, lr1=0.001)
        # full refinement (shorter for speed)
        steps_full = 650 if kdel <= 32 else 750
        Xnew = refine_full(Xnew, steps_full)
        v = _max_cos(Xnew)
        if v < best_val - 2e-12:
            # LP polish if promising or periodic near-best
            do_lp = False
            if v < best_val - 5e-7 and time.time() < SEARCH_END - 4:
                do_lp = True
            elif v < best_val and (trial % 5 == 0) and time.time() < SEARCH_END - 5:
                do_lp = True
            if do_lp:
                try:
                    Xq = _lp_polish(Xnew, min(time.time() + 1.6, SEARCH_END))
                    vq = _max_cos(Xq)
                    if vq < v:
                        Xnew, v = Xq, vq
                except Exception:
                    pass
            if v < best_val:
                best_val = v
                x_best = Xnew.copy()
        # occasional small jitter fallback to keep diversity if stuck: every 12th trial do plain jitter refine
        if trial % 12 == 0 and time.time() < SEARCH_END - 3:
            try:
                Xj = x_best + 0.02 * rng.randn(n, d)
                Xj = _nm(Xj)
                Xj = refine_full(Xj, 400)
                vj = _max_cos(Xj)
                if vj < best_val:
                    best_val = vj
                    x_best = Xj.copy()
            except Exception:
                pass

    # final tighten + polish
    try:
        if time.time() < HARD_END - 4:
            x_best = refine_full(x_best, 350, beta0=150.0, beta1=900.0, lr0=0.004, lr1=0.0003)
    except Exception:
        pass
    remain = min(HARD_END - 1.0, time.time() + 24.0)
    if remain > time.time() + 1.0:
        try:
            x_best = _lp_polish(x_best, remain)
        except Exception:
            pass
    x_best = _nm(x_best)
    return x_best
