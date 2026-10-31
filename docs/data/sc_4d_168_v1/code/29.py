import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 142.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and getattr(c, 'shape', None) == (168, 4):
            c = np.asarray(c, dtype=float)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if np.all(np.isfinite(c)):
                cands.append(c)
    if len(cands) == 0:
        x_best = rng.randn(168, 4)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        x_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(x_best)
    n, d = 168, 4

    def refine_full(x_init, steps, mu=0.9, beta0=30.0, beta1=400.0, lr0=0.012, lr1=0.0008):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s > 200 and time.time() >= SEARCH_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def batch_opt_cands(cand, Xcur, iters=80, b0=30.0, b1=250.0, l0=0.08, l1=0.005):
        c = cand.copy()
        for it in range(iters):
            f = it / max(1, iters - 1)
            beta = b0 * (b1 / b0) ** f
            lr = l0 * (1.0 - f) + l1 * f
            Dd = c @ Xcur.T
            cm = Dd.max(axis=1, keepdims=True)
            W = np.exp(np.clip(beta * (Dd - cm), -20.0, 0.0))
            W /= (W.sum(axis=1, keepdims=True) + 1e-30)
            grad = W @ Xcur
            dot = np.sum(grad * c, axis=1, keepdims=True)
            gt = grad - dot * c
            c = c - lr * gt
            c /= np.linalg.norm(c, axis=1, keepdims=True)
        return c

    def deepest_hole_dir(X, ns=8000):
        S = rng.randn(ns, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X.T
        m = D.max(axis=1)
        return S[int(np.argmin(m))]

    def partial_relax(x_init, active, steps=300, mu=0.9, beta0=30.0, beta1=280.0, lr0=0.012, lr1=0.001):
        x = x_init.copy()
        vel = np.zeros_like(x)
        idx = np.where(active)[0]
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
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel[idx] = mu * vel[idx] + (1.0 - mu) * gt[idx]
            x[idx] = x[idx] - lr * vel[idx]
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
        return x

    CAPS = [24, 32, 40]
    CENTRES = ["random", "hole", "high"]
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2.0:
            break
        k = CAPS[(trial - 1) % len(CAPS)]
        # cycle centres slower, also vary with trial
        ctype = CENTRES[((trial - 1) // len(CAPS)) % len(CENTRES)]
        # optionally expand with contact neighbours flag
        expand = (rng.rand() < 0.3)
        x = x_best.copy()
        # contact degrees for high centre
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        cmax0 = float(np.max(G0))
        deg0 = np.sum(G0 > cmax0 - 0.005, axis=1)
        if ctype == "random":
            centre = rng.randn(4)
            centre /= np.linalg.norm(centre)
        elif ctype == "hole":
            try:
                centre = deepest_hole_dir(x, ns=6000)
            except Exception:
                centre = rng.randn(4); centre /= np.linalg.norm(centre)
        else:
            i0 = int(np.argmax(deg0 + rng.rand(n) * 0.5))
            centre = x[i0].copy()
        dots_c = x @ centre
        del_idx = np.argsort(-dots_c, kind="stable")[:k]
        if expand:
            # also delete contact neighbours: replace farthest few of cap with neighbours of cap core
            core = del_idx[:8]
            nbr_set = set(del_idx.tolist())
            for ci in core:
                nbrs = np.argsort(-G0[ci], kind="stable")[:4]
                for nb in nbrs:
                    nbr_set.add(int(nb))
                    if len(nbr_set) >= k + 6:
                        break
                if len(nbr_set) >= k + 6:
                    break
            # keep size exactly k by taking most aligned among expanded ∪ cap? keep k largest dots with small random tiebreak
            cand_list = list(nbr_set)
            cd = dots_c[cand_list] + 1e-6 * rng.rand(len(cand_list))
            order = np.argsort(-cd, kind="stable")[:k]
            del_idx = np.array([cand_list[o] for o in order], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        X_keep = x[keep_mask]  # (n-k,4)
        # reinsert one at a time in random order
        perm = rng.permutation(k)
        inv_assign = np.empty(k, dtype=int)
        # map insertion step -> position in del_idx
        ordered_slots = del_idx[perm]
        cur_list = [X_keep]
        # maintain stacked array incrementally
        try:
            Xcur = X_keep.copy()
        except Exception:
            continue
        new_pts = []
        NSAMP = 16384
        fail = False
        for step in range(k):
            if time.time() > SEARCH_END - 1.0:
                fail = True
                break
            try:
                S = rng.randn(NSAMP, 4)
                S /= np.linalg.norm(S, axis=1, keepdims=True)
                D = S @ Xcur.T
                worst = D.max(axis=1)
                # 8 candidates minimising max cosine
                p8 = np.argpartition(worst, 8)[:8]
                cand8 = S[p8]
                cand8r = batch_opt_cands(cand8, Xcur, iters=80)
                Df = cand8r @ Xcur.T
                wf = Df.max(axis=1)
                best_c = cand8r[int(np.argmin(wf))]
            except Exception:
                # fallback greedy
                S = rng.randn(4000, 4)
                S /= np.linalg.norm(S, axis=1, keepdims=True)
                D = S @ Xcur.T
                worst = D.max(axis=1)
                best_c = S[int(np.argmin(worst))]
            new_pts.append(best_c)
            try:
                Xcur = np.vstack([Xcur, best_c[None, :]])
            except Exception:
                fail = True
                break
        if fail:
            continue
        x_new = x.copy()
        for s_i, slot in enumerate(ordered_slots):
            x_new[slot] = new_pts[s_i]
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        # jointly relax only points within cosine > 0.75 of a reinserted point
        try:
            Gr = x_new @ x_new.T
            # max dot to any reinserted slot
            sub = Gr[:, ordered_slots]
            mx = sub.max(axis=1)
            active = mx > 0.75
            # ensure at least reinserted are active
            active[ordered_slots] = True
            x_new = partial_relax(x_new, active, steps=300)
        except Exception:
            pass
        # full refinement
        try:
            xo = refine_full(x_new, steps=900, beta0=25.0, beta1=420.0, lr0=0.012, lr1=0.0007)
        except Exception:
            continue
        v = _max_cos(xo)
        improved = False
        if v < best_val:
            best_val = v
            x_best = xo.copy()
            improved = True
        do_lp = False
        lp_budget = 0.0
        if improved:
            do_lp = True; lp_budget = 2.0
        elif v < best_val + 0.0015 and (trial % 3 == 0):
            do_lp = True; lp_budget = 0.9
        elif v < best_val + 0.004 and (trial % 10 == 0):
            do_lp = True; lp_budget = 0.7
        if do_lp and time.time() < SEARCH_END - 0.5:
            try:
                xp = _lp_polish(xo, min(time.time() + lp_budget, SEARCH_END + 4.0, HARD_END - 6.0))
                vp = _max_cos(xp)
                if vp < best_val:
                    best_val = vp
                    x_best = xp.copy()
            except Exception:
                pass
    rem_end = min(HARD_END - 1.0, time.time() + max(5.0, HARD_END - time.time() - 1.0))
    if time.time() < HARD_END - 2.0:
        try:
            x_best = _lp_polish(x_best, rem_end)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
