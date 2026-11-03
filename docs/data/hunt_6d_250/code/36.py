import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _single_refine(p0, F, iters=60, b0=150.0, b1=3000.0, lr=0.025, seed_mom=None):
    p = p0 / (np.linalg.norm(p0) + 1e-16)
    m = np.zeros(6)
    v = np.zeros(6)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0); lb1 = np.log(b1)
    for s in range(iters):
        frac = s / (iters - 1) if iters > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        d = F @ p
        cm = float(np.max(d))
        e = np.clip(beta * (d - cm), -30.0, 0.0)
        w = np.exp(e)
        S = float(np.sum(w))
        if S < 1e-100:
            continue
        grad = (w @ F) / S
        gt = grad - float(np.dot(grad, p)) * p
        t += 1
        m = b1m * m + (1.0 - b1m) * gt
        v = b2m * v + (1.0 - b2m) * (gt * gt)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = float(np.linalg.norm(step))
        cap = 5.0 * lr
        if sn > cap:
            step *= (cap / (sn + 1e-16))
        p = p - step
        nr = float(np.linalg.norm(p))
        if nr == 0 or not np.isfinite(nr):
            p = p0 / (np.linalg.norm(p0) + 1e-16)
            m[:] = 0; v[:] = 0; t = 0
            continue
        p /= nr
        md = float(np.dot(m, p))
        m -= md * p
    return p / (np.linalg.norm(p) + 1e-16)

def _block_joint(Block, Frozen, steps=150, b0=120.0, b1=4000.0, lr=0.003):
    B = Block.copy()
    B = _norm(B)
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    b1m = 0.9; b2m = 0.999; eps = 1e-8
    t = 0
    lb0 = np.log(b0); lb1 = np.log(b1)
    k = B.shape[0]
    has_intra = k > 1
    if has_intra:
        iu = np.triu_indices(k, 1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        C = B @ Frozen.T
        cm_c = float(np.max(C)) if C.size else -2.0
        if has_intra:
            G = B @ B.T
            # max over off-diag
            cm_g = float(np.max(G[iu])) if iu[0].size else -2.0
            cm = cm_c if cm_c > cm_g else cm_g
            Wc = np.exp(np.clip(beta * (C - cm), -30.0, 0.0))
            Wi = np.exp(np.clip(beta * (G - cm), -30.0, 0.0))
            np.fill_diagonal(Wi, 0.0)
            Sc = float(np.sum(Wc))
            Si = float(np.sum(Wi)) * 0.5
            S = Sc + Si
            if S < 1e-100:
                continue
            Grad = (Wc @ Frozen + Wi @ B) / S
        else:
            cm = cm_c
            Wc = np.exp(np.clip(beta * (C - cm), -30.0, 0.0))
            S = float(np.sum(Wc))
            if S < 1e-100:
                continue
            Grad = (Wc @ Frozen) / S
        dot = np.sum(Grad * B, axis=1, keepdims=True)
        Grad = Grad - dot * B
        t += 1
        m = b1m * m + (1.0 - b1m) * Grad
        v = b2m * v + (1.0 - b2m) * (Grad * Grad)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr
        sc = np.minimum(1.0, cap / sn)
        step = step * sc
        B = B - step
        B = _norm(B)
        md = np.sum(m * B, axis=1, keepdims=True)
        m -= md * B
    return _norm(B)

def _adam_global(X0, steps, b0, b1, lr):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1m = 0.9; b2m = 0.999; eps = 1e-8
    t = 0
    lb0 = np.log(b0); lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = (G @ X) * (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m = b1m * m + (1.0 - b1m) * GX
        v = b2m * v + (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t); c2 = 1.0 - (b2m ** t)
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        step *= np.minimum(1.0, cap / sn)
        X -= step
        X = _norm(X)
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _lp_polish(x, deadline, k_top=3500, r0=1.0e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        mval = float(np.max(c_all))
        sel = c_all > mval - max(20.0 * radius, 1e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
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
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _component(X, thresh):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    flat = int(np.argmax(G))
    i0, j0 = divmod(flat, n)
    mx = float(G[i0, j0])
    adj = G > (mx - 2e-5)
    # BFS from i0
    seen = np.zeros(n, dtype=bool)
    seen[i0] = True
    stack = [int(i0)]
    while stack:
        u = stack.pop()
        nbrs = np.where(adj[u])[0]
        for w in nbrs:
            ww = int(w)
            if not seen[ww]:
                seen[ww] = True
                stack.append(ww)
    comp = np.where(seen)[0].tolist()
    return G, mx, int(i0), int(j0), comp

def _pick_excision(G, X, comp, target, i_star, j_star):
    comp_set = set(int(c) for c in comp)
    comp_set.add(i_star); comp_set.add(j_star)
    comp = sorted(comp_set)
    selected = []
    sset = set()
    for s in (i_star, j_star):
        if s not in sset:
            selected.append(s); sset.add(s)
    # grow within component by connectivity strength
    while len(selected) < min(target, len(comp)):
        cands = [c for c in comp if c not in sset]
        if not cands:
            break
        # score = max edge to selected
        sub = G[np.ix_(np.array(cands, dtype=int), np.array(selected, dtype=int))]
        scores = np.max(sub, axis=1)
        bi = int(np.argmax(scores))
        pick = cands[bi]
        selected.append(pick); sset.add(pick)
    # if still short, expand by proximity to selected set
    n = X.shape[0]
    if len(selected) < target:
        Spts = X[np.array(selected, dtype=int)]
        while len(selected) < target:
            remaining = [i for i in range(n) if i not in sset]
            if not remaining:
                break
            R = X[np.array(remaining, dtype=int)] @ Spts.T
            mxv = np.max(R, axis=1)
            bi = int(np.argmax(mxv))
            pick = remaining[bi]
            selected.append(pick); sset.add(pick)
            Spts = X[np.array(selected, dtype=int)]
    # if too many (should not), truncate already handled
    return sorted(selected)[:target]

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    REGROW_END = t0 + 140.0
    n, d = 250, 6
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (n, d):
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        z = rng0.standard_normal((n, d))
        cands.append(_norm(z))
    best = min(cands, key=_max_cos).copy()
    best = _norm(best)
    best_val = _max_cos(best)
    fallback = best.copy()

    sizes_cycle = [25, 40, 15, 30, 20, 35, 25, 40, 15, 30, 22, 28]
    seeds_cycle = [101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333,
                   444, 555, 666, 777, 888, 999, 1234, 5678]
    NC = 8192
    trial = 0
    cfg_idx = 0
    while time.time() < REGROW_END and time.time() < HARD - 30:
        target = sizes_cycle[cfg_idx % len(sizes_cycle)]
        seed = seeds_cycle[cfg_idx % len(seeds_cycle)] + trial * 7919 + target * 131
        cfg_idx += 1
        trial += 1
        if time.time() > REGROW_END - 2.0:
            break
        try:
            G, mx, i_star, j_star, comp = _component(best, mx_thresh := None) if False else _component(best, None)
        except Exception:
            continue
        try:
            exc = _pick_excision(G, best, comp, int(target), i_star, j_star)
        except Exception:
            continue
        k = len(exc)
        if k < 5:
            continue
        exc_set = set(exc)
        frozen_idx = [i for i in range(n) if i not in exc_set]
        Frozen0 = best[np.array(frozen_idx, dtype=int)]
        rng = np.random.default_rng(int(seed) % (2**32 - 1))
        grown = []  # list of vectors
        ok = True
        for pos in range(k):
            if time.time() > REGROW_END - 1.0 or time.time() > HARD - 28:
                ok = False
                break
            try:
                cand = rng.standard_normal((NC, d))
                cand = _norm(cand)
                if grown:
                    ref = np.vstack([Frozen0, np.vstack(grown)])
                else:
                    ref = Frozen0
                S = cand @ ref.T
                mxv = np.max(S, axis=1)
                bi = int(np.argmin(mxv))
                p0 = cand[bi]
                p = _single_refine(p0, ref, iters=60, b0=150.0, b1=3000.0, lr=0.025)
                grown.append(p)
            except Exception:
                ok = False
                break
            # joint re-opt every 5 insertions
            if len(grown) % 5 == 0 and len(grown) >= 5:
                try:
                    B = np.vstack(grown)
                    Bj = _block_joint(B, Frozen0, steps=120, b0=120.0, b1=3500.0, lr=0.004)
                    grown = [Bj[i] for i in range(Bj.shape[0])]
                except Exception:
                    pass
        if not ok or len(grown) != k:
            continue
        try:
            B = np.vstack(grown)
            # final joint of inserted block
            B = _block_joint(B, Frozen0, steps=200, b0=150.0, b1=5000.0, lr=0.003)
            Xnew = best.copy()
            for ii, idx in enumerate(exc):
                Xnew[int(idx)] = B[ii]
            Xnew = _norm(Xnew)
            # short collective polish: light global adam (few steps) focused?
            # do a quick block-vs-all + tiny global to settle
            # evaluate first
            nv = _max_cos(Xnew)
            # if promising (within 2e-3 of best) do extra polish
            if nv < best_val + 3e-3:
                try:
                    # joint again with slightly higher beta
                    B2 = _block_joint(B, Frozen0, steps=120, b0=500.0, b1=8000.0, lr=0.0015)
                    X2 = best.copy()
                    for ii, idx in enumerate(exc):
                        X2[int(idx)] = B2[ii]
                    X2 = _norm(X2)
                    nv2 = _max_cos(X2)
                    if nv2 < nv:
                        Xnew, nv = X2, nv2
                except Exception:
                    pass
                # small global adam to tie plateau
                try:
                    if nv < best_val + 1e-3 and time.time() < REGROW_END - 5:
                        Xp = _adam_global(Xnew, 250, 200.0, 4000.0, 0.001)
                        vp = _max_cos(Xp)
                        if vp < nv:
                            Xnew, nv = Xp, vp
                except Exception:
                    pass
            if nv < best_val - 1e-12:
                best, best_val = _norm(Xnew), nv
        except Exception:
            continue
        # early exit if time low
        if time.time() >= REGROW_END:
            break

    # short collective polish with remaining time
    try:
        if time.time() < HARD - 8:
            # annealed adam squeeze
            steps_left = 600 if time.time() < HARD - 40 else 250
            Xp = _adam_global(best, steps_left, 300.0, 7000.0, 0.0007)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
    except Exception:
        pass
    try:
        if time.time() < HARD - 12:
            best = _lp_polish(best, min(HARD - 1.0, time.time() + 18.0), k_top=3500, r0=1.0e-3)
            best_val = _max_cos(best)
    except Exception:
        pass
    try:
        if time.time() < HARD - 6:
            Xp = _adam_global(best, 300, 800.0, 9000.0, 0.0004)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
    except Exception:
        pass
    best = _norm(best)
    # safety fallback
    try:
        fv = _max_cos(fallback)
        bv = _max_cos(best)
        if not np.isfinite(bv) or bv > fv:
            # keep the better of the two (prefer improvement, else fallback)
            if bv > fv:
                best = fallback
    except Exception:
        best = fallback
    best = _norm(np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0))
    # ensure no zero rows
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (~np.isfinite(nr[:, 0])) | (nr[:, 0] < 1e-12)
    if np.any(bad):
        rngf = np.random.default_rng(999)
        for i in np.where(bad)[0]:
            v = rngf.standard_normal(6)
            v /= np.linalg.norm(v) + 1e-16
            best[i] = v
    best = _norm(best)
    return best
