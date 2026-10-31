import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normr(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, 1e-12)

def _contact_mask(X, tol=1e-7):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = float(c.max())
    return (c > m - tol), m

def _degrees(X, tol=1e-7):
    n = X.shape[0]
    mask, m = _contact_mask(X, tol)
    iu = np.triu_indices(n, 1)
    deg = np.zeros(n, dtype=int)
    I = iu[0][mask]; J = iu[1][mask]
    for a, b in zip(I, J):
        deg[a] += 1; deg[b] += 1
    return deg, mask, m

def _anneal_full(x, steps, beta0, beta1, lr0, mu):
    n = x.shape[0]
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        Gt = A - (A * x).sum(axis=1, keepdims=True) * x
        v = mu * v + Gt
        v -= (v * x).sum(axis=1, keepdims=True) * x
        x = x - lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _anneal_masked(x, opt_idx, steps, beta0, beta1, lr0, mu):
    if len(opt_idx) == 0:
        return x
    opt_idx = np.asarray(opt_idx, dtype=int)
    v = np.zeros((len(opt_idx), 4), dtype=np.float64)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        # clip exponent for stability
        E = beta * (G - gmax)
        np.clip(E, -60.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        Gt = A - (A * x).sum(axis=1, keepdims=True) * x
        Go = Gt[opt_idx]
        v = mu * v + Go
        v -= (v * x[opt_idx]).sum(axis=1, keepdims=True) * x[opt_idx]
        x[opt_idx] = x[opt_idx] - lr * v
        x[opt_idx] /= np.maximum(np.linalg.norm(x[opt_idx], axis=1, keepdims=True), 1e-12)
    return x

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normr(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
            m = float(c.max())
        except Exception:
            break
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest only to bound LP size
            idx = np.argsort(c[sel])[-4000:]
            I, J = I[idx], J[idx]
            p = len(I)
            csv = c[sel][idx]
        else:
            csv = c[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normr(y)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_holes(H, Xr, steps=100, beta0=40.0, beta1=600.0, lr=0.04):
    # H (Hn,4), Xr (M,4): minimize max dot
    Hn = H.shape[0]
    if Hn == 0:
        return H
    lb = np.log(beta1 / beta0)
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        G = H @ Xr.T
        gmax = G.max(axis=1, keepdims=True)
        E = beta * (G - gmax)
        np.clip(E, -60.0, 0.0, out=E)
        W = np.exp(E) + 1e-300
        W /= W.sum(axis=1, keepdims=True)
        Grad = W @ Xr
        Gt = Grad - (Grad * H).sum(axis=1, keepdims=True) * H
        H = H - lr * (1.0 - 0.5 * frac) * Gt
        H = _normr(H)
    return H

def solve():
    t0 = time.time()
    TLIM = 170.0
    rng = np.random.default_rng(292016)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nr) > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        Xb = rng.standard_normal((n, d)); Xb = _normr(Xb)
    else:
        vals = [_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    try:
        best_mask, _ = _contact_mask(X_best, 1e-7)
    except Exception:
        best_mask = None
    pool = [X_best.copy()]
    pool_vals = [best_val]
    pool_masks = [best_mask]
    ks = [4, 8, 12, 16, 20]
    it = 0
    explore_end = t0 + TLIM - 14.0
    NSAMP = 65536
    CHUNK = 8192
    while time.time() < explore_end:
        it += 1
        rem = explore_end - time.time()
        if rem < 2.5:
            break
        # pick base: mostly best, sometimes diverse basin
        if len(pool) > 1 and rng.random() < 0.30:
            bi = int(rng.integers(len(pool)))
            base = pool[bi]
            base_val = pool_vals[bi]
        else:
            base = X_best
            base_val = best_val
        try:
            deg, cmask, mval = _degrees(base, 1e-7)
        except Exception:
            deg = np.full(n, 6, dtype=int)
            mval = base_val
        ratt = np.where(deg == 0)[0]
        low = np.where(deg <= 4)[0]
        high = np.where(deg >= 8)[0]
        # row maxima for worst points
        try:
            Gf = base @ base.T
            np.fill_diagonal(Gf, -2.0)
            rowmax = Gf.max(axis=1)
            worst = np.argsort(rowmax)[-24:]
        except Exception:
            worst = np.arange(n)[-24:]
        k = ks[it % len(ks)]
        if rng.random() < 0.5:
            k = int(rng.choice(np.array(ks)))
        # build deletion set mixture
        delset = set()
        if len(ratt) > 0:
            # include at least one rattler
            delset.add(int(ratt[int(rng.integers(len(ratt)))]))
        # targets: ~35% low, ~30% high, rest worst/random
        n_low = min(len(low), int(0.35 * k) + (1 if rng.random() < 0.5 else 0))
        n_high = min(len(high), int(0.30 * k) + (1 if rng.random() < 0.5 else 0))
        if n_low > 0 and len(low) > 0:
            perm = rng.permutation(low)
            for v in perm[:n_low * 2]:
                if len([x for x in delset if x in set(low)]) >= n_low:
                    break
                delset.add(int(v))
                if len(delset) >= k:
                    break
        if len(delset) < k and n_high > 0 and len(high) > 0:
            perm = rng.permutation(high)
            for v in perm:
                if len(delset) >= k:
                    break
                # avoid adding too many adjacent? just add
                delset.add(int(v))
                # count high added
                if sum(1 for x in delset if x in set(high)) >= n_high + 2:
                    break
        if len(delset) < k:
            perm = rng.permutation(worst)
            for v in perm:
                if len(delset) >= k:
                    break
                delset.add(int(v))
        while len(delset) < k:
            delset.add(int(rng.integers(n)))
        del_idx = np.array(sorted(delset)[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool); keep_mask[del_idx] = False
        Xr = base[keep_mask]  # (n-k,4)
        # sample NSAMP holes
        try:
            best_h = None; best_s = None
            # we need top few hundred: keep running top 400
            TOP = 400
            cand_pool = np.empty((0, 4), dtype=np.float64)
            score_pool = np.empty((0,), dtype=np.float64)
            for s0 in range(0, NSAMP, CHUNK):
                sz = min(CHUNK, NSAMP - s0)
                S = rng.standard_normal((sz, 4)); S = _normr(S)
                # max dot vs Xr in chunks of Xr to save memory? direct is sz x (n-k) ~ 8192x288 ~2.3M fine
                sc = S @ Xr.T
                mx = sc.max(axis=1)
                cand_pool = np.concatenate([cand_pool, S], axis=0) if cand_pool.shape[0] > 0 else S
                score_pool = np.concatenate([score_pool, mx], axis=0) if score_pool.size > 0 else mx
                # prune to TOP*2 to bound memory
                if cand_pool.shape[0] > TOP * 4:
                    oi = np.argsort(score_pool)[:TOP * 2]
                    cand_pool = cand_pool[oi]; score_pool = score_pool[oi]
            order = np.argsort(score_pool)[:TOP]
            H0 = cand_pool[order]
            Hscores = score_pool[order]
        except Exception:
            continue
        # refine best few hundred holes
        try:
            Hn_ref = min(300, H0.shape[0])
            Hr = _refine_holes(H0[:Hn_ref].copy(), Xr, steps=90, beta0=40.0, beta1=600.0, lr=0.05)
            # rescore refined
            rsc = Hr @ Xr.T
            rmax = rsc.max(axis=1)
            rorder = np.argsort(rmax)
            Hr = Hr[rorder]; rmax = rmax[rorder]
        except Exception:
            continue
        # greedy insertion under temporary separation target below 0.90347
        cap = min(base_val, 0.90347)
        # start strict then relax
        targets = [cap - 3e-04, cap - 1.5e-04, cap - 5e-05, cap + 0.0, cap + 2e-04]
        # ensure first target is below 0.90347
        targets = [min(t, 0.90347 - 1e-05) if i < 3 else t for i, t in enumerate(targets)]
        picked = None
        used_target = targets[0]
        try:
            for tt in targets:
                if time.time() > explore_end - 1.0:
                    break
                sel = []
                sel_scores = []
                for hi in range(Hr.shape[0]):
                    if len(sel) >= k:
                        break
                    h = Hr[hi]
                    if rmax[hi] >= tt:
                        # since sorted, later ones worse; but still need check if later could be ok? no, sorted ascending so break
                        # allow skipping? sorted so all later >= current, break
                        break
                    if len(sel) > 0:
                        dp = float(np.max(np.asarray(sel) @ h))
                        if dp >= tt:
                            continue
                    sel.append(h); sel_scores.append(rmax[hi])
                # if not enough, try scanning all (not just prefix) for compatible set
                if len(sel) < k:
                    sel = []
                    for hi in range(Hr.shape[0]):
                        if len(sel) >= k:
                            break
                        h = Hr[hi]
                        dm_ret = float(h @ Xr.T.max(axis=0).max()) if False else float((Xr @ h).max())
                        if dm_ret >= tt:
                            continue
                        if len(sel) > 0:
                            dp = float(np.max(np.asarray(sel) @ h))
                            if dp >= tt:
                                continue
                        sel.append(h)
                if len(sel) >= k:
                    picked = np.array(sel[:k])
                    used_target = tt
                    break
            if picked is None:
                continue
        except Exception:
            continue
        # assemble full config
        try:
            X = np.empty((n, 4), dtype=np.float64)
            X[keep_mask] = Xr
            X[del_idx] = picked[:k]
            X = _normr(X)
        except Exception:
            continue
        # local optimisation: inserted + neighbours with cos>0.75
        try:
            dots_ins = X[del_idx] @ X.T  # k x n
            neigh = np.where(dots_ins.max(axis=0) > 0.75)[0]
            opt_idx = np.unique(np.concatenate([del_idx, neigh]))
            # limit opt size to avoid full problem disguised as local
            if len(opt_idx) > 120:
                # keep inserted + closest neighbours
                mxn = dots_ins.max(axis=0)
                o2 = np.argsort(mxn)[-120:]
                opt_idx = np.unique(np.concatenate([del_idx, o2]))
            X = _anneal_masked(X, opt_idx, steps=320, beta0=30.0, beta1=500.0, lr0=0.06, mu=0.85)
            # annealed softmax global short
            X = _anneal_full(X, steps=260, beta0=25.0, beta1=420.0, lr0=0.055, mu=0.86)
        except Exception:
            continue
        try:
            v_pre = _max_cos(X)
        except Exception:
            continue
        # gate LP polish: only if promising or diverse enough to warrant
        gate = min(best_val + 4e-04, cap + 4e-04)
        if v_pre > gate and (it % 4 != 0):
            # keep occasional unpolished diverse check without LP? skip to save time
            continue
        dl = min(time.time() + 2.2, explore_end)
        if dl - time.time() < 0.6:
            continue
        try:
            Y = _lp_polish_deadline(X, dl)
            vy = _max_cos(Y)
        except Exception:
            continue
        # contact-graph novelty
        try:
            ymask, _ = _contact_mask(Y, 1e-7)
            # diff vs best
            if best_mask is not None and len(ymask) == len(best_mask):
                diff_best = int(np.sum(ymask ^ best_mask))
            else:
                diff_best = 999
            # diff vs pool min
            mindiff = diff_best
            for pm in pool_masks:
                if pm is not None and len(pm) == len(ymask):
                    dd = int(np.sum(ymask ^ pm))
                    if dd < mindiff:
                        mindiff = dd
        except Exception:
            mindiff = 999; diff_best = 999
        # banking rule: retain if diff>=20 even without record
        if mindiff >= 20:
            if len(pool) < 6:
                pool.append(Y.copy()); pool_vals.append(vy); pool_masks.append(ymask)
            else:
                # replace worst pool member if new is better or more novel
                wi = int(np.argmax(pool_vals))
                if vy < pool_vals[wi]:
                    pool[wi] = Y.copy(); pool_vals[wi] = vy; pool_masks[wi] = ymask
        if vy < best_val:
            X_best = Y.copy(); best_val = float(vy)
            best_mask = ymask
            pool[0] = X_best.copy(); pool_vals[0] = best_val; pool_masks[0] = best_mask
            # early exit if strong record? continue to improve
        elif vy < best_val + 1e-06 and diff_best >= 20:
            # near-record but novel: keep as is already banked
            pass
    # final polish best
    try:
        if time.time() < t0 + TLIM - 1.0:
            X_best = _lp_polish_deadline(X_best, t0 + TLIM - 0.7)
    except Exception:
        pass
    X_best = _normr(X_best)
    return X_best
