import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, t_dead=None, init_radius=1e-3, cap=2500):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead - 0.5:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            # keep tightest cap
            thr = np.partition(c_all, -cap)[-cap]
            sel = c_all >= thr
            # if still too many, truncate
            idx = np.where(sel)[0]
            if len(idx) > cap + 200:
                order = np.argsort(-c_all[idx])[:cap]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2*d] = J[:, None] * d + ar
        cols_mat[:, 2*d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2*d] = x[I]
        vals_mat[:, 2*d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _get_duals(X, tol=1e-7, cap=2500, radius=1e-3):
    n, d = X.shape
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    sel = c_all > m - tol
    if int(sel.sum()) == 0:
        sel = c_all >= m - 1e-12
    if int(sel.sum()) > cap:
        thr = np.partition(c_all, -cap)[-cap]
        nsel = c_all >= thr
        idx = np.where(nsel)[0]
        if len(idx) > cap:
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(nsel)
            mask[idx[order]] = True
            nsel = mask
        sel = nsel
    I = iu[0][sel]; J = iu[1][sel]
    csel = c_all[sel]
    p = len(I)
    duals = np.ones(p)
    try:
        nv = n * d + 1
        ar = np.arange(d)
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2*d] = J[:, None] * d + ar
        cols_mat[:, 2*d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2*d] = x[I]
        vals_mat[:, 2*d] = -1.0
        vals = vals_mat.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status == 0:
            marg = None
            try:
                marg = np.asarray(res.ineqlin.marginals, dtype=np.float64)
            except Exception:
                try:
                    marg = np.asarray(res['ineqlin']['marginals'], dtype=np.float64)
                except Exception:
                    marg = None
            if marg is not None and marg.shape[0] == p and np.all(np.isfinite(marg)):
                duals = np.abs(marg)
                if duals.max() <= 0:
                    duals = np.ones(p)
            else:
                # fallback: weight by closeness to max
                duals = np.ones(p)
    except Exception:
        pass
    return I, J, csel, duals, m

def _refine_single_batch(V, Y, beta=170.0, iters=110, lr0=0.12, lr1=0.008):
    S_n = V.shape[0]
    for it in range(iters):
        S = V @ Y.T
        Sm = S.max(axis=1, keepdims=True)
        E = np.exp(beta * (S - Sm))
        Ws = E.sum(axis=1, keepdims=True) + 1e-300
        W = E / Ws
        Gr = W @ Y
        dot = np.einsum('ij,ij->i', Gr, V)[:, None]
        Gt = Gr - dot * V
        frac = it / max(1, iters - 1)
        lr = lr0 + (lr1 - lr0) * frac
        V = V - lr * Gt
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def _global_smooth(X, steps, k0=40.0, k1=220.0, t_dead=None):
    n, d = X.shape
    for s in range(steps):
        if t_dead is not None and time.time() >= t_dead - 1.0:
            break
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        lr = 0.02 * (1.0 - frac) + 0.003 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / (nrm + 1e-12)) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    T_DEAD = t0 + 155.0
    rng = np.random.default_rng(7)
    # load incumbent
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9) and np.all(np.isfinite(a)):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nrm > 1e-12):
                    cand.append(a / nrm)
        except Exception:
            pass
    if len(cand) == 0:
        a = rng.standard_normal((181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    scores = [_max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    x_inc = cand[bi].copy()
    b_inc = float(scores[bi])

    # one LP polish
    rem = T_DEAD - time.time()
    if rem > 8:
        x_pol = _lp_polish(x_inc, budget=min(10.0, rem - 5.0), t_dead=T_DEAD, init_radius=1e-3)
        v = _max_cos(x_pol)
        if v < b_inc and np.all(np.isfinite(x_pol)):
            x_inc, b_inc = x_pol, v
    x_base = x_inc.copy()

    # dual stress
    try:
        I, J, csel, duals, m = _get_duals(x_base, tol=1e-7, cap=2500, radius=1e-3)
    except Exception:
        G = x_base @ x_base.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(181, 1)
        c_all = G[iu]
        sel = c_all > m - 1e-7
        I = iu[0][sel]; J = iu[1][sel]
        duals = np.ones(len(I))
    n, d = x_base.shape
    # contacts per point
    Gf = x_base @ x_base.T
    np.fill_diagonal(Gf, -2.0)
    cnt = (Gf > m - 1e-9).sum(axis=1)
    rattlers = list(np.where(cnt == 0)[0])
    # point stress
    pstress = np.zeros(n)
    np.add.at(pstress, I, np.abs(duals))
    np.add.at(pstress, J, np.abs(duals))
    # if duals degenerate uniform, use counts
    if pstress.max() <= 1e-12:
        pstress = cnt.astype(float)
    # ensure rattlers excluded from seed
    # build adjacency for contact graph (use tol 1e-7 pairs I,J)
    adj = [set() for _ in range(n)]
    edge_w = {}
    for ii, jj, w in zip(I, J, np.abs(duals)):
        adj[ii].add(jj); adj[jj].add(ii)
        edge_w[(ii, jj)] = w
        edge_w[(jj, ii)] = w
    # pick core cluster size 14
    KCORE = 14
    # order candidates by stress descending excluding rattlers
    ratt_set = set(rattlers)
    # if less than 2 rattlers, add lowest-stress as rattler-like
    if len(rattlers) < 2:
        order_lo = np.argsort(pstress)
        for idx in order_lo:
            if int(idx) not in ratt_set:
                rattlers.append(int(idx))
                ratt_set.add(int(idx))
            if len(rattlers) >= 2:
                break
    else:
        rattlers = rattlers[:2]
        ratt_set = set(rattlers)
    # seed: max stress not rattler
    cand_idx = [i for i in range(n) if i not in ratt_set]
    cand_idx.sort(key=lambda i: pstress[i], reverse=True)
    seed = cand_idx[0] if len(cand_idx) > 0 else 0
    cluster = set([seed])
    while len(cluster) < KCORE:
        # frontier neighbours of cluster
        best_c = -1
        best_s = -1e100
        for u in list(cluster):
            for v in adj[u]:
                if v in cluster or v in ratt_set:
                    continue
                s = pstress[v]
                if s > best_s:
                    best_s = s
                    best_c = v
        if best_c == -1:
            # disconnected: pick next highest stress outside
            for v in cand_idx:
                if v not in cluster:
                    best_c = v
                    break
            if best_c == -1:
                break
        cluster.add(int(best_c))
        if len(cluster) >= n - 2:
            break
    remove_idx = list(ratt_set.union(cluster))
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove_idx] = False
    Kmat = x_base[keep_mask].copy()  # frozen
    removed_old = x_base[remove_idx].copy()

    # reinsert one at a time
    Y = Kmat.copy()
    # pre-generate hole pools? generate per insertion
    n_ins = len(remove_idx)
    new_pts = []
    for k in range(n_ins):
        if time.time() >= T_DEAD - 25.0:
            # not enough time: fill remaining with old positions greedily projected
            # just append remaining old points
            for kk in range(k, n_ins):
                new_pts.append(removed_old[kk - k if False else 0] if False else removed_old[kk])
            # Actually removed_old order corresponds; use them directly
            # rebuild: new_pts already has k items, need rest
            # fill with old
            # (we already appended one extra, fix)
            # remove the wrongly appended and add correct
            # Simpler: if we broke at k, add all remaining old
            # new_pts currently has k items (since we appended after optimize). At break point k, it has k items.
            # So add removed_old[k:]? but removed_old order vs insertion order may differ; just add them.
            # We already may have added one incorrectly if break at top; handle:
            # At this branch we haven't optimized k yet, so new_pts has k items. Add old[k:].
            # If we entered here, new_pts length == k, so extend:
            # (no double count)
            pass
            # extend with remaining old positions (best effort: deepest hole single pick quickly)
            # To avoid complexity, just extend with old vectors
            # new_pts currently length k, need n_ins-k more
            need = n_ins - len(new_pts)
            # take last 'need' old vectors
            # Use old vectors not yet used: just cycle removed_old
            for t in range(need):
                new_pts.append(removed_old[(len(new_pts)) % n_ins].copy())
            break
        # 50k sphere samples for holes
        try:
            C = rng.standard_normal((50000, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ Y.T
            mx = S.max(axis=1)
            # deepest holes: smallest mx
            hidx = np.argpartition(mx, 20)[:20]
            # sort them
            horder = hidx[np.argsort(mx[hidx])]
            holes = C[horder]
        except Exception:
            holes = rng.standard_normal((20, d))
            holes /= np.linalg.norm(holes, axis=1, keepdims=True)
        # starts: holes + 200 random + maybe old position
        R = rng.standard_normal((200, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        V0 = np.concatenate([holes, R], axis=0)
        # add corresponding old vector as start too
        try:
            ov = removed_old[k % len(removed_old)].reshape(1, -1)
            ov = ov / np.linalg.norm(ov, axis=1, keepdims=True)
            V0 = np.concatenate([V0, ov], axis=0)
        except Exception:
            pass
        # refine batch
        try:
            Vopt = _refine_single_batch(V0, Y, beta=170.0, iters=110, lr0=0.12, lr1=0.008)
            Sv = Vopt @ Y.T
            mv = Sv.max(axis=1)
            bi2 = int(np.argmin(mv))
            best_v = Vopt[bi2].copy()
            best_v /= np.linalg.norm(best_v)
        except Exception:
            # fallback to best hole
            Sv0 = V0 @ Y.T
            mv0 = Sv0.max(axis=1)
            best_v = V0[int(np.argmin(mv0))].copy()
        new_pts.append(best_v)
        Y = np.concatenate([Y, best_v.reshape(1, -1)], axis=0)

    # assemble full configuration: frozen + new in original order? order doesn't matter
    # Build Xnew = concat Kmat + new_pts
    try:
        NP = np.array(new_pts)
        if NP.shape[0] != n_ins:
            # pad/truncate
            if NP.shape[0] < n_ins:
                extra = np.tile(removed_old[0:1], (n_ins - NP.shape[0], 1))
                NP = np.concatenate([NP, extra], axis=0)
            else:
                NP = NP[:n_ins]
        Xnew = np.concatenate([Kmat, NP], axis=0)
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        vnew = _max_cos(Xnew)
    except Exception:
        Xnew = x_base.copy()
        vnew = b_inc

    # short exponential-repulsion smoothing (unfrozen)
    rem2 = T_DEAD - time.time()
    if rem2 > 6 and Xnew.shape[0] == 181:
        steps = 90 if rem2 > 30 else 40
        Xsm = _global_smooth(Xnew, steps=steps, k0=50.0, k1=220.0, t_dead=T_DEAD)
        try:
            vsm = _max_cos(Xsm)
        except Exception:
            vsm = 1e9
        # keep better between Xnew and Xsm for final polish start
        if vsm < vnew:
            Xnew, vnew = Xsm, vsm

    # update incumbent if improved before final polish
    if vnew < b_inc and np.all(np.isfinite(Xnew)):
        x_inc, b_inc = Xnew, vnew

    # final LP polish with remaining time
    rem3 = T_DEAD - time.time()
    if rem3 > 4:
        Xf = _lp_polish(Xnew if 'Xnew' in locals() else x_inc, budget=max(1.0, rem3 - 1.0), t_dead=T_DEAD, init_radius=8e-4)
        try:
            vf = _max_cos(Xf)
            if vf < b_inc and np.all(np.isfinite(Xf)):
                x_inc, b_inc = Xf, vf
        except Exception:
            pass

    x_inc = x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)
    # tiny shrink not needed; ensure finite
    x_inc = np.where(np.isfinite(x_inc), x_inc, 0.0)
    # renormalize after fix
    nrm = np.linalg.norm(x_inc, axis=1, keepdims=True) + 1e-300
    x_inc = x_inc / nrm
    return x_inc
