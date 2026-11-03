import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, budget, t_dead):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    t0 = time.time()
    last_I = None; last_J = None; last_dual = None
    while True:
        now = time.time()
        if now - t0 >= budget:
            break
        if now >= t_dead - 0.5:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2000:
            # keep tightest 2000
            idx_all = np.where(sel)[0]
            cs = c[idx_all]
            o = np.argpartition(-cs, 2000)[:2000]
            sel2 = np.zeros_like(sel)
            sel2[idx_all[o]] = True
            sel = sel2
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
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
        # try capture duals
        try:
            marg = None
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                except Exception:
                    try:
                        marg = np.asarray(res.ineqlin["marginals"], dtype=float)
                    except Exception:
                        marg = None
            if marg is not None and marg.shape[0] == p:
                last_I = I.copy(); last_J = J.copy(); last_dual = np.abs(marg)
            else:
                # dict access
                try:
                    d2 = res["ineqlin"]["marginals"]
                    d2 = np.asarray(d2, dtype=float)
                    if d2.shape[0] == p:
                        last_I = I.copy(); last_J = J.copy(); last_dual = np.abs(d2)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best, last_I, last_J, last_dual

def _extract_duals(x, t_dead):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iu]
    m = float(c.max())
    tol = 1e-7
    sel = c > m - tol
    if int(sel.sum()) == 0:
        sel = c > m - 1e-9
    if int(sel.sum()) > 2500:
        idx_all = np.where(sel)[0]
        cs = c[idx_all]
        o = np.argpartition(-cs, 2500)[:2500]
        sel2 = np.zeros_like(sel)
        sel2[idx_all[o]] = True
        sel = sel2
    I = iu[0][sel]; J = iu[1][sel]
    csel = c[sel]
    p = len(I)
    if p == 0 or time.time() >= t_dead - 1.0:
        return I, J, np.ones(len(I))
    nv = n * d + 1
    ar = np.arange(d)
    radius = 1e-3
    cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
    cols_mat[:, :d] = I[:, None] * d + ar
    cols_mat[:, d:2 * d] = J[:, None] * d + ar
    cols_mat[:, 2 * d] = n * d
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = cols_mat.ravel()
    vals_mat = np.empty((p, 2 * d + 1))
    vals_mat[:, :d] = x[J]
    vals_mat[:, d:2 * d] = x[I]
    vals_mat[:, 2 * d] = -1.0
    vals = vals_mat.ravel()
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status == 0:
            marg = None
            try:
                if hasattr(res, "ineqlin") and res.ineqlin is not None:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
            except Exception:
                marg = None
            if marg is None:
                try:
                    marg = np.asarray(res["ineqlin"]["marginals"], dtype=float)
                except Exception:
                    marg = None
            if marg is not None and marg.shape[0] == p:
                return I, J, np.abs(marg)
    except Exception:
        pass
    # fallback: weight by closeness
    w = (csel - (m - tol)) / tol + 0.1
    w = np.maximum(w, 1e-3)
    return I, J, w

def _batch_lse_opt(V0, F, k0=120.0, k1=350.0, iters=260, lr0=0.12, lr1=0.01):
    V = V0.copy()
    B = V.shape[0]
    for it in range(iters):
        frac = it / max(1, iters - 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        S = V @ F.T
        m = S.max(axis=1, keepdims=True)
        # avoid overflow: clip
        E = np.exp(np.clip(k * (S - m), -50, 0))
        den = E.sum(axis=1, keepdims=True) + 1e-30
        # grad = (E @ F)/den
        G = (E @ F) / den
        dot = np.sum(G * V, axis=1, keepdims=True)
        Gt = G - dot * V
        # normalize step per-row by norm to stabilize
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # limit huge steps
        V = V - lr * Gt / np.maximum(nrm, 1.0) * 1.0
        # alternative: plain when norm small
        # use plain lr*Gt but scaled: we already divided when >1
        # for small norms keep plain: adjust
        # (above divides only if >1 due to maximum, good)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

def _smooth_exp(X, steps, k0=40.0, k1=250.0, lr0=0.02, lr1=0.002, t_dead=None):
    n = X.shape[0]
    for s in range(steps):
        if t_dead is not None and time.time() >= t_dead - 2.0:
            break
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
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
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    T0 = time.time()
    T_DEAD = T0 + 155.0
    rng = np.random.default_rng(7)
    # load incumbent
    cands = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9) and np.all(np.isfinite(a)):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.normal(size=(181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    scores = [_max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    x_inc = cands[bi].copy()
    b_inc = scores[bi]
    n, d = x_inc.shape

    # 1) one LP polish
    try:
        rem = min(10.0, max(0.0, T_DEAD - time.time() - 100.0))
        # ensure at least a short polish if plenty time; else fixed 6s
        if rem < 5.0:
            # if near deadline logic not yet consumed, just use 6s if available
            avail = T_DEAD - time.time() - 60.0
            rem = max(0.0, min(8.0, avail))
            if avail < 8.0:
                rem = max(0.0, min(6.0, T_DEAD - time.time() - 20.0))
        if rem > 1.0:
            xp, bp, lI, lJ, lD = _lp_polish_deadline(x_inc, rem, T_DEAD)
            if bp < b_inc and np.all(np.isfinite(xp)):
                x_inc, b_inc = xp, bp
            x_pol = xp
        else:
            x_pol = x_inc
            lI = lJ = lD = None
    except Exception:
        x_pol = x_inc
        lI = lJ = lD = None

    if time.time() >= T_DEAD - 30.0:
        return x_inc

    x_cur = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
    # 2) dual multipliers
    try:
        if lI is None or lD is None or len(lD) != int((x_cur @ x_cur.T)[np.triu_indices(n,1)].__len__() if False else 0) or lI is None:
            pass
    except Exception:
        pass
    # always re-extract for consistency (one LP)
    try:
        I_du, J_du, W_du = _extract_duals(x_cur, T_DEAD)
    except Exception:
        I_du, J_du, W_du = np.array([], dtype=int), np.array([], dtype=int), np.array([])

    # contacts per point
    try:
        Gc = x_cur @ x_cur.T
        np.fill_diagonal(Gc, -2.0)
        m_all = float(Gc.max())
        tol_c = 1e-7
        Cmask = Gc > (m_all - tol_c)
        cnt = Cmask.sum(axis=1)
    except Exception:
        cnt = np.zeros(n)
        m_all = b_inc

    # point stress from duals
    stress = np.zeros(n)
    if len(W_du) > 0 and len(I_du) == len(W_du):
        try:
            np.add.at(stress, I_du, W_du)
            np.add.at(stress, J_du, W_du)
        except Exception:
            pass
    # fallback if all zero
    if stress.max() <= 0:
        stress = cnt.astype(float) + 0.1

    # rattlers: zero contacts
    rattlers = np.where(cnt == 0)[0].tolist()
    # ensure exactly 2: take lowest cnt if needed
    if len(rattlers) < 2:
        order = np.argsort(cnt + stress * 1e-6)
        rattlers = order[:2].tolist()
    elif len(rattlers) > 2:
        # take 2 with lowest stress among them? just first 2
        rattlers = rattlers[:2]
    # highest-stress connected cluster expansion
    KDEL_CORE = 14  # 10-18 range
    # adjacency from tight pairs (use dual pairs + contact mask)
    # build neighbor lists from I_du,J_du
    neigh = [set() for _ in range(n)]
    try:
        for a, b in zip(I_du.tolist(), J_du.tolist()):
            neigh[a].add(b); neigh[b].add(a)
    except Exception:
        pass
    # if graph empty, use contact mask
    # find seed: max stress excluding rattlers
    mask_ok = np.ones(n, dtype=bool)
    mask_ok[rattlers] = False
    seed = int(np.argmax(np.where(mask_ok, stress, -1)))
    cluster = set([seed])
    # greedy expand by highest stress frontier to keep connected
    while len(cluster) < KDEL_CORE:
        frontier = set()
        for u in list(cluster):
            for v in neigh[u]:
                if v not in cluster and v not in rattlers:
                    frontier.add(v)
        if len(frontier) == 0:
            # disconnected: add highest-stress remaining
            rem_idx = [i for i in range(n) if i not in cluster and i not in rattlers]
            if len(rem_idx) == 0:
                break
            rem_stress = stress[rem_idx]
            cluster.add(int(rem_idx[int(np.argmax(rem_stress))]))
        else:
            fl = list(frontier)
            bv = fl[int(np.argmax(stress[fl]))]
            cluster.add(int(bv))
        if time.time() >= T_DEAD - 25.0:
            break
    remove_idx = sorted(set(rattlers) | cluster)
    keep_idx = [i for i in range(n) if i not in set(remove_idx)]
    # safety: keep at least 150
    if len(keep_idx) < 150:
        # reduce core
        remove_idx = sorted(set(rattlers) | list(cluster)[:10])
        keep_idx = [i for i in range(n) if i not in set(remove_idx)]

    F = x_cur[keep_idx].copy()
    # 3) sequential reinsertion with frozen set
    # need at least 200 random starts + 50000 samples per point
    try:
        for ri, ridx in enumerate(remove_idx):
            if time.time() >= T_DEAD - 22.0:
                break
            M = F.shape[0]
            # 50k samples for holes
            NS = 50000
            # chunk to save memory/time: 2 chunks of 25k
            best_hole_vals = None
            best_hole_vecs = None
            # generate in one go (50k x 9) is fine
            C = rng.normal(size=(NS, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = C @ F.T
            mx = S.max(axis=1)
            # deepest holes: smallest mx
            nh = 10
            part = np.argpartition(mx, nh)[:nh]
            # order them
            part = part[np.argsort(mx[part])]
            holes = C[part].copy()
            # 200 random starts
            NR = 200
            R = rng.normal(size=(NR, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            V0 = np.concatenate([R, holes], axis=0)
            # optimize batch
            # adapt iters to remaining time: fewer if late
            iters = 260
            time_left = T_DEAD - time.time()
            remaining_pts = len(remove_idx) - ri
            if time_left < remaining_pts * 4.0:
                iters = 120
            if time_left < remaining_pts * 2.0:
                iters = 60
            Vopt = _batch_lse_opt(V0, F, k0=120.0, k1=350.0, iters=iters, lr0=0.12, lr1=0.01)
            # evaluate true max
            Sv = Vopt @ F.T
            mv = Sv.max(axis=1)
            bi2 = int(np.argmin(mv))
            vstar = Vopt[bi2].copy()
            vstar /= np.linalg.norm(vstar)
            # append
            F = np.concatenate([F, vstar[None, :]], axis=0)
            del C, S, mx, V0, Vopt, Sv
    except Exception:
        pass

    if time.time() >= T_DEAD - 20.0:
        return x_inc
    # rebuild full set: F should be 181 points (keep + reinserted)
    try:
        if F.shape[0] != 181:
            # fallback: if incomplete, fill with original removed points
            if F.shape[0] < 181:
                need = 181 - F.shape[0]
                orig_removed = x_cur[remove_idx[:need]].copy()
                F = np.concatenate([F, orig_removed], axis=0)
            else:
                F = F[:181].copy()
        Xnew = F / np.linalg.norm(F, axis=1, keepdims=True)
    except Exception:
        return x_inc

    # quick check: if Xnew much worse, still continue with smoothing+polish
    # 4) short exponential-repulsion smoothing
    try:
        steps_sm = 150
        # reduce if low time
        if T_DEAD - time.time() < 25.0:
            steps_sm = 60
        Xnew = _smooth_exp(Xnew, steps_sm, k0=40.0, k1=250.0, lr0=0.02, lr1=0.002, t_dead=T_DEAD)
    except Exception:
        pass

    if time.time() >= T_DEAD - 15.0:
        # return best between Xnew and incumbent
        try:
            v = _max_cos(Xnew)
            if v < b_inc:
                return Xnew
        except Exception:
            pass
        return x_inc

    # 5) final LP polish under deadline
    try:
        budget_final = max(0.0, T_DEAD - time.time() - 1.0)
        if budget_final > 2.0:
            Xp, bp2, _, _, _ = _lp_polish_deadline(Xnew, budget_final, T_DEAD)
            # choose best among Xp, Xnew, incumbent
            cands2 = [(b_inc, x_inc)]
            try:
                cands2.append((_max_cos(Xnew), Xnew))
            except Exception:
                pass
            try:
                cands2.append((_max_cos(Xp), Xp))
            except Exception:
                pass
            cands2.sort(key=lambda t: t[0])
            return cands2[0][1]
        else:
            # no time: pick best of Xnew/incumbent
            try:
                if _max_cos(Xnew) < b_inc:
                    return Xnew
            except Exception:
                pass
            return x_inc
    except Exception:
        try:
            if _max_cos(Xnew) < b_inc:
                return Xnew
        except Exception:
            pass
        return x_inc
