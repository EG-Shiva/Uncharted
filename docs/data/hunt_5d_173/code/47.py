import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline, r0=2e-3, cap=3500):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    try:
        while time.time() < deadline and radius > 1e-14:
            Gfull = X @ X.T
            c = Gfull[iu0]
            m = float(c.max())
            thr = m - max(20.0 * radius, 2e-9)
            sel = c > thr
            if np.count_nonzero(sel) > cap:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:cap]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu0[0][sel]
            J = iu0[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -c[sel]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
    except Exception:
        pass
    return X

def _refine_void(Y, v0, s=200.0, iters=120, lr=0.05):
    v = v0.copy()
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return v0
    v /= nv
    for _ in range(iters):
        dd = Y @ v
        dm = float(dd.max())
        w = np.exp(s * (dd - dm))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = Y.T @ w
        g = g - (g @ v) * v
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        # normalize gradient step
        v = v - lr * g / max(1.0, gn * 5.0)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        v /= nrm
    return v

def _greedy_void(existing, rng, n_cand=50000, chunk=10000):
    d = existing.shape[1]
    best_pt = None
    best_score = 2.0
    nch = n_cand // chunk
    for _ in range(nch):
        C = rng.randn(chunk, d)
        C = _norm(C)
        dots = C @ existing.T
        md = dots.max(axis=1)
        i = int(np.argmin(md))
        if float(md[i]) < best_score:
            best_score = float(md[i])
            best_pt = C[i].copy()
    return best_pt

def _local_repel_patch(Xall, patch_idx, iters=260, alpha=150.0, lr0=0.03):
    n, d = Xall.shape
    k = len(patch_idx)
    # map global -> position in patch for self-masking
    pos_of = {g: t for t, g in enumerate(patch_idx)}
    lr = lr0
    for it in range(iters):
        P = Xall[patch_idx]
        G = P @ Xall.T  # k x n
        # mask self dots
        for t, g in enumerate(patch_idx):
            G[t, g] = -2.0
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        # force = W @ Xall
        F = W @ Xall
        # tangent projection per patch point
        dot = (F * P).sum(axis=1, keepdims=True)
        Ft = F - dot * P
        # normalize step
        fn = np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-12
        # scale: use mean weight
        P = P - lr * Ft / np.maximum(1.0, fn)
        P = _norm(P)
        Xall[patch_idx] = P
        lr *= 0.9985
        # ramp alpha slightly
        if it % 80 == 79:
            alpha = min(alpha * 1.25, 320.0)
    return Xall

def _global_repel(X, iters_list=(180, 220, 260), alphas=(35.0, 90.0, 200.0), lrs=(0.08, 0.04, 0.02), mu=0.6, decay=0.9985):
    X = _norm(X)
    n, d = X.shape
    V = np.zeros_like(X)
    for alpha0, iters, lr0 in zip(alphas, iters_list, lrs):
        alpha = float(alpha0)
        lr = float(lr0)
        for _ in range(int(iters)):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ X) / rs
            dot = (A * X).sum(axis=1, keepdims=True)
            Ft = A - dot * X
            V = mu * V + (1.0 - mu) * Ft
            X = X - lr * V
            X = _norm(X)
            lr *= decay
    return X

def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    rng = np.random.RandomState(7)
    try:
        base = np.load("best.npy").astype(float)
    except Exception:
        try:
            base = np.load("record.npy").astype(float)
        except Exception:
            base = rng.randn(173, 5)
    base = _norm(np.asarray(base, dtype=float))
    if base.shape != (173, 5):
        tmp = _norm(rng.randn(173, 5))
        try:
            m = min(base.shape[0], 173)
            dd = min(base.shape[1], 5)
            tmp[:m, :dd] = base[:m, :dd]
            base = _norm(tmp)
        except Exception:
            base = tmp
    best = base.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n, d = 173, 5
    tabu = set()
    trial = 0
    # thresholds for contact graph
    while time.time() < t_end - 14.0:
        trial += 1
        # --- analyse contact structure of cur ---
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        cthr = mcur - 1e-7
        cnt = (G > cthr).sum(axis=1)
        # identify rattler and hub
        rattler = int(np.argmin(cnt))
        hub = int(np.argmax(cnt))
        mode = trial % 3
        center = hub
        if mode == 0:
            # 13-contact point (max contacts)
            center = hub
        elif mode == 1:
            # rattler's nearest jammed cluster: nearest neighbour of rattler
            dots_r = cur @ cur[rattler]
            dots_r[rattler] = -2.0
            nn = int(np.argmax(dots_r))
            # if nn has few contacts, step to its most-contacted neighbour
            if cnt[nn] < 7:
                # neighbours of nn sorted by dot
                order = np.argsort(-(cur @ cur[nn]))
                for cand in order[1:12]:
                    if cnt[int(cand)] >= 7 and int(cand) not in tabu:
                        nn = int(cand)
                        break
            center = nn
        else:
            # random point with >=9 contacts, avoid tabu
            elig = np.where(cnt >= 9)[0]
            if len(elig) == 0:
                elig = np.where(cnt >= 7)[0]
            # filter tabu
            filt = [int(e) for e in elig if int(e) not in tabu]
            if len(filt) == 0:
                filt = [int(e) for e in elig]
            center = int(filt[rng.randint(len(filt))])
        if center in tabu and rng.rand() < 0.8:
            # try to find non-tabu alternative nearby with many contacts
            order_c = np.argsort(-cnt)
            for cand in order_c:
                if int(cand) not in tabu and cnt[int(cand)] >= 7:
                    center = int(cand)
                    break
        # neighbourhood: center + 8-14 contact neighbours
        dots_c = cur @ cur[center]
        dots_c[center] = -2.0
        order_n = np.argsort(-dots_c)
        # number to delete: 9..15 total
        k_total = int(rng.randint(9, 16))  # 9..15
        # prefer true contacts first
        neigh = []
        for cand in order_n:
            if len(neigh) >= k_total - 1:
                break
            # skip tabu to keep neighbourhoods non-overlapping (unless need fill)
            if int(cand) in tabu and len(neigh) < 6:
                continue
            neigh.append(int(cand))
        del_idx = np.array([center] + neigh[:k_total - 1], dtype=int)
        # ensure non-overlap bookkeeping: if heavily overlapping tabu, still proceed
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        frozen = cur[keep_mask]
        # --- sequential greedy maximin rebuild with 50000 candidates each ---
        Xnew = cur.copy()
        new_pts = []
        # existing set grows
        for t in range(len(del_idx)):
            if time.time() > t_end - 14.0:
                break
            if len(new_pts) == 0:
                existing = frozen
            else:
                existing = np.vstack([frozen, np.stack(new_pts, axis=0)])
            pt = _greedy_void(existing, rng, n_cand=50000, chunk=10000)
            pt = _refine_void(existing, pt, s=220.0, iters=100, lr=0.06)
            new_pts.append(pt)
        if len(new_pts) != len(del_idx):
            continue
        Xnew[del_idx] = np.stack(new_pts, axis=0)
        Xnew = _norm(Xnew)
        # --- locally repel patch points (frozen surroundings) ---
        try:
            Xnew = _local_repel_patch(Xnew, del_idx, iters=220, alpha=140.0, lr0=0.035)
        except Exception:
            pass
        if time.time() > t_end - 13.0:
            # still evaluate
            pass
        # --- unfreeze: annealed repulsion ---
        try:
            # vary schedule slightly per trial
            j = rng.rand()
            if j < 0.5:
                Xnew = _global_repel(Xnew, iters_list=(150, 180, 220), alphas=(30.0, 85.0, 190.0), lrs=(0.07, 0.035, 0.018), mu=0.55 + rng.rand() * 0.3, decay=0.9985)
            else:
                Xnew = _global_repel(Xnew, iters_list=(120, 200, 280), alphas=(40.0, 110.0, 240.0), lrs=(0.09, 0.04, 0.015), mu=0.55 + rng.rand() * 0.3, decay=0.9988)
        except Exception:
            pass
        v_rep = _max_cos(Xnew)
        # only LP-polish promising ones to save time
        thresh = min(best_val + 0.004, cur_val + 0.005)
        if v_rep < thresh and time.time() < t_end - 12.0:
            try:
                Xnew = _lp_polish(Xnew, deadline=min(time.time() + 1.6, t_end - 12.0), r0=2e-3)
            except Exception:
                pass
            v_new = _max_cos(Xnew)
        else:
            v_new = v_rep
        if v_new < best_val - 1e-12:
            best = Xnew.copy()
            best_val = float(v_new)
            cur = Xnew.copy()
            cur_val = float(v_new)
            # add neighbourhood to tabu to force non-overlapping next cycles, but keep size bounded
            for g in del_idx:
                tabu.add(int(g))
            if len(tabu) > 70:
                # drop oldest arbitrary: clear half
                tabu = set(list(tabu)[-40:])
        elif v_new < cur_val:
            cur = Xnew.copy()
            cur_val = float(v_new)
            for g in del_idx:
                tabu.add(int(g))
            if len(tabu) > 70:
                tabu = set(list(tabu)[-40:])
        else:
            # occasional acceptance to change basin + tabu anyway to enforce cycling
            if v_new < 0.90 and rng.rand() < 0.08:
                cur = Xnew.copy()
                cur_val = float(v_new)
            for g in del_idx:
                tabu.add(int(g))
            if len(tabu) > 80:
                tabu.clear()
        # periodically restart from best to avoid drift, clear tabu occasionally
        if trial % 18 == 0 and rng.rand() < 0.6:
            cur = best.copy()
            cur_val = best_val
        if trial % 30 == 0:
            tabu.clear()
        if best_val < 0.70:
            break
    try:
        best = _lp_polish(best, deadline=t_end - 1.0, r0=1.5e-3)
    except Exception:
        pass
    best = _norm(best)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
