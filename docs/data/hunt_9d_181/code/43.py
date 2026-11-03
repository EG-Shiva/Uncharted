import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_dual(X, budget, t_dead, init_radius=1e-3, cap=1500):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    last_I = None; last_J = None; last_dual = None; last_m = best
    while True:
        if time.time() - t0 >= budget:
            break
        if time.time() >= t_dead:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c[sel]
        p = len(I)
        if p > cap:
            idx = np.argpartition(-csel, cap)[:cap]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = cap
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2*d] = J[:, None] * d + ar
        cols_mat[:, 2*d] = n * d
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2*d+1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2*d] = x[I]
        vals_mat[:, 2*d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        # extract duals
        duals = None
        try:
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                mg = getattr(res.ineqlin, 'marginals', None)
                if mg is not None:
                    duals = np.abs(np.asarray(mg, dtype=float)).ravel()
        except Exception:
            duals = None
        if duals is None or duals.size != p:
            try:
                mg2 = getattr(res, 'marginals', None)
                if mg2 is not None:
                    duals = np.abs(np.asarray(mg2, dtype=float)).ravel()[:p]
            except Exception:
                duals = None
        if duals is None or duals.size != p or not np.all(np.isfinite(duals)):
            duals = np.ones(p)
        else:
            # if all zero, fallback to uniform
            if duals.max() <= 0:
                duals = np.ones(p)
        last_I = I.copy(); last_J = J.copy(); last_dual = duals.copy(); last_m = m
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x, best, (last_I, last_J, last_dual, last_m)

def _lp_polish(X, budget, t_dead, init_radius=1e-3, cap=1500):
    x, b, _ = _lp_polish_dual(X, budget, t_dead, init_radius, cap)
    return x

def _smooth_exp(X, steps, t_dead, k0=40.0, k1=250.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        if s % 20 == 0 and time.time() >= t_dead:
            break
        frac = s / max(steps-1, 1)
        k = k0 + (k1-k0)*frac
        lr = lr0 + (lr1-lr0)*frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k*(G-gmax), -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot*X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr/nrm)*Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _optimize_insert_batch(C, Kmat, beta=80.0, steps=250, lr0=0.3, lr1=0.02):
    # C: (B,d) starts on sphere, Kmat: (M,d) frozen
    B, d = C.shape
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps-1, 1)
        lr = lr0 + (lr1-lr0)*frac
        S = C @ Kmat.T  # (B,M)
        smax = S.max(axis=1, keepdims=True)
        W = np.exp(beta*(S - smax))
        Ws = W.sum(axis=1, keepdims=True) + 1e-30
        Gr = (W @ Kmat) / Ws  # (B,d)
        dot = np.einsum('ij,ij->i', Gr, C)[:, None]
        Gt = Gr - dot*C
        # normalize step per row by norm
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        C = C - (lr / 1.0) * (Gt / nrm) * 0.15
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    DEADLINE = t0 + 155.0
    HARD = t0 + 170.0
    rng = np.random.default_rng(0)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        tmp = rng.normal(size=(181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [_max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    x_best = cand[bi].copy()
    b_best = scores[bi]

    # one LP polish with dual capture
    rem = min(7.0, max(0.5, DEADLINE - time.time() - 100.0))
    # ensure at least a short polish
    if time.time() < DEADLINE:
        try:
            xp, bp, dualinfo = _lp_polish_dual(x_best, budget=min(7.0, max(1.0, DEADLINE-time.time()-60)), t_dead=DEADLINE, init_radius=2e-3)
            if bp < b_best and np.all(np.finite(xp)):
                x_best, b_best = xp, bp
            x_cur = xp
        except Exception:
            x_cur = x_best.copy()
            dualinfo = (None, None, None, b_best)
    else:
        return x_best
    # try up to 2 surgery cycles with different sizes
    for cycle, Ksize in enumerate([14, 11]):
        if time.time() >= DEADLINE - 25.0:
            break
        if time.time() >= HARD - 30.0:
            break
        try:
            X = x_cur.copy()
            n, d = X.shape
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # contacts for rattler detection
            Gc = G.copy()
            np.fill_diagonal(Gc, 1.0)
            cnt = np.sum(Gc > m - 1e-7, axis=1) - 1
            rattlers = np.where(cnt <= 0)[0].tolist()
            # pad rattlers to 2 if needed
            if len(rattlers) < 2:
                order = np.argsort(cnt)
                for o in order:
                    if int(o) not in rattlers:
                        rattlers.append(int(o))
                    if len(rattlers) >= 2:
                        break
            else:
                # keep exactly 2 lowest
                rattlers = sorted(rattlers, key=lambda i: cnt[i])[:2]
            # dual-weighted stress
            last_I, last_J, last_dual, last_m = dualinfo
            stress = np.zeros(n)
            if last_I is not None and last_J is not None and last_dual is not None:
                # accumulate
                for ii, jj, w in zip(last_I, last_J, last_dual):
                    stress[int(ii)] += float(w)
                    stress[int(jj)] += float(w)
                # if stress all zero fallback to contact count
                if stress.max() <= 0:
                    stress = cnt.astype(float)
            else:
                stress = cnt.astype(float)
            # never pick rattlers as core
            for r in rattlers:
                stress[r] = -1e99
            # adjacency for connectivity: use tight pairs m-1e-7
            # build neighbor lists
            tight = (Gc > m - 1e-7)
            # seed = max stress
            seed = int(np.argmax(stress))
            # greedy BFS expansion limited to Ksize picking highest stress neighbors
            cluster = [seed]
            in_cl = np.zeros(n, dtype=bool)
            in_cl[seed] = True
            # candidate frontier: neighbors of cluster
            while len(cluster) < Ksize:
                # find all neighbors of current cluster not yet in cluster and not rattler
                front = set()
                for u in cluster:
                    nbrs = np.where(tight[u])[0]
                    for v in nbrs:
                        v = int(v)
                        if in_cl[v]:
                            continue
                        if v in rattlers:
                            continue
                        front.add(v)
                if len(front) == 0:
                    # add global highest stress outside
                    order = np.argsort(-stress)
                    added = False
                    for v in order:
                        v = int(v)
                        if in_cl[v] or v in rattlers:
                            continue
                        front.add(v)
                        added = True
                        break
                    if not added:
                        break
                # pick highest stress in frontier
                pick = max(list(front), key=lambda v: stress[v])
                cluster.append(pick)
                in_cl[pick] = True
            remove_idx = sorted(set(rattlers) | set(cluster))
            # keep frozen
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[remove_idx] = False
            Kfrozen0 = X[keep_mask].copy()
            kdel = len(remove_idx)
            # sequential reinsertion with frozen set growing
            inserted = []
            # pre-generate hole samples per insertion on the fly
            for j in range(kdel):
                if time.time() >= DEADLINE - 20.0:
                    break
                Kmat = np.vstack([Kfrozen0] + inserted) if len(inserted) > 0 else Kfrozen0
                # 50k sphere samples for deepest holes
                NS = 50000
                # chunk to save memory/time: 2 chunks of 25k
                best_vals = None
                best_vecs = None
                cand_holes = []
                for _ch in range(2):
                    R = rng.normal(size=(NS//2, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    S = R @ Kmat.T
                    mx = S.max(axis=1)
                    # keep top few deepest
                    kkeep = 4
                    idx = np.argpartition(mx, kkeep-1)[:kkeep]
                    for ii in idx:
                        cand_holes.append(R[int(ii)])
                    # also track global best for diagnostics (not needed)
                cand_holes = np.array(cand_holes)  # (8,d)
                # 200 random starts
                R2 = rng.normal(size=(200, d))
                R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
                C0 = np.vstack([R2, cand_holes])
                # batch optimize
                Copt = _optimize_insert_batch(C0, Kmat, beta=80.0, steps=220, lr0=0.3, lr1=0.02)
                # evaluate true max
                S = Copt @ Kmat.T
                mx = S.max(axis=1)
                bi2 = int(np.argmin(mx))
                inserted.append(Copt[bi2:bi2+1].copy())
                # early abort if deadline
                if time.time() >= DEADLINE - 15.0:
                    break
            if len(inserted) < kdel:
                # incomplete due to deadline
                # fill remaining with random holes quickly
                Kmat = np.vstack([Kfrozen0] + inserted) if len(inserted) > 0 else Kfrozen0
                while len(inserted) < kdel:
                    R = rng.normal(size=(5000, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    S = R @ Kmat.T
                    mx = S.max(axis=1)
                    inserted.append(R[int(np.argmin(mx)):int(np.argmin(mx))+1].copy())
                    Kmat = np.vstack([Kfrozen0] + inserted)
            Kfrozen0_full = np.vstack([Kfrozen0] + inserted)
            # rebuild full X: place frozen back and inserted in removed slots
            Xnew = np.empty_like(X)
            Xnew[keep_mask] = Kfrozen0
            ins_mat = np.vstack(inserted)
            for k, ri in enumerate(remove_idx):
                Xnew[ri] = ins_mat[k]
            Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            # short exponential-repulsion smoothing (unfrozen)
            if time.time() < DEADLINE - 12.0:
                Xnew = _smooth_exp(Xnew, steps=150, t_dead=DEADLINE-10.0, k0=50.0, k1=220.0, lr0=0.015, lr1=0.002)
            # finish with LP polish under deadline
            if time.time() < DEADLINE - 5.0:
                budg = max(1.0, min(25.0, DEADLINE - time.time() - 1.0))
                Xnew = _lp_polish(Xnew, budget=budg, t_dead=DEADLINE, init_radius=8e-3, cap=1500)
            vnew = _max_cos(Xnew)
            if vnew < b_best - 1e-15:
                x_best, b_best = Xnew, vnew
                x_cur = Xnew.copy()
                # refresh dualinfo for next cycle with quick polish info
                # do a tiny dual capture if time allows
                if time.time() < DEADLINE - 20.0:
                    try:
                        xp2, bp2, di2 = _lp_polish_dual(x_cur, budget=2.0, t_dead=DEADLINE, init_radius=5e-4, cap=1500)
                        if bp2 < b_best:
                            x_best, b_best = xp2, bp2
                            x_cur = xp2.copy()
                        dualinfo = di2
                    except Exception:
                        pass
            else:
                # keep current for next cycle perturbation? move to new basin slightly
                # set x_cur to Xnew if it is close, to diversify next cluster choice
                # but don't lose incumbent
                # use Xnew as start for next cycle only if not much worse
                if vnew < _max_cos(x_cur) + 0.005:
                    x_cur = Xnew.copy()
                    # need fresh dual info for new x_cur
                    if time.time() < DEADLINE - 20.0:
                        try:
                            xp3, bp3, di3 = _lp_polish_dual(x_cur, budget=2.0, t_dead=DEADLINE, init_radius=5e-4, cap=1500)
                            x_cur = xp3.copy()
                            dualinfo = di3
                        except Exception:
                            pass
        except Exception:
            continue
    # final safety: ensure best returned
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
