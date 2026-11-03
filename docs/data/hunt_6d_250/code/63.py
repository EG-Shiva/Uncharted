import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _rand_Q(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _metrics_from_G(G, m, tol=1e-4):
    # G: (n,n) with diag -2, m = max
    n = G.shape[0]
    Adj = G > (m - tol)
    np.fill_diagonal(Adj, False)
    deg = Adj.sum(axis=1).astype(int)
    cnt = int(Adj.sum() // 2)
    mn = int(deg.min()) if n else 0
    mean = float(deg.mean()) if n else 0.0
    # connected via DSU
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    iu0, iu1 = np.where(Adj)
    for a, b in zip(iu0.tolist(), iu1.tolist()):
        if b > a:
            union(a, b)
    r0 = find(0)
    conn = all(find(i) == r0 for i in range(n))
    return cnt, mn, mean, bool(conn)

def _smooth_single(X0, beta, steps, lr, mu=0.86):
    X = X0.copy()
    V = np.zeros_like(X)
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = beta * (G - m)
        np.clip(D, -30.0, 0.0, out=D)
        W = np.exp(D)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        V = mu * V + tang
        cur = lr * (1.0 - 0.4 * it / max(1, steps))
        X = X - cur * V
        X = _norm_rows(X)
    return X

def _lp_polish(x, best, deadline, k_top=3000, r0=2e-3, maxit=25):
    n, d = x.shape
    x = _norm_rows(x)
    try:
        bv = _max_cos(x)
    except Exception:
        return x, best
    if bv < best:
        best = bv
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 0.8 and radius > 1e-13 and it < maxit:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0 * radius, 5e-7)
        sel = c_all > m - window
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
        try:
            cols = np.concatenate([I[:, None] * d + ar,
                                   J[:, None] * d + ar,
                                   np.full((p, 1), n * d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + np.asarray(res.x[:-1], dtype=np.float64).reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            x = y; best = v
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x, best

def solve():
    t_start = time.time()
    BUDGET = 173.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(7)
    n, d = 250, 6
    R = 24
    # load incumbent(s)
    inc_best = None; inc_rec = None
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = _norm_rows(z)
                if np.all(np.isfinite(z)):
                    if fn == "best.npy":
                        inc_best = z.copy()
                    else:
                        inc_rec = z.copy()
        except Exception:
            pass
    if inc_best is None:
        if inc_rec is not None:
            inc_best = inc_rec.copy()
        else:
            z = rng.standard_normal((n, d))
            inc_best = _norm_rows(z)
    if inc_rec is None:
        inc_rec = inc_best.copy()
    elite = inc_best.copy()
    elite_val = _max_cos(elite)

    # ---- initialise R replicas: random / perturbed incumbent / crossover / rotated ----
    Xrep = np.empty((R, n, d), dtype=np.float64)
    sigmas = [0.01, 0.03, 0.07, 0.15]
    for r in range(R):
        if r < 8:
            Z = rng.standard_normal((n, d))
            Xrep[r] = _norm_rows(Z)
        elif r < 12:
            s = sigmas[(r - 8) % len(sigmas)]
            Z = inc_best + s * rng.standard_normal((n, d))
            Xrep[r] = _norm_rows(Z)
        elif r < 18:
            Q = _rand_Q(d, rng)
            rec_rot = inc_rec @ Q.T
            mask = rng.random(n) < 0.5
            Z = np.where(mask[:, None], inc_best, rec_rot)
            Z = Z + 0.005 * rng.standard_normal((n, d))
            Xrep[r] = _norm_rows(Z)
        else:
            Q = _rand_Q(d, rng)
            Z = inc_best @ Q.T + (0.02 + 0.01 * (r % 3)) * rng.standard_normal((n, d))
            Xrep[r] = _norm_rows(Z)
    Vrep = np.zeros_like(Xrep)

    # ---- p-energy homotopy schedule ----
    p_sched = [20, 50, 100, 200, 500, 1000]
    steps_sched = [120, 120, 130, 150, 170, 200]
    lr_sched = [0.015, 0.012, 0.010, 0.008, 0.005, 0.003]
    mu = 0.82
    ar_n = np.arange(n)
    global_best = elite.copy()
    global_val = elite_val

    for si, p in enumerate(p_sched):
        steps = steps_sched[si]
        lr0 = lr_sched[si]
        exp paran = p - 1
        for it in range(steps):
            if remaining() < 55 and si < 5:
                # keep enough time for final stages + LP; shorten early if slow
                pass
            cur_lr = lr0 * (1.0 - 0.3 * it / max(1, steps))
            # vectorised Gram: (R,n,n)
            Xt = np.transpose(Xrep, (0, 2, 1))
            G = np.matmul(Xrep, Xt)
            # set diag -2 via flat view
            G.reshape(R, -1)[:, ::n + 1] = -2.0
            m = np.max(G, axis=(1, 2))
            m_safe = m + 1e-12
            Gpos = G
            np.maximum(Gpos, 0.0, out=Gpos)  # in-place, diag already -2 -> 0
            # ratio in [0,1]
            Ratio = Gpos / m_safe[:, None, None]
            np.clip(Ratio, 0.0, 1.0, out=Ratio)
            # p-energy weights: (ratio)^{p-1}
            W = np.power(Ratio, paran)
            W.reshape(R, -1)[:, ::n + 1] = 0.0
            rs = W.sum(axis=2, keepdims=True) + 1e-12
            Mean = np.matmul(W, Xrep) / rs
            dot = np.sum(Mean * Xrep, axis=2, keepdims=True)
            Tang = Mean - dot * Xrep
            # clip tangent norm to 1 for stability
            tn = np.sqrt(np.sum(Tang * Tang, axis=2, keepdims=True)) + 1e-12
            scale = np.minimum(1.0, 1.0 / tn)
            Tang *= scale
            Vrep = mu * Vrep + Tang
            Xrep = Xrep - cur_lr * Vrep
            nr = np.sqrt(np.sum(Xrep * Xrep, axis=2, keepdims=True))
            np.maximum(nr, 1e-12, out=nr)
            Xrep /= nr
            if (it % 25) == 24 and remaining() < 12:
                break
            if not np.all(np.isfinite(Xrep)):
                # reset non-finite replicas
                bad = ~np.all(np.isfinite(Xrep), axis=(1, 2))
                for bi in np.where(bad)[0]:
                    Z = rng.standard_normal((n, d))
                    Xrep[bi] = _norm_rows(Z)
                    Vrep[bi] = 0.0
        # ---- periodic rigidity diagnostics per replica ----
        try:
            m_r = np.empty(R)
            cnt_r = np.empty(R, dtype=int)
            mind_r = np.empty(R, dtype=int)
            meand_r = np.empty(R)
            conn_r = np.empty(R, dtype=bool)
            # recompute Grams for metrics (reuse last G? recompute exactly)
            for rr in range(R):
                Xr = Xrep[rr] / np.maximum(np.linalg.norm(Xrep[rr], axis=1, keepdims=True), 1e-12)
                Xrep[rr] = Xr
                Gr = Xr @ Xr.T
                np.fill_diagonal(Gr, -2.0)
                mr = float(np.max(Gr))
                m_r[rr] = mr
                cnt, mn, me, co = _metrics_from_G(Gr, mr, tol=1e-4)
                cnt_r[rr] = cnt; mind_r[rr] = mn; meand_r[rr] = me; conn_r[rr] = co
                if mr < global_val:
                    global_val = mr
                    global_best = Xr.copy()
            # evolutionary reseeding favouring rigidity (>=5 contacts/pt target)
            if si < len(p_sched) - 1 and remaining() > 30:
                order = np.argsort(m_r)
                # rigidity score
                rig = mind_r.astype(float) + 0.2 * meand_r + np.where(conn_r, 2.0, 0.0)
                top_half = order[:12]
                best_score_idx = int(order[0])
                # best rigid among reasonable scores (within 0.01 of median)
                med = float(np.median(m_r))
                pool = [i for i in top_half if m_r[i] < med + 0.01]
                if not pool:
                    pool = top_half.tolist()
                best_rig_idx = int(pool[int(np.argmax(rig[pool]))])
                worst = order[-6:]
                for k, wi in enumerate(worst):
                    src = best_score_idx if k < 3 else best_rig_idx
                    sigma = 0.008 if si < 3 else 0.003
                    Xrep[int(wi)] = _norm_rows(Xrep[src] + sigma * rng.standard_normal((n, d)))
                    Vrep[int(wi)] = 0.0
        except Exception:
            pass
        if remaining() < 28:
            break

    # ---- select best three structurally distinct replicas by rigidity-aware diversity ----
    try:
        m_final = np.empty(R)
        Grams = []
        for rr in range(R):
            m_final[rr] = _max_cos(Xrep[rr])
            Grams.append(Xrep[rr] @ Xrep[rr].T)
        Grams = np.stack(Grams, axis=0)
        order = np.argsort(m_final)
        selected = []
        for idx in order:
            idx = int(idx)
            if not selected:
                selected.append(idx)
            else:
                dmin = min(float(np.linalg.norm(Grams[idx] - Grams[s])) for s in selected)
                if dmin > 1.0 or len(selected) < 1:
                    selected.append(idx)
                elif len(order) < 6:
                    selected.append(idx)
                else:
                    # allow if score very close to best
                    if m_final[idx] < m_final[selected[0]] + 2e-4 and dmin > 0.3:
                        selected.append(idx)
            if len(selected) >= 3:
                break
        # fill if needed
        for idx in order:
            if len(selected) >= 3:
                break
            if int(idx) not in selected:
                selected.append(int(idx))
        selected = selected[:3]
    except Exception:
        selected = [int(np.argmin([_max_cos(Xrep[r]) for r in range(R)]))]

    # include global best / elite as fallback candidate
    cands = []
    for s in selected:
        cands.append(Xrep[s].copy())
    # polish each of the three with smooth high-beta + sparse tangent LP
    best_out = global_best.copy()
    best_val = global_val
    if elite_val < best_val:
        best_out = elite.copy(); best_val = elite_val
    for ci, Xc in enumerate(cands):
        if remaining() < 8:
            break
        try:
            # short smooth equalisation
            Xp = _smooth_single(_norm_rows(Xc), 3000.0, 150, 0.004)
            v = _max_cos(Xp)
            if v < _max_cos(Xc):
                Xc = Xp
            # sparse tangent LP with per-candidate deadline
            per = remaining() / (len(cands) - ci) - 0.5
            dl = time.time() + max(2.0, min(per, 25.0))
            Xp2, vp2 = _lp_polish(Xc, min(best_val, _max_cos(Xc)), dl)
            vv = _max_cos(Xp2)
            if vv < best_val:
                best_val = vv
                best_out = Xp2.copy()
        except Exception:
            continue
    # final exact comparison with incumbent
    try:
        bv = _max_cos(best_out)
        if elite_val < bv:
            best_out = elite.copy()
    except Exception:
        best_out = elite.copy()
    best_out = _norm_rows(best_out)
    return best_out
