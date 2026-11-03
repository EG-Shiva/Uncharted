import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _build_T(x):
    # Householder basis for tangent: T[i] is (d x dm) orthonormal, T[i].T @ x[i]=0
    n, d = x.shape
    dm = d - 1
    s = np.sign(x[:, 0])
    s[s == 0] = 1.0
    u = x.copy()
    u[:, 0] += s
    n2 = np.sum(u * u, axis=1)  # (n,)
    n2[n2 < 1e-20] = 1e-20
    T = np.zeros((n, d, dm))
    # T[r,c] = delta_{r,c+1} - 2*u_r*u_{c+1}/n2
    # vectorized: base identity part
    for c in range(dm):
        T[:, c + 1, c] = 1.0
    # rank-1 correction
    # u_r * u_{c+1}
    Uc = u[:, 1:]  # n x dm
    coef = (2.0 / n2)  # n
    # T[:,r,c] -= coef * u[:,r,None] * Uc[:,c]
    # do with broadcasting
    T -= (coef[:, None, None] * u[:, :, None] * Uc[:, None, :])
    return T

def _rigidity_rows(x, T, I, J):
    # return matrix A (p x N) where N=n*dm, row k has T[I]^T x[J] at block I and T[J]^T x[I] at block J
    n, d = x.shape
    dm = d - 1
    N = n * dm
    p = len(I)
    if p == 0:
        return np.zeros((0, N))
    TI = T[I]  # p x d x dm
    TJ = T[J]
    xJ = x[J]  # p x d
    xI = x[I]
    # segI[p,q] = sum_r TI[p,r,q]*xJ[p,r]
    segI = np.einsum('prq,pr->pq', TI, xJ)
    segJ = np.einsum('prq,pr->pq', TJ, xI)
    A = np.zeros((p, N))
    # fill by loop (p~750, fast)
    for k in range(p):
        A[k, I[k] * dm:(I[k] + 1) * dm] = segI[k]
        A[k, J[k] * dm:(J[k] + 1) * dm] = segJ[k]
    return A

def _active_descend(y, steps, lr0, lr1, w0, w1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        win = w0 + (w1 - w0) * frac
        if win < 1e-12:
            win = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G - (m - win)) / win
        np.clip(W, 0.0, 1.0, out=W)
        np.fill_diagonal(W, 0.0)
        # boost tightest
        # W = W*W*(3-2*W) smoothstep? keep linear for speed
        S = (W * G).sum(axis=1)
        A = W @ y
        F = A - S[:, None] * y
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F /= cnt
        y = y - lr * F
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y

def _lp_polish(x, seconds, p_cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def _deepest_hole_replace(y, idx, rng, n_cand=2000):
    n, d = y.shape
    mask = np.ones(n, dtype=bool); mask[idx] = False
    Yr = y[mask]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ Yr.T).max(axis=1)
    bi = int(np.argmin(scores))
    y2 = y.copy()
    y2[idx] = cand[bi]
    return y2

def solve():
    t0 = time.time()
    HARD = 168.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    np.random.seed(0)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur_best = best.copy(); cur_val = float(best_val)
    dm = d - 1; N = n * dm
    CONTACT_WIN = 1e-7
    iu0, iu1 = np.triu_indices(n, 1)
    steps_list = [0.04, 0.07, 0.11, 0.15, 0.03, 0.09, 0.06, 0.12]
    drop_list = [2, 3, 4, 5, 3, 2, 4, 3]
    it = 0
    # precompute once T for incumbent (rebuild each cycle as incumbent changes rarely)
    while time.time() - t0 < HARD - 12.0:
        it += 1
        remain = HARD - (time.time() - t0)
        if remain < 8.0:
            break
        x0 = cur_best
        G0 = x0 @ x0.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        # contacts within 1e-7
        # use triu values
        Gall = x0 @ x0.T
        c_all = Gall[iu0, iu1]
        c_mask = c_all > m0 - CONTACT_WIN
        I_all = iu0[c_mask]; J_all = iu1[c_mask]
        p_all = len(I_all)
        if p_all < 10:
            # fallback random shake
            s = steps_list[it % len(steps_list)]
            pert = rng.standard_normal(size=(n, d)) * 0.05
            pert -= (pert * x0).sum(axis=1, keepdims=True) * x0
            y = x0 + pert
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _active_descend(y, 200, 0.15, 0.02, 0.01, 0.002)
            lp_sec = min(3.0, remain - 4.0)
            if lp_sec > 0.5:
                y = _lp_polish(y, lp_sec)
            v = _max_cos(y)
            if v < cur_val:
                cur_val = v; cur_best = y.copy()
            continue
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, I_all, 1); np.add.at(deg, J_all, 1)
        hubs = np.argsort(-deg)[:6]
        hub = int(hubs[it % len(hubs)])
        # neighbors of hub
        neigh_mask = (I_all == hub) | (J_all == hub)
        neigh_idx = np.where(neigh_mask)[0]
        if len(neigh_idx) == 0:
            # rattler as hub: pick random hub with contacts
            hub = int(hubs[(it + 1) % len(hubs)])
            neigh_mask = (I_all == hub) | (J_all == hub)
            neigh_idx = np.where(neigh_mask)[0]
            if len(neigh_idx) == 0:
                continue
        nd = drop_list[it % len(drop_list)]
        nd = min(nd, len(neigh_idx))
        # cycle drop choice: tightest vs random
        if it % 2 == 0:
            # tightest contacts of hub
            cv = c_all[c_mask][neigh_idx]
            order = np.argsort(-cv)
            drop_rel = order[:nd]
            drop_idx = neigh_idx[drop_rel]
        else:
            perm = rng.permutation(len(neigh_idx))
            drop_idx = neigh_idx[perm[:nd]]
        keep_mask = np.ones(p_all, dtype=bool)
        keep_mask[drop_idx] = False
        I_keep = I_all[keep_mask]; J_keep = J_all[keep_mask]
        I_drop = I_all[~keep_mask]; J_drop = J_all[~keep_mask]
        # build tangent basis and rigidity
        try:
            T = _build_T(x0)
            A_keep = _rigidity_rows(x0, T, I_keep, J_keep)
            A_drop = _rigidity_rows(x0, T, I_drop, J_drop)
        except Exception:
            continue
        p_keep = A_keep.shape[0]
        # form M = A A^T + reg
        try:
            M = A_keep @ A_keep.T
            reg = 1e-9 * (np.trace(M) / max(p_keep, 1) + 1e-12)
            if reg < 1e-12:
                reg = 1e-9
            M = M + reg * np.eye(p_keep)
        except Exception:
            continue
        # generate candidates
        n_cand = 4
        best_v = None; best_score = 1e100
        # neighbor set for hub-focused
        hub_region = set([hub])
        for k in neigh_idx[:20]:
            a = int(I_all[k]); b = int(J_all[k])
            hub_region.add(a); hub_region.add(b)
        hub_region = np.array(sorted(hub_region), dtype=int)
        for c in range(n_cand):
            r = rng.standard_normal(size=N)
            if c >= 2:
                # hub-focused: suppress outside region
                rr = r.reshape(n, dm)
                scale = np.ones(n) * 0.2
                scale[hub_region] = 3.0
                # extra boost hub itself
                scale[hub] = 5.0
                rr = rr * scale[:, None]
                r = rr.ravel()
            try:
                b = A_keep @ r
                # solve M y = b
                yv = np.linalg.solve(M, b)
                v = r - A_keep.T @ yv
            except Exception:
                continue
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                continue
            vn = v / nv
            try:
                dd = A_drop @ vn
            except Exception:
                continue
            m_plus = float(dd.max()) if dd.size else 0.0
            m_minus = float((-dd).max()) if dd.size else 0.0
            score = m_plus if m_plus < m_minus else m_minus
            # prefer modes that move hub appreciably
            vmat = vn.reshape(n, dm)
            hub_n = float(np.linalg.norm(vmat[hub]))
            mean_n = float(np.linalg.norm(vmat) / np.sqrt(n))
            if hub_n < 0.3 * mean_n and len(neigh_idx) > 0:
                score += 0.02  # penalty
            if score < best_score:
                best_score = score
                # store sign giving best
                if m_minus < m_plus:
                    best_v = -vn
                else:
                    best_v = vn
        if best_v is None:
            continue
        s_step = float(steps_list[it % len(steps_list)])
        # scale so max per-point norm = s_step
        vmat = best_v.reshape(n, dm)
        per = np.linalg.norm(vmat, axis=1).max()
        if per < 1e-14:
            continue
        v_scaled = best_v * (s_step / per)
        vmat_s = v_scaled.reshape(n, dm)
        # map to ambient dx = T @ vseg
        # dx_i = T[i] @ vseg_i
        dx = np.einsum('nij,nj->ni', T, vmat_s)
        norms = np.linalg.norm(vmat_s, axis=1)  # == |dx|
        # exponential map
        y = x0.copy()
        for i in range(n):
            a = norms[i]
            if a < 1e-15:
                continue
            # cap to avoid wrap
            if a > 0.6:
                scale_c = 0.6 / a
                a = 0.6
                # rescale dx direction? approximate
            y[i] = np.cos(a) * x0[i] + np.sin(a) * (dx[i] / (np.linalg.norm(dx[i]) + 1e-20))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # rattler hole reinsert for zero-degree points (within 1e-7)
        try:
            Gy = y @ y.T
            np.fill_diagonal(Gy, -2.0)
            my = float(Gy.max())
            degy = (Gy > my - 1e-7).sum(axis=1)
            zeros = np.where(degy == 0)[0]
            for z in zeros[:2]:
                if time.time() - t0 > HARD - 10:
                    break
                y = _deepest_hole_replace(y, int(z), rng, n_cand=1500)
        except Exception:
            pass
        # cheap repulsion to repair
        try:
            y = _active_descend(y, 220, 0.12, 0.02, 0.012, 0.002)
        except Exception:
            pass
        if time.time() - t0 > HARD - 8.0:
            # no time for LP, just evaluate
            try:
                v = _max_cos(y)
                if v < cur_val:
                    cur_val = v; cur_best = y.copy()
            except Exception:
                pass
            continue
        lp_sec = min(3.5, HARD - (time.time() - t0) - 2.0)
        if lp_sec < 0.7:
            try:
                v = _max_cos(y)
                if v < cur_val:
                    cur_val = v; cur_best = y.copy()
            except Exception:
                pass
            continue
        try:
            y2 = _lp_polish(y, lp_sec, p_cap=1100)
            v2 = _max_cos(y2)
            if v2 < cur_val - 1e-12:
                cur_val = v2; cur_best = y2.copy()
            # also consider unpolished y if LP worsened? LP keeps best internally so fine
        except Exception:
            try:
                v = _max_cos(y)
                if v < cur_val:
                    cur_val = v; cur_best = y.copy()
            except Exception:
                pass
    cur_best = cur_best / np.linalg.norm(cur_best, axis=1, keepdims=True)
    return cur_best
