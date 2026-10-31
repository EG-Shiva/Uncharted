import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 200:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _tangent_bases(x):
    n, d = x.shape
    T = np.empty((n, d, d - 1))
    E = np.eye(d)
    for i in range(n):
        M = np.empty((d, d))
        M[:, 0] = x[i]
        M[:, 1:] = E[:, :d - 1]
        # ensure full rank: if x close to e0.. fix by using different columns
        # QR will handle, but if degenerate replace
        try:
            Q, _ = np.linalg.qr(M)
        except Exception:
            Q, _ = np.linalg.qr(np.eye(d))
        if abs(float(Q[:, 0] @ x[i])) < 0.9:
            # first column not aligned, rebuild with different order
            M2 = np.empty((d, d))
            # put x last then qr and take complement
            M2[:, -1] = x[i]
            M2[:, :-1] = E[:, :d - 1] if False else np.eye(d)[:, 1:]
            # simpler: use svd null space
            u, _, _ = np.linalg.svd(x[i].reshape(1, -1))
            # fallback: gram schmidt manually
            Q, _ = np.linalg.qr(M)
        # align sign so Q0 ~ x
        if float(Q[:, 0] @ x[i]) < 0:
            Q[:, 0] = -Q[:, 0]
        # If Q0 not parallel (numerical), project: build tangent via Householder
        if abs(float(Q[:, 0] @ x[i])) < 0.99:
            # Householder reflection mapping e0 to x
            e0 = np.zeros(d); e0[0] = 1.0
            v = e0 - x[i]
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                Q = np.eye(d)
            else:
                v = v / nv
                H = np.eye(d) - 2.0 * np.outer(v, v)
                # H maps e0 to x (up to sign); columns 1.. are tangent
                Q = H[:, ::-1]  # move x column to front? simpler recompute
                # ensure first column is x
                # H[:,0] should be x
                Q = H
                if float(Q[:, 0] @ x[i]) < 0.99:
                    # try transpose
                    pass
        T[i] = Q[:, 1:]
    return T

def _global_rotation_basis(x, T):
    n, d = x.shape
    Nv = n * (d - 1)
    # 6 skew bases for d=4
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    R = np.zeros((Nv, len(pairs)))
    for k, (a, b) in enumerate(pairs):
        for i in range(n):
            xi = x[i]
            Axi = np.zeros(d)
            Axi[a] = -xi[b]
            Axi[b] = xi[a]
            # project onto tangent: T_i^T Axi
            R[i * (d - 1):(i + 1) * (d - 1), k] = T[i].T @ Axi
    return R

def _build_J(x, T, pairs_I, pairs_J):
    n, d = x.shape
    dm = d - 1
    p = len(pairs_I)
    Nv = n * dm
    J = np.zeros((p, Nv))
    for t in range(p):
        i = int(pairs_I[t]); j = int(pairs_J[t])
        vi = T[i].T @ x[j]
        vj = T[j].T @ x[i]
        J[t, i * dm:(i + 1) * dm] = vi
        J[t, j * dm:(j + 1) * dm] = vj
    return J

def _simplex_proj(w):
    # Duchi et al.
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(w) + 1)
    cond = u - css / ind > 0
    if not np.any(cond):
        return np.full_like(w, 1.0 / len(w))
    rho = ind[cond][-1]
    theta = css[cond][-1] / float(rho)
    return np.maximum(w - theta, 0.0)

def _estimate_weights(J, P, iters=1200):
    # minimize w^T J P J^T w over simplex (projected gradient)
    p = J.shape[0]
    if p == 0:
        return np.zeros(0)
    JP = J @ P
    Q = JP @ JP.T  # p x p PSD
    # normalize scale
    sc = float(np.max(np.diag(Q))) if p > 0 else 1.0
    if sc <= 0 or not np.isfinite(sc):
        return np.full(p, 1.0 / p)
    Q = Q / sc
    w = np.full(p, 1.0 / p)
    # projected gradient with decreasing step
    for it in range(iters):
        g = Q @ w
        eta = 0.5 / (1.0 + it * 0.02)
        w = w - eta * g
        w = _simplex_proj(w)
        if it % 300 == 299:
            # sharpen slightly? keep
            pass
    return w

def _deg_hash(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > m - tol, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    FLIP_END = t0 + 120.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    # short initial polish to reach exactly jammed state (fast)
    if time.time() < t0 + 170:
        try:
            best_x, best_v = _lp_polish(best_x, min(time.time() + 12.0, HARD_END - 60.0))
        except Exception:
            pass
    global_best_x = best_x.copy()
    global_best_v = float(best_v)
    base_x = best_x.copy()
    base_m = float(best_v)
    seen = set()
    try:
        seen.add(_deg_hash(global_best_x))
    except Exception:
        pass

    # ---- build contact system for base ----
    try:
        G0 = base_x @ base_x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        iu = np.triu_indices(n, 1)
        c_all = G0[iu]
        active_mask = c_all > (m0 - 1e-9)
        act_idx = np.where(active_mask)[0]
        AI = iu[0][act_idx]
        AJ = iu[1][act_idx]
        p = len(AJ)
        # degrees with active tol
        deg_active = np.zeros(n, dtype=int)
        for ii, jj in zip(AI, AJ):
            deg_active[ii] += 1
            deg_active[jj] += 1
        # tangent bases, rotation projector
        T = _tangent_bases(base_x)
        R = _global_rotation_basis(base_x, T)
        Qq, _ = np.linalg.qr(R)  # Nv x 6
        Nv = n * (d - 1)
        P = np.eye(Nv) - Qq @ Qq.T
        J = _build_J(base_x, T, AI, AJ)
        # dual weights via simplex QP (nonneg equilibrium)
        if p > 0 and p < 3000:
            w = _estimate_weights(J, P, iters=900)
        else:
            w = np.full(p, 1.0 / max(p, 1))
        # rank contacts by small weight, boost degree-3/4 vertices
        score = w.copy()
        # degree bonus: prefer contacts incident to low-degree vertices
        for t in range(p):
            i = int(AI[t]); j = int(AJ[t])
            dm = min(int(deg_active[i]), int(deg_active[j]))
            if dm <= 3:
                score[t] -= 1e6  # force to front? use large offset then sort by w
                # instead handle via tuple sort
                pass
        order = []
        # tuple sort: (is_lowdeg?0:1, w)
        keys = []
        for t in range(p):
            i = int(AI[t]); j = int(AJ[t])
            dm = min(int(deg_active[i]), int(deg_active[j]))
            low = 0 if dm <= 4 else 1
            keys.append((low, float(w[t])))
        order = np.argsort(np.array([k[1] + k[0] * 10.0 for k in keys]))
        # take best 20 weak contacts
        cand_e = order[:min(20, p)]
        # precompute full Gram for slack search
        Gfull = base_x @ base_x.T
        np.fill_diagonal(Gfull, -2.0)
        # build flip tasks: (e_pos, f_pair)
        tasks = []
        for epos in cand_e:
            ei = int(AI[epos]); ej = int(AJ[epos])
            # candidate f sharing vertex with e, slack < 2e-4, non-contact
            for v in (ei, ej):
                row = Gfull[v]
                # slack = m0 - row
                for u in range(n):
                    if u == v:
                        continue
                    g = float(row[u])
                    if g > m0 - 1e-9:
                        continue  # already active (contact)
                    slack = m0 - g
                    if slack < 2e-4 and slack > -1e-12:
                        a_ = min(v, u); b_ = max(v, u)
                        # avoid duplicating e itself
                        if (a_ == ei and b_ == ej) or (a_ == ej and b_ == ei):
                            continue
                        tasks.append((int(epos), a_, b_, slack))
        # keep only best few f per e (smallest slack first)
        from collections import defaultdict
        grouped = defaultdict(list)
        for (epos, a_, b_, sl) in tasks:
            grouped[epos].append((sl, a_, b_))
        tasks2 = []
        for epos, lst in grouped.items():
            lst.sort()
            for sl, a_, b_ in lst[:4]:
                tasks2.append((epos, a_, b_, sl))
        # sort tasks by (e weakness, slack)
        e_rank = {int(e): r for r, e in enumerate(cand_e)}
        tasks2.sort(key=lambda t: (e_rank.get(int(t[0]), 99), t[3]))
        # limit total tasks to ~60
        tasks2 = tasks2[:70]

        # mechanism per e cache
        mech_cache = {}
        def get_mech(epos):
            if epos in mech_cache:
                return mech_cache[epos]
            mask = np.ones(p, dtype=bool)
            mask[epos] = False
            Jrem = J[mask]
            M = Jrem @ P
            try:
                _, _, Vt = np.linalg.svd(M, full_matrices=False)
                wv = Vt[-1, :]
            except Exception:
                wv = rng.standard_normal(size=(Nv,))
                wv = wv - Qq @ (Qq.T @ wv)
            v = P @ wv
            nv = np.linalg.norm(v)
            if nv < 1e-12 or not np.isfinite(nv):
                mech_cache[epos] = None
                return None
            v = v / nv
            mech_cache[epos] = v
            return v

        ALPHAS = [0.03, 0.02, 0.015, 0.01, 0.006, 0.003, 0.001]
        for (epos, fa, fb, sl) in tasks2:
            if time.time() > FLIP_END:
                break
            v = get_mech(int(epos))
            if v is None:
                continue
            dm = d - 1
            # row vector for f
            jf = np.zeros(Nv)
            i = fa; j = fb
            jf[i * dm:(i + 1) * dm] = T[i].T @ base_x[j]
            jf[j * dm:(j + 1) * dm] = T[j].T @ base_x[i]
            dot = float(jf @ v)
            if dot > 0:
                vv = -v
                dot = -dot
            else:
                vv = v
            if dot >= -1e-12:
                continue  # not decreasing
            # build dx0 (n,4)
            Vmat = vv.reshape(n, dm)
            DX = np.empty((n, d))
            for ii in range(n):
                DX[ii] = T[ii] @ Vmat[ii]
            # projected line search walk (fixed direction, re-tangentized)
            x_cur = base_x.copy()
            # tangentize DX to x_cur initially (already tangent to base)
            cur_f = float(x_cur[fa] @ x_cur[fb])
            improved = False
            for step in range(7):
                if time.time() > FLIP_END:
                    break
                # re-tangentize direction to current x_cur
                proj = np.sum(DX * x_cur, axis=1, keepdims=True)
                Dt = DX - proj * x_cur
                # normalize scale to original norm per row? keep global scale
                best_alpha = None
                best_y = None
                best_y_max = None
                best_y_f = None
                # try alphas, pick one that most decreases f while feasible
                cur_max = _max_cos(x_cur)
                for alpha in ALPHAS:
                    y = x_cur + alpha * Dt
                    y = _norm(y)
                    ym = _max_cos(y)
                    tol = 3e-4  # step tolerance (global cap)
                    if ym > m0 + tol:
                        continue
                    yf = float(y[fa] @ y[fb])
                    if yf < cur_f - 1e-7:
                        # prefer largest decrease; alphas ordered large first,
                        # but check all to find max decrease
                        if best_y_f is None or yf < best_y_f:
                            best_y_f = yf
                            best_y = y
                            best_y_max = ym
                            best_alpha = alpha
                if best_y is None:
                    break
                x_cur = best_y
                cur_f = float(best_y_f)
                improved = True
                if cur_f <= m0 - 1e-9:
                    break  # reached active level
                if cur_f <= m0 - 5e-5:
                    # good enough, try polish
                    pass
            if not improved:
                continue
            # run trust-region LP polish on walked point
            try:
                ypol, vpol = _lp_polish(x_cur, min(time.time() + 1.6, HARD_END - 25.0))
            except Exception:
                continue
            try:
                h = _deg_hash(ypol)
            except Exception:
                h = None
            is_new = (h not in seen) if h is not None else True
            if vpol < global_best_v - 1e-12 or (is_new and vpol <= global_best_v + 5e-4):
                # retain if genuinely different hash or improves
                if h is not None:
                    seen.add(h)
                if vpol < global_best_v:
                    global_best_x = ypol.copy()
                    global_best_v = float(vpol)
                    # if improved over base, could update base? keep base fixed for mechanism consistency
                    # but if big improvement, refresh base structures? skip to save time
                    if global_best_v < m0 - 1e-7:
                        base_x = ypol.copy()
                        m0 = float(vpol)
                        # rebuild quickly for remaining tasks? break to rebuild
                        break
            # also if new hash but worse, keep exploring (don't discard time)
    except Exception:
        pass

    # ---- final polish of best ----
    try:
        gx, gv = _lp_polish(global_best_x, HARD_END - 2.0)
        if gv < global_best_v:
            global_best_x = gx
            global_best_v = gv
    except Exception:
        pass
    # safety normalize
    global_best_x = _norm(global_best_x)
    return global_best_x
