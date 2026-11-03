import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)
    enc_all = iu0 * n + iu1
    # map encode->position for fast mask
    enc_to_pos = {int(enc_all[k]): k for k in range(P)}
    rng = np.random.RandomState(20261103)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def contact_set_deg(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = G > (m - tol)
        deg = C.sum(axis=1)
        # encode set
        a, b = np.where(np.triu(C, 1))
        s = set((int(x) * n + int(y)) for x, y in zip(a, b))
        return m, deg, s, G

    def run_free(X0, beta, maxiter):
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def run_capped(X0, capped_mask, cap_val, beta, maxiter, Kpen=20000.0):
        cap_idx = np.where(capped_mask)[0]
        unc_idx = np.where(~capped_mask)[0]
        if len(cap_idx) == 0 or len(unc_idx) == 0:
            return run_free(X0, beta, maxiter)
        cu0 = iu0[unc_idx]; cu1 = iu1[unc_idx]
        cc0 = iu0[cap_idx]; cc1 = iu1[cap_idx]
        nc = len(cap_idx)
        y0 = X0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            gu = g[unc_idx]
            m = float(np.max(gu))
            e = np.exp(beta * (gu - m))
            S = float(np.sum(e)) + 1e-300
            loss_free = m + np.log(S) / beta
            wu = e / S
            gc = g[cap_idx]
            viol = gc - cap_val
            pos = np.maximum(viol, 0.0)
            pen = Kpen * float(np.sum(pos * pos)) / float(nc)
            # linear part to push harder
            pen2 = 5.0 * float(np.sum(pos)) / float(nc)
            loss = loss_free + pen + pen2
            # weights
            pw = (2.0 * Kpen * pos + 5.0) / float(nc)
            pw = pw * (viol > 0.0)
            W = np.zeros((n, n))
            W[cu0, cu1] = wu
            W[cc0, cc1] = pw
            W = W + W.T
            # symmetrized? unc weights already symmetric fill via transpose; capped same.
            # Note W currently has half weights on each off-diag? For unc we set one triangle then symmetrize.
            # Above we set W[cu0,cu1]=wu then W=W+W.T gives symmetric. For capped same. Good.
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(X0)

    def lp_polish(x, deadline, r_init=8e-4, max_steps=9):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5 and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3000:
                k2 = 3000
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            try:
                y = _norm(x + res.x[:-1].reshape(n, d))
            except Exception:
                radius *= 0.5
                continue
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.015)
            else:
                radius *= 0.5
        return bx, best

    # initial incumbent contact set
    m0, deg0, set0, _ = contact_set_deg(x_best, tol=1e-7)
    x_cur = x_best.copy()
    cur_val = best_val
    cur_set = set0
    # pool for diversity
    pool = [(best_val, x_best.copy(), set0)]

    trial = 0
    # precompute order helpers
    while time.time() - t0 < BUDGET - 14 and trial < 60:
        trial += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 14:
            break
        try:
            m_c, deg_c, set_c, Gc = contact_set_deg(x_cur, tol=1e-7)
        except Exception:
            x_cur = x_best.copy(); cur_val = best_val
            continue
        # choose hub among top degrees
        order = np.argsort(-deg_c)
        # hubs: top 5
        hub_choice = order[rng.randint(0, min(6, n))]
        # decide cap type: alternate star vs cluster
        use_cluster = (trial % 3 == 0)
        cap_pairs = []
        if not use_cluster:
            # star of hub: contacts incident to hub within tol, else nearest
            row = Gc[hub_choice].copy()
            row[hub_choice] = -2
            # contacts first
            cont = np.where(row > m_c - 1e-7)[0]
            if len(cont) >= 6:
                cap_pairs = [(int(min(hub_choice, j)), int(max(hub_choice, j))) for j in cont]
                if len(cap_pairs) > 15:
                    # keep most stressed: largest row values
                    vals = row[cont]
                    idx = np.argsort(-vals)[:15]
                    cap_pairs = [cap_pairs[k] for k in idx]
            else:
                nn = np.argsort(-row)[:10]
                cap_pairs = [(int(min(hub_choice, j)), int(max(hub_choice, j))) for j in nn[:10]]
        else:
            # small cluster BFS from hub using contact graph
            # adjacency via contacts tol 5e-7 (slightly looser to get connectivity)
            try:
                Gtmp = x_cur @ x_cur.T
                np.fill_diagonal(Gtmp, -2.0)
                mc2 = float(Gtmp.max())
                adj = [set(np.where(Gtmp[i] > mc2 - 5e-7)[0].tolist()) for i in range(n)]
            except Exception:
                adj = [set() for _ in range(n)]
            # BFS collect edges
            seen_nodes = {int(hub_choice)}
            edge_list = []
            frontier = [int(hub_choice)]
            # gather incident edges first
            neigh = list(adj[int(hub_choice)])
            rng.shuffle(neigh)
            for nb in neigh[:15]:
                a, b = (int(hub_choice), int(nb)) if hub_choice < nb else (int(nb), int(hub_choice))
                edge_list.append((a, b))
                if nb not in seen_nodes:
                    seen_nodes.add(int(nb))
            # expand one more ring if too few
            if len(edge_list) < 6:
                # add nearest non-contact pairs of hub
                row = Gc[hub_choice].copy(); row[hub_choice] = -2
                nn = np.argsort(-row)[:10]
                for j in nn:
                    a, b = (int(hub_choice), int(j)) if hub_choice < j else (int(j), int(hub_choice))
                    if (a, b) not in edge_list:
                        edge_list.append((a, b))
                    if len(edge_list) >= 8:
                        break
            # truncate / sample to 6-15
            if len(edge_list) > 15:
                sel = rng.choice(len(edge_list), 12, replace=False)
                edge_list = [edge_list[k] for k in sel]
            cap_pairs = edge_list
        if len(cap_pairs) < 6:
            continue
        if len(cap_pairs) > 16:
            cap_pairs = cap_pairs[:16]
        # delta and beta
        deltas = [0.001, 0.0015, 0.002, 0.0025, 0.003]
        delta = deltas[rng.randint(0, len(deltas))]
        betas = [1200.0, 1600.0, 2000.0, 2500.0]
        beta_hi = float(betas[rng.randint(0, len(betas))])
        cap_val = float(m_c - delta)
        # build mask
        capped_mask = np.zeros(P, dtype=bool)
        for (a, b) in cap_pairs:
            key = int(a) * n + int(b)
            pos = enc_to_pos.get(key, None)
            if pos is not None:
                capped_mask[pos] = True
        # perturb start: tangential noise, stronger on involved nodes
        involved = set()
        for (a, b) in cap_pairs:
            involved.add(a); involved.add(b)
        involved = list(involved)
        try:
            Nz = rng.randn(n, d)
            Nz -= (Nz * x_cur).sum(axis=1, keepdims=True) * x_cur
            nnorm = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
            Nz = Nz / nnorm
            amp = np.zeros((n, 1))
            amp[:] = 0.012
            for v in involved:
                amp[v] = 0.035 + 0.02 * rng.rand()
            # rattler gets moderate kick to integrate
            rm = Gc.max(axis=1)
            ratt = int(np.argmin(rm))
            amp[ratt] = max(float(amp[ratt]), 0.03)
            x_start = _norm(x_cur + amp * Nz)
        except Exception:
            x_start = x_cur.copy()
        # capped re-jam
        try:
            Xc = run_capped(x_start, capped_mask, cap_val, beta_hi, maxiter=90)
        except Exception:
            continue
        if time.time() - t0 > BUDGET - 12:
            # evaluate quickly
            try:
                vc = _max_cos(Xc)
                if vc < best_val:
                    xp, vp = lp_polish(Xc, deadline=time.time() + 3.0)
                    if vp < best_val:
                        x_best, best_val = xp.copy(), vp
            except Exception:
                pass
            break
        # release: free soften
        try:
            Xf = run_free(Xc, 380.0, maxiter=50)
        except Exception:
            Xf = Xc
        # polish
        try:
            Xp, Vp = lp_polish(Xf, deadline=time.time() + 4.0)
        except Exception:
            continue
        try:
            m_new, deg_new, set_new, Gn = contact_set_deg(Xp, tol=1e-7)
        except Exception:
            continue
        # diversity: jaccard distance from incumbent best set
        try:
            inter = len(set_new & set0)
            union = len(set_new | set0) + 1e-12
            jd = 1.0 - inter / union
        except Exception:
            jd = 0.0
        hub_deg_new = float(deg_new[hub_choice]) if hub_choice < n else 0.0
        hub_deg_old = float(deg_c[hub_choice])
        ratt_new = int(np.min(deg_new))
        # update global best
        if Vp < best_val - 1e-12:
            x_best, best_val = Xp.copy(), Vp
            set0 = set_new
            # move cur to new basin
            x_cur, cur_val, cur_set = Xp.copy(), Vp, set_new
            pool.append((Vp, Xp.copy(), set_new))
        else:
            # accept as exploration if diverse and not too bad and structurally good
            cond_score = (Vp < cur_val + 3e-4) and (Vp < best_val + 5e-4)
            cond_struct = (hub_deg_new < hub_deg_old - 0.5) or (ratt_new >= 1) or (jd > 0.03)
            if cond_score and (cond_struct or jd > 0.06):
                # prefer diverse: replace cur
                x_cur, cur_val, cur_set = Xp.copy(), Vp, set_new
            elif Vp < cur_val:
                x_cur, cur_val, cur_set = Xp.copy(), Vp, set_new
            else:
                # occasionally restart from best with noise to try new hub
                if rng.rand() < 0.25:
                    try:
                        Nz2 = rng.randn(n, d)
                        Nz2 -= (Nz2 * x_best).sum(axis=1, keepdims=True) * x_best
                        Nz2 /= (np.linalg.norm(Nz2, axis=1, keepdims=True) + 1e-12)
                        x_cur = _norm(x_best + 0.015 * Nz2)
                        cur_val = _max_cos(x_cur)
                        _, _, cur_set, _ = contact_set_deg(x_cur, tol=1e-7)
                    except Exception:
                        pass
        # end trial loop
    return _norm(np.asarray(x_best, dtype=float))
