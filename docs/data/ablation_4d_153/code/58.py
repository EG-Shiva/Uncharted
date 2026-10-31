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

def _lp_polish_deadline(x, deadline, r0=8e-4, cap=2000, maxit=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            o = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[o]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _degree_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FLIP_END = t0 + 118.0
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1)
                if np.min(nr) > 1e-9:
                    a = a / nr[:, None]
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = _norm(rng0.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    # fallback copy
    fallback_x = best_x.copy()
    fallback_v = float(best_v)
    try:
        x0 = best_x.copy()
        m0 = float(best_v)
        iu = np.triu_indices(n, 1)
        G0 = x0 @ x0.T
        c0 = G0[iu]
        # contact threshold for Jacobian
        tolJ = 5e-7
        cmask = c0 > (m0 - tolJ)
        cidx = np.where(cmask)[0]
        # if too few/many, adjust to top ~455
        if cidx.size < 300 or cidx.size > 700:
            order = np.argsort(-c0)
            K = min(max(453, cidx.size), len(order))
            K = min(K, 500)
            cidx = order[:K]
        p = int(cidx.size)
        EI = iu[0][cidx].astype(int)
        EJ = iu[1][cidx].astype(int)
        edges = list(zip(EI.tolist(), EJ.tolist()))
        # tangent bases Bs (n,4,3)
        Bs = np.zeros((n, 4, 3))
        for i in range(n):
            v = x0[i]
            m = int(np.argmax(np.abs(v)))
            cols = [k for k in range(4) if k != m]
            E = np.eye(4)[:, cols]
            coeff = v @ E
            Brow = E - v[:, None] * coeff[None, :]
            try:
                Qb, _ = np.linalg.qr(Brow)
            except Exception:
                Qb = Brow
            # ensure tangent (qr keeps span)
            Bs[i] = Qb
        # rotation basis Qmat (N x 6)
        kdim = 3
        N = n * kdim
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        Qmat = np.zeros((N, 6))
        for cc, (pp, qq) in enumerate(pairs):
            for i in range(n):
                U = np.zeros(4)
                U[pp] = -x0[i, qq]
                U[qq] = x0[i, pp]
                Qmat[i * kdim:(i + 1) * kdim, cc] = Bs[i].T @ U
        try:
            Qq, _ = np.linalg.qr(Qmat)
            Qmat = Qq
        except Exception:
            pass
        # Jt (p x N)
        Jt = np.zeros((p, N))
        for r in range(p):
            ii = int(EI[r]); jj = int(EJ[r])
            Jt[r, ii * kdim:(ii + 1) * kdim] = Bs[ii].T @ x0[jj]
            Jt[r, jj * kdim:(jj + 1) * kdim] = Bs[jj].T @ x0[ii]
        # NNLS dual weights
        try:
            from scipy.optimize import nnls
            Cpen = 5.0
            A = np.vstack([Jt.T, Cpen * np.ones((1, p))])
            b = np.zeros(N + 1); b[-1] = Cpen
            # scale for stability
            w, _ = nnls(A, b, maxiter=10 * p)
            if np.sum(w) <= 1e-12 or not np.all(np.finite(w)):
                raise ValueError("bad w")
        except Exception:
            w = np.ones(p)  # uniform fallback
        # contact graph adjacency
        neigh = [[] for _ in range(n)]
        for (a_, b_) in edges:
            neigh[a_].append(b_)
            neigh[b_].append(a_)
        deg = np.array([len(v) for v in neigh], dtype=int)
        for i in range(n):
            neigh[i] = sorted(neigh[i], key=lambda v: (deg[v], v))
        contact_set = set()
        for (a_, b_) in edges:
            contact_set.add((min(a_, b_), max(a_, b_)))
        order_w = np.argsort(w, kind="stable")
        nseed = min(25, p)
        seed_edges = [(int(EI[order_w[s]]), int(EJ[order_w[s]])) for s in range(nseed)]
        # grow clusters 6-10 vertices
        clusters = []
        seen_cl = set()
        sizes_cycle = [6, 8, 10, 7, 9]
        for si, (sa, sb) in enumerate(seed_edges):
            target = sizes_cycle[si % len(sizes_cycle)]
            visited = []
            vset = set()
            queue = [sa, sb]
            for q in queue:
                if q not in vset:
                    vset.add(q); visited.append(q)
            ptr = 0
            while len(vset) < target and ptr < len(queue):
                cur = queue[ptr]; ptr += 1
                for nb in neigh[cur]:
                    if nb not in vset:
                        vset.add(nb); visited.append(nb); queue.append(nb)
                        if len(vset) >= target:
                            break
                # if queue exhausted, add any neighbor of visited set
                if ptr >= len(queue) and len(vset) < target:
                    # find smallest-degree node adjacent to set but not in set
                    cand = None
                    for vv in list(vset):
                        for nb in neigh[vv]:
                            if nb not in vset:
                                cand = nb
                                break
                        if cand is not None:
                            break
                    if cand is not None:
                        queue.append(cand)
                    else:
                        break
            if len(vset) < 6:
                continue
            fs = frozenset(vset)
            if fs in seen_cl:
                continue
            seen_cl.add(fs)
            clusters.append(sorted(list(vset)))
            if len(clusters) >= 14:
                break
        if len(clusters) == 0:
            # fallback single cluster
            clusters = [list(range(8))]
        tabu = set()
        tabu.add(_degree_hash(x0))
        global_best_x = x0.copy()
        global_best_v = float(m0)
        # precompute incident lists per cluster for deletions
        amps = [0.002, 0.005, 0.012, 0.025, 0.05]
        # collect candidates first (cheap line search), then polish subset
        cand_list = []
        # limit SVD work: for each cluster x k
        for ci, cl in enumerate(clusters):
            if time.time() > FLIP_END - 5.0:
                break
            clset = set(cl)
            # incident contact indices
            incid = []
            for r in range(p):
                ii = int(EI[r]); jj = int(EJ[r])
                if (ii in clset) or (jj in clset):
                    incid.append(r)
            if len(incid) == 0:
                continue
            incid_w = [(float(w[r]), r) for r in incid]
            incid_w.sort(key=lambda t: t[0])
            for kdel in (4, 6, 8):
                if time.time() > FLIP_END - 5.0:
                    break
                kk = min(kdel, len(incid_w))
                if kk < 2:
                    continue
                del_rows = set([r for (_, r) in incid_w[:kk]])
                keep = np.array([r for r in range(p) if r not in del_rows], dtype=int)
                if keep.size == 0:
                    continue
                Jr = Jt[keep]
                # augmented matrix with rotations
                try:
                    mu = 10.0
                    Aaug = np.vstack([Jr, mu * Qmat.T])
                    # SVD
                    _, _, Vt = np.linalg.svd(Aaug, full_matrices=False)
                    mechs = Vt[-3:, :].copy()  # 3 x N smallest
                except Exception:
                    continue
                # del edges info for demotion check
                del_pairs = [(int(EI[r]), int(EJ[r])) for r in del_rows]
                for mi in range(mechs.shape[0]):
                    v = mechs[mi, :]
                    nv = np.linalg.norm(v)
                    if nv < 1e-12 or not np.isfinite(nv):
                        continue
                    v = v / nv
                    # map to DX
                    vv = v.reshape(n, kdim)
                    DX = np.einsum('nij,nj->ni', Bs, vv)
                    # normalize fro to 1 (already ~1 since Bs orthonormal and v unit)
                    fn = np.linalg.norm(DX)
                    if fn > 1e-12:
                        DX = DX / fn
                    for a in amps:
                        for sgn in (1.0, -1.0):
                            xt = x0 + (sgn * a) * DX
                            xt = _norm(xt)
                            G = xt @ xt.T
                            # max
                            # cheap max via triu
                            mnew = float(np.max(G - 2.0 * np.eye(n)))
                            if not np.isfinite(mnew):
                                continue
                            if mnew > m0 + 0.003:
                                continue
                            # demotion count
                            cnt = 0
                            for (aa_, bb_) in del_pairs:
                                if (m0 - float(G[aa_, bb_])) > 1e-5:
                                    cnt += 1
                            if cnt < 2:
                                continue
                            cand_list.append((mnew, ci, kdel, mi, sgn * a, DX.copy()))
        # sort candidates by max
        cand_list.sort(key=lambda t: t[0])
        # polish top candidates (limit count to fit time)
        max_polish = 40
        npol = 0
        for (mnew, ci, kdel, mi, amp, DX) in cand_list:
            if time.time() > FLIP_END:
                break
            if npol >= max_polish:
                break
            xt = _norm(x0 + amp * DX)
            h = _degree_hash(xt)
            if h in tabu:
                continue
            tabu.add(h)
            dl = min(time.time() + 1.2, FLIP_END)
            if dl <= time.time() + 0.15:
                break
            try:
                xp, vp = _lp_polish_deadline(xt, dl, r0=8e-4, cap=2000, maxit=200)
            except Exception:
                continue
            npol += 1
            try:
                hp = _degree_hash(xp)
                if hp in tabu and not (vp < global_best_v - 1e-12):
                    # still record tabu but skip
                    pass
                tabu.add(hp)
            except Exception:
                pass
            if vp < global_best_v - 1e-12:
                # require real topology change? accept any improvement
                global_best_v = float(vp)
                global_best_x = xp.copy()
                if global_best_v < m0 - 2e-6:
                    # keep going; do not rebase x0 to keep enumeration stable
                    pass
            # keep tabu size small
            if len(tabu) > 500:
                # trim arbitrarily (keep last 300) - convert to list
                tabu = set(list(tabu)[-300:])
        # final thorough polish of global best with remainder
        try:
            if time.time() < HARD_END - 1.0:
                global_best_x, global_best_v = _lp_polish_deadline(global_best_x, HARD_END - 0.7, r0=1e-3, cap=2500, maxit=400)
        except Exception:
            pass
        global_best_x = _norm(global_best_x)
        # safety: if not better than fallback, return best of the two
        try:
            gv = _max_cos(global_best_x)
            if not np.isfinite(gv) or gv > fallback_v + 1e-12:
                return fallback_x
        except Exception:
            return fallback_x
        return global_best_x
    except Exception:
        try:
            return fallback_x
        except Exception:
            rng = np.random.default_rng(0)
            return _norm(rng.standard_normal(size=(153, 4)))
