import time
import numpy as np
from collections import deque, defaultdict

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(x):
    n = x.shape[0]
    T = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        axes = [a for a in range(4) if a != k]
        V = np.zeros((4, 3), dtype=np.float64)
        for c, a in enumerate(axes):
            e = np.zeros(4)
            e[a] = 1.0
            V[:, c] = e - float(xi[a]) * xi
        Q, _ = np.linalg.qr(V)
        T[i] = Q
    return T

def _global_Q(x, T):
    n = x.shape[0]
    m = 3 * n
    Qraw = np.zeros((m, 6), dtype=np.float64)
    col = 0
    for a in range(4):
        for b in range(a + 1, 4):
            for i in range(n):
                xi = x[i]
                y = np.zeros(4)
                y[a] = xi[b]
                y[b] = -xi[a]
                Qraw[i*3:(i+1)*3, col] = T[i].T @ y
            col += 1
    Q, _ = np.linalg.qr(Qraw)
    return Q

def _contact_J(x, T, I, J):
    p = I.shape[0]
    m = 3 * x.shape[0]
    Jac = np.zeros((p, m), dtype=np.float64)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        Jac[k, i*3:(i+1)*3] = T[i].T @ x[j]
        Jac[k, j*3:(j+1)*3] = T[j].T @ x[i]
    return Jac

def _lp_polish(x, deadline, r0=0.003, cap=2500, maxit=120):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1e9
    best_x = x.copy()
    iu0 = np.triu_indices(n, 1)
    radius = float(r0)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            radius *= 0.5
            continue
        if idx.size > cap:
            part = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[part]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cI = I[:, None]*d + np.arange(d)
            cJ = J[:, None]*d + np.arange(d)
            cols = np.concatenate([cI, cJ, np.full((p,1), n*d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
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
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm_rows(x + dx)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _deg_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m)*1e7)), m

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FLIP_END = t0 + 120.0
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = _norm_rows(rng0.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_x = best_x.copy(); global_v = best_v
    try:
        global_x, global_v = _lp_polish(global_x, min(HARD_END-5.0, t0+10.0), r0=0.003, maxit=100)
    except Exception:
        pass
    base_x = global_x.copy(); base_v = global_v
    seen = set()
    try:
        hk0, _ = _deg_hash(base_x, tol=5e-7)
        seen.add(hk0)
    except Exception:
        pass
    tabu = []
    try:
        G0 = base_x @ base_x.T
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        m0 = float(np.max(c_all))
        ACT_TOL = 1e-7
        act_mask = c_all > (m0 - ACT_TOL)
        act_idx = np.where(act_mask)[0]
        Iact = iu_all[0][act_idx]; Jact = iu_all[1][act_idx]
        p_act = act_idx.size
        if p_act < 20 or p_act > 3000:
            raise ValueError("bad active count")
        T = _tangent_bases(base_x)
        Q = _global_Q(base_x, T)
        m3n = 3*n
        Jfull = _contact_J(base_x, T, Iact, Jact)
        # NNLS dual weights
        try:
            from scipy.optimize import nnls
            Aeq = Jfull.T
            alpha = 1.0
            M = np.vstack([Aeq, alpha*np.ones((1, p_act))])
            b = np.zeros(M.shape[0]); b[-1] = alpha
            w, _ = nnls(M, b, maxiter=10*p_act)
            if np.sum(w) <= 1e-12 or np.any(~np.isfinite(w)):
                raise ValueError("nnls bad")
            w = w / float(np.sum(w))
        except Exception:
            w = np.ones(p_act)/max(1, p_act)
        order_weak = np.argsort(w)
        # adjacency sorted by w
        adj = [[] for _ in range(n)]
        for k in range(p_act):
            i = int(Iact[k]); j = int(Jact[k])
            wk = float(w[k])
            adj[i].append((j, k, wk))
            adj[j].append((i, k, wk))
        for i in range(n):
            adj[i].sort(key=lambda t: t[2])
        # grow clusters 6-10 vertices by BFS from lowest-w contacts
        num_seeds = min(12, p_act)
        seed_ks = order_weak[:num_seeds]
        target_sizes = [6, 8, 10]
        clusters = []
        cseen = set()
        for sk in seed_ks:
            si = int(Iact[int(sk)]); sj = int(Jact[int(sk)])
            for ts in target_sizes:
                try:
                    visited = set([si, sj])
                    qq = deque([si, sj])
                    while len(visited) < ts and len(qq) > 0:
                        u = qq.popleft()
                        for (nbr, kk, ww) in adj[u]:
                            if nbr not in visited:
                                visited.add(nbr)
                                qq.append(nbr)
                                if len(visited) >= ts:
                                    break
                    if len(visited) < 6:
                        continue
                    # trim to ts if larger (should not)
                    # keep as frozenset key
                    key = (tuple(sorted(visited)), ts)
                    if key in cseen:
                        continue
                    cseen.add(key)
                    clusters.append(set(visited))
                except Exception:
                    continue
        # projector out global rotations
        try:
            P = np.eye(m3n) - Q @ Q.T
        except Exception:
            P = np.eye(m3n)
        amps = [0.002, 0.0035, 0.006, 0.01, 0.017, 0.03, 0.05]
        kvals = [4, 6, 8]
        # precompute for speed: Jfull @ Q
        try:
            JQ = Jfull @ Q
        except Exception:
            JQ = None
        # iterate clusters x k
        # limit mechanisms to smallest 2 singular vectors
        nmech = 2
        for cset in clusters:
            if time.time() > FLIP_END:
                break
            # contacts in and around cluster
            try:
                mask_in = np.zeros(p_act, dtype=bool)
                for k in range(p_act):
                    if int(Iact[k]) in cset or int(Jact[k]) in cset:
                        mask_in[k] = True
                cand_ks = np.where(mask_in)[0]
                if cand_ks.size < 4:
                    continue
                # sort cand by w ascending (weakest first)
                cand_ks = cand_ks[np.argsort(w[cand_ks])]
            except Exception:
                continue
            for kdel in kvals:
                if time.time() > FLIP_END:
                    break
                if cand_ks.size < kdel:
                    continue
                del_ks = cand_ks[:kdel]
                # avoid duplicate delete sets (tabu on delete tuple)
                delkey = tuple(sorted([int(x) for x in del_ks]))
                if delkey in tabu:
                    continue
                # build Jrem
                try:
                    keep = np.ones(p_act, dtype=bool)
                    keep[del_ks] = False
                    Jrem = Jfull[keep]
                    # A = Jrem @ P  via  Jrem - (Jrem@Q)Q^T
                    JremQ = Jrem @ Q
                    A = Jrem - JremQ @ Q.T
                    Gram = A.T @ A
                    # symmetrize
                    Gram = 0.5*(Gram+Gram.T)
                    vals_e, vecs_e = np.linalg.eigh(Gram)
                    # smallest nmech
                    mechs = []
                    for mi in range(nmech):
                        v = vecs_e[:, mi].copy()
                        # project out Q again for safety
                        coeff = Q.T @ v
                        v = v - Q @ coeff
                        nv = float(np.linalg.norm(v))
                        if nv < 1e-12 or not np.isfinite(nv):
                            continue
                        v /= nv
                        mechs.append(v)
                    if len(mechs) == 0:
                        continue
                except Exception:
                    continue
                for v in mechs:
                    if time.time() > FLIP_END:
                        break
                    try:
                        Vmat = v.reshape(n, 3)
                        D = np.einsum('nij,nj->ni', T, Vmat)
                        # line search both signs
                        for sgn in (1.0, -1.0):
                            if time.time() > FLIP_END:
                                break
                            best_amp = None
                            best_my = 1e9
                            best_Y = None
                            for al in amps:
                                try:
                                    Y = _norm_rows(base_x + (sgn*al)*D)
                                    G = Y @ Y.T
                                    np.fill_diagonal(G, -2.0)
                                    my = float(np.max(G))
                                    if my > m0 + 1e-4:
                                        continue
                                    # demoted slacks w.r.t m0
                                    cnt = 0
                                    for dk in del_ks:
                                        di = int(Iact[int(dk)]); dj = int(Jact[int(dk)])
                                        gval = float(Y[di] @ Y[dj])
                                        if (m0 - gval) > 1e-5:
                                            cnt += 1
                                    if cnt < 2:
                                        continue
                                    if my < best_my:
                                        best_my = my
                                        best_amp = al
                                        best_Y = Y
                                except Exception:
                                    continue
                            if best_Y is None:
                                continue
                            # LP polish candidate 1-2s
                            dl = min(FLIP_END, time.time()+1.5)
                            if dl >= HARD_END-15.0:
                                break
                            try:
                                Yp, vp = _lp_polish(best_Y, dl, r0=0.003, maxit=60)
                            except Exception:
                                continue
                            try:
                                hk, mm = _deg_hash(Yp, tol=5e-7)
                            except Exception:
                                continue
                            is_new = (hk not in seen)
                            if is_new:
                                seen.add(hk)
                            else:
                                if vp >= global_v:
                                    continue
                            # keep tabu of delete set to avoid repeat
                            if len(tabu) > 200:
                                tabu.pop(0)
                            tabu.append(delkey)
                            if vp < global_v - 1e-12:
                                global_x = Yp.copy(); global_v = vp
                            # early exit if improved a lot? keep searching
                    except Exception:
                        continue
    except Exception:
        pass
    # final polish of global best
    try:
        rem = HARD_END - time.time() - 1.0
        if rem > 3.0:
            global_x, global_v = _lp_polish(global_x, time.time()+min(rem, 40.0), r0=0.002, maxit=150)
    except Exception:
        pass
    try:
        global_x = _norm_rows(global_x)
    except Exception:
        pass
    return global_x
