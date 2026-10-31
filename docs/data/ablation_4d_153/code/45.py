import time
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish_deadline(x, deadline, r0=1e-3, cap=2500, maxit=300):
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

def tangent_basis(x):
    n, d = x.shape
    k = d - 1
    T = np.zeros((n * d, n * k))
    for i in range(n):
        v = x[i]
        # null space of v^T via svd
        try:
            _, _, vh = np.linalg.svd(v.reshape(1, 4))
            B = vh[1:, :].T  # 4x3
        except Exception:
            # fallback Gram-Schmidt with fixed vectors
            B = np.zeros((4, 3))
            cand = np.eye(4)
            cnt = 0
            for c in range(4):
                w = cand[:, c] - np.dot(cand[:, c], v) * v
                for q in range(cnt):
                    w = w - np.dot(w, B[:, q]) * B[:, q]
                nw = np.linalg.norm(w)
                if nw > 1e-8:
                    B[:, cnt] = w / nw
                    cnt += 1
                    if cnt == 3:
                        break
        T[i*d:(i+1)*d, i*k:(i+1)*k] = B
    return T

def rotation_basis(T, x):
    n, d = x.shape
    k = d - 1
    N = n * k
    G = np.zeros((N, 6))
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for c, (p, q) in enumerate(pairs):
        U = np.zeros_like(x)
        U[:, p] = -x[:, q]
        U[:, q] = x[:, p]
        uvec = U.ravel()
        G[:, c] = T.T @ uvec
    # orthonormalize
    try:
        Q, _ = np.linalg.qr(G)
    except Exception:
        Q, _ = np.linalg.qr(G + 1e-12*np.random.randn(*G.shape))
    return Q  # N x 6

def build_Jt(edges, x, T):
    # edges: list of (i,j)
    p = len(edges)
    n, d = x.shape
    nd = n * d
    if p == 0:
        return np.zeros((0, T.shape[1]))
    J = np.zeros((p, nd))
    for r, (i, j) in enumerate(edges):
        J[r, i*d:(i+1)*d] = x[j]
        J[r, j*d:(j+1)*d] = x[i]
    Jt = J @ T
    return Jt

def row_Jt(pair, x, T):
    n, d = x.shape
    nd = n * d
    row = np.zeros(nd)
    a, b = pair
    row[a*d:(a+1)*d] = x[b]
    row[b*d:(b+1)*d] = x[a]
    return row @ T

def degree_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1).tolist()
    return (tuple(sorted(deg)), round(float(m)*1e7))

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FLIP_END = t0 + 120.0
    n, d = 153, 4
    # load best
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = _norm(rng0.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    m0_ref = best_v

    iu = np.triu_indices(n, 1)
    G0 = best_x @ best_x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    c0 = (best_x @ best_x.T)[iu]
    slack0 = m0 - c0

    # active contacts within 1e-9
    act_mask = slack0 < 1e-9
    if np.count_nonzero(act_mask) < 100:
        act_mask = slack0 < 1e-7
    if np.count_nonzero(act_mask) < 100:
        act_mask = slack0 < 1e-6
    cI = iu[0][act_mask]; cJ = iu[1][act_mask]
    contacts = [(int(cI[t]), int(cJ[t])) for t in range(len(cI))]
    p0 = len(contacts)
    # degrees
    deg = np.zeros(n, dtype=int)
    for (u, v) in contacts:
        deg[u] += 1; deg[v] += 1

    # tangent + rotation basis at best_x
    try:
        T0 = tangent_basis(best_x)
        Q0 = rotation_basis(T0, best_x)
        Jt_full = build_Jt(contacts, best_x, T0)
        # NNLS dual weights: A w ~= b, A = [Jt^T; alpha*1^T]
        alpha = 2.0
        A_nn = np.vstack([Jt_full.T, alpha*np.ones((1, p0))])
        b_nn = np.zeros(A_nn.shape[0]); b_nn[-1] = alpha
        try:
            w, _ = nnls(A_nn, b_nn, maxiter=5000)
        except Exception:
            w = np.ones(p0)/p0
        if np.sum(w) <= 0 or not np.all(np.isfinite(w)):
            w = np.ones(p0)/p0
    except Exception:
        w = np.ones(max(p0,1))/max(p0,1)
        T0 = None; Q0 = None

    # rank contacts by small weight, boost degree-3/4
    order_e = []
    try:
        warr = np.asarray(w, dtype=float)
        # bonus: subtract for low-degree endpoints
        score_e = warr.copy()
        for t, (u, v) in enumerate(contacts):
            md = min(int(deg[u]), int(deg[v]))
            if md <= 3:
                score_e[t] -= 1e6  # force top? use large boost but keep order by w within group
            elif md == 4:
                score_e[t] -= 5e5
        # implement grouped sort: first deg<=4 sorted by w, then rest by w
        idx_all = np.argsort(warr)
        low = [t for t in idx_all if min(int(deg[contacts[t][0]]), int(deg[contacts[t][1]])) <= 4]
        high = [t for t in idx_all if min(int(deg[contacts[t][0]]), int(deg[contacts[t][1]])) <= 4]
        # actually low already; high = rest
        high = [t for t in idx_all if t not in set(low)]
        ranked = low + high
        # take best 20
        ranked20 = ranked[:20]
    except Exception:
        ranked20 = list(range(min(20, p0)))

    # non-contact pairs with slack < 2e-4
    nc_mask = (slack0 >= 1e-9) & (slack0 < 2e-4)
    nc_idx = np.where(nc_mask)[0]
    # map vertex -> list of nc pairs
    from collections import defaultdict
    nc_by_v = defaultdict(list)
    nc_pairs = []
    for idx in nc_idx:
        a = int(iu[0][idx]); b = int(iu[1][idx]); s = float(slack0[idx])
        nc_pairs.append((a, b, s))
        nc_by_v[a].append((a, b, s))
        nc_by_v[b].append((a, b, s))
    # if too few, relax to 5e-4
    if len(nc_pairs) < 20:
        nc_mask2 = (slack0 >= 1e-9) & (slack0 < 5e-4)
        nc_idx = np.where(nc_mask2)[0]
        nc_by_v = defaultdict(list); nc_pairs = []
        for idx in nc_idx:
            a = int(iu[0][idx]); b = int(iu[1][idx]); s = float(slack0[idx])
            nc_pairs.append((a, b, s))
            nc_by_v[a].append((a, b, s))
            nc_by_v[b].append((a, b, s))

    # build (e,f) candidates
    cands = []
    for ei in ranked20:
        if ei < 0 or ei >= len(contacts):
            continue
        e = contacts[ei]
        we = float(w[ei]) if ei < len(w) else 1.0
        # f sharing vertex with e
        fs = []
        for v in e:
            for (a, b, s) in nc_by_v.get(v, []):
                # ensure f != e (always since nc)
                fs.append((a, b, s))
        # dedup
        seenf = set()
        ufs = []
        for (a, b, s) in fs:
            key = (min(a,b), max(a,b))
            if key in seenf:
                continue
            seenf.add(key)
            ufs.append((a, b, s))
        # sort by slack ascending
        ufs.sort(key=lambda t: t[2])
        for (a, b, s) in ufs[:6]:
            cands.append((e, (a, b), we, s))
    # rank cands by we + slack
    # normalize: sort by (we, s)
    cands.sort(key=lambda t: (t[2], t[3]))
    # cap to ~30 attempts
    cands = cands[:30]

    seen_hashes = set()
    try:
        seen_hashes.add(degree_hash(global_best_x))
    except Exception:
        pass

    # precompute T0,Q0 for mechanism; if failed, fallback to random?
    if T0 is None:
        try:
            T0 = tangent_basis(global_best_x)
            Q0 = rotation_basis(T0, global_best_x)
        except Exception:
            T0 = None

    step_sizes = [0.03, 0.02, 0.01, 0.005, 0.002, 0.001]

    for (e, f, we, sf) in cands:
        if time.time() >= FLIP_END - 4:
            break
        if time.time() >= HARD_END - 35:
            break
        try:
            # build remaining edges (delete e)
            rem_edges = [c for c in contacts if not ((c[0]==e[0] and c[1]==e[1]) or (c[0]==e[1] and c[1]==e[0]))]
            # use current global_best_x as base? spec says current maximum; use global_best basin
            xb = global_best_x.copy()
            m_cur = float(_max_cos(xb))
            # recompute T,Q at xb (xb == best initially, so reuse; if best changed, recompute)
            # for speed reuse T0/Q0 if xb is original
            Tloc = T0
            Qloc = Q0
            # if global best changed from initial, recompute
            # simple check: if m_cur != m0_ref beyond tol, recompute basis
            if abs(m_cur - m0) > 1e-7:
                try:
                    Tloc = tangent_basis(xb)
                    Qloc = rotation_basis(Tloc, xb)
                except Exception:
                    continue
            if Tloc is None:
                continue
            Jt_rem = build_Jt(rem_edges, xb, Tloc)
            N = Tloc.shape[1]
            # mechanism: min ||Jt_rem z|| s.t. ||z||=1, Q^T z=0
            try:
                M = Jt_rem.T @ Jt_rem  # N x N
                P = np.eye(N) - Qloc @ Qloc.T
                Mp = P @ M @ P + 10.0 * (Qloc @ Qloc.T)
                # eigh
                vals_e, vecs_e = np.linalg.eigh(Mp)
                # pick smallest eigenvector orthogonal to Q (first ones may be Q directions with val ~10)
                # Q directions have eigenvalue ~10; mechanism is smallest among remainder.
                # So find first vector with |Q^T v| small
                z = None
                for k in range(N):
                    v = vecs_e[:, k]
                    qn = np.linalg.norm(Qloc.T @ v)
                    if qn < 0.1:
                        z = v.copy()
                        break
                if z is None:
                    z = vecs_e[:, 0].copy()
                    # project out Q
                    z = P @ z
                    nz = np.linalg.norm(z)
                    if nz < 1e-12:
                        continue
                    z /= nz
            except Exception:
                continue
            # orient so linearised cosine of f decreases
            try:
                rf = row_Jt(f, xb, Tloc)
                dot = float(rf @ z)
            except Exception:
                continue
            if abs(dot) < 1e-12:
                continue
            if dot > 0:
                z = -z; dot = -dot
            U = (Tloc @ z).reshape(n, d)
            # line search
            x_cur = xb.copy()
            # current f dot
            Gc = x_cur @ x_cur.T
            cf_cur = float(Gc[f[0], f[1]])
            m_allow_base = m_cur
            took = 0
            for _it in range(14):
                if time.time() >= FLIP_END:
                    break
                improved = False
                for s in step_sizes:
                    tol = 5e-5 + s * 2e-3
                    xt = x_cur + s * U
                    xt = _norm(xt)
                    Gt = xt @ xt.T
                    np.fill_diagonal(Gt, -2.0)
                    mt = float(np.max(Gt))
                    cf = float((xt[f[0]] @ xt[f[1]]))
                    if not (cf < cf_cur - 1e-12):
                        continue
                    if mt <= m_allow_base + tol:
                        # accept
                        x_cur = xt
                        cf_cur = cf
                        took += 1
                        improved = True
                        break
                if not improved:
                    break
                # check if f reached active level
                Gt2 = x_cur @ x_cur.T
                np.fill_diagonal(Gt2, -2.0)
                mt2 = float(np.max(Gt2))
                if (mt2 - cf_cur) < 1e-9:
                    break
            if took == 0:
                continue
            # polish
            polish_deadline = min(time.time() + 4.0, FLIP_END, HARD_END - 30)
            if polish_deadline - time.time() < 0.8:
                continue
            xp, vp = lp_polish_deadline(x_cur, polish_deadline)
            try:
                h = degree_hash(xp)
            except Exception:
                h = None
            is_new_hash = (h not in seen_hashes) if h is not None else True
            if h is not None:
                seen_hashes.add(h)
            if vp < global_best_v - 1e-12:
                global_best_x = xp.copy(); global_best_v = vp
                m_allow_base = vp
                # update contacts/weights? keep original for remaining flips per spec (share vertex with e from original)
                # but allow improved ref
            elif is_new_hash:
                # retain different basin even if slightly worse? spec: retain only if genuinely different hash or improves.
                # keep track but don't replace global best unless better; store for diversity?
                # If different hash and within small tolerance, keep as alternate base for further flips? 
                # We keep global best unchanged, but record hash. Optionally if vp < m0_ref + 2e-4 keep as base for one more polish? 
                # To respect "retain result only if...", we simply don't revert; we just note hash.
                # Keep best among new-hash within tolerance as secondary? For simplicity, if vp is close and hash new, we still keep global best.
                # But to enable topology change exploration, with small prob use it as base for next flip if much different?
                pass
        except Exception:
            continue

    # final polish of global best with remaining time
    try:
        final_deadline = HARD_END - 1.0
        global_best_x, global_best_v = lp_polish_deadline(global_best_x, final_deadline, r0=1e-3)
    except Exception:
        pass
    # ensure fallback: if global best worse than loaded best, reload
    try:
        fb = None; fv = 1e9
        for fn in ("best.npy", "record.npy"):
            try:
                a = np.load(fn)
                if a.shape == (n, d):
                    a = _norm(a)
                    v = _max_cos(a)
                    if v < fv:
                        fv = v; fb = a.copy()
            except Exception:
                pass
        if fb is not None and fv < global_best_v:
            global_best_x = fb; global_best_v = fv
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
