import time
import numpy as np
from collections import defaultdict


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def lp_polish_deadline(x, deadline, r0=2e-3, cap=2500, maxit=100):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = float(r0)
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
        I = iu0[0][idx]
        J = iu0[1][idx]
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
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best


def build_T(x):
    n = x.shape[0]
    N = 3 * n
    T = np.zeros((n * 4, N), dtype=float)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        axes = [a for a in range(4) if a != k]
        V = np.zeros((4, 3), dtype=float)
        for cc, a in enumerate(axes):
            e = np.zeros(4)
            e[a] = 1.0
            V[:, cc] = e - float(xi[a]) * xi
        try:
            Q, _ = np.linalg.qr(V)
        except Exception:
            Q = V
        T[i * 4:(i + 1) * 4, i * 3:(i + 1) * 3] = Q
    return T


def build_Q(T, x):
    n = x.shape[0]
    N = T.shape[1]
    G = np.zeros((N, 6), dtype=float)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for c, (p, q) in enumerate(pairs):
        U = np.zeros_like(x)
        U[:, p] = -x[:, q]
        U[:, q] = x[:, p]
        uvec = U.ravel()
        G[:, c] = T.T @ uvec
    try:
        Q, _ = np.linalg.qr(G)
    except Exception:
        Q, _ = np.linalg.qr(G + 1e-12 * np.random.randn(*G.shape))
    return Q


def build_Jt(edges, x, T):
    N = T.shape[1]
    p = len(edges)
    if p == 0:
        return np.zeros((0, N), dtype=float)
    Jt = np.zeros((p, N), dtype=float)
    for r, (i, j) in enumerate(edges):
        Bi = T[i * 4:(i + 1) * 4, i * 3:(i + 1) * 3]
        Bj = T[j * 4:(j + 1) * 4, j * 3:(j + 1) * 3]
        Jt[r, i * 3:(i + 1) * 3] = Bi.T @ x[j]
        Jt[r, j * 3:(j + 1) * 3] = Bj.T @ x[i]
    return Jt


def row_Jt_vec(pair, x, T):
    N = T.shape[1]
    a, b = int(pair[0]), int(pair[1])
    row = np.zeros(N, dtype=float)
    Ba = T[a * 4:(a + 1) * 4, a * 3:(a + 1) * 3]
    Bb = T[b * 4:(b + 1) * 4, b * 3:(b + 1) * 3]
    row[a * 3:(a + 1) * 3] = Ba.T @ x[b]
    row[b * 3:(b + 1) * 3] = Bb.T @ x[a]
    return row


def degree_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FLIP_END = t0 + 120.0
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = _norm(rng0.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v

    # short initial polish to lock baseline (bounded)
    try:
        px, pv = lp_polish_deadline(global_best_x, min(HARD_END - 60.0, t0 + 10.0), r0=1e-3, maxit=60)
        if pv < global_best_v and np.all(np.isfinite(px)):
            global_best_x = px
            global_best_v = pv
    except Exception:
        pass

    iu = np.triu_indices(n, 1)
    try:
        G0 = global_best_x @ global_best_x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        Gall = global_best_x @ global_best_x.T
        c0 = Gall[iu]
        slack0 = m0 - c0
    except Exception:
        return _norm(global_best_x)

    act_mask = slack0 < 1e-9
    if np.count_nonzero(act_mask) < 100:
        act_mask = slack0 < 1e-7
    if np.count_nonzero(act_mask) < 100:
        act_mask = slack0 < 1e-6
    cI = iu[0][act_mask]
    cJ = iu[1][act_mask]
    contacts = [(int(cI[t]), int(cJ[t])) for t in range(len(cI))]
    p0 = len(contacts)
    if p0 < 50:
        # fallback: just polish
        try:
            fx, fv = lp_polish_deadline(global_best_x, HARD_END - 2.0, r0=2e-3, maxit=200)
            if fv < global_best_v:
                global_best_x = fx
        except Exception:
            pass
        return _norm(global_best_x)

    deg = np.zeros(n, dtype=int)
    for (u, v) in contacts:
        deg[u] += 1
        deg[v] += 1

    # dual weights via NNLS
    T0 = None
    Q0 = None
    w = np.ones(p0) / p0
    try:
        from scipy.optimize import nnls
        T0 = build_T(global_best_x)
        Q0 = build_Q(T0, global_best_x)
        Jt_full = build_Jt(contacts, global_best_x, T0)
        alpha = 2.0
        A_nn = np.vstack([Jt_full.T, alpha * np.ones((1, p0))])
        b_nn = np.zeros(A_nn.shape[0])
        b_nn[-1] = alpha
        w_nn, _ = nnls(A_nn, b_nn, maxiter=5000)
        if np.all(np.isfinite(w_nn)) and np.sum(w_nn) > 0:
            w = w_nn
    except Exception:
        try:
            if T0 is None:
                T0 = build_T(global_best_x)
                Q0 = build_Q(T0, global_best_x)
        except Exception:
            pass

    # rank contacts: small weight first, degree 3/4 boosted
    try:
        warr = np.asarray(w, dtype=float)
        idx_all = np.argsort(warr)
        low = [t for t in idx_all if min(int(deg[contacts[t][0]]), int(deg[contacts[t][1]])) <= 4]
        lowset = set(low)
        high = [t for t in idx_all if t not in lowset]
        ranked = low + high
        ranked20 = ranked[:20]
    except Exception:
        ranked20 = list(range(min(20, p0)))

    # near contacts
    nc_mask = (slack0 >= 1e-9) & (slack0 < 2e-4)
    nc_idx = np.where(nc_mask)[0]
    if len(nc_idx) < 20:
        nc_mask = (slack0 >= 1e-9) & (slack0 < 5e-4)
        nc_idx = np.where(nc_mask)[0]
    nc_by_v = defaultdict(list)
    for idx in nc_idx:
        a = int(iu[0][idx])
        b = int(iu[1][idx])
        s = float(slack0[idx])
        nc_by_v[a].append((a, b, s))
        nc_by_v[b].append((a, b, s))

    cands = []
    for ei in ranked20:
        if ei < 0 or ei >= len(contacts):
            continue
        e = contacts[ei]
        try:
            we = float(w[ei])
        except Exception:
            we = 1.0
        fs = []
        for vtx in e:
            for (a, b, s) in nc_by_v.get(vtx, []):
                fs.append((a, b, s))
        seenf = set()
        ufs = []
        for (a, b, s) in fs:
            key = (min(a, b), max(a, b))
            if key in seenf:
                continue
            seenf.add(key)
            ufs.append((a, b, s))
        ufs.sort(key=lambda t: t[2])
        for (a, b, s) in ufs[:6]:
            cands.append((e, (a, b), we, s))
    cands.sort(key=lambda t: (t[2], t[3]))
    cands = cands[:30]

    seen_hashes = set()
    try:
        seen_hashes.add(degree_hash(global_best_x))
    except Exception:
        pass

    step_sizes = [0.03, 0.02, 0.01, 0.005, 0.002, 0.001]

    for (e, f, we, sf) in cands:
        if time.time() >= FLIP_END - 4.0:
            break
        if time.time() >= HARD_END - 40.0:
            break
        try:
            xb = global_best_x.copy()
            m_cur = float(_max_cos(xb))
            # local bases (reuse if unchanged)
            if T0 is None:
                Tloc = build_T(xb)
                Qloc = build_Q(Tloc, xb)
            else:
                # check drift: reuse T0/Q0 since xb == global best basis point
                Tloc = T0
                Qloc = Q0
            rem_edges = [c for c in contacts if not ((c[0] == e[0] and c[1] == e[1]) or (c[0] == e[1] and c[1] == e[0]))]
            Jt_rem = build_Jt(rem_edges, xb, Tloc)
            N = Tloc.shape[1]
            # mechanism: smallest eig of projected Gram
            try:
                M = Jt_rem.T @ Jt_rem
                P = np.eye(N) - Qloc @ Qloc.T
                Mp = P @ M @ P + 10.0 * (Qloc @ Qloc.T)
                vals_e, vecs_e = np.linalg.eigh(Mp)
                z = None
                for k in range(N):
                    v = vecs_e[:, k]
                    qn = float(np.linalg.norm(Qloc.T @ v))
                    if qn < 0.1:
                        z = v.copy()
                        break
                if z is None:
                    continue
                nz = float(np.linalg.norm(z))
                if nz < 1e-12 or not np.all(np.isfinite(z)):
                    continue
                z = z / nz
            except Exception:
                continue
            # orient so f decreases
            try:
                rf = row_Jt_vec(f, xb, Tloc)
                if float(rf @ z) > 0:
                    z = -z
            except Exception:
                pass
            try:
                Udir = (Tloc @ z).reshape(n, d)
            except Exception:
                continue
            cur = xb.copy()
            # line search walk
            for _step_iter in range(12):
                if time.time() >= FLIP_END - 2.0:
                    break
                Gcur = cur @ cur.T
                np.fill_diagonal(Gcur, -2.0)
                cur_f_cos = float(Gcur[f[0], f[1]])
                moved = False
                for s in step_sizes:
                    y = cur + s * Udir
                    # need Udir recomputed? keep fixed direction for simplicity,
                    # re-project each time: recompute Udir from original orientation is ok
                    y = _norm(y)
                    G = y @ y.T
                    np.fill_diagonal(G, -2.0)
                    mf = float(G[f[0], f[1]])
                    mg = float(np.max(G))
                    if mf < cur_f_cos - 1e-9 and mg <= m_cur + 2e-4:
                        cur = y
                        # update direction to stay tangent at new point? keep same Udir
                        # recompute Udir by projecting old ambient direction to new tangent
                        try:
                            amb = Udir.ravel()
                            # project: amb - (x·amb)x per row
                            dots = np.sum(cur.reshape(-1) * amb if False else np.zeros(1)))
                            Ud = Udir.copy()
                            # proper tangent projection at cur
                            proj = np.sum(Ud * cur, axis=1, keepdims=True) * cur
                            Ud = Ud - proj
                            # re-orthogonalize vs rotations? keep simple, renormalize
                            nn = float(np.linalg.norm(Ud))
                            if nn > 1e-12:
                                Udir = Ud / nn
                        except Exception:
                            pass
                        moved = True
                        break
                if not moved:
                    break
                G2 = cur @ cur.T
                np.fill_diagonal(G2, -2.0)
                mg2 = float(np.max(G2))
                mf2 = float(G2[f[0], f[1]])
                if mg2 - mf2 < 5e-7:
                    break
            # polish walked point
            remain = min(HARD_END - 5.0, time.time() + 8.0)
            if remain <= time.time() + 0.5:
                break
            try:
                yp, vp = lp_polish_deadline(cur, remain, r0=2e-3, maxit=80)
            except Exception:
                continue
            try:
                hh = degree_hash(yp)
            except Exception:
                hh = None
            is_new = (hh is not None and hh not in seen_hashes)
            if hh is not None:
                seen_hashes.add(hh)
            if vp < global_best_v - 1e-12 or (is_new and vp <= global_best_v + 5e-7):
                # accept new basin: update global if improved, else keep exploring from new hash?
                if vp < global_best_v:
                    global_best_x = yp.copy()
                    global_best_v = vp
                    # refresh structures for remaining flips
                    try:
                        G0 = global_best_x @ global_best_x.T
                        np.fill_diagonal(G0, -2.0)
                        m0 = float(G0.max())
                        Gall2 = global_best_x @ global_best_x.T
                        c0b = Gall2[iu]
                        # update contacts/deg/T0/Q0 quickly for next iterations
                        act2 = (m0 - c0b) < 1e-9
                        if np.count_nonzero(act2) >= 50:
                            cI2 = iu[0][act2]
                            cJ2 = iu[1][act2]
                            contacts = [(int(cI2[t]), int(cJ2[t])) for t in range(len(cI2))]
                            p0 = len(contacts)
                            deg = np.zeros(n, dtype=int)
                            for (u, v) in contacts:
                                deg[u] += 1
                                deg[v] += 1
                            T0 = build_T(global_best_x)
                            Q0 = build_Q(T0, global_best_x)
                            m_cur_new = m0
                    except Exception:
                        pass
                else:
                    # different hash but not better: do not replace global best, continue
                    pass
            # stop early if big improvement
            if global_best_v < 0.8491021761744 - 1e-7:
                break
        except Exception:
            continue

    # final polish with remaining time
    try:
        if time.time() < HARD_END - 3.0:
            fx, fv = lp_polish_deadline(global_best_x, HARD_END - 1.5, r0=2e-3, maxit=200)
            if fv < global_best_v and np.all(np.isfinite(fx)):
                global_best_x = fx
                global_best_v = fv
    except Exception:
        pass
    return _norm(global_best_x)
