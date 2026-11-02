import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _smooth(x, steps, a0, a1, eta0, deadline=None):
    x = x.copy()
    for s in range(steps):
        if deadline is not None and (s % 60 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * ((a1 / max(a0, 1e-9)) ** frac)
        eta = eta0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W @ x
        wg = (W * G).sum(axis=1, keepdims=True)
        Tang = S - wg * x
        wsum = W.sum(axis=1, keepdims=True) + 1e-12
        step = Tang / wsum
        x = x - eta * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, max_pairs=600):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.25:
            break
        Gfull = x @ x.T
        c_full = Gfull[iu0]
        m = c_full.max()
        thresh = m - max(20 * radius, 1e-9)
        sel_idx = np.where(c_full > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > max_pairs:
            part = np.argpartition(c_full[sel_idx], -max_pairs)[-max_pairs:]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        p = len(I)
        b_ub = -c_full[sel_idx]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _fingerprint(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = np.where(c > m - tol)[0]
    s = set((iu[0][sel] * 1000 + iu[1][sel]).tolist())
    return s, m

def _procrustes_rms(A, B):
    # min_O ||A - B O||_F / sqrt(n)
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        O = U @ Vt
        D = A - B @ O
        return float(np.linalg.norm(D) / np.sqrt(A.shape[0]))
    except Exception:
        return 1e9

def _jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / max(1, union)

def _connected_cluster(adj, rng, k_target):
    n = len(adj)
    seed = int(rng.integers(n))
    visited = [seed]
    seen = set([seed])
    queue = [seed]
    while queue and len(visited) < k_target:
        u = queue.pop(0)
        nbrs = adj[u]
        rng.shuffle(nbrs)
        for w in nbrs:
            if w not in seen:
                seen.add(w)
                visited.append(w)
                queue.append(w)
                if len(visited) >= k_target:
                    break
    if len(visited) < k_target:
        # fill randomly if disconnected
        extra = rng.choice(n, size=k_target - len(visited), replace=False)
        for e in extra:
            if int(e) not in seen:
                visited.append(int(e))
    return np.array(visited[:k_target], dtype=int)

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = t0 + 140.0
    rng = np.random.default_rng(12345)
    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (189, 12) and np.all(np.isfinite(c)):
                c = _norm(c)
                cands.append(c)
        except Exception:
            pass
    if not cands:
        c = rng.normal(size=(189, 12))
        cands.append(_norm(c))
    best_x = cands[0]
    best_v = _max_cos(best_x)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_v:
            best_x, best_v = c.copy(), v
    best_x = best_x.copy()
    # archive: list of dicts x, v, fp
    arch_x = [best_x.copy()]
    arch_v = [best_v]
    fp0, _ = _fingerprint(best_x, tol=1e-6)
    arch_fp = [fp0]
    def is_distinct(x_new, fp_new):
        for ax in arch_x:
            r = _procrustes_rms(ax, x_new)
            if r < 0.03:
                # need fingerprint check too
                # find matching fp
                idx = arch_x.index(ax) if False else None
                pass
        # compare to each member
        for k, ax in enumerate(arch_x):
            r = _procrustes_rms(ax, x_new)
            j = _jaccard(arch_fp[k], fp_new)
            if r < 0.03 and j > 0.90:
                return False, k
        return True, -1
    # quick diversity seeding: 2 noisy copies smoothed briefly
    for si in range(2):
        if time.time() > EXPLORE_END - 100:
            break
        y = best_x + rng.normal(scale=0.02, size=best_x.shape)
        y = _norm(y)
        y = _smooth(y, steps=120, a0=25.0, a1=250.0, eta0=0.05, deadline=EXPLORE_END - 2.0)
        v = _max_cos(y)
        fp, _ = _fingerprint(y, tol=1e-6)
        distinct, _ = is_distinct(y, fp)
        if distinct and len(arch_x) < 6:
            arch_x.append(y); arch_v.append(v); arch_fp.append(fp)
            if v < best_v:
                best_x, best_v = y.copy(), v
    it = 0
    while True:
        now = time.time()
        if now > EXPLORE_END - 1.2:
            break
        if it > 200:
            break
        frac = (now - t0) / max(1e-9, (EXPLORE_END - t0))
        frac = min(max(frac, 0.0), 1.0)
        T = 1e-4 * ((1e-2) ** frac)  # 1e-4 -> 1e-6
        # pick base uniformly for diversity (slight bias to best every 4th)
        if it % 4 == 0:
            bi = int(np.argmin(arch_v))
        else:
            bi = int(rng.integers(len(arch_x)))
        base = arch_x[bi]
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        mb = float(Gb.max())
        # adjacency for cluster (tol 0.005 gives connected graph)
        thr_adj = mb - 0.005
        adj = [np.where(Gb[i] > thr_adj)[0].tolist() for i in range(base.shape[0])]
        # ensure each has at least itself excluded; if isolated, connect to nearest
        per = Gb.max(axis=1)
        nn = np.argmax(Gb, axis=1)
        for i in range(len(adj)):
            if len(adj[i]) == 0:
                adj[i] = [int(nn[i])]
            # remove self if present (not, diag -2)
            adj[i] = [w for w in adj[i] if w != i]
            if len(adj[i]) == 0:
                adj[i] = [int(nn[i])]
        y = base.copy()
        mode = rng.random()
        if mode < 0.5:
            # rotate whole connected cluster through random 2-plane
            k = int(rng.integers(5, 16))
            cl = _connected_cluster(adj, rng, k)
            # random 2-plane
            Ru = rng.normal(size=(12, 2))
            Q, _ = np.linalg.qr(Ru)
            th = float(rng.uniform(0.05, 0.2)) * (1.0 if rng.random() < 0.5 else -1.0)
            ct, st = np.cos(th), np.sin(th)
            R2 = np.array([[ct, -st], [st, ct]])
            for idx in cl:
                xx = y[idx]
                c2 = Q.T @ xx
                c2r = R2 @ c2
                y[idx] = xx + Q @ (c2r - c2)
            y = _norm(y)
        else:
            # displace along smallest singular vectors of active-contact Jacobian
            # pick 22 high-load points
            deg = (Gb > mb - 0.01).sum(axis=1)
            k2 = 22
            sel_pts = np.argsort(deg)[-k2:]
            sel_set = set(sel_pts.tolist())
            pos = {int(vv): kk for kk, vv in enumerate(sel_pts)}
            # collect active pairs among sel_pts with tight threshold
            iu_s = []
            for a in range(k2):
                for b in range(a + 1, k2):
                    i = int(sel_pts[a]); j = int(sel_pts[b])
                    if Gb[i, j] > mb - 0.01:
                        iu_s.append((a, b, i, j))
            if len(iu_s) < 8:
                # fallback: take closest pairs among sel
                vals = []
                for a in range(k2):
                    for b in range(a + 1, k2):
                        i = int(sel_pts[a]); j = int(sel_pts[b])
                        vals.append((Gb[i, j], a, b, i, j))
                vals.sort(reverse=True)
                iu_s = [(a, b, i, j) for (_, a, b, i, j) in vals[:30]]
            P = len(iu_s)
            J = np.zeros((P, k2 * 12))
            Xb = base
            for r, (a, b, i, j) in enumerate(iu_s):
                J[r, a * 12:(a + 1) * 12] = Xb[j]
                J[r, b * 12:(b + 1) * 12] = Xb[i]
            try:
                _, _, Vt = np.linalg.svd(J, full_matrices=False)
                # take two smallest
                for vv_idx in [-1, -2]:
                    if -vv_idx <= Vt.shape[0]:
                        vec = Vt[vv_idx]
                        D = vec.reshape(k2, 12)
                        # tangent project
                        for a in range(k2):
                            i = int(sel_pts[a])
                            xi = Xb[i]
                            D[a] -= float(D[a] @ xi) * xi
                        mx = np.linalg.norm(D, axis=1).max()
                        if mx > 1e-12:
                            scale = float(rng.uniform(0.05, 0.12)) / mx
                            D *= scale
                            y[sel_pts] = y[sel_pts] + D * (1.0 if vv_idx == -1 else 0.6)
                        break
            except Exception:
                y[sel_pts] += rng.normal(scale=0.05, size=(k2, 12))
            # tiny extra noise everywhere
            y += rng.normal(scale=0.004, size=y.shape)
            y = _norm(y)
        # smooth briefly
        if time.time() > EXPLORE_END - 1.0:
            break
        y = _smooth(y, steps=160, a0=30.0, a1=280.0, eta0=0.05, deadline=EXPLORE_END - 1.0)
        if time.time() > EXPLORE_END - 0.8:
            v = _max_cos(y)
            # SA accept without LP if no time
            if v < best_v or rng.random() < np.exp(-max(0.0, v - best_v) / T):
                fp, _ = _fingerprint(y, tol=1e-6)
                distinct, match = is_distinct(y, fp)
                if distinct:
                    if len(arch_x) < 6:
                        arch_x.append(y); arch_v.append(v); arch_fp.append(fp)
                    else:
                        wi = int(np.argmax(arch_v))
                        if v < arch_v[wi]:
                            arch_x[wi] = y; arch_v[wi] = v; arch_fp[wi] = fp
                elif v < arch_v[match]:
                    arch_x[match] = y; arch_v[match] = v; arch_fp[match] = fp
                if v < best_v:
                    best_x, best_v = y.copy(), v
            it += 1
            continue
        # briefly LP-polish, cap 2s (use 0.9s)
        lp_dead = min(time.time() + 0.9, EXPLORE_END - 0.3)
        y = _lp_polish(y, deadline=lp_dead, max_pairs=550)
        if not np.all(np.isfinite(y)):
            it += 1
            continue
        y = _norm(y)
        v = _max_cos(y)
        fp_new, _ = _fingerprint(y, tol=1e-6)
        distinct, match = is_distinct(y, fp_new)
        # SA acceptance vs global best
        accept = False
        if v < best_v - 1e-12:
            accept = True
        else:
            p = np.exp(-max(0.0, v - best_v) / max(T, 1e-12))
            if rng.random() < p:
                accept = True
        if accept:
            if distinct:
                if len(arch_x) < 6:
                    arch_x.append(y.copy()); arch_v.append(v); arch_fp.append(fp_new)
                else:
                    wi = int(np.argmax(arch_v))
                    if v < arch_v[wi] - 1e-12 or rng.random() < 0.3:
                        # replace worst to keep diversity cycling
                        arch_x[wi] = y.copy(); arch_v[wi] = v; arch_fp[wi] = fp_new
            else:
                if v < arch_v[match]:
                    arch_x[match] = y.copy(); arch_v[match] = v; arch_fp[match] = fp_new
            if v < best_v - 1e-12:
                best_x, best_v = y.copy(), v
        else:
            # even if not accepted globally, keep if it improves its matching member
            if not distinct and v < arch_v[match]:
                arch_x[match] = y.copy(); arch_v[match] = v; arch_fp[match] = fp_new
        it += 1
    # final polish of archive best
    bi = int(np.argmin(arch_v))
    best_x = arch_x[bi].copy()
    best_v = arch_v[bi]
    remain = HARD_END - time.time()
    if remain > 2.0:
        fin_dead = time.time() + min(remain - 0.7, 30.0)
        best_x = _lp_polish(best_x, deadline=fin_dead, max_pairs=900)
        best_x = _norm(best_x)
    return best_x
