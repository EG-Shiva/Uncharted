import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, tol=1e-6):
    n = x.shape[0]
    G = x @ x.T
    m = float(np.max(G - 2.0*np.eye(n)))
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - tol
    pairs = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    deg = np.zeros(n, dtype=int)
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    return m, pairs, deg

def _fingerprint(x, tol=1e-6):
    m, pairs, deg = _contacts(x, tol)
    return m, pairs, tuple(sorted(deg.tolist()))

def _proc_dist(A, B):
    # min_O ||A - B O||_F / sqrt(n)
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        O = U @ Vt
        D = A - B @ O
        return float(np.linalg.norm(D) / np.sqrt(A.shape[0]))
    except Exception:
        return 1.0

def _is_distinct(x, v, arch, tol=1e-6, rms_thr=0.03):
    _, pairs, degt = _fingerprint(x, tol)
    for (av, ax, _, _, _) in arch:
        _, apairs, adeg = _fingerprint(ax, tol)
        # fingerprint difference: symmetric difference size
        inter = len(pairs & apairs)
        union = len(pairs | apairs)
        jacc = inter / max(1, union)
        # degree L1
        l1 = sum(abs(a-b) for a, b in zip(degt, adeg))
        d = _proc_dist(x, ax)
        if d < rms_thr and jacc > 0.995 and l1 < 20:
            return False
    return True

def _lp_polish(x, budget, radius_init=1e-3):
    t_end = time.time() + budget
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap constraints for speed
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(c[sel], -2500)[-2500:]
            I = I[idx]; J = J[idx]
            csel = c[sel][idx]
            p = len(I)
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # time check inside loop via while condition
    return x, best

def _smooth(x, steps=120, alpha=80.0, lr=0.05):
    n, d = x.shape
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        # decay lr slightly
        cur_lr = lr * (1.0 - 0.5 * s / max(1, steps))
        x = x - cur_lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _get_cluster(rng, adj, n, smin=5, smax=15):
    s = int(rng.integers(smin, smax + 1))
    seed = int(rng.integers(0, n))
    cluster = [seed]
    seen = set(cluster)
    frontier = [seed]
    while len(cluster) < s and frontier:
        u = frontier.pop(0)
        nbrs = adj.get(u, [])
        rng.shuffle(nbrs)
        for w in nbrs:
            if w not in seen:
                seen.add(w)
                cluster.append(w)
                frontier.append(w)
                if len(cluster) >= s:
                    break
    # if graph too sparse, fill randomly
    while len(cluster) < s:
        w = int(rng.integers(0, n))
        if w not in seen:
            seen.add(w)
            cluster.append(w)
    return np.array(cluster, dtype=int)

def _rotate_cluster(x, cluster, rng):
    n, d = x.shape
    theta = float(rng.uniform(0.05, 0.2))
    if rng.random() < 0.5:
        theta = -theta
    A = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]; v = Q[:, 1]
    y = x.copy()
    cu = x[cluster] @ u  # (k,)
    cv = x[cluster] @ v
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    cu2 = cos_t * cu - sin_t * cv
    cv2 = sin_t * cu + cos_t * cv
    # x_perp + cu2*u + cv2*v
    base = x[cluster] - np.outer(cu, u) - np.outer(cv, v)
    y[cluster] = base + np.outer(cu2, u) + np.outer(cv2, v)
    y[cluster] /= np.linalg.norm(y[cluster], axis=1, keepdims=True)
    return y

def _floppy_step(x, cluster, rng, tol_active=3e-4):
    n, d = x.shape
    k = len(cluster)
    pos = {int(v): i for i, v in enumerate(cluster)}
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # collect active pairs touching cluster
    iu_pairs = []
    # only check pairs where at least one in cluster for speed
    # gather candidates: for each c in cluster, find j with G[c,j] > m - tol_active
    for c in cluster:
        row = G[int(c)]
        js = np.where(row > m - tol_active)[0]
        for j in js:
            j = int(j)
            if j == int(c):
                continue
            a = min(int(c), j); b = max(int(c), j)
            iu_pairs.append((a, b))
    # dedup
    iu_pairs = sorted(set(iu_pairs))
    if len(iu_pairs) > 400:
        # keep tightest
        vals = np.array([G[a, b] for a, b in iu_pairs])
        idx = np.argpartition(vals, -400)[-400:]
        iu_pairs = [iu_pairs[i] for i in idx]
    p = len(iu_pairs)
    y = x.copy()
    if p == 0:
        # random tangent kick
        D = rng.standard_normal((k, d)) * 0.03
        dot = np.sum(D * x[cluster], axis=1, keepdims=True)
        D -= dot * x[cluster]
        y[cluster] = x[cluster] + D
        y[cluster] /= np.linalg.norm(y[cluster], axis=1, keepdims=True)
        return y
    # Build Jacobian p x (k*d)
    Jm = np.zeros((p, k * d))
    for r, (a, b) in enumerate(iu_pairs):
        if a in pos and b in pos:
            Jm[r, pos[a]*d:(pos[a]+1)*d] = x[b]
            Jm[r, pos[b]*d:(pos[b]+1)*d] = x[a]
        elif a in pos:
            Jm[r, pos[a]*d:(pos[a]+1)*d] = x[b]
        elif b in pos:
            Jm[r, pos[b]*d:(pos[b]+1)*d] = x[a]
    try:
        # smallest singular vector
        _, S, Vt = np.linalg.svd(Jm, full_matrices=False)
        # pick one of last 3 randomly for diversity
        q = min(3, Vt.shape[0])
        pick = int(rng.integers(0, q))
        vec = Vt[-(pick+1)]
    except Exception:
        vec = rng.standard_normal(k*d)
    D = vec.reshape(k, d)
    # project to tangent
    dot = np.sum(D * x[cluster], axis=1, keepdims=True)
    D -= dot * x[cluster]
    dn = np.linalg.norm(D) + 1e-30
    D /= dn
    step = float(rng.uniform(0.05, 0.15)) * np.sqrt(k)
    # step scaled so per-point move ~0.05-0.15
    D = D * (step / np.sqrt(k))
    y[cluster] = x[cluster] + D
    y[cluster] /= np.linalg.norm(y[cluster], axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    T_END = t0 + 178.0
    T_EXPL = t0 + 140.0
    rng = np.random.default_rng(7)
    # load start
    cand = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=np.float64)
            if a.shape == (189, 12):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if cand is None or v < cand[0]:
                    cand = (v, a)
        except Exception:
            pass
    if cand is None:
        a = rng.standard_normal((189, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand = (_max_cos(a), a)
    best_v, best_x = cand[0], cand[1].copy()
    cur_v, cur_x = best_v, best_x.copy()
    arch = [(best_v, best_x.copy(), 0)]
    it = 0
    # prebuild: ensure archive has at least diversity via tiny kicks if time allows (skip)
    while time.time() < T_EXPL and time.time() < T_END - 8.0:
        elapsed = time.time() - t0
        frac = elapsed / 140.0
        T = 1e-4 * (1e-2) ** min(1.0, max(0.0, frac))  # 1e-4 -> 1e-6
        # choose base: mostly cur, sometimes best/arch random
        r = rng.random()
        if r < 0.65:
            base_x, base_v = cur_x, cur_v
        elif r < 0.85:
            base_x, base_v = best_x, best_v
        else:
            ai = int(rng.integers(0, len(arch)))
            base_x, base_v = arch[ai][1], arch[ai][0]
        # build adjacency at 1e-6 for cluster
        try:
            _, pairs_b, _ = _contacts(base_x, tol=1e-6)
        except Exception:
            pairs_b = set()
        adj = {}
        for a_, b_ in pairs_b:
            adj.setdefault(a_, []).append(b_)
            adj.setdefault(b_, []).append(a_)
        # copy lists to allow shuffle without affecting? _get_cluster shuffles in place so copy
        adj2 = {k: list(v) for k, v in adj.items()}
        cluster = _get_cluster(rng, adj2, 189, 5, 15)
        if rng.random() < 0.5:
            off = _rotate_cluster(base_x, cluster, rng)
        else:
            off = _floppy_step(base_x, cluster, rng)
        # optional tiny global jitter
        if rng.random() < 0.3:
            j = rng.choice(189, size=int(rng.integers(1, 4)), replace=False)
            off[j] += 0.01 * rng.standard_normal((len(j), 12))
            off[j] /= np.linalg.norm(off[j], axis=1, keepdims=True)
        # smooth briefly
        try:
            off = _smooth(off, steps=90, alpha=60.0, lr=0.06)
        except Exception:
            pass
        # briefly LP-polish, cap 2s
        remain_expl = T_EXPL - time.time()
        if remain_expl < 0.3:
            break
        budget = min(2.0, remain_expl, max(0.5, T_END - time.time() - 8.0))
        try:
            off_p, off_v = _lp_polish(off, budget=budget)
        except Exception:
            it += 1
            continue
        # archive update if distinct
        distinct = _is_distinct(off_p, off_v, [(a[0], a[1], 0) for a in arch])
        if distinct:
            if len(arch) < 6:
                arch.append((off_v, off_p.copy(), it))
            else:
                # replace worst if better
                worst_i = int(np.argmax([a[0] for a in arch]))
                if off_v < arch[worst_i][0]:
                    arch[worst_i] = (off_v, off_p.copy(), it)
        else:
            # if not distinct but better than matching member, replace it
            # find closest member
            best_d = 1e9; best_i = -1
            for i, (av, ax, _) in enumerate(arch):
                d = _proc_dist(off_p, ax)
                if d < best_d:
                    best_d = d; best_i = i
            if best_i >= 0 and off_v < arch[best_i][0] - 1e-12:
                arch[best_i] = (off_v, off_p.copy(), it)
        if off_v < best_v:
            best_v, best_x = off_v, off_p.copy()
        # SA acceptance for cur (spec: exp(-(v - v_best)/T))
        if off_v <= cur_v:
            cur_v, cur_x = off_v, off_p.copy()
        else:
            # use v_best at time of comparison (best_v)
            prob = np.exp(-(off_v - best_v) / max(T, 1e-12))
            if rng.random() < prob:
                cur_v, cur_x = off_v, off_p.copy()
        it += 1
        # hard guard: check clock
        if time.time() >= T_EXPL:
            break
    # final polish of archive best
    # find best in arch
    arch.sort(key=lambda t: t[0])
    if arch[0][0] < best_v:
        best_v, best_x = arch[0][0], arch[0][1].copy()
    else:
        # ensure best_x is archived best if close
        pass
    remain = T_END - time.time() - 1.0
    if remain > 1.0:
        try:
            # spend remaining polishing best
            best_x, best_v = _lp_polish(best_x, budget=min(remain, 35.0), radius_init=5e-4)
        except Exception:
            pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
