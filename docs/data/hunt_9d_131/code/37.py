import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def fast_polish(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        c_all = (x @ x.T)[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        if p > 5000:
            # keep worst 5000 to bound LP size
            idx = np.argsort(-c)[:5000]
            I, J, c = I[idx], J[idx], c[idx]
            p = 5000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_pool():
    # all nonzero {-1,0,1}^9
    N = 3 ** 9
    powers = 3 ** np.arange(9)
    idx = np.arange(N)
    # (N,9) base-3 digits
    D = ((idx[:, None] // powers[None, :]) % 3) - 1  # -1,0,1
    D = D.astype(float)
    # remove zero row
    mask = np.any(D != 0, axis=1)
    D = D[mask]
    nrm = np.linalg.norm(D, axis=1, keepdims=True)
    P1 = D / nrm
    # subsampled {-2,-1,0,1,2}^9
    rng = np.random.default_rng(12345)
    M2 = 5000
    R = rng.integers(-2, 3, size=(M2, 9)).astype(float)
    z = np.all(R == 0, axis=1)
    R[z, 0] = 1.0
    n2 = np.linalg.norm(R, axis=1, keepdims=True)
    P2 = R / np.maximum(n2, 1e-12)
    P = np.vstack([P1, P2])
    return P

def greedy_select(pool, start):
    M = pool.shape[0]
    s0 = pool[start]
    best_dots = pool @ s0
    selected = [int(start)]
    in_sel = np.zeros(M, dtype=bool)
    in_sel[start] = True
    best_dots[start] = np.inf
    # incremental greedy minimizing running max dot
    for _ in range(1, 131):
        idx = int(np.argmin(best_dots))
        selected.append(idx)
        in_sel[idx] = True
        d = pool @ pool[idx]
        # max update
        # use out= to avoid alloc? simple:
        np.maximum(best_dots, d, out=best_dots)
        best_dots[in_sel] = np.inf
    return np.array(selected, dtype=int)

def swap_improve(sel_idx, pool, time_limit=1.5):
    t0 = time.time()
    sel_idx = sel_idx.copy()
    n = len(sel_idx)
    S = pool[sel_idx]
    in_pool = np.zeros(pool.shape[0], dtype=bool)
    in_pool[sel_idx] = True
    def cur_max_G(G):
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    G = S @ S.T
    Mcur = cur_max_G(G.copy())
    # count near-max pairs
    for _ in range(40):
        if time.time() - t0 > time_limit:
            break
        G = S @ S.T
        # per-i max without i
        M_without = np.empty(n)
        for i in range(n):
            # mask row/col i
            row = G[i].copy(); col = G[:, i].copy()
            # temporarily set to -2
            G[i, :] = -2.0; G[:, i] = -2.0
            M_without[i] = float(G.max())
            G[i, :] = row; G[:, i] = col
        np.fill_diagonal(G, -2.0)
        Mcur = float(G.max())
        cnt_old = int(np.sum(G[np.triu_indices(n, 1)] >= Mcur - 1e-9))
        D = pool @ S.T  # Mx n
        # top-2 per row
        A1 = np.argmax(D, axis=1)
        R1 = D[np.arange(D.shape[0]), A1]
        # second max: mask argmax
        # avoid full copy: compute via where
        # create masked version on the fly in chunks to save memory
        # simple: set those entries to -2 in a copy (3M floats ~24MB ok)
        D2 = D.copy()
        D2[np.arange(D.shape[0]), A1] = -2.0
        R2 = D2.max(axis=1)
        best_gain = 0
        best_swap = None
        # scan each i for best p
        # precompute for speed: for each i, cross = where(A1!=i, R1, R2)
        for i in range(n):
            cross = np.where(A1 != i, R1, R2)
            cross[in_pool] = np.inf
            # candidate max
            # find p minimizing cross
            p = int(np.argmin(cross))
            cmin = float(cross[p])
            cand = cmin if cmin > M_without[i] else M_without[i]
            if cand < Mcur - 1e-12:
                gain = Mcur - cand
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, p, cand)
            elif abs(cand - Mcur) < 1e-12:
                # count reduction?
                # new cross count
                cc = int(np.sum(D[p] >= Mcur - 1e-9)) - (1 if D[p, i] >= Mcur - 1e-9 else 0)
                # old pairs not involving i at max
                # count old pairs with max excluding i
                # approximate: cnt_old - pairs involving i at max
                inv = int(np.sum(G[i] >= Mcur - 1e-9))
                cnt_wo = cnt_old - inv
                cnt_new = cnt_wo + cc
                if cnt_new < cnt_old - 1e-9:
                    # tie-break: prefer larger reduction
                    gain = (cnt_old - cnt_new) * 1e-13
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, p, cand)
        if best_swap is None:
            break
        i, p, _ = best_swap
        # apply swap
        in_pool[sel_idx[i]] = False
        sel_idx[i] = p
        in_pool[p] = True
        S = pool[sel_idx]
    return sel_idx, S

def analyze(X, tol=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G >= m - tol
    deg = C.sum(axis=1)
    iu = np.triu_indices(X.shape[0], 1)
    sel = (G[iu] >= m - tol)
    edges = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    ratt = int(np.sum(deg == 0))
    return m, deg, edges, ratt

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    incumbent_edges = None
    try:
        _, _, incumbent_edges, _ = analyze(best, tol=1e-7)
    except Exception:
        incumbent_edges = set()

    # ---- structured discrete seeds ----
    try:
        pool = build_pool()
    except Exception:
        pool = None
    discrete_polished = []
    archive = []
    if pool is not None:
        M = pool.shape[0]
        starts = [0, M // 4, M // 2, (3 * M) // 4]
        # ensure starts point to ternary part diverse: also force specific types if found
        cands = []
        for si, st in enumerate(starts):
            try:
                sel_idx = greedy_select(pool, int(st) % M)
                sel_idx2, Smat = swap_improve(sel_idx, pool, time_limit=1.2)
                cands.append(Smat.copy())
            except Exception:
                continue
            if time.time() > t0 + 25:
                break
        # rotated copies via random orthogonal
        rot_cands = []
        for Smat in list(cands):
            try:
                A = rng.standard_normal((d, d))
                Q, _ = np.linalg.qr(A)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                rot_cands.append((Smat @ Q.T).copy())
            except Exception:
                pass
        all_seeds = cands + rot_cands  # up to 8
        # polish each seed briefly (40-60s total budget, cap at ~50s from t0)
        seed_deadline = min(t0 + 55, HARD_END - 90)
        for k, Sd in enumerate(all_seeds):
            if time.time() > seed_deadline:
                break
            X = Sd.copy()
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # varied hyperparams deterministically
            lr0 = 0.014 + 0.004 * (k % 4)
            lr1 = 0.0004 + 0.0001 * (k % 3)
            m0 = 0.03 + 0.005 * (k % 2)
            m1 = 0.002 + 0.0005 * (k % 2)
            try:
                X = fast_polish(X, 2200, lr0, lr1, m0, m1)
                X = fast_polish(X, 400, 0.0006, 0.00015, 0.002, 0.001)
            except Exception:
                pass
            v = max_cos(X)
            discrete_polished.append((v, X.copy()))
            # diversity retention: no zero-contact + edge overlap <70%
            try:
                mv, deg, edges, ratt = analyze(X, tol=1e-4)
                ov = 1.0
                if incumbent_edges is not None and len(incumbent_edges) > 0:
                    inter = len(edges.intersection(incumbent_edges))
                    ov = inter / max(1, len(incumbent_edges))
                if ratt == 0 and ov < 0.70:
                    archive.append((v, X.copy(), ov, ratt))
                elif ratt == 0 and v < best_val + 0.02 and ov < 0.70:
                    archive.append((v, X.copy(), ov, ratt))
            except Exception:
                pass
            # also light LP for top seeds if time permits
            if k < 2 and time.time() < seed_deadline - 4:
                try:
                    X2 = lp_polish_deadline(X, min(time.time() + 2.5, seed_deadline))
                    v2 = max_cos(X2)
                    discrete_polished.append((v2, X2.copy()))
                    if v2 < best_val:
                        best_val = v2
                        best = X2.copy()
                except Exception:
                    pass
        # update global best from discrete polished
        for v, X in discrete_polished:
            if v < best_val:
                best_val = v
                best = X.copy()
        # if discrete best is much worse, keep incumbent best (already)
        # refresh incumbent edges for overlap reference (keep original record basin ref)
        pass

    # ---- elitist basin-hopping from best (and diverse archive) ----
    # contact-degree probs favor hubs
    def cur_probs(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 1e-7)
        deg = C.sum(axis=1).astype(float)
        w = (deg + 1.0) ** 1.5
        return w / w.sum(), deg

    try:
        probs, _ = cur_probs(best, best_val)
    except Exception:
        probs = np.full(n, 1.0 / n)
    # pool of restart sources: best + up to 2 diverse archive members
    sources = [best.copy()]
    if len(archive) > 0:
        archive_sorted = sorted(archive, key=lambda t: t[0])[:2]
        for _, Xa, _, _ in archive_sorted:
            sources.append(Xa.copy())
    it = 0
    hop_end = HARD_END - 32
    while time.time() < hop_end:
        it += 1
        if it % 12 == 1:
            try:
                probs, _ = cur_probs(best, best_val)
            except Exception:
                pass
        # choose source: mostly best, occasionally diverse
        if len(sources) > 1 and (it % 9 == 8):
            Y = sources[it % len(sources)].copy() if it % 18 == 8 else best.copy()
            # if from archive, add moderate kick to escape
            Y += 0.015 * rng.standard_normal(Y.shape)
        else:
            Y = best.copy()
            cycle = it % 7
            if cycle == 0:
                Y += 0.008 * rng.standard_normal(Y.shape)
            elif cycle == 1:
                Y += 0.022 * rng.standard_normal(Y.shape)
            elif cycle == 2:
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.10 * rng.standard_normal((1, d))
            elif cycle == 3:
                idx = rng.choice(n, 2, replace=False, p=probs)
                Y[idx] += 0.10 * rng.standard_normal((2, d))
            elif cycle == 4:
                idx = rng.choice(n, 3, replace=False, p=probs)
                Y[idx] += 0.16 * rng.standard_normal((3, d))
            elif cycle == 5:
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.35 * rng.standard_normal((1, d))
            else:
                Y += 0.012 * rng.standard_normal(Y.shape)
                idx = rng.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.12 * rng.standard_normal((1, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        steps = int(rng.integers(1800, 2600))
        try:
            Y = fast_polish(Y, steps, lr0, lr1, m0, m1)
            Y = fast_polish(Y, 350, 0.0006, 0.00015, 0.002, 0.001)
        except Exception:
            continue
        try:
            v = max_cos(Y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = Y
            best_val = v
            try:
                probs, _ = cur_probs(best, best_val)
            except Exception:
                pass
        if time.time() > hop_end:
            break

    remain = HARD_END - time.time()
    if remain > 6:
        try:
            best = lp_polish_deadline(best, time.time() + min(remain - 2, 28))
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
