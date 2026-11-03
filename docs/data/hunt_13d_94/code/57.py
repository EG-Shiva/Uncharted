import time
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_lse(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = float(G.max())
        W = np.exp(k * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _adam_quantile(x, steps, M, lr0, lr1, iu0, iu1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        vals = G[iu0, iu1]
        if M >= vals.size:
            t = float(vals.min())
        else:
            t = float(np.partition(vals, -M)[-M])
        Mm = G - t
        Mm[Mm < 0] = 0.0
        np.fill_diagonal(Mm, 0.0)
        cnt = float((Mm > 0).sum() * 0.5)
        if cnt < 1.0:
            cnt = 1.0
        Grad = (Mm @ x) / cnt
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _align_procrustes(A, B, iters=4):
    # A,B: (n,d) unit rows. Returns perm + R such that (B[perm] @ R) ~ A
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(iters):
        BR = B @ R
        C = A @ BR.T  # similarity
        _, col = linear_sum_assignment(-C)
        # row i matched to col? linear_sum_assignment returns row_ind,col_ind
        # we want perm such that B_perm[i] matches A[i]: need inverse
        # C[i,j]: A[i] vs BR[j]. Assignment pairs (r_i,c_i). Build perm:
        # Use full assignment: row_ind sorted
        r, c = linear_sum_assignment(-C)
        # r is 0..n-1 sorted, c gives match for each A row
        perm = c  # B index matched to A[i]
        Bp = B[perm]
        M = Bp.T @ A
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
        except Exception:
            break
    # final perm recompute with final R
    BR = B @ R
    C = A @ BR.T
    r, c = linear_sum_assignment(-C)
    perm = c
    Bp = B[perm]
    M = Bp.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
    except Exception:
        pass
    return perm, R

def _slerp_children(A, Bperm_R, alpha, rng, jitter=0.03):
    dot = (A * Bperm_R).sum(axis=1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    om = np.arccos(dot)  # (n,1)
    so = np.sin(om)
    # fallback for tiny angles
    small = (so[:, 0] < 1e-12)
    w1 = np.sin((1 - alpha) * om) / np.maximum(so, 1e-12)
    w2 = np.sin(alpha * om) / np.maximum(so, 1e-12)
    C = w1 * A + w2 * Bperm_R
    C[small] = (1 - alpha) * A[small] + alpha * Bperm_R[small]
    if jitter > 0:
        N = rng.standard_normal(size=C.shape)
        dotn = (N * C).sum(axis=1, keepdims=True)
        N = N - dotn * C
        C = C + jitter * N
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    T0 = time.time()
    END = T0 + 165.0
    HARD = T0 + 175.0
    rng = np.random.default_rng(0)
    # load incumbent
    inc = None; inc_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < inc_val:
                    inc_val = v; inc = a.copy()
        except Exception:
            pass
    if inc is None:
        inc = rng.standard_normal(size=(94, 13))
        inc /= np.linalg.norm(inc, axis=1, keepdims=True)
        inc_val = _max_cos(inc)
    n, d = 94, 13
    iu0, iu1 = np.triu_indices(n, 1)
    global_best = inc.copy(); global_val = inc_val

    # --- pool of diverse parents: jittered + 2 random, short annealed Adam ---
    sigmas = [0.15, 0.3, 0.5, 0.8]
    n_jit = 10
    pool = []
    # include incumbent itself polished slightly? keep raw as candidate
    pool.append((global_val, inc.copy()))
    for i in range(n_jit):
        sig = sigmas[i % len(sigmas)]
        y = inc + sig * rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # short annealed LogSumExp Adam ~400 steps k 30->300
        y = _adam_lse(y, steps=400, k0=30, k1=300, lr0=0.02, lr1=0.004)
        v = _max_cos(y)
        pool.append((v, y))
        if v < global_val:
            global_val = v; global_best = y.copy()
        if time.time() > END - 100:
            break
    for _ in range(2):
        if time.time() > END - 95:
            break
        y = rng.standard_normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _adam_lse(y, steps=400, k0=30, k1=300, lr0=0.025, lr1=0.005)
        v = _max_cos(y)
        pool.append((v, y))
        if v < global_val:
            global_val = v; global_best = y.copy()

    # retain best few, but keep diversity: sort and keep top ~8 + ensure incumbent kept
    pool.sort(key=lambda t: t[0])
    # elite archive size
    NES = min(8, len(pool))
    elites = [p[1].copy() for p in pool[:NES]]
    elite_vals = [p[0] for p in pool[:NES]]
    # never discard original incumbent: ensure it is in elites
    found = any(_max_cos(e) <= inc_val + 1e-12 for e in elites)
    if not found:
        elites[-1] = inc.copy(); elite_vals[-1] = inc_val
    # order elites by val
    order = np.argsort(elite_vals)
    elites = [elites[i] for i in order]
    elite_vals = [elite_vals[i] for i in order]
    if elite_vals[0] < global_val:
        global_val = elite_vals[0]; global_best = elites[0].copy()

    alphas = [0.3, 0.5, 0.7]
    pair_list = []
    for i in range(len(elites)):
        for j in range(i + 1, len(elites)):
            pair_list.append((i, j))
    rng.shuffle(pair_list)
    pi = 0
    it = 0
    # --- recombination among elites until 165s ---
    while time.time() < END - 8.0:
        if time.time() > HARD - 5.0:
            break
        it += 1
        if len(pair_list) == 0:
            # rebuild pairs
            for i in range(len(elites)):
                for j in range(i + 1, len(elites)):
                    pair_list.append((i, j))
            rng.shuffle(pair_list)
        i, j = pair_list[pi % len(pair_list)]; pi += 1
        # refresh indices if elites changed size (keep fixed size)
        i = i % len(elites); j = j % len(elites)
        if i == j:
            continue
        A = elites[i]; B = elites[j]
        va = elite_vals[i]; vb = elite_vals[j]
        worse = max(va, vb)
        # alternate direction randomly
        if rng.random() < 0.5:
            A, B = B, A
        try:
            perm, R = _align_procrustes(A, B, iters=4)
        except Exception:
            continue
        BpR = B[perm] @ R
        alpha = alphas[rng.integers(0, len(alphas))]
        # occasionally try all three alphas in one go if time allows (first few iters)
        alpha_set = [alpha]
        if it <= 3:
            alpha_set = alphas
        for alpha in alpha_set:
            if time.time() > END - 8.0:
                break
            try:
                C = _slerp_children(A, BpR, alpha, rng, jitter=0.03)
            except Exception:
                continue
            # polish: top-600 quantile crush then short LSE
            try:
                C = _adam_quantile(C, steps=300, M=600, lr0=0.02, lr1=0.004, iu0=iu0, iu1=iu1)
                C = _adam_lse(C, steps=150, k0=60, k1=400, lr0=0.010, lr1=0.003)
            except Exception:
                continue
            cv = _max_cos(C)
            # LP polish only if promising (beats worse parent or close) and time allows
            if cv < worse + 0.002 and time.time() < END - 25.0:
                try:
                    # at most ~2-3s per child
                    dl = min(time.time() + 2.5, END - 8.0)
                    # cap 20s total concept: we limit per-child; overall bounded by loop deadline
                    C2, cv2 = _lp_polish_deadline(C, deadline=dl)
                    if cv2 < cv and np.all(np.isfinite(C2)):
                        C, cv = C2, cv2
                except Exception:
                    pass
            if cv < global_val - 1e-15:
                global_val = cv; global_best = C.copy()
            # keep elite archive: use child only if beats worse parent
            if cv < worse - 1e-15:
                # replace worst elite
                wi = int(np.argmax(elite_vals))
                elites[wi] = C.copy(); elite_vals[wi] = cv
                # rebuild pair list occasionally
                if it % 6 == 0:
                    pair_list = []
                    for a_ in range(len(elites)):
                        for b_ in range(len(elites)):
                            if a_ < b_:
                                pair_list.append((a_, b_))
                    rng.shuffle(pair_list); pi = 0
            if time.time() > HARD - 5.0:
                break

    # final safety: never discard original incumbent
    try:
        if _max_cos(global_best) > inc_val + 1e-12:
            global_best = inc.copy(); global_val = inc_val
    except Exception:
        pass
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
