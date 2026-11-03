import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _adam_quench(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b_1 = 0.9
    b_2 = 0.999
    eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        step = np.where(over, step * (0.05 / (sn + 1e-12)), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _lp_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return X


def _contact_hash(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    gap = 0.004
    deg = tuple(sorted((G > m - gap).sum(axis=1).tolist()))
    return (deg, int(round(m * 3000)))


def _sparse_seed(rng, n, d):
    X = np.zeros((n, d))
    for i in range(n):
        while True:
            v = rng.integers(-2, 3, size=d).astype(float)
            mask = rng.random(d) < 0.45
            v[mask] = 0.0
            # ensure not too sparse
            if np.count_nonzero(v) < 4:
                continue
            nr = float(np.linalg.norm(v))
            if nr < 1e-9:
                continue
            X[i] = v / nr
            break
    return X


def _greedy_seed(rng, n, d, pool=900):
    C = rng.standard_normal((pool, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    avail = np.ones(pool, dtype=bool)
    start = int(rng.integers(0, pool))
    sel = [start]
    avail[start] = False
    S = C[start:start + 1]
    for _ in range(1, n):
        idx = np.where(avail)[0]
        D = C[idx] @ S.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        pick = int(idx[bi])
        sel.append(pick)
        avail[pick] = False
        S = C[np.array(sel)]
    return C[np.array(sel)]


def _align_procrustes(A, B):
    n, d = A.shape
    Q = np.eye(d)
    perm = np.arange(n)
    for _ in range(3):
        Bq = B @ Q
        D = A @ Bq.T
        r, c = linear_sum_assignment(-D)
        p = np.empty(n, dtype=int)
        p[r] = c
        perm = p
        M = B[perm].T @ A
        try:
            U, _, Vt = np.linalg.svd(M)
            Q = U @ Vt
        except Exception:
            break
    return Q, perm


def _hole_replace(X, victim, rng, n_cand=2000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2


def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    T_SEED_END = t0 + 55.0
    T_RECOMB_END = t0 + 130.0
    rng = np.random.default_rng(20261102)
    n, d = 58, 22

    best_overall = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best_overall = v.copy()
        except Exception:
            continue
    if best_overall is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_overall = z
        best_val = _max_cos(z)

    archive = []  # list of (score, X)
    seen = set()

    def try_archive(X):
        nonlocal best_overall, best_val
        try:
            s = _max_cos(X)
        except Exception:
            return False
        h = _contact_hash(X)
        if h in seen:
            return False
        seen.add(h)
        if s < best_val - 1e-12:
            best_val = s
            best_overall = X.copy()
        # keep archive sorted, cap 16
        archive.append((s, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[16:]
        return True

    # ---- Phase 1: seeding 60s ----
    fam = 0
    while time.time() < T_SEED_END and time.time() < T_END - 30:
        fam += 1
        kind = fam % 3
        try:
            if kind == 0:
                X = rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            elif kind == 1:
                X = _sparse_seed(rng, n, d)
            else:
                X = _greedy_seed(rng, n, d, pool=900)
            # fast quench, no LP
            X = _adam_quench(X, 350, 60, 600, 0.006, 0.0008)
            X = _adam_quench(X, 200, 800, 2500, 0.0012, 0.0003)
            try_archive(X)
        except Exception:
            continue
        if len(archive) >= 16 and fam > 40:
            # keep generating to fill time with diversity
            pass

    # if archive too small, add random quenches quickly
    while len(archive) < 4 and time.time() < T_RECOMB_END:
        try:
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = _adam_quench(X, 300, 60, 600, 0.006, 0.0008)
            try_archive(X)
        except Exception:
            break

    # ---- Phase 2: recombination 70s ----
    it = 0
    while time.time() < T_RECOMB_END and time.time() < T_END - 28:
        it += 1
        if len(archive) < 2:
            break
        try:
            a = int(rng.integers(0, len(archive)))
            b = int(rng.integers(0, len(archive)))
            if a == b:
                continue
            A = archive[a][1]
            B = archive[b][1]
            Q, perm = _align_procrustes(A, B)
            Bq = B @ Q
            # pair-wise crossover: for each matched pair pick one side
            takeA = np.zeros(n, dtype=bool)
            idx = rng.choice(n, size=n // 2, replace=False)
            takeA[idx] = True
            # ensure exactly 29 each
            child = np.empty_like(A)
            Bp = Bq[perm]
            for i in range(n):
                if takeA[i]:
                    child[i] = A[i]
                else:
                    child[i] = Bp[i]
            child /= np.linalg.norm(child, axis=1, keepdims=True)
            # repair collisions: worst 2 points -> hole reinsert
            for _ in range(2):
                G = child @ child.T
                np.fill_diagonal(G, -2.0)
                m0 = float(G.max())
                # threshold for collision: near current best
                if m0 < best_val + 0.002:
                    break
                k = int(np.argmax(G))
                i0, j0 = k // n, k % n
                # victim = more crowded
                si = float(G[i0].sum())
                sj = float(G[j0].sum())
                victim = int(i0) if si > sj else int(j0)
                child = _hole_replace(child, victim, rng, n_cand=1500)
            # short quench
            child = _adam_quench(child, 300, 70, 700, 0.005, 0.0006)
            child = _adam_quench(child, 150, 900, 2500, 0.001, 0.0003)
            s_pre = _max_cos(child)
            # LP only if competitive
            med = archive[len(archive) // 2][0] if archive else best_val
            thresh = min(med, best_val + 0.003)
            if s_pre < thresh and time.time() < T_RECOMB_END - 3:
                try:
                    child2 = _lp_polish(child, 0.8, T_RECOMB_END)
                    s2 = _max_cos(child2)
                    if s2 < s_pre:
                        child = child2
                        s_pre = s2
                except Exception:
                    pass
            try_archive(child)
        except Exception:
            continue
        # occasional fresh seed to keep diversity
        if it % 12 == 0 and time.time() < T_RECOMB_END - 5:
            try:
                X = rng.standard_normal((n, d))
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _adam_quench(X, 300, 60, 600, 0.006, 0.0008)
                try_archive(X)
            except Exception:
                pass

    # ---- Phase 3: final LP on finalists ----
    try:
        cands = []
        cands.append((best_val, best_overall.copy()))
        for s, X in archive[:3]:
            cands.append((s, X.copy()))
        # dedup roughly by score closeness is fine, just polish best 3 unique
        cands.sort(key=lambda t: t[0])
        polished_best = best_overall
        polished_val = best_val
        remain = (T_END - time.time()) - 1.0
        if remain > 3:
            per = min(8.0, remain / min(3, len(cands)))
            for _, X in cands[:3]:
                if time.time() >= T_END - 2:
                    break
                try:
                    Y = _lp_polish(X, per, T_END - 1.0)
                    v = _max_cos(Y)
                    if v < polished_val:
                        polished_val = v
                        polished_best = Y.copy()
                except Exception:
                    continue
        best_overall = polished_best
        best_val = polished_val
    except Exception:
        pass

    best_overall = best_overall / np.linalg.norm(best_overall, axis=1, keepdims=True)
    best_overall = np.nan_to_num(best_overall, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_overall, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_overall = best_overall / nr
    return best_overall
