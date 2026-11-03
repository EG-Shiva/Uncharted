import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_lse(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - np.eye(n) * 2.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
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

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
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

def _greedy_delete_to(X, target):
    X = X.copy()
    while X.shape[0] > target:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        idx = int(np.argmax(rm))
        X = np.delete(X, idx, axis=0)
    return X

def _best_insertion(X, rng, n_samp=20000):
    n, d = X.shape
    best_c = None
    best_s = 1e9
    # chunked to bound memory
    chunk = 5000
    done = 0
    cand_best = None
    # generate in chunks and keep global best
    all_best_val = 1e9
    all_best_vec = None
    while done < n_samp:
        m = min(chunk, n_samp - done)
        C = rng.standard_normal((m, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = (C @ X.T).max(axis=1)
        j = int(np.argmin(S))
        if float(S[j]) < all_best_val:
            all_best_val = float(S[j])
            all_best_vec = C[j].copy()
        done += m
    return all_best_vec

def solve():
    T0 = time.time()
    BUDGET = 172.0
    HOP_END = T0 + 120.0
    LP_END = T0 + BUDGET - 1.5
    rng_master = np.random.default_rng(7)
    # load incumbent
    incumb = None
    incumb_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < incumb_val:
                    incumb_val = v
                    incumb = a.copy()
        except Exception:
            pass
    if incumb is None:
        incumb = rng_master.standard_normal((94, 13))
        incumb /= np.linalg.norm(incumb, axis=1, keepdims=True)
        incumb_val = _max_cos(incumb)
    try:
        rec = np.load("record.npy").astype(float)
        rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
    except Exception:
        rec = incumb.copy()

    cands = []
    # helper to push candidate
    def push(x):
        try:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            if x.shape == (94, 13) and np.all(np.isfinite(x)):
                cands.append(x.copy())
        except Exception:
            pass

    # ---- Chain 1: record -> 88 -> grow ----
    try:
        if time.time() < HOP_END:
            rng = np.random.default_rng(100)
            X88 = _greedy_delete_to(rec, 88)
            # annealed LSE Adam beta 30->300 lr 0.01->0.001
            X88 = _adam_lse(X88, steps=900, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            # grow one at a time to 94
            X = X88
            while X.shape[0] < 94 and time.time() < HOP_END:
                v = _best_insertion(X, rng, n_samp=20000)
                X = np.vstack([X, v.reshape(1, -1)])
                X = _adam_lse(X, steps=300, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            if X.shape[0] == 94:
                X = _adam_lse(X, steps=700, k0=80.0, k1=500.0, lr0=0.008, lr1=0.001)
                push(X)
    except Exception:
        pass

    # ---- Chain 2: random 88 -> grow ----
    try:
        if time.time() < HOP_END:
            rng = np.random.default_rng(201)
            X88 = rng.standard_normal((88, 13))
            X88 /= np.linalg.norm(X88, axis=1, keepdims=True)
            X88 = _adam_lse(X88, steps=1000, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            X = X88
            while X.shape[0] < 94 and time.time() < HOP_END:
                v = _best_insertion(X, rng, n_samp=20000)
                X = np.vstack([X, v.reshape(1, -1)])
                X = _adam_lse(X, steps=300, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            if X.shape[0] == 94:
                X = _adam_lse(X, steps=700, k0=80.0, k1=500.0, lr0=0.008, lr1=0.001)
                push(X)
    except Exception:
        pass

    # ---- Chain 3: record/best -> 90/92 staged ----
    try:
        if time.time() < HOP_END:
            rng = np.random.default_rng(302)
            # start from incumb subsampled to 90 with different deletion (random tie break)
            base = incumb.copy()
            # random deletion of 4 to reach 90
            X90 = _greedy_delete_to(base, 90)
            X90 = _adam_lse(X90, steps=700, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            X = X90
            # go 90->92->93->94 with polish after each insertion
            while X.shape[0] < 94 and time.time() < HOP_END:
                v = _best_insertion(X, rng, n_samp=20000)
                X = np.vstack([X, v.reshape(1, -1)])
                X = _adam_lse(X, steps=300, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
                # short hinge-like sharpen
                X = _adam_lse(X, steps=150, k0=150.0, k1=400.0, lr0=0.004, lr1=0.001)
            if X.shape[0] == 94:
                X = _adam_lse(X, steps=600, k0=100.0, k1=600.0, lr0=0.006, lr1=0.0008)
                push(X)
    except Exception:
        pass

    # ---- Chain 4: N=96 squeezed then delete to 94 ----
    try:
        if time.time() < HOP_END:
            rng = np.random.default_rng(403)
            X96 = rng.standard_normal((96, 13))
            X96 /= np.linalg.norm(X96, axis=1, keepdims=True)
            X96 = _adam_lse(X96, steps=900, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            # delete worst by summed positive contact excess, twice
            X = X96
            for target in [95, 94]:
                if X.shape[0] <= target:
                    continue
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(X.shape[0], 1)
                med = float(np.median(G[iu]))
                # summed positive excess over median
                E = np.maximum(0.0, G - med)
                # don't count diagonal (already -2 -> 0)
                score = E.sum(axis=1)
                idx = int(np.argmax(score))
                X = np.delete(X, idx, axis=0)
                X = _adam_lse(X, steps=350, k0=30.0, k1=300.0, lr0=0.01, lr1=0.001)
            if X.shape[0] == 94 and time.time() < HOP_END:
                X = _adam_lse(X, steps=600, k0=100.0, k1=600.0, lr0=0.006, lr1=0.0008)
                push(X)
    except Exception:
        pass

    # always include incumbent
    push(incumb)
    if len(cands) == 0:
        return incumb

    # evaluate and compare basins by sorted Gram spectra
    vals = []
    for x in cands:
        try:
            vals.append(_max_cos(x))
        except Exception:
            vals.append(1e9)
    order = np.argsort(np.array(vals))
    cands = [cands[i] for i in order]
    vals = [vals[i] for i in order]

    # pick best two distinct outcomes (Gram spectrum distance)
    def gram_sorted(x):
        G = x @ x.T
        iu = np.triu_indices(x.shape[0], 1)
        return np.sort(G[iu])

    best_two_idx = [0]
    try:
        g0 = gram_sorted(cands[0])
        best_dist = -1
        best_j = -1
        for j in range(1, len(cands)):
            gj = gram_sorted(cands[j])
            d = float(np.linalg.norm(g0 - gj))
            if d > 1e-6 and (best_j < 0 or vals[j] < vals[best_j] - 1e-12 or d > best_dist):
                # prefer low val among distinct
                pass
        # simply pick the best j with distance > 1e-4, lowest val
        for j in range(1, len(cands)):
            gj = gram_sorted(cands[j])
            d = float(np.linalg.norm(g0 - gj))
            if d > 1e-4:
                best_j = j
                break
        if best_j >= 0:
            best_two_idx.append(best_j)
    except Exception:
        pass

    # LP-polish best two distinct outcomes within remaining budget
    polished = []
    polished_vals = []
    # split remaining time
    t_rem = max(1.0, LP_END - time.time())
    per = t_rem / max(1, len(best_two_idx))
    for k, idx in enumerate(best_two_idx):
        try:
            dl = min(LP_END, time.time() + per)
            # pre-sharpen with Adam before LP
            x = cands[idx].copy()
            if time.time() < dl - 1.0:
                # quick sharpen, limited steps if little time
                x = _adam_lse(x, steps=300, k0=150.0, k1=700.0, lr0=0.004, lr1=0.0008)
            x2, v2 = _lp_polish_deadline(x, dl)
            polished.append(x2)
            polished_vals.append(v2)
        except Exception:
            polished.append(cands[idx])
            polished_vals.append(vals[idx])
    # also keep unpolished others as fallback
    for i, x in enumerate(cands):
        if i not in best_two_idx:
            polished.append(x)
            polished_vals.append(vals[i])
    # include incumbent safety
    polished.append(incumb)
    polished_vals.append(incumb_val)
    bi = int(np.argmin(np.array(polished_vals)))
    best_out = polished[bi] / np.linalg.norm(polished[bi], axis=1, keepdims=True)
    # verify: if not better than incumb, return incumb (still valid)
    try:
        bv = _max_cos(best_out)
        if bv <= incumb_val and np.all(np.isfinite(best_out)):
            return best_out
        else:
            # return whichever is lower
            if bv < incumb_val:
                return best_out
            return incumb
    except Exception:
        return incumb
