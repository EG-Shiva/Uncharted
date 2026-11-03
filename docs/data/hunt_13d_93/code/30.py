import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _worst_pair(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    return k // n, k % n, float(G.max())

def _cluster_around(X, a, b, L):
    n = X.shape[0]
    G = X @ X.T
    # neighbours of a and b (excluding self and each other handled by union)
    ra = G[a].copy(); ra[a] = -2
    rb = G[b].copy(); rb[b] = -2
    na = np.argsort(-ra)[:L]
    nb = np.argsort(-rb)[:L]
    s = set([int(a), int(b)])
    for v in na: s.add(int(v))
    for v in nb: s.add(int(v))
    return sorted(s)

def _top_set(X, top=100):
    n = X.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0, iu1]
    if top >= len(c):
        idx = np.arange(len(c))
    else:
        idx = np.argpartition(-c, top)[:top]
    s = set()
    for k in idx:
        i = int(iu0[k]); j = int(iu1[k])
        if i > j: i, j = j, i
        s.add((i, j))
    return s

def _jaccard(A, B):
    if not A and not B: return 1.0
    inter = len(A & B)
    union = len(A | B)
    return inter / max(1, union)

def _refine_hole(v, Y, steps=200, tau=0.006, lr=0.025):
    v = v / max(1e-12, np.linalg.norm(v))
    for _ in range(steps):
        dots = Y @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Y
        gv = float(g @ v)
        gt = g - gv * v
        ng = float(np.linalg.norm(gt))
        if ng < 1e-12 or not np.isfinite(ng):
            break
        gt /= ng
        v = v - lr * gt
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            break
        v /= nv
    return v

def _anneal_subset(X, idx, steps, T0, T1, lr0, lr1, mu=0.85):
    # optimize only rows idx, rest frozen
    n, d = X.shape
    V = np.zeros((len(idx), d))
    idx = np.asarray(idx, dtype=int)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T0 != T1 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        if T < 1e-12: T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        gsub = gt[idx]
        V = mu * V + (1.0 - mu) * gsub
        # re-project velocity
        dotv = np.einsum('ij,ij->i', V, X[idx])[:, None]
        Vt = V - dotv * X[idx]
        X[idx] = X[idx] - lr * Vt
        # renormalize subset
        nr = np.linalg.norm(X[idx], axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X[idx] = X[idx] / nr
    return X

def _anneal_full(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T0 != T1 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        if T < 1e-12: T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        grad = W @ X
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        V = mu * V + (1.0 - mu) * gt
        dotv = np.einsum('ij,ij->i', V, X)[:, None]
        Vt = V - dotv * X
        X = X - lr * Vt
        X = _norm(X)
    return X

def _hard_topK(X, iters, step0, K):
    best = _max_cos(X)
    step = step0
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m = len(iu0)
    K = int(min(K, m))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < m:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(m)
        I = iu0[idx]; J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = _norm(X - step * gt)
        vv = _max_cos(Xt)
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.3, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best_X = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (93, 13):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best_X)
    # pool: keep best + up to 5 diverse
    pool_X = [best_X.copy()]
    pool_v = [best_val]
    # add second distinct seed if available
    try:
        r = np.load("record.npy")
        r = _norm(np.asarray(r, dtype=np.float64))
        if r.shape == (93, 13):
            rv = _max_cos(r)
            if abs(rv - best_val) > 1e-12:
                pool_X.append(r.copy()); pool_v.append(rv)
    except Exception:
        pass
    best_top = _top_set(best_X, 100)
    trial = 0
    k_choices = [3, 5, 8, 12]
    while True:
        if time.time() - t0 > BUDGET - 12:
            break
        trial += 1
        # pick seed: 70% best, else random pool
        if len(pool_X) == 1 or rng.random() < 0.7:
            seed_idx = int(np.argmin(np.array(pool_v)))
        else:
            seed_idx = int(rng.integers(0, len(pool_X)))
        Xs = pool_X[seed_idx].copy()
        # bottleneck + cluster
        try:
            a, b, _ = _worst_pair(Xs)
        except Exception:
            continue
        L = int(10 + rng.integers(0, 6))  # 10..15
        C = _cluster_around(Xs, a, b, L)
        k = k_choices[int(rng.integers(0, len(k_choices)))]
        k = min(k, len(C))
        if k < 2:
            continue
        mode = int(rng.integers(0, 4))  # 0:a,1:b,2:neither,3:random
        Cset = set(C)
        del_idx = None
        if mode == 0:
            rest = [c for c in C if c != a]
            if len(rest) < k - 1:
                continue
            pick = rng.choice(np.array(rest), k - 1, replace=False) if k - 1 > 0 else np.array([], dtype=int)
            del_idx = np.array([a] + [int(x) for x in pick], dtype=int)
        elif mode == 1:
            rest = [c for c in C if c != b]
            if len(rest) < k - 1:
                continue
            pick = rng.choice(np.array(rest), k - 1, replace=False) if k - 1 > 0 else np.array([], dtype=int)
            del_idx = np.array([b] + [int(x) for x in pick], dtype=int)
        elif mode == 2:
            rest = [c for c in C if c != a and c != b]
            if len(rest) < k:
                # fall back to random from C
                del_idx = np.array(rng.choice(np.array(C), k, replace=False), dtype=int)
            else:
                del_idx = np.array(rng.choice(np.array(rest), k, replace=False), dtype=int)
        else:
            del_idx = np.array(rng.choice(np.array(C), k, replace=False), dtype=int)
        del_idx = np.sort(del_idx)
        keep_mask = np.ones(n, dtype=bool); keep_mask[del_idx] = False
        Keep = Xs[keep_mask].copy()  # (n-k, d)
        # greedy reinsert
        cur_keep = Keep.copy()
        new_vecs = []
        ok = True
        for _ in range(k):
            if time.time() - t0 > BUDGET - 10:
                ok = False; break
            Cand = _norm(rng.standard_normal((50000, d)))
            D = Cand @ cur_keep.T
            worst = D.max(axis=1)
            i0 = int(np.argmin(worst))
            v0 = Cand[i0].copy()
            v = _refine_hole(v0, cur_keep, steps=200, tau=0.006, lr=0.025)
            new_vecs.append(v)
            cur_keep = np.vstack([cur_keep, v.reshape(1, -1)])
        if not ok:
            break
        Xc = Xs.copy()
        for t_, idx_ in enumerate(del_idx):
            Xc[int(idx_)] = new_vecs[t_]
        Xc = _norm(Xc)
        # perturb reinserted cluster by 1e-3..1e-2
        sigma = float(10.0 ** rng.uniform(-3.0, -2.0))
        Xc[del_idx] = _norm(Xc[del_idx] + sigma * rng.standard_normal((k, d)))
        # optimize cluster with remainder frozen
        try:
            Xc = _anneal_subset(Xc, del_idx, steps=350, T0=8e-4, T1=8e-5, lr0=0.02, lr1=0.004, mu=0.85)
        except Exception:
            pass
        # unfreeze: short annealed softmax polish + hard topK
        try:
            # scale polish by k to save time
            if k <= 5:
                Xc = _anneal_full(Xc, steps=450, T0=3e-4, T1=3e-5, lr0=0.012, lr1=0.002, mu=0.85)
                Xc, vc = _hard_topK(Xc, iters=350, step0=0.0015, K=250)
                Xc, vc = _hard_topK(Xc, iters=250, step0=0.0006, K=120)
            else:
                Xc = _anneal_full(Xc, steps=600, T0=4e-4, T1=3e-5, lr0=0.014, lr1=0.002, mu=0.85)
                Xc, vc = _hard_topK(Xc, iters=400, step0=0.0015, K=300)
                Xc, vc = _hard_topK(Xc, iters=250, step0=0.0006, K=150)
        except Exception:
            continue
        vc = _max_cos(Xc)
        # acceptance: improve or diverse top-100
        try:
            top_c = _top_set(Xc, 100)
            jac = _jaccard(best_top, top_c)
        except Exception:
            jac = 0.0
        if vc < best_val - 1e-12:
            best_val = vc; best_X = Xc.copy()
            best_top = _top_set(best_X, 100)
            pool_X.append(Xc.copy()); pool_v.append(vc)
            # keep pool bounded: retain best 6 by value, but ensure diversity? just keep best 6
            if len(pool_X) > 6:
                order = np.argsort(np.array(pool_v))
                pool_X = [pool_X[i] for i in order[:6]]
                pool_v = [pool_v[i] for i in order[:6]]
        elif jac < 0.85:
            # valuable different topology even if slightly worse
            # only keep if not too bad (within 0.01) to avoid polluting pool
            if vc < best_val + 0.008:
                pool_X.append(Xc.copy()); pool_v.append(vc)
                if len(pool_X) > 6:
                    # drop worst non-best entry
                    bi = int(np.argmin(np.array(pool_v)))
                    # remove worst that is not best
                    vals = np.array(pool_v)
                    wi = int(np.argmax(vals))
                    if wi != bi:
                        del pool_X[wi]; del pool_v[wi]
                    else:
                        # worst is best? drop second worst
                        order = np.argsort(vals)
                        drop = order[-1]
                        del pool_X[drop]; del pool_v[drop]
        # else discard (same plateau copy)
    best_X = _norm(best_X)
    return best_X
