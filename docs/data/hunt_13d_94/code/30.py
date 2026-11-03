import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish_deadline(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        sel_thresh = m - max(20 * radius, 1e-9)
        sel = c > sel_thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[sel]
        J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best


def _batch_reinsert(Frozen, k, B, steps, beta0, beta1, lr0, lr1, rng, d):
    Nf = Frozen.shape[0]
    F = rng.randn(B, k, d)
    F /= np.linalg.norm(F, axis=2, keepdims=True) + 1e-12
    m1 = np.zeros_like(F)
    v1 = np.zeros_like(F)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Eye = np.eye(k)
    OneMinusEye = 1.0 - Eye
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac if beta0 != beta1 else beta0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        C = np.einsum('bkd,nd->bkn', F, Frozen)
        D = np.einsum('bkd,bjd->bkj', F, F)
        Cmax = C.max(axis=(1, 2))
        Dminus = D - 2.0 * Eye[None, :, :]
        Dmax = Dminus.max(axis=(1, 2))
        mval = np.maximum(Cmax, Dmax)
        EC = np.exp(beta * (C - mval[:, None, None]))
        ED = np.exp(beta * (Dminus - mval[:, None, None])) * OneMinusEye[None, :, :]
        Z = EC.sum(axis=(1, 2)) + ED.sum(axis=(1, 2)) * 0.5 + 1e-30
        Gc = np.einsum('bkn,nd->bkd', EC, Frozen) / Z[:, None, None]
        Gd = np.einsum('bkj,bjd->bkd', ED, F) / Z[:, None, None]
        grad = Gc + Gd
        dot = (grad * F).sum(axis=2, keepdims=True)
        gt = grad - dot * F
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        F = F - upd
        F /= np.linalg.norm(F, axis=2, keepdims=True) + 1e-12
    return F


def _global_adam(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    Y = X.copy()
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac if beta0 != beta1 else beta0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        E = np.exp(beta * (G - mval))
        np.fill_diagonal(E, 0.0)
        Z = E.sum() * 0.5 + 1e-30
        grad = (E / Z) @ Y
        dot = (grad * Y).sum(axis=1, keepdims=True)
        gt = grad - dot * Y
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        Y = Y - upd
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    return Y


def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    EXPL_END = t0 + 152.0
    rng = np.random.RandomState(12345)
    n, d = 94, 13

    def load_norm(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (94, 13):
                return None
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                return None
            return a
        except Exception:
            return None

    xb = load_norm("best.npy")
    xr = load_norm("record.npy")
    cands = [a for a in [xb, xr] if a is not None]
    if not cands:
        a = rng.randn(n, d)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands = [a]
    X_best = cands[0].copy()
    best_val = _max_cos(X_best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            X_best = c.copy()

    B = 32
    attempt = 0
    k_cycle = [8, 12, 16, 12, 8, 16]

    while time.time() < EXPL_END:
        attempt += 1
        if time.time() > HARD_END - 6:
            break
        base = X_best
        Gfull = base @ base.T
        np.fill_diagonal(Gfull, -2.0)
        cmax = Gfull.max()
        contacts = np.count_nonzero(Gfull > cmax - 5e-4, axis=1)
        # seed with high contact degree, randomized among top
        order_hi = np.argsort(-contacts)
        top_pool = order_hi[:10]
        seed = int(top_pool[rng.randint(len(top_pool))])
        k = k_cycle[(attempt - 1) % len(k_cycle)]
        # cluster: seed + closest angular neighbours (largest dots)
        dots_to_seed = Gfull[seed].copy()
        dots_to_seed[seed] = 2.0
        # argsort descending excluding seed
        nn = np.argsort(-dots_to_seed)
        nn = nn[nn != seed][:k - 1]
        cluster = np.concatenate([[seed], nn])
        cluster = np.asarray(cluster, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[cluster] = False
        Frozen = base[mask]
        OrigCluster = base[cluster]
        frozen_max = _max_cos(Frozen) if len(Frozen) > 1 else -2.0

        # annealed batched reinsertion inside cavity
        steps = 900 if k <= 8 else (1000 if k <= 12 else 1100)
        # vary beta schedule slightly per attempt for diversity
        beta0 = 25.0
        beta1 = 350.0 if (attempt % 2 == 0) else 280.0
        lr0, lr1 = 0.03, 0.002
        Fbatch = _batch_reinsert(Frozen, k, B, steps, beta0, beta1, lr0, lr1, rng, d)

        # rank completions by exact max cosine
        vals = np.empty(B)
        cross_max = np.empty(B)
        intra_max = np.empty(B)
        for b in range(B):
            Yb = Fbatch[b]
            Cb = Yb @ Frozen.T
            Db = Yb @ Yb.T
            np.fill_diagonal(Db, -2.0)
            cm = float(Cb.max())
            dm = float(Db.max())
            cross_max[b] = cm
            intra_max[b] = dm
            vals[b] = max(cm, dm, frozen_max)
        order = np.argsort(vals)
        # distance from original cluster (chamfer): larger = farther
        dists = np.empty(B)
        for b in range(B):
            Dm = Fbatch[b] @ OrigCluster.T
            # for each orig point, best match in new set
            best_match = Dm.max(axis=0)
            dists[b] = 1.0 - float(best_match.mean())
        # select best 3-5 with diversity: greedy
        K = 4
        # candidate pool: top 12 by value
        pool = order[:12]
        selected = [int(pool[0])]
        # greedily add candidates that are good and far from selected
        while len(selected) < K and len(selected) < len(pool):
            best_score = 1e9
            best_c = None
            for c in pool:
                c = int(c)
                if c in selected:
                    continue
                # min distance to already selected (in solution space)
                dd = min(float(np.linalg.norm(Fbatch[c] - Fbatch[s])) for s in selected)
                # score: value + small penalty for closeness
                score = vals[c] - 0.002 * min(dd, 1.0)
                # also bonus for distance from original if values tie
                score -= 0.001 * min(dists[c], 0.5)
                if score < best_score:
                    best_score = score
                    best_c = c
            if best_c is None:
                break
            selected.append(best_c)
        # ensure one far candidate if all are close to original
        if max(dists[s] for s in selected) < 0.02:
            far = int(pool[np.argmax(dists[pool])])
            if far not in selected:
                selected[-1] = far

        # unfreeze and refine only best selected
        refined = []
        for s in selected:
            if time.time() > EXPL_END:
                break
            Xcand = np.empty_like(base)
            Xcand[mask] = Frozen
            Xcand[cluster] = Fbatch[s]
            # short global Adam unfreeze
            Xref = _global_adam(Xcand, 250, 120.0, 300.0, 0.008, 0.002)
            vref = _max_cos(Xref)
            refined.append((vref, Xref))
        if not refined:
            continue
        refined.sort(key=lambda t: t[0])
        # try LP polish on the single best refined if promising
        v0, X0 = refined[0]
        # also consider second best if very close but farther (topology backup)
        cands_to_polish = refined[:2]
        for v_r, X_r in cands_to_polish:
            if time.time() > EXPL_END + 4:
                break
            if v_r > best_val + 0.008 and v_r > v0 + 0.003:
                continue
            budget = 1.2
            if v_r < best_val + 0.001:
                budget = 2.2
            if v_r < best_val - 1e-12:
                budget = 2.5
            endt = min(time.time() + budget, HARD_END - 0.5)
            if endt <= time.time() + 0.1:
                continue
            Xp, vp = _lp_polish_deadline(X_r, endt)
            if vp < best_val - 1e-12:
                X_best = Xp
                best_val = vp
            elif v_r < best_val - 1e-12:
                X_best = X_r
                best_val = v_r
        # also direct accept without polish if refined beats best
        for v_r, X_r in refined:
            if v_r < best_val - 1e-12:
                X_best = X_r
                best_val = v_r
                break

    if time.time() < HARD_END - 0.5:
        X_best, best_val = _lp_polish_deadline(X_best, HARD_END - 0.2)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
