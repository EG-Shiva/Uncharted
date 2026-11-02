import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _norm_batch(X):
    n = np.linalg.norm(X, axis=2, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _batch_max_cos(X):
    # X: (P,N,D)
    G = np.matmul(X, np.transpose(X, (0, 2, 1)))
    P, N, _ = G.shape
    dm = np.eye(N, dtype=bool)
    # set diag to -2 via mask per batch
    for i in range(P):
        G[i][dm] = -2.0
    return np.max(G, axis=(1, 2))

def _batch_quench(X0, steps, a0, a1, lr0, lr1, deadline=None):
    X = X0.copy()
    P, N, D = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    dm = np.eye(N, dtype=bool)
    b1t_pow = 1.0
    b2t_pow = 1.0
    for t in range(1, steps + 1):
        if deadline is not None and (t % 100 == 0) and time.time() > deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        for i in range(P):
            G[i][dm] = -2.0
        M = np.max(G, axis=(1, 2), keepdims=True)
        E = np.exp(alpha * (G - M))
        for i in range(P):
            E[i][dm] = 0.0
        S = np.sum(E, axis=(1, 2))
        S = np.maximum(S, 1e-300)
        EX = np.matmul(E, X)
        grad = EX / S[:, None, None]
        proj = np.sum(grad * X, axis=2, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        b1t_pow *= b1
        b2t_pow *= b2
        mhat = m / (1.0 - b1t_pow)
        vhat = v / (1.0 - b2t_pow)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _norm_batch(X)
    return X

def _fingerprint(x):
    n = x.shape[0]
    g = x @ x.T
    iu = np.triu_indices(n, 1)
    c = g[iu]
    m = float(np.max(c))
    # top spectrum quantized
    cs = np.sort(c)[::-1][:40]
    q = tuple(np.round(cs, 4).tolist())
    contacts = int(np.sum(c > m - 1e-3))
    # degree vector sorted quantized
    Gfull = g.copy()
    np.fill_diagonal(Gfull, -2.0)
    deg = np.sum(Gfull > m - 1e-3, axis=1)
    degq = tuple(sorted(deg.tolist()))
    return (q, contacts, degq, m)

def _lp_polish_deadline(x, deadline):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos_single(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 40:
        it += 1
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1500:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1500]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos_single(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 148.0
    SEARCH_END = t0 + 118.0
    rng = np.random.default_rng(20261101)
    n, d = 55, 21

    archive = []  # list of (score, X, fp)
    seen = set()
    best_score = 1e9
    best_X = None
    total_inits = 0

    batch_id = 0
    # multi-start population search: at least 100 inits
    # use batches of 50 with brief annealed quenches
    while time.time() < SEARCH_END:
        batch_id += 1
        # adapt batch size to remaining time
        remain = SEARCH_END - (time.time() - t0) - 0.0
        # estimate; break if too little time for another batch
        if remain < 3.0 and total_inits >= 100:
            break
        B = 50
        if batch_id == 1:
            steps, a0, a1, lr0, lr1 = 450, 25.0, 420.0, 0.02, 0.002
        elif batch_id == 2:
            steps, a0, a1, lr0, lr1 = 450, 35.0, 500.0, 0.018, 0.0018
        elif batch_id == 3:
            steps, a0, a1, lr0, lr1 = 500, 20.0, 600.0, 0.022, 0.0015
        else:
            steps, a0, a1, lr0, lr1 = 400, 30.0, 550.0, 0.015, 0.0015
            # smaller batch for extra rounds
            B = 40
        X0 = rng.standard_normal(size=(B, n, d))
        X0 = _norm_batch(X0)
        total_inits += B
        # brief annealed Riesz/log-sum-exp Adam repulsion
        Xq = _batch_quench(X0, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1,
                           deadline=SEARCH_END)
        scores = _batch_max_cos(Xq)
        for b in range(B):
            sc = float(scores[b])
            xb = _norm_rows(Xq[b].copy())
            fp = _fingerprint(xb)
            key = (fp[0], fp[1])  # spectrum + contact count
            dup = key in seen
            seen.add(key)
            # diversity: penalize exact duplicates for archive ordering but keep true score
            archive.append((sc, xb, fp, dup))
            if sc < best_score:
                best_score = sc
                best_X = xb.copy()
        # stop after >=150 inits or time
        if total_inits >= 150:
            # allow one more diverse round if time permits
            if total_inits >= 190 or time.time() > SEARCH_END - 5:
                break
        if batch_id >= 5:
            break

    if best_X is None:
        best_X = _norm_rows(rng.standard_normal(size=(n, d)))
        best_score = _max_cos_single(best_X)

    # refine top diverse candidates with a second anneal (batched)
    # select candidates below 0.12 for refinement
    archive_sorted = sorted(archive, key=lambda t: t[0])
    # fingerprint diversity filter: keep first occurrence of each key
    filt = []
    used_keys = set()
    for sc, xb, fp, dup in archive_sorted:
        k = (fp[0], fp[1])
        if k in used_keys:
            continue
        used_keys.add(k)
        filt.append((sc, xb, fp))
        if len(filt) >= 12:
            break
    if len(filt) > 0 and time.time() < HARD_END - 30.0:
        # stack top up to 10 for joint refinement
        M = min(len(filt), 10)
        Xr = np.stack([filt[i][1] for i in range(M)], axis=0)
        Xr = _batch_quench(Xr, steps=350, a0=150.0, a1=800.0, lr0=0.008, lr1=0.001,
                           deadline=HARD_END - 25.0)
        rscores = _batch_max_cos(Xr)
        for i in range(M):
            sc = float(rscores[i])
            xb = _norm_rows(Xr[i].copy())
            fp = _fingerprint(xb)
            archive.append((sc, xb, fp, False))
            if sc < best_score:
                best_score = sc
                best_X = xb.copy()

    # LP-polish only candidates already below ~0.075, at most 2s each, diverse first
    # re-sort including refined
    archive_sorted = sorted(archive, key=lambda t: t[0])
    # build LP queue: below 0.075, non-duplicate first, penalize jammed signature
    queue = []
    for sc, xb, fp, *rest in archive_sorted:
        if sc > 0.075:
            continue
        contacts = fp[1]
        # penalize incumbent-like jammed basin (300-360 contacts, max<0.07) for ordering
        jammed = (250 <= contacts <= 400 and sc < 0.07)
        penalty = 0.002 if jammed else 0.0
        queue.append((sc + penalty, sc, xb, fp))
    queue.sort(key=lambda t: t[0])
    # also consider best overall even if above threshold? spec says only below 0.075
    # cap number of LP attempts by time
    lp_attempts = 0
    for _, sc, xb, fp in queue:
        if lp_attempts >= 10:
            break
        if time.time() > HARD_END - 8.0:
            break
        # at most 2s each
        budget = 2.0
        # leave room for final comparison
        if HARD_END - time.time() < budget + 3.0:
            budget = max(0.5, HARD_END - time.time() - 3.0)
        if budget < 0.4:
            break
        deadline = time.time() + budget
        try:
            xp, vp = _lp_polish_deadline(xb, deadline)
        except Exception:
            continue
        lp_attempts += 1
        if vp < best_score:
            best_score = vp
            best_X = xp.copy()
        if time.time() >= HARD_END - 5.0:
            break

    # final comparison against published best (allowed only at end)
    try:
        for fn in ("best.npy", "record.npy"):
            try:
                ref = np.load(fn)
                ref = np.asarray(ref, dtype=np.float64)
                if ref.shape == (55, 21):
                    ref = _norm_rows(ref)
                    rs = _max_cos_single(ref)
                    if rs < best_score:
                        best_score = rs
                        best_X = ref.copy()
            except Exception:
                pass
    except Exception:
        pass

    best_X = _norm_rows(np.asarray(best_X, dtype=np.float64))
    # strict feasibility: no zero rows, no NaN
    assert best_X.shape == (55, 21)
    assert np.all(np.isfinite(best_X))
    nr = np.linalg.norm(best_X, axis=1)
    assert np.all(nr > 1e-9)
    return best_X
