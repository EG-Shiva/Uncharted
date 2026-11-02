import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_core(Y, steps, thr=0.164, w_soft=0.4, lr0=0.012, lr1=0.0012, T0=0.012, T1=0.002):
    n, d = Y.shape
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    ll = np.log(lr1 / lr0)
    lt = np.log(T1 / T0)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        lr = lr0 * np.exp(ll * frac)
        T = T0 * np.exp(lt * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        # hinge grad
        H = G - thr
        H[H < 0] = 0.0
        np.fill_diagonal(H, 0.0)
        Gh = 2.0 * (H @ Y)
        # soft-max grad
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            Gs = np.zeros_like(Y)
        else:
            Gs = (E @ Y) / S
        Gr = Gh + w_soft * Gs
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gt
        vv *= b2; vv += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1t)
        vhat = vv / (1 - b2t)
        Y = Y - lr * mhat / (np.sqrt(vhat) + eps)
        Y = _norm(Y)
    return Y

def _refine_candidates(cands, Y, T0=0.010, T1=0.001, lr0=0.08, lr1=0.005, iters=150):
    # cands (C,d), refine each by hole maximization vs Y
    C = cands.copy()
    C = _norm(C)
    ll = np.log(lr1 / lr0)
    lt = np.log(T1 / T0)
    for it in range(iters):
        frac = it / (iters - 1) if iters > 1 else 0.0
        T = T0 * np.exp(lt * frac)
        lr = lr0 * np.exp(ll * frac)
        S = C @ Y.T  # (C,m)
        gm = S.max(axis=1, keepdims=True)
        W = np.exp((S - gm) / T)
        Ws = W.sum(axis=1, keepdims=True)
        Ws[Ws == 0] = 1.0
        W = W / Ws
        Gdir = W @ Y  # (C,d)
        dot = np.sum(Gdir * C, axis=1, keepdims=True)
        Gt = Gdir - dot * C
        C = C - lr * Gt
        C = _norm(C)
    return C

def _local_polish(Y, active_idx, steps=250, lr0=0.006, lr1=0.0006, T0=0.008, T1=0.0015):
    # only update rows in active_idx, others fixed; minimize softmax max
    n, d = Y.shape
    m = np.zeros((len(active_idx), d))
    vv = np.zeros((len(active_idx), d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    ll = np.log(lr1 / lr0)
    lt = np.log(T1 / T0)
    A = np.asarray(active_idx, dtype=int)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        lr = lr0 * np.exp(ll * frac)
        T = T0 * np.exp(lt * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        Gr = 2.0 * (E @ Y) / S
        Gt = Gr[A] - np.sum(Gr[A] * Y[A], axis=1, keepdims=True) * Y[A]
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gt
        vv *= b2; vv += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1t)
        vhat = vv / (1 - b2t)
        Y[A] = Y[A] - lr * mhat / (np.sqrt(vhat) + eps)
        Y[A] = _norm(Y[A])
    return Y

def _lp_polish_capped(x, t_end, max_constraints=1500):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    try:
        iu = np.triu_indices(n, 1)
    except Exception:
        return x
    best = _maxcos(x)
    radius = 1e-3
    while time.time() < t_end - 0.4 and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_constraints:
            idx = np.argpartition(-c, max_constraints)[:max_constraints]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 165.0
    RECORD = 0.168263492524
    # load base
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None and a.shape == (76, 15):
            a = _norm(a.astype(float))
            if np.all(np.isfinite(a)):
                cands.append(a)
    if not cands:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        return _norm(X)
    base = cands[0]
    bv = _maxcos(base)
    for c in cands[1:]:
        v = _maxcos(c)
        if v < bv:
            base, bv = c.copy(), v
    n, d = 76, 15
    # contact analysis at slack 1e-9
    G0 = base @ base.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    contact_mask = G0 > m0 - 1e-9
    deg = contact_mask.sum(axis=1)
    # sum of near-max cosines per point (only contacts)
    ssum = (np.where(contact_mask, G0, 0.0)).sum(axis=1)
    # rank worst: high degree, then high ssum
    order = np.lexsort((-ssum, -deg))  # descending deg then ssum? lexsort last key primary
    # Actually lexsort: keys (secondary, primary); we want primary -deg
    # So order[0] is worst core point
    best_global = base.copy()
    bv_global = bv

    # try deletion sizes 8, 6, 10
    for trial_k, k in enumerate([8, 6, 10]):
        if time.time() > deadline - 35:
            break
        del_idx = np.sort(order[:k])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Y = base[keep_mask].copy()  # (n-k, d)
        # re-optimize core with hinge 0.164 + softmax
        steps_core = 1400
        # reduce steps if low on time
        if time.time() > deadline - 60:
            steps_core = 800
        Y = _anneal_core(Y, steps=steps_core, thr=0.164, w_soft=0.4)
        # quick check: if core still very bad, continue a bit
        # reinsert one at a time
        rng = np.random.default_rng(9000 + k * 100 + trial_k)
        # reinsert in order of increasing "hole difficulty": start with worst deleted last?
        # insert deleted points sorted by ssum ascending (easier first) to stabilize
        del_order = del_idx[np.argsort(ssum[del_idx])]
        cur = Y
        ok = True
        for ins_i, _orig in enumerate(del_order):
            if time.time() > deadline - 28:
                ok = False
                break
            m_cur = cur.shape[0]
            # sample >=50000 candidates
            NS = 50000
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ cur.T).max(axis=1)
            # best 20 indices
            part = np.argpartition(scores, 20)[:20]
            # order them by score
            part = part[np.argsort(scores[part])]
            cand20 = C[part]
            # refine best 20 by hole maximization
            refined = _refine_candidates(cand20, cur, iters=150)
            # evaluate true max
            rscores = (refined @ cur.T).max(axis=1)
            bi = int(np.argmin(rscores))
            h = refined[bi] / np.linalg.norm(refined[bi])
            # append
            cur = np.vstack([cur, h.reshape(1, d)])
            # short tangent polish restricted to inserted point + 20 nearest neighbours
            # nearest = largest dot with new point
            dots = (cur[:-1] @ cur[-1])
            nn = np.argsort(-dots)[:20]
            active = np.concatenate([nn, np.array([cur.shape[0] - 1])])
            # keep it fast: 200 steps
            cur = _local_polish(cur, active, steps=200)
            # optional tiny extra hole tweak for inserted point only if still high
        if not ok:
            # evaluate partial? need 76 rows, skip if incomplete
            continue
        if cur.shape[0] != 76:
            continue
        # short global anneal to settle (fast, 400 steps, low lr)
        if time.time() < deadline - 25:
            cur = _anneal_core(cur, steps=400, thr=0.164, w_soft=0.6, lr0=0.004, lr1=0.0004, T0=0.004, T1=0.0008)
        v = _maxcos(cur)
        if v < bv_global and np.all(np.isfinite(cur)):
            best_global = cur.copy()
            bv_global = v
        # also try to keep any beating record even if not best? best_global already does
    # final global LP polish capped at 20 seconds
    remain = deadline - time.time()
    if remain > 3:
        cap = min(remain - 1.0, 20.0)
        t_end = time.time() + cap
        try:
            polished = _lp_polish_capped(best_global, t_end)
            vp = _maxcos(polished)
            if vp < bv_global and np.all(np.isfinite(polished)):
                best_global = polished
                bv_global = vp
        except Exception:
            pass
    best_global = _norm(best_global)
    if not np.all(np.isfinite(best_global)) or best_global.shape != (76, 15):
        return _norm(base)
    return best_global
