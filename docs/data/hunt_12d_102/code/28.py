import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _batch_scores(X):
    B, N, D = X.shape
    G = np.matmul(X, X.transpose(0, 2, 1))
    idx = np.arange(N)
    G[:, idx, idx] = -2.0
    return G.max(axis=(1, 2))

def _fingerprints(X, K=300):
    B, N, D = X.shape
    iu = np.triu_indices(N, 1)
    F = np.empty((B, K))
    for b in range(B):
        g = (X[b] @ X[b].T)[iu]
        g = np.sort(g)[::-1]
        F[b] = g[:K]
    return F

def _greedy_sample(rng, n=102, d=12, pool=8000):
    cand = rng.standard_normal((pool, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sel0 = int(rng.integers(pool))
    sel_mask = np.zeros(pool, dtype=bool)
    sel_mask[sel0] = True
    maxdot = cand @ cand[sel0]
    maxdot[sel0] = np.inf
    sel = [sel0]
    for _ in range(1, n):
        idx = int(np.argmin(np.where(sel_mask, np.inf, maxdot)))
        sel.append(idx)
        sel_mask[idx] = True
        maxdot = np.maximum(maxdot, cand @ cand[idx])
        maxdot[idx] = np.inf
    return cand[np.array(sel)]

def _isotropise(X, rng, iters=3):
    Y = X.copy()
    n, d = Y.shape
    target = n / d
    for _ in range(iters):
        C = Y.T @ Y
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, 1e-10)
        sc = np.sqrt(target / w)
        Wm = (V * sc) @ V.T
        Y = Y @ Wm
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # random rotation
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Y = Y @ Q
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _batched_quench(X, steps, a0, a1, lr0, lr1, deadline=1e18):
    B, N, D = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    idx = np.arange(N)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(1, steps + 1):
        frac = s / steps
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = np.matmul(X, X.transpose(0, 2, 1))
        G[:, idx, idx] = -2.0
        M = G.max(axis=(1, 2))
        E = np.exp(alpha * (G - M[:, None, None]))
        E[:, idx, idx] = 0.0
        S = E.sum(axis=(1, 2))
        S[S == 0] = 1.0
        P = E / S[:, None, None]
        Grad = np.matmul(P, X)
        dot = np.sum(Grad * X, axis=2, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        d = np.sum(upd * X, axis=2, keepdims=True)
        upd_t = upd - d * X
        X -= lr * upd_t
        nrm = np.linalg.norm(X, axis=2, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        if (s % 250 == 0) and time.time() > deadline:
            break
    return X

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos_single(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = _max_cos_single(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(0)
    N, D, B = 102, 12, 16
    # reference record / best
    ref = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (N, D) and np.all(np.isfinite(a)):
                ref = a
                break
        except Exception:
            continue
    if ref is None:
        ref = rng.standard_normal((N, D))
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)

    pop = np.empty((B, N, D))
    # 0-1: perturbed record
    for i, sig in enumerate([0.01, 0.035]):
        y = ref + sig * rng.standard_normal((N, D))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        pop[i] = y
    # 2-5: isotropised record + noise/rotation
    for i in range(2, 6):
        y = _isotropise(ref, rng, iters=3)
        y = y + (0.005 + 0.005 * (i - 2)) * rng.standard_normal((N, D))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        pop[i] = y
    # 6-10: pure random
    for i in range(6, 11):
        y = rng.standard_normal((N, D))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        pop[i] = y
    # 11-15: greedy farthest point
    for i in range(11, 16):
        try:
            pop[i] = _greedy_sample(rng, n=N, d=D, pool=6000)
        except Exception:
            y = rng.standard_normal((N, D))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            pop[i] = y

    scores = _batch_scores(pop)
    best_idx = int(np.argmin(scores))
    gbest_score = float(scores[best_idx])
    gbest = pop[best_idx].copy()
    archive_X = [pop[b].copy() for b in range(B)]
    archive_s = [float(scores[b]) for b in range(B)]

    cycle = 0
    # batched search, reserve ~26s for LP polish
    while time.time() < T_END - 28.0 and cycle < 60:
        cycle += 1
        # anneal sharpness progressively: early broad, late sharp
        if cycle <= 2:
            a0, a1, lr0, lr1, steps = 20.0, 80.0, 0.008, 0.003, 900
        elif cycle <= 5:
            a0, a1, lr0, lr1, steps = 40.0, 150.0, 0.006, 0.002, 900
        else:
            a0, a1, lr0, lr1, steps = 80.0, 250.0, 0.005, 0.0012, 800
        # shrink steps if little time
        if time.time() > T_END - 60.0:
            steps = 600
        pop = _batched_quench(pop, steps, a0, a1, lr0, lr1, deadline=T_END - 28.0)
        scores = _batch_scores(pop)
        for b in range(B):
            archive_X.append(pop[b].copy())
            archive_s.append(float(scores[b]))
        bi = int(np.argmin(scores))
        if float(scores[bi]) < gbest_score:
            gbest_score = float(scores[bi])
            gbest = pop[bi].copy()
        if time.time() >= T_END - 28.0:
            break
        # diversity-aware elitism
        F = _fingerprints(pop, K=300)
        order = np.argsort(scores)
        keep = []
        for b in order:
            ok = True
            for k in keep:
                if np.linalg.norm(F[b] - F[k]) < 2e-3:
                    ok = False
                    break
            if ok:
                keep.append(b)
            if len(keep) >= 4:
                break
        if len(keep) == 0:
            keep = list(order[:4])
        # refill population: keep elites, mutate rest
        newpop = np.empty_like(pop)
        for j, b in enumerate(keep):
            newpop[j] = pop[b]
        # fill remaining
        for j in range(len(keep), B):
            if j >= B - 2 and cycle % 2 == 0:
                # fresh blood: random or greedy
                if rng.random() < 0.5:
                    y = rng.standard_normal((N, D))
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                else:
                    try:
                        y = _greedy_sample(rng, n=N, d=D, pool=4000)
                    except Exception:
                        y = rng.standard_normal((N, D))
                        y /= np.linalg.norm(y, axis=1, keepdims=True)
                newpop[j] = y
            else:
                parent = pop[rng.choice(np.array(keep))]
                sig = float(rng.uniform(0.02, 0.14))
                # occasionally kick only worst points of parent
                if rng.random() < 0.4:
                    Gp = parent @ parent.T
                    np.fill_diagonal(Gp, -2.0)
                    rm = Gp.max(axis=1)
                    worst = np.argsort(-rm)[:8]
                    y = parent.copy()
                    y[worst] += sig * 2.0 * rng.standard_normal((len(worst), D))
                    y += 0.005 * rng.standard_normal((N, D))
                else:
                    y = parent + sig * rng.standard_normal((N, D))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                newpop[j] = y
        pop = newpop
        # keep global best inside population
        pop[0] = gbest.copy()

    # select best three genuinely distinct elites from archive
    try:
        A = np.array(archive_X)  # (M,N,D) may be large; do selection by scores
        S = np.array(archive_s)
        o = np.argsort(S)
        # fingerprints for top 60 only to save time
        top = o[:min(60, len(o))]
        Ft = _fingerprints(A[top], K=400)
        elites = []
        elites_idx = []
        for ii, gi in enumerate(top):
            ok = True
            for jj in range(len(elites)):
                if np.linalg.norm(Ft[ii] - Ft[elites_idx[jj]]) < 1.5e-3:
                    ok = False
                    break
            if ok:
                elites.append(A[gi].copy())
                elites_idx.append(ii)
            if len(elites) >= 3:
                break
        if len(elites) == 0:
            elites = [gbest.copy()]
    except Exception:
        elites = [gbest.copy()]

    # LP polish at most 8s each
    best_final = gbest.copy()
    best_final_score = _max_cos_single(best_final)
    for e in elites[:3]:
        if time.time() >= T_END - 2.0:
            break
        try:
            y = _lp_polish(e.copy(), seconds=8.0, t_end=T_END - 1.0)
            s = _max_cos_single(y)
            if s < best_final_score:
                best_final_score = s
                best_final = y.copy()
        except Exception:
            continue
    best_final /= np.linalg.norm(best_final, axis=1, keepdims=True)
    return best_final
