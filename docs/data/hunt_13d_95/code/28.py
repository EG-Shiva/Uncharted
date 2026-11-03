import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_stats(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rowmax = G.max(axis=1)
    contacts = (G > m - 0.002).sum(axis=1)
    return m, rowmax, contacts

def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    if t_end is None:
        t_end = start + seconds
    else:
        t_end = min(t_end, start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def fingerprint(X):
    G = X @ X.T
    np.fill_diagonal(G, 1.0)
    # eigenvalues invariant to perm/rotation
    try:
        ev = np.linalg.eigvalsh(G)
    except Exception:
        ev = np.zeros(X.shape[0])
    ev = np.sort(ev)
    G2 = G.copy()
    np.fill_diagonal(G2, -2.0)
    m = float(G2.max())
    cc = (G2 > m - 0.002).sum(axis=1)
    cc = np.sort(cc.astype(float))
    return ev, cc, m

def make_child(p1, p2, rng, k):
    n, d = p1.shape
    _, rowmax1, cont1 = per_point_stats(p1)
    _, rowmax2, cont2 = per_point_stats(p2)
    order_worst = np.argsort(-rowmax1)[:k]
    order_donor = np.lexsort((rowmax2, cont2))[:k]
    child = p1.copy()
    if rng.random() < 0.5:
        # direct transplant (random relative rotation preserved)
        child[order_worst] = p2[order_donor]
    else:
        # rotation-aligned transplant via assignment + Procrustes
        try:
            D = p1 @ p2.T
            rr, cc = linear_sum_assignment(-D)
            P2perm = p2[cc]
            M = P2perm.T @ p1
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            p2r = p2 @ Q
        except Exception:
            p2r = p2
        child[order_worst] = p2r[order_donor]
    # small tangent jitter to break exact degeneracy
    nz = rng.standard_normal(size=(k, d)) * 0.005
    for ii, idx in enumerate(order_worst):
        v = nz[ii] - float(nz[ii] @ child[idx]) * child[idx]
        child[idx] = child[idx] + v
        nn = np.linalg.norm(child[idx])
        if nn < 1e-12:
            vv = rng.standard_normal(size=d); vv /= np.linalg.norm(vv)
            child[idx] = vv
        else:
            child[idx] /= nn
    return child

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    B, N, D = 24, 95, 13
    # load best for reference / fallback
    try:
        xbest0 = np.load("best.npy").astype(float)
        xbest0 /= np.linalg.norm(xbest0, axis=1, keepdims=True)
        best0 = max_cos(xbest0)
    except Exception:
        xbest0 = None; best0 = 0.2440978569264
    # ---- init population: half Gaussian, half low-discrepancy (Halton->normal) ----
    X = np.empty((B, N, D))
    for b in range(12):
        a = rng.standard_normal(size=(N, D))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        X[b] = a
    try:
        from scipy.stats import qmc
        from scipy.special import ndtri
        eng = qmc.Halton(d=D, scramble=True, seed=12345)
        u = eng.random(n=(B - 12) * N)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z = ndtri(u).reshape((B - 12, N, D))
        # random orthogonal scramble per member for diversity
        for bi in range(B - 12):
            q, _ = np.linalg.qr(rng.standard_normal(size=(D, D)))
            z[bi] = z[bi] @ q
            z[bi] /= np.linalg.norm(z[bi], axis=1, keepdims=True)
        X[12:] = z
    except Exception:
        for b in range(12, B):
            a = rng.standard_normal(size=(N, D))
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            X[b] = a
    m1 = np.zeros_like(X); m2 = np.zeros_like(X)
    b1, b2a, eps = 0.9, 0.999, 1e-8
    gstep = 0
    betas = [20.0, 80.0, 250.0, 700.0]
    lrs = [0.01, 0.006, 0.0035, 0.002]
    phase1_end = t0 + 75.0
    last_prune = t0
    ar = np.arange(N)
    # batched Adam with periodic prune/crossover
    si = 0
    while time.time() < phase1_end:
        beta = betas[si % 4]; lr = lrs[si % 4]
        si += 1
        # run ~200-step chunk
        for _ in range(200):
            if time.time() >= phase1_end:
                break
            gstep += 1
            G = np.matmul(X, np.transpose(X, (0, 2, 1)))
            # exclude diagonal from softmax
            G[:, ar, ar] = -2.0
            gmax = G.max(axis=(1, 2))
            # exp
            W = np.exp(beta * (G - gmax[:, None, None]))
            W[:, ar, ar] = 0.0
            ssum = W.sum(axis=(1, 2))[:, None, None] + 1e-30
            W /= ssum
            grad = np.matmul(W, X)
            dot = np.sum(grad * X, axis=2, keepdims=True)
            gt = grad - dot * X
            m1 = b1 * m1 + (1 - b1) * gt
            m2 = b2a * m2 + (1 - b2a) * (gt * gt)
            c1 = 1 - b1 ** gstep; c2 = 1 - b2a ** gstep
            m1h = m1 / c1; m2h = m2 / c2
            step = lr * m1h / (np.sqrt(m2h) + eps)
            # tangent project step
            sdot = np.sum(step * X, axis=2, keepdims=True)
            step = step - sdot * X
            # clip
            sn = np.linalg.norm(step, axis=2, keepdims=True) + 1e-12
            clip = np.minimum(1.0, 0.05 / sn)
            step = step * clip
            X = X - step
            X /= np.linalg.norm(X, axis=2, keepdims=True)
        if time.time() >= phase1_end:
            break
        # prune every ~10s
        if time.time() - last_prune >= 10.0:
            last_prune = time.time()
            scores = np.empty(B)
            evs = []; ccs = []
            for b in range(B):
                ev, cc, mm = fingerprint(X[b])
                evs.append(ev); ccs.append(cc); scores[b] = mm
            order = np.argsort(scores)
            kept = []
            is_dup = np.zeros(B, dtype=bool)
            for idx in order:
                dup = False
                for k2 in kept:
                    de = float(np.linalg.norm(evs[idx] - evs[k2]))
                    dc = float(np.abs(ccs[idx] - ccs[k2]).sum())
                    if de < 0.05 and dc < 8:
                        dup = True
                        break
                if dup:
                    is_dup[idx] = True
                else:
                    kept.append(idx)
            # replace duplicates by crossover children of survivors
            surv = [i for i in order if not is_dup[i]]
            if len(surv) == 0:
                surv = order[:6].tolist()
            for b in range(B):
                if is_dup[b]:
                    a1 = surv[int(rng.integers(len(surv)))]
                    a2 = surv[int(rng.integers(len(surv)))]
                    if a1 == a2 and len(surv) > 1:
                        a2 = surv[(surv.index(a1) + 1) % len(surv)]
                    k = int(rng.integers(15, 26))
                    X[b] = make_child(X[a1], X[a2], rng, k)
                    m1[b] = 0; m2[b] = 0
            # also replace worst 2 with crossover of top-2 distinct for fresh topology
            if len(kept) >= 2:
                worst = order[-2:]
                for wi, b in enumerate(worst):
                    if is_dup[b]:
                        continue
                    a1 = kept[wi % len(kept)]; a2 = kept[(wi + 1) % len(kept)]
                    k = int(rng.integers(15, 26))
                    X[b] = make_child(X[a1], X[a2], rng, k)
                    m1[b] = 0; m2[b] = 0
            gstep = max(gstep - 200, 0)
    # evaluate elites
    scores = np.array([max_cos(X[b]) for b in range(B)])
    order = np.argsort(scores)
    # pick top three mutually distinct
    elites = []
    elites_evs = []; elites_ccs = []
    for idx in order:
        ev, cc, _ = fingerprint(X[idx])
        ok = True
        for e2 in range(len(elites)):
            if float(np.linalg.norm(ev - elites_evs[e2])) < 0.05 and float(np.abs(cc - elites_ccs[e2]).sum()) < 8:
                ok = False; break
        if ok:
            elites.append(idx); elites_evs.append(ev); elites_ccs.append(cc)
        if len(elites) >= 3:
            break
    if len(elites) == 0:
        elites = order[:3].tolist()
    # LP polish top three ~15s each (time-aware)
    polished = []
    for e in elites:
        rem = deadline - time.time() - 8.0
        if rem <= 1.0:
            polished.append(X[e] / np.linalg.norm(X[e], axis=1, keepdims=True))
            continue
        secs = min(15.0, rem / (4 - len(polished)) if len(polished) < 3 else rem)
        # ensure leave time for final long polish
        secs = min(secs, max(1.0, deadline - time.time() - 20.0) if len(elites) > 1 else rem)
        if secs < 1.0:
            secs = min(15.0, rem)
        xp = lp_polish(X[e], seconds=secs)
        polished.append(xp)
        if time.time() >= deadline - 22.0:
            break
    # winner
    if len(polished) > 0:
        vals = [max_cos(p) for p in polished]
        wi = int(np.argmin(vals))
        winner = polished[wi]
        wval = vals[wi]
    else:
        winner = X[order[0]]
        wval = scores[order[0]]
    # long polish winner with remaining time
    rem = deadline - time.time() - 0.8
    if rem > 2.0:
        winner = lp_polish(winner, seconds=rem)
        wval = max_cos(winner)
    winner /= np.linalg.norm(winner, axis=1, keepdims=True)
    # return only if beats best.npy, else fallback
    if xbest0 is not None:
        if wval < best0 - 1e-12:
            return winner
        else:
            return xbest0
    else:
        return winner
