import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _find_void(x_fixed, rng, n_samples=8000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    return cand[idx].copy()

def _personal_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _fingerprint(x):
    return np.sort(_personal_worst(x))

def _fp_dist(a, b):
    return float(np.linalg.norm(a - b))

def _align(A, B, iters=4):
    Q = np.eye(4)
    perm = np.arange(A.shape[0])
    for _ in range(iters):
        Bq = B @ Q
        S = A @ Bq.T
        r, c = linear_sum_assignment(-S)
        order = np.argsort(r)
        perm = c[order]
        Bp = B[perm]
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
    return Q, perm

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        xrec = np.load("best.npy")
    except Exception:
        try:
            xrec = np.load("record.npy")
        except Exception:
            xrec = rng.standard_normal((272, 4))
    xrec = _norm(np.asarray(xrec, dtype=float))
    if xrec.shape != (272, 4):
        flat = xrec.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        xrec = _norm(flat[:need].reshape(272, 4))
    vrec = _max_cos(xrec)
    # population: list of dicts
    pop_X = [xrec.copy()]
    pop_V = [vrec]
    pop_F = [_fingerprint(xrec)]
    # Phase 1: diverse population by strong kicks + anneal
    sigmas = [0.30, 0.45, 0.25, 0.55, 0.35, 0.40, 0.28, 0.50]
    for si, sigma in enumerate(sigmas):
        if time.time() > t0 + 45 or time.time() > deadline - 120:
            break
        x = xrec + sigma * rng.standard_normal((272, 4))
        x = _norm(x)
        # topology change: kill worst 3-5 into voids before anneal
        nk = 3 + (si % 3)
        for _ in range(nk):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            # pick among worst few
            order = np.argsort(-pw)
            victim = int(order[rng.integers(0, min(8, len(order)))])
            mask = np.ones(272, dtype=bool)
            mask[victim] = False
            x[victim] = _find_void(x[mask], rng, n_samples=6000)
        tau0, tau1 = (0.025, 0.002) if si % 2 == 0 else (0.015, 0.0012)
        x = _anneal(x, 400, tau0, tau1, 0.010, 0.001)
        vv = _max_cos(x)
        # retain even if 0.905-0.93, but discard catastrophic >0.95 duplicates check
        if vv < 0.96:
            # diversity check vs existing: require min fp dist > 1e-6 else still keep if better
            fp = _fingerprint(x)
            dmin = min(_fp_dist(fp, f) for f in pop_F) if pop_F else 1.0
            if dmin > 5e-5 or vv < min(pop_V) or len(pop_X) < 9:
                pop_X.append(x.copy())
                pop_V.append(vv)
                pop_F.append(fp)
    # ensure 6-10 codes: if too few, add jittered copies
    while len(pop_X) < 6 and time.time() < deadline - 100:
        x = xrec + 0.35 * rng.standard_normal((272, 4))
        x = _norm(x)
        x = _anneal(x, 250, 0.02, 0.002, 0.008, 0.001)
        pop_X.append(x.copy()); pop_V.append(_max_cos(x)); pop_F.append(_fingerprint(x))
    MAXP = 10
    def prune():
        # always keep index 0 (record fallback) and best + diverse
        n = len(pop_X)
        if n <= MAXP:
            return
        order = np.argsort(np.array(pop_V))
        keep = set()
        keep.add(0)
        for k in order[:3]:
            keep.add(int(k))
        # greedily add most diverse
        while len(keep) < MAXP:
            best_c, best_d = -1, -1.0
            for i in range(n):
                if i in keep:
                    continue
                dmin = min(_fp_dist(pop_F[i], pop_F[j]) for j in keep)
                # bonus for good score
                score = dmin - 0.02 * (pop_V[i] - min(pop_V))
                if score > best_d:
                    best_d = score
                    best_c = i
            if best_c < 0:
                break
            keep.add(best_c)
        keep = sorted(keep)
        pop_X[:] = [pop_X[i] for i in keep]
        pop_V[:] = [pop_V[i] for i in keep]
        pop_F[:] = [pop_F[i] for i in keep]
    prune()
    # Phase 2: recombination
    nchild = 0
    # precompute best index
    while time.time() < deadline - 32:
        if len(pop_X) < 2:
            break
        vals = np.array(pop_V)
        i_best = int(np.argmin(vals))
        # most graph-diverse from best
        dists = np.array([_fp_dist(pop_F[i], pop_F[i_best]) if i != i_best else -1.0 for i in range(len(pop_X))])
        # occasionally pick second best for exploitation
        if nchild % 4 == 3:
            order = np.argsort(vals)
            i1 = int(order[0]); i2 = int(order[1]) if len(order) > 1 else int(order[0])
        else:
            i1 = i_best
            i2 = int(np.argmax(dists))
            if i2 == i1:
                # random pair
                i2 = int(rng.integers(0, len(pop_X)))
                if i2 == i1:
                    i2 = (i1 + 1) % len(pop_X)
        A = pop_X[i1]; B = pop_X[i2]
        try:
            Q, perm = _align(A, B, iters=3)
        except Exception:
            Q = np.eye(4)
        Bq = B @ Q
        # several random great spheres per pair
        nsph = 2
        for s in range(nsph):
            if time.time() > deadline - 32:
                break
            u = rng.standard_normal((4,))
            u /= np.linalg.norm(u)
            h = 136 + int(rng.integers(-28, 29))
            h = max(90, min(182, h))
            sA = A @ u
            sB = Bq @ u
            idxA = np.argsort(-sA)[:h]
            idxB = np.argsort(sB)[:272 - h]
            child = np.concatenate([A[idxA], Bq[idxB]], axis=0)
            child = _norm(child)
            # repair closest cross-boundary pairs by void reinsertion (1-2 kills)
            for _ in range(2):
                G = child @ child.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                victim = int(np.argmax(pw))
                if pw[victim] < 0.90 and rng.random() < 0.7:
                    break
                mask = np.ones(272, dtype=bool); mask[victim] = False
                child[victim] = _find_void(child[mask], rng, n_samples=6000)
            # anneal child
            child = _anneal(child, 350, 0.012, 0.001, 0.008, 0.0008)
            vv = _max_cos(child)
            fp = _fingerprint(child)
            # retain if not near-duplicate or if improved
            dmin = min(_fp_dist(fp, f) for f in pop_F) if pop_F else 1e9
            if vv < max(pop_V) + 0.005 or dmin > 1e-4:
                # avoid exact duplicates of record basin
                if dmin < 2e-6 and vv >= min(pop_V) - 1e-12:
                    pass
                else:
                    pop_X.append(child.copy()); pop_V.append(vv); pop_F.append(fp)
                    prune()
            nchild += 1
            if nchild > 24:
                break
        if nchild > 24:
            break
        # small random restart injection to keep diversity if stagnating
        if len(pop_X) >= MAXP and nchild % 6 == 5 and time.time() < deadline - 60:
            pass
    # Phase 3: polish best candidates
    vals = np.array(pop_V)
    order = np.argsort(vals)
    # candidate indices: best overall + best diverse (ensure record kept)
    cands = [int(order[0])]
    # add most diverse among top half
    if len(order) > 1:
        # pick one with good score but large distance to best
        best_fp = pop_F[cands[0]]
        scored = []
        for i in order[1:6]:
            d = _fp_dist(pop_F[int(i)], best_fp)
            scored.append((d, int(i)))
        scored.sort(reverse=True)
        if scored:
            cands.append(scored[0][1])
    x_best = pop_X[cands[0]].copy()
    best_val = pop_V[cands[0]]
    # also consider record explicitly
    if vrec < best_val:
        x_best = xrec.copy(); best_val = vrec
    remain = deadline - time.time() - 1.5
    if remain > 6:
        # polish best candidate first
        per = min(remain * 0.65, 22.0)
        xp = _lp_polish(x_best, per, deadline - 0.5)
        vp = _max_cos(xp)
        if vp <= best_val:
            x_best, best_val = xp, vp
        # polish second candidate with leftover
        leftover = deadline - time.time() - 1.0
        if len(cands) > 1 and leftover > 6:
            x2 = _lp_polish(pop_X[cands[1]], min(leftover, 12.0), deadline - 0.5)
            v2 = _max_cos(x2)
            if v2 < best_val:
                x_best, best_val = x2, v2
    # safe fallback: if polish failed / nonfinite, return record
    if not np.all(np.isfinite(x_best)):
        x_best = xrec.copy()
    x_best = _norm(x_best)
    # final safety: if our best is worse than record, return record-polished? return best anyway but ensure record fallback
    if _max_cos(x_best) > vrec + 1e-12:
        # return record (still valid, avoids regression) unless child essentially equal
        x_best = xrec.copy()
    x_best = _norm(x_best)
    return x_best
