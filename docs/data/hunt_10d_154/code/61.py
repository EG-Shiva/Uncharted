import time
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _adam(x, steps, lr, T0, T1, deadline=1e18):
    x = _norm(x.copy())
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if (t % 200) == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-4:
            T = 8e-4
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        x = _norm(x)
    return x


def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x.copy())
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 30:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            cvals = c[sel][order]
        else:
            cvals = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cvals, A_eq=A_eq,
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
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _hole_repair(child, rng, n_cand=3000):
    n, d = child.shape
    child = _norm(child.copy())
    # find near duplicates
    G = child @ child.T
    np.fill_diagonal(G, -2.0)
    # threshold for duplicate
    dup_thr = 0.90
    # pre-generate hole candidates: pick farthest from set
    cand = _norm(rng.standard_normal(size=(n_cand, d)))
    # for each duplicate pair, replace one end with a hole candidate
    # do at most a few replacements
    reps = 0
    iu = np.triu_indices(n, 1)
    # quick loop
    for _ in range(6):
        G = child @ child.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        if mx < dup_thr:
            break
        ij = int(np.argmax(G.max(axis=1)))
        # row max of ij
        # generate best hole for this position: minimize max dot to others
        others = np.delete(child, ij, axis=0)
        md = (cand @ others.T).max(axis=1)
        bi = int(np.argmin(md))
        child[ij] = cand[bi]
        reps += 1
    # one greedy pass on worst points: try hole replacement if it lowers global max
    cur = _mc(child)
    G = child @ child.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    order = np.argsort(-rowmax)[:4]
    for vv in order:
        vv = int(vv)
        others = np.delete(child, vv, axis=0)
        md = (cand @ others.T).max(axis=1)
        # try few best candidates
        trial_idx = np.argpartition(md, 4)[:4]
        for bi in trial_idx:
            y = cand[int(bi)]
            nv = float((others @ y).max())
            if nv < cur - 1e-12:
                child[vv] = y
                cur = _mc(child)
                break
    return _norm(child)


def _align_and_match(A, B):
    n = A.shape[0]
    # stage 1: match without rotation
    try:
        r1, c1 = linear_sum_assignment(-(A @ B.T))
    except Exception:
        return B.copy(), np.arange(n)
    Bordered = np.empty_like(B)
    Bordered[r1] = B[c1]
    # Procrustes rotation via SVD: min || Bordered R - A ||
    M = Bordered.T @ A
    try:
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    except Exception:
        R = np.eye(A.shape[1])
    Bal = B @ R
    # stage 2: re-match after alignment
    try:
        r2, c2 = linear_sum_assignment(-(A @ Bal.T))
    except Exception:
        return Bal, np.arange(n)
    # map from A-index -> Bal-index
    mp = np.empty(n, dtype=int)
    mp[r2] = c2
    return Bal, mp


def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _mc(best)
    incumbent = best.copy()
    inc_val = best_val

    # ---- build 6-8 parents with capped coarse optimisation ----
    parents = [incumbent.copy()]
    pvals = [inc_val]
    coarse_dead = t0 + 70.0

    def add_parent(X):
        parents.append(_norm(X))
        pvals.append(_mc(parents[-1]))

    # large perturbations 0.3, 0.6, 1.0
    for sig in (0.3, 0.6, 1.0):
        if time.time() > coarse_dead:
            break
        X = incumbent + sig * rng.standard_normal(size=(n, d))
        X = _norm(X)
        X = _adam(X, steps=700, lr=0.03, T0=0.03, T1=0.004, deadline=coarse_dead)
        add_parent(X)

    # structured lattice-like inits (sparse signed patterns)
    for s in range(2):
        if time.time() > coarse_dead:
            break
        X = np.zeros((n, d))
        for i in range(n):
            k = int(rng.integers(2, 6))
            cols = rng.choice(d, size=k, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=k)
            X[i, cols] = signs
            # mix with small dense noise for diversity
            X[i] = X[i] + 0.2 * rng.standard_normal(d)
        X = _norm(X)
        X = _adam(X, steps=700, lr=0.03, T0=0.03, T1=0.004, deadline=coarse_dead)
        add_parent(X)

    # purely random starts
    for s in range(2):
        if time.time() > coarse_dead:
            break
        X = _norm(rng.standard_normal(size=(n, d)))
        X = _adam(X, steps=700, lr=0.03, T0=0.03, T1=0.004, deadline=coarse_dead)
        add_parent(X)

    # update global best with parents
    for X, v in zip(parents, pvals):
        if v < best_val:
            best_val = v
            best = X.copy()

    # ---- recombination children ----
    # distinctness: keep all; recombine incumbent with each donor
    block_sizes = [28, 40, 52, 64, 34, 46, 58, 30]
    children = []
    cvals = []
    mid_dead = DEAD - 28.0
    donor_order = list(range(1, len(parents)))
    # sort donors by structural difference (assignment score) descending difference
    # compute difference metric: mean matched inner after alignment
    scored = []
    for di in donor_order:
        try:
            Bal, mp = _align_and_match(incumbent, parents[di])
            score = float((incumbent * Bal[mp]).sum(axis=1).mean())
        except Exception:
            score = 1.0
        scored.append((score, di))
    scored.sort(key=lambda t: t[0])  # most different first
    donor_order = [di for _, di in scored]

    bi = 0
    for di in donor_order:
        if time.time() > mid_dead:
            break
        donor = parents[di]
        try:
            Bal, mp = _align_and_match(incumbent, donor)
        except Exception:
            continue
        k = block_sizes[bi % len(block_sizes)]
        bi += 1
        if k < 25:
            k = 25
        if k > 70:
            k = 70
        seed = int(rng.integers(n))
        sims = incumbent @ incumbent[seed]
        block = np.argsort(-sims)[:k]
        child = incumbent.copy()
        for j in block:
            child[int(j)] = Bal[int(mp[int(j)])]
        child = _hole_repair(child, rng, n_cand=2500)
        # intermediate-temperature Adam relax
        child = _adam(child, steps=1000, lr=0.02, T0=0.012, T1=0.0015, deadline=mid_dead)
        # light extra hole pass after relax for worst point
        v = _mc(child)
        children.append(child)
        cvals.append(v)
        if v < best_val:
            best_val = v
            best = child.copy()

    # one extra generation: recombine best child with incumbent if improved basin found
    if len(children) > 0 and time.time() < mid_dead - 8.0:
        try:
            order = np.argsort(np.array(cvals))
            top = children[int(order[0])]
            Bal, mp = _align_and_match(top, incumbent)
            k = 36
            seed = int(rng.integers(n))
            sims = top @ top[seed]
            block = np.argsort(-sims)[:k]
            child2 = top.copy()
            for j in block:
                child2[int(j)] = Bal[int(mp[int(j)])]
            child2 = _hole_repair(child2, rng, n_cand=2000)
            child2 = _adam(child2, steps=800, lr=0.02, T0=0.010, T1=0.0012, deadline=mid_dead)
            v2 = _mc(child2)
            children.append(child2)
            cvals.append(v2)
            if v2 < best_val:
                best_val = v2
                best = child2.copy()
        except Exception:
            pass

    # ---- final LP polish solely on best 1-2 children ----
    if len(children) > 0:
        order = np.argsort(np.array(cvals))
        n_polish = 2 if (DEAD - time.time()) > 18.0 else 1
        for ii in range(min(n_polish, len(order))):
            if time.time() > DEAD - 3.0:
                break
            cand = children[int(order[ii])]
            # only polish if competitive
            if cvals[int(order[ii])] > best_val + 0.004 and ii == 1:
                continue
            budget = (DEAD - 2.0 - time.time()) / (n_polish - ii)
            budget = max(2.0, min(budget, 14.0))
            try:
                pol = _lp_polish(cand, seconds=budget, deadline=DEAD - 1.0)
            except Exception:
                continue
            v = _mc(pol)
            if v < best_val:
                best_val = v
                best = pol.copy()
    else:
        # fallback: polish incumbent briefly
        if time.time() < DEAD - 5.0:
            try:
                pol = _lp_polish(incumbent, seconds=min(8.0, DEAD - 2.0 - time.time()), deadline=DEAD - 1.0)
                v = _mc(pol)
                if v < best_val:
                    best_val = v
                    best = pol.copy()
            except Exception:
                pass

    best = _norm(best)
    # final safety: remove NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    best = _norm(best)
    return best
