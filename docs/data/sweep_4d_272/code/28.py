import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(x, steps, tau0, tau1, lr0, lr1, rng=None):
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
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        mcur = C.max()
        sel = C > mcur - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def _void_for(others, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    sc = s.max(axis=1)
    bi = int(np.argmin(sc))
    return cand[bi]

def _align(X, Y, iters=3):
    # alternating Hungarian + Procrustes over O(4) incl. reflections
    R = np.eye(4)
    perm = np.arange(X.shape[0])
    for _ in range(iters):
        YR = Y @ R
        C = X @ YR.T
        try:
            _, col = linear_sum_assignment(-C)
        except Exception:
            break
        perm = col
        Yp = Y[perm]
        M = Yp.T @ X
        try:
            U, _, Vt = np.linalg.svd(M)
        except Exception:
            break
        R = U @ Vt
    Y_aligned = Y[perm] @ R
    Y_aligned = _norm(Y_aligned)
    return Y_aligned, perm, R

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _norm(tmp)
    best_val = _max_cos(x0)
    x_best = x0.copy()

    pop = [(best_val, x0.copy())]

    # Phase 1: diverse population by strong kicks + annealing (accept 0.905-0.93)
    sigmas = [0.25, 0.35, 0.20, 0.30, 0.15, 0.28, 0.22]
    for k, sig in enumerate(sigmas):
        if time.time() > t0 + 55:
            break
        y = x0 + sig * rng.standard_normal((272, 4))
        y = _norm(y)
        # topology change: kill 3-5 worst into voids before anneal
        for _ in range(3):
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            idx = int(G.argmax())
            ii, jj = divmod(idx, 272)
            # victim = endpoint with larger row-max
            ri = G[ii].max(); rj = G[jj].max()
            victim = ii if ri >= rj else jj
            others = np.delete(y, victim, axis=0)
            y[victim] = _void_for(others, rng, n_cand=4000)
        y = _anneal(y, steps=700, tau0=0.022, tau1=0.002, lr0=0.012, lr1=0.001)
        v = _max_cos(y)
        # keep even if mediocre, but avoid exact duplicates of record
        pop.append((v, y.copy()))
        if v < best_val:
            best_val, x_best = v, y.copy()

    # diversity filter: drop near-duplicates (mean best-match inner > 0.9995)
    # keep record + up to 7 most diverse: use greedy on score+distance
    # simple: sort by score, keep if far from kept ones (approx via sorted assignment-free distance)
    def _graph_sig(x, thr=0.895):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        deg = (G > thr).sum(axis=1)
        return np.sort(deg), G
    # retain all for crossover; just sort
    pop = sorted(pop, key=lambda t: t[0])

    # Phase 2: pairwise crossover with alignment + great-sphere splice
    nchild = 0
    # parent index pairs: combine best with others and diverse-diverse
    order = list(range(min(len(pop), 8)))
    pairs = []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            pairs.append((order[a], order[b]))
    rng2 = np.random.default_rng(1)
    rng2.shuffle(pairs)
    for (ai, bi) in pairs:
        if time.time() > deadline - 30:
            break
        if nchild >= 10:
            break
        # need current pop values (refresh sorted)
        Xa = pop[ai][1]; Xb = pop[bi][1]
        try:
            Yal, _, _ = _align(Xa, Xb, iters=3)
        except Exception:
            continue
        # several random great spheres per pair (2 tries)
        for trial in range(2):
            if time.time() > deadline - 30:
                break
            u = rng2.standard_normal(4); u /= np.linalg.norm(u)
            mid = Xa + Yal
            dots = mid @ u
            takeA = dots > 0
            # balance guard: if too lopsided, shift threshold to median
            if takeA.sum() < 60 or takeA.sum() > 212:
                thr = np.median(dots)
                takeA = dots > thr
            child = np.where(takeA[:, None], Xa, Yal)
            child = _norm(child)
            # repair closest cross-boundary pairs by void reinsertion
            for _ in range(3):
                G = child @ child.T
                np.fill_diagonal(G, -2.0)
                idx = int(G.argmax())
                ii, jj = divmod(idx, 272)
                if G[ii, jj] < 0.90:
                    break
                # kill one of the pair that came from different side if possible
                diff = takeA[ii] != takeA[jj]
                if diff:
                    victim = ii if rng2.random() < 0.5 else jj
                else:
                    # kill endpoint with larger row max
                    victim = ii if G[ii].max() >= G[jj].max() else jj
                others = np.delete(child, victim, axis=0)
                child[victim] = _void_for(others, rng2, n_cand=5000)
            child = _anneal(child, steps=600, tau0=0.015, tau1=0.0012, lr0=0.010, lr1=0.0008)
            v = _max_cos(child)
            if v < 0.925 or v < best_val + 0.005:
                # LP polish promising children briefly
                rem = min(4.0, max(0.5, deadline - 28 - time.time()))
                if rem > 0.6:
                    child = _lp_polish(child, seconds=rem, deadline=deadline - 22)
                    v = _max_cos(child)
            # graph-diversity check: accept if not near-duplicate
            dup = False
            for (ov, ox) in pop:
                try:
                    Yt, _, _ = _align(child, ox, iters=1)
                    dmean = float((child * Yt).sum(axis=1).mean())
                except Exception:
                    dmean = 0.0
                if dmean > 0.9995 and abs(v - ov) < 1e-6:
                    dup = True
                    break
            if not dup:
                pop.append((v, child.copy()))
                pop = sorted(pop, key=lambda t: t[0])
                # keep size bounded but retain diversity: keep best 4 + 4 most diverse (by degree hist)
                if len(pop) > 10:
                    # keep best 5 and 5 with largest degree-hist distance to best
                    bdeg, _ = _graph_sig(pop[0][1])
                    def dh(t):
                        d, _ = _graph_sig(t[1])
                        return float(np.abs(d.astype(float) - bdeg.astype(float)).sum())
                    rest = pop[1:]
                    rest_sorted = sorted(rest, key=dh, reverse=True)
                    keep_rest = []
                    # always keep 2nd best for elitism
                    keep_rest.append(pop[1])
                    for t in rest_sorted:
                        if len(keep_rest) >= 9:
                            break
                        if all(t[1] is not k[1] for k in keep_rest):
                            # avoid double-adding pop[1]
                            if np.abs(t[0] - pop[1][0]) < 1e-12 and t[1].shape == pop[1][1].shape and np.allclose(t[1], pop[1][1]):
                                continue
                            keep_rest.append(t)
                    pop = [pop[0]] + keep_rest[:9]
                    pop = sorted(pop, key=lambda t: t[0])
            if v < best_val:
                best_val, x_best = v, child.copy()
            nchild += 1

    # Final: polish overall best with remaining time (keep fallback to record)
    if time.time() < deadline - 2:
        rem = max(0.5, deadline - 3 - time.time())
        # polish top-2 candidates briefly and take better
        cands = [x_best, x0]
        # also consider pop best
        if pop:
            if _max_cos(pop[0][1]) < _max_cos(cands[0]) + 1e-12:
                cands[0] = pop[0][1]
        best_c = cands[0]
        best_c_v = _max_cos(best_c)
        if rem > 1:
            polished = _lp_polish(best_c, seconds=min(rem, 22.0), deadline=deadline - 1)
            pv = _max_cos(polished)
            if pv < best_c_v:
                best_c, best_c_v = polished, pv
        # safe fallback: never return worse than record by much; return best found
        if best_c_v <= _max_cos(x0) + 1e-9:
            x_best = best_c
        else:
            x_best = x0 if _max_cos(x0) < best_c_v else best_c
    x_best = _norm(x_best)
    # tiny shrink guard
    return x_best
