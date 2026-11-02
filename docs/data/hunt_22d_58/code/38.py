import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _quench(X, steps, b0=25.0, b1=450.0, lr0=0.08, lr1=0.006, mu=0.82, clip=0.05, abort_thresh=0.09, abort_at=100):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if s == abort_at and m > abort_thresh:
            return None, 1e9
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        X = _norm(X)
    return X, _max_cos(X)

def _lp_polish(X, seconds, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 25:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _fingerprint(X, gap=5e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    A = (G > m - gap).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = np.sort(A.sum(axis=1))
    try:
        ev = np.linalg.eigvalsh(A)
        ev = np.sort(ev)
    except Exception:
        ev = np.zeros(X.shape[0])
    return deg, ev, m

def _fp_differs(fp1, fp2):
    d1, e1, _ = fp1
    d2, e2, _ = fp2
    dl1 = float(np.abs(d1 - d2).sum())
    el2 = float(np.linalg.norm(e1 - e2))
    return (dl1 > 4.0) or (el2 > 0.5)

def _align(B, A):
    # Hungarian on cosines then Procrustes B->A
    S = A @ B.T  # 58x58
    r, c = linear_sum_assignment(-S)
    Bp = np.zeros_like(B)
    Bp[r] = B[c]
    M = Bp.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
    except Exception:
        Q = np.eye(A.shape[1])
    return Bp @ Q

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 142.0
    rng = np.random.default_rng(0)
    n, d = 58, 22
    seeds = []
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape == (n, d):
                seeds.append(_norm(v))
        except Exception:
            pass
    base = seeds[0].copy() if len(seeds) > 0 else _norm(rng.standard_normal((n, d)))
    # build initial seed list: loaded + perturbed + random
    inits = []
    if len(seeds) > 0:
        for s in seeds:
            inits.append(s.copy())
    sigmas = [0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.18, 0.28]
    for sg in sigmas:
        inits.append(_norm(base + sg * rng.standard_normal((n, d))))
    # add random starts to reach ~20
    while len(inits) < 20:
        inits.append(_norm(rng.standard_normal((n, d))))
    inits = inits[:22]

    archive = []  # list of (X, val, fp)
    best_X = base.copy()
    best_val = _max_cos(best_X)

    # Phase 1: fast quenches
    for k, init in enumerate(inits):
        if time.time() > EXPLORE_END - 40:
            break
        # vary quench length slightly for diversity
        steps = 380 if k < 8 else 320
        Xq, vq = _quench(init.copy(), steps)
        if Xq is None:
            continue
        if vq > 0.09:
            continue
        # brief polish only for promising ones, very short
        if vq < 0.075 and time.time() < EXPLORE_END - 50:
            try:
                Xq = _lp_polish(Xq, 0.6, EXPLORE_END - 40)
                vq = _max_cos(Xq)
            except Exception:
                pass
        fp = _fingerprint(Xq)
        # keep archive of non-isomorphic basins: add if differs from all or improves
        distinct = True
        for (_, _, fpe) in [(a[0], a[1], a[2]) for a in archive]:
            if not _fp_differs(fp, fpe):
                distinct = False
                break
        if distinct or vq < best_val:
            archive.append((Xq, vq, fp))
        if vq < best_val:
            best_val = vq; best_X = Xq.copy()
        if len(archive) >= 24:
            break

    if len(archive) == 0:
        archive.append((best_X, best_val, _fingerprint(best_X)))

    # sort archive by val
    archive.sort(key=lambda t: t[1])
    # keep diverse + good: trim to 16
    if len(archive) > 16:
        archive = archive[:16]

    # Phase 2: recombination
    fracs = [0.25, 0.5, 0.75]
    pair_tried = set()
    rng2 = np.random.default_rng(1)
    # precompute order of pairs
    while time.time() < EXPLORE_END:
        if len(archive) < 2:
            break
        # pick two distinct parents, bias to best
        idx = np.arange(len(archive))
        # random pair
        a = int(rng2.integers(0, len(archive)))
        b = int(rng2.integers(0, len(archive)))
        if a == b:
            continue
        if (a, b) in pair_tried and len(pair_tried) > len(archive) * (len(archive) - 1) // 2:
            pair_tried.clear()
        if (a, b) in pair_tried or (b, a) in pair_tried:
            # still allow different frac
            pass
        Xa = archive[a][0]; Xb = archive[b][0]
        fpa = archive[a][2]; fpb = archive[b][2]
        try:
            Xb_al = _align(Xb, Xa)
        except Exception:
            continue
        for fr in fracs:
            if time.time() >= EXPLORE_END:
                break
            # spherical-ish interpolation: linear + normalize
            Xc = _norm((1.0 - fr) * Xa + fr * Xb_al)
            # small jitter to break symmetry
            Xc = _norm(Xc + 0.005 * rng2.standard_normal(Xc.shape))
            Xc, vc = _quench(Xc, 260)
            if Xc is None:
                continue
            if vc > 0.09:
                continue
            fpc = _fingerprint(Xc)
            diff_a = _fp_differs(fpc, fpa)
            diff_b = _fp_differs(fpc, fpb)
            improves = vc < min(archive[a][1], archive[b][1]) - 1e-12
            if (diff_a and diff_b) or improves or vc < best_val:
                # brief LP only if promising or distinct and good
                if vc < 0.075:
                    try:
                        Xc = _lp_polish(Xc, 0.5, EXPLORE_END)
                        vc = _max_cos(Xc)
                        fpc = _fingerprint(Xc)
                    except Exception:
                        pass
                archive.append((Xc, vc, fpc))
                if vc < best_val:
                    best_val = vc; best_X = Xc.copy()
            else:
                # even if not distinct, keep if global best
                if vc < best_val:
                    best_X = Xc.copy(); best_val = vc
                    archive.append((Xc, vc, fpc))
        pair_tried.add((a, b))
        # keep archive bounded, keep best 16
        archive.sort(key=lambda t: t[1])
        if len(archive) > 18:
            # keep best 10 + 8 diverse random from rest
            top = archive[:10]
            rest = archive[10:]
            keep = []
            for t in rest:
                ok = True
                for s in top + keep:
                    if not _fp_differs(t[2], s[2]):
                        ok = False
                        break
                if ok:
                    keep.append(t)
                if len(keep) >= 8:
                    break
            archive = top + keep
        if time.time() >= EXPLORE_END:
            break

    # Final: LP-polish single best
    archive.sort(key=lambda t: t[1])
    best_X = archive[0][0].copy()
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 2:
            best_X = _lp_polish(best_X, remain, DEADLINE)
    except Exception:
        pass
    best_X = _norm(best_X)
    return best_X
