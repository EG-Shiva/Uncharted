import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    x = x / n
    bad = ~np.all(np.isfinite(x), axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _gd(X0, steps=300, b0=60.0, b1=900.0, lr0=0.011, lr1=0.0025, mu=0.88, max_step=0.04):
    X = X0.copy()
    vel = np.zeros_like(X)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-12
        grad = W @ X
        d = np.sum(grad * X, axis=1, keepdims=True)
        gt = (grad - d * X) / rs
        vel = mu * vel - lr * gt
        vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, max_step / vn)
        vel = vel * sc
        X = X + vel
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _lp_limit(x, max_iters=15, r0=2e-3, rmax=0.02, t_end=None):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    for _ in range(max_iters):
        if t_end is not None and time.time() > t_end:
            break
        c_all = (x @ x.T)[iu0, iu1]
        m = float(np.max(c_all))
        thr = m - max(15.0 * radius, 2e-9)
        sel = c_all > thr
        if np.count_nonzero(sel) == 0:
            break
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            idx2 = idx[order]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx2] = True
        I, J = iu0[sel], iu1[sel]
        cs = c_all[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
            if radius < 1e-13:
                break
    return bx, best

def _lp_polish_timed(x, t_end):
    x = _normalize(x)
    best = _max_cos(x)
    bx = x.copy()
    n, d = x.shape
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 0.8 > t_end and radius < 4e-4:
            break
        c_all = (x @ x.T)[iu0, iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        if np.count_nonzero(sel) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:3000]
            idx2 = idx[order]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx2] = True
            sel = sel2
        I, J = iu0[sel], iu1[sel]
        cs = c_all[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx

def _rand_orth(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q

def _align(A, B, rng, iters=6, n_init=2):
    n = A.shape[0]
    best_score = -1e100
    best_perm = np.arange(n)
    best_Q = np.eye(4)
    inits = [np.eye(4)]
    for _ in range(n_init - 1):
        inits.append(_rand_orth(rng))
    for Q0 in inits:
        Q = Q0.copy()
        perm = np.arange(n)
        for _ in range(iters):
            BQ = B @ Q
            C = A @ BQ.T
            try:
                ri, ci = linear_sum_assignment(-C)
            except Exception:
                break
            perm = np.empty(n, dtype=int)
            perm[ri] = ci
            Bp = B[perm]
            M = Bp.T @ A
            try:
                U, _, Vt = np.linalg.svd(M)
            except Exception:
                break
            Q = U @ Vt
            sc = float(np.sum(C[ri, ci]))
            if sc > best_score:
                best_score = sc
                best_perm = perm.copy()
                best_Q = Q.copy()
    return best_perm, best_Q

def _splice(A, B2, u):
    n = A.shape[0]
    sA = A @ u
    sB = B2 @ u
    northA = sA > 0
    northB = sB > 0
    idxA = np.where(northA)[0]
    idxB = np.where(~northB)[0]
    child_pts = []
    src = []
    adot = []
    for i in idxA:
        child_pts.append(A[i]); src.append(0); adot.append(abs(float(sA[i])))
    for j in idxB:
        child_pts.append(B2[j]); src.append(1); adot.append(abs(float(sB[j])))
    k = len(child_pts)
    if k > n:
        order = np.argsort(np.array(adot))
        drop = k - n
        keep = np.ones(k, dtype=bool)
        keep[order[:drop]] = False
        child_pts = [child_pts[i] for i in range(k) if keep[i]]
        src = [src[i] for i in range(k) if keep[i]]
    elif k < n:
        need = n - k
        remA = np.where(~northA)[0]
        remB = np.where(northB)[0]
        cand = []
        for i in remA:
            cand.append((abs(float(sA[i])), A[i], 0))
        for j in remB:
            cand.append((abs(float(sB[j])), B2[j], 1))
        cand.sort(key=lambda t: t[0])
        for t in range(min(need, len(cand))):
            child_pts.append(cand[t][1]); src.append(cand[t][2])
        while len(child_pts) < n:
            child_pts.append(A[0]); src.append(0)
    C = np.stack(child_pts, axis=0)
    return _normalize(C), np.array(src, dtype=int)

def _void_point(others, rng, n_cand=5000):
    S = rng.standard_normal((n_cand, 4))
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    dots = S @ others.T
    md = np.max(dots, axis=1)
    bi = int(np.argmin(md))
    return S[bi].copy()

def _repair(child, src, rng, n_fix=3):
    n = child.shape[0]
    G = child @ child.T
    np.fill_diagonal(G, -2.0)
    pw = np.max(G, axis=1)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    order = np.argsort(-c)[:30]
    to_fix = []
    seen = set()
    for t in order:
        i = int(iu0[t]); j = int(iu1[t])
        cross = (src[i] != src[j]) if src is not None and len(src) == n else True
        for v in (i, j):
            if v in seen:
                continue
            if cross or len(to_fix) < n_fix:
                # prefer cross-boundary
                if cross:
                    to_fix.append(v); seen.add(v)
                    break
        if len(to_fix) >= n_fix:
            break
    if len(to_fix) == 0:
        to_fix = list(np.argsort(-pw)[:n_fix])
    # also add global worst if not included
    for v in list(np.argsort(-pw)[:2]):
        if v not in seen and len(to_fix) < n_fix + 1:
            to_fix.append(int(v)); seen.add(v)
    tmp = child.copy()
    for w in to_fix:
        mask = np.ones(n, dtype=bool); mask[w] = False
        others = tmp[mask]
        b = _void_point(others, rng, n_cand=5000)
        # tiny jitter to avoid exact ties
        b = b + 0.002 * rng.standard_normal(4)
        b = b / max(np.linalg.norm(b), 1e-12)
        tmp[w] = b
    return _normalize(tmp)

def _signature(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    thr = m - 2e-3
    B = (G > thr).astype(float)
    deg = np.sum(B, axis=1)
    deg = np.sort(deg)
    iu = np.triu_indices(X.shape[0], 1)
    c = (X @ X.T)[iu]
    top = np.sort(c)[-1500:]
    return deg, top

def _div_dist(s1, s2):
    d1, t1 = s1; d2, t2 = s2
    return float(np.linalg.norm(d1 - d2) + 0.2 * np.linalg.norm(t1 - t2))

def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_POP = t0 + 55.0
    T_CROSS = t0 + 152.0
    rng = np.random.default_rng(20261031)
    try:
        rec = np.load("best.npy")
    except Exception:
        rec = np.load("record.npy")
    rec = _normalize(np.asarray(rec, dtype=np.float64))
    if rec.shape != (272, 4):
        rec = _normalize(rng.standard_normal((272, 4)))
    rec_val = _max_cos(rec)
    try:
        rec_p, rec_pv = _lp_limit(rec, max_iters=12, r0=1e-3)
        if rec_pv < rec_val:
            rec, rec_val = rec_p, rec_pv
    except Exception:
        pass
    best = rec.copy()
    best_val = rec_val
    pops = [(best.copy(), best_val)]
    # population of distinct optima by strong kicks
    sigmas = [0.30, 0.45, 0.35, 0.55, 0.30, 0.50, 0.40]
    for k in range(7):
        if time.time() > T_POP:
            break
        Xp = best.copy()
        mode = k % 4
        if mode == 0:
            Xp = Xp + sigmas[k] * rng.standard_normal(Xp.shape)
        elif mode == 1:
            Xp = Xp + sigmas[k] * rng.standard_normal(Xp.shape)
            idx = rng.choice(272, size=25, replace=False)
            Xp[idx] = rng.standard_normal((25, 4))
        elif mode == 2:
            idx = rng.choice(272, size=40, replace=False)
            Xp[idx] = rng.standard_normal((40, 4))
            mask = np.ones(272, dtype=bool); mask[idx] = False
            Xp[mask] += 0.15 * rng.standard_normal((272 - 40, 4))
        else:
            # near-random start for diversity
            Xp = rng.standard_normal((272, 4))
        Xp = _normalize(Xp)
        steps = 300 if k < 5 else 220
        Xg = _gd(Xp, steps=steps, b0=50.0, b1=800.0, lr0=0.011, lr1=0.0025, mu=0.88, max_step=0.045)
        try:
            Xl, vl = _lp_limit(Xg, max_iters=10, r0=1.5e-3, t_end=T_POP + 2.0)
        except Exception:
            Xl, vl = Xg, _max_cos(Xg)
        # retain even if 0.905-0.93 (diversity)
        pops.append((Xl.copy(), float(vl)))
        if vl < best_val:
            best = Xl.copy(); best_val = float(vl)
            pops[0] = (best.copy(), best_val)
    # diversity filter: drop near-duplicates of record using signature distance
    try:
        sigs = [_signature(p[0]) for p in pops]
        keep = [0]
        for i in range(1, len(pops)):
            dup = False
            for j in keep:
                if _div_dist(sigs[i], sigs[j]) < 8.0:
                    # check value closeness too -> likely same basin
                    if abs(pops[i][1] - pops[j][1]) < 2e-4:
                        dup = True
                        break
            if not dup:
                keep.append(i)
            # always keep at least 5: if too few, keep anyway
        if len(keep) < 4:
            keep = list(range(len(pops)))
        pops = [pops[i] for i in keep]
    except Exception:
        pass
    # ensure at least 4 members (pad with perturbed if collapsed)
    while len(pops) < 4 and time.time() < T_CROSS:
        Xp = _normalize(best + 0.4 * rng.standard_normal(best.shape))
        Xg = _gd(Xp, steps=200)
        try:
            Xl, vl = _lp_limit(Xg, max_iters=8, r0=1.5e-3)
        except Exception:
            Xl, vl = Xg, _max_cos(Xg)
        pops.append((Xl, float(vl)))
    # crossover pairs: best with each other + diverse-diverse
    pairs = []
    for i in range(1, len(pops)):
        pairs.append((0, i))
    if len(pops) > 3:
        pairs.append((1, 2))
        if len(pops) > 4:
            pairs.append((3, 4 if len(pops) > 4 else 1))
            if len(pops) > 5:
                pairs.append((1, 4))
    # limit pairs to fit time
    offspring = []
    for (a, b) in pairs:
        if time.time() > T_CROSS:
            break
        if a >= len(pops) or b >= len(pops):
            continue
        A = pops[a][0].copy()
        B = pops[b][0].copy()
        try:
            perm, Q = _align(A, B, rng, iters=6, n_init=2)
        except Exception:
            perm = np.arange(272); Q = np.eye(4)
        B2 = B[perm] @ Q
        B2 = _normalize(B2)
        for trial in range(2):
            if time.time() > T_CROSS:
                break
            u = rng.standard_normal(4); u = u / max(np.linalg.norm(u), 1e-12)
            try:
                C, src = _splice(A, B2, u)
            except Exception:
                continue
            try:
                C = _repair(C, src, rng, n_fix=3)
            except Exception:
                pass
            Cg = _gd(C, steps=300, b0=60.0, b1=900.0, lr0=0.011, lr1=0.0025, mu=0.88, max_step=0.04)
            try:
                Cl, vl = _lp_limit(Cg, max_iters=14, r0=1.5e-3, t_end=T_CROSS + 1.0)
            except Exception:
                Cl, vl = Cg, _max_cos(Cg)
            offspring.append((Cl.copy(), float(vl)))
            if vl < best_val:
                best = Cl.copy(); best_val = float(vl)
                pops[0] = (best.copy(), best_val)
    # second generation: best + most diverse offspring
    if time.time() < T_CROSS - 8.0 and len(offspring) > 0:
        try:
            bsig = _signature(best)
            osigs = [_signature(o[0]) for o in offspring]
            dists = [_div_dist(bsig, s) for s in osigs]
            # most diverse among decent ones
            order = np.argsort(-np.array(dists))
            # pick top-2 diverse with val < 0.93
            cand_idx = [i for i in order if offspring[i][1] < 0.935][:2]
            for ci in cand_idx:
                if time.time() > T_CROSS:
                    break
                A = best.copy()
                B = offspring[ci][0].copy()
                try:
                    perm, Q = _align(A, B, rng, iters=6, n_init=2)
                except Exception:
                    perm = np.arange(272); Q = np.eye(4)
                B2 = _normalize(B[perm] @ Q)
                for trial in range(2):
                    if time.time() > T_CROSS:
                        break
                    u = rng.standard_normal(4); u = u / max(np.linalg.norm(u), 1e-12)
                    C, src = _splice(A, B2, u)
                    C = _repair(C, src, rng, n_fix=3)
                    Cg = _gd(C, steps=280)
                    Cl, vl = _lp_limit(Cg, max_iters=12, r0=1.5e-3, t_end=T_CROSS + 1.0)
                    if vl < best_val:
                        best = Cl.copy(); best_val = float(vl)
        except Exception:
            pass
    # final polish of safe best
    if time.time() < T_HARD - 3.0:
        try:
            best = _lp_polish_timed(best, T_HARD - 1.0)
        except Exception:
            pass
    best = _normalize(best)
    return best
