import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n

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
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _find_void(x_fixed, rng, n_samples=8000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    return cand[idx].copy()

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

def _align(A, B, iters=3):
    # align B to A: permute + right-orthogonal (incl. reflections)
    n = A.shape[0]
    R = np.eye(4)
    perm = np.arange(n)
    for _ in range(iters):
        Brot = B @ R
        S = A @ Brot.T
        try:
            r, c = linear_sum_assignment(-S)
        except Exception:
            # greedy fallback
            c = np.argmax(S, axis=1)
            # make unique greedily
            used = np.zeros(n, dtype=bool)
            perm_g = np.empty(n, dtype=int)
            order = np.argsort(-S.max(axis=1))
            for i in order:
                j = int(np.argmax(np.where(used, -1e99, S[i])))
                perm_g[i] = j
                used[j] = True
            perm = perm_g
            Bp = B[perm]
            K = Bp.T @ A
            # need M = Bp^T A for R=U Vt
            Mm = Bp.T @ A
            try:
                U, s, Vt = np.linalg.svd(Mm)
                R = U @ Vt
            except Exception:
                pass
            continue
        p = np.empty(n, dtype=int)
        p[r] = c
        perm = p
        Bp = B[perm]
        Mm = Bp.T @ A
        try:
            U, s, Vt = np.linalg.svd(Mm)
            R = U @ Vt
        except Exception:
            pass
    return B[perm] @ R, perm, R

def _splice(A, Bal, u, rng):
    n = A.shape[0]
    s = A @ u
    t = Bal @ u
    maskA = s >= 0
    maskB = t < 0
    pts = []
    src = []  # 0 from A, 1 from B
    proj = []
    for i in np.where(maskA)[0]:
        pts.append(A[i]); src.append(0); proj.append(s[i])
    for j in np.where(maskB)[0]:
        pts.append(Bal[j]); src.append(1); proj.append(t[j])
    pts = np.array(pts) if len(pts) else np.zeros((0, 4))
    src = np.array(src, dtype=int) if len(src) else np.zeros((0,), dtype=int)
    proj = np.array(proj) if len(proj) else np.zeros((0,))
    m = pts.shape[0]
    if m > n:
        # drop boundary points first (smallest |proj|)
        order = np.argsort(np.abs(proj))
        drop = np.zeros(m, dtype=bool)
        drop[order[:m - n]] = True
        pts = pts[~drop]; src = src[~drop]; proj = proj[~drop]
    elif m < n:
        need = n - m
        rem_idx_A = np.where(~maskA)[0]
        rem_idx_B = np.where(~maskB)[0]
        cand_pts, cand_abs = [], []
        for i in rem_idx_A:
            cand_pts.append(A[i]); cand_abs.append(abs(float(s[i])))
        for j in rem_idx_B:
            cand_pts.append(Bal[j]); cand_abs.append(abs(float(t[j])))
        cand_pts = np.array(cand_pts)
        cand_abs = np.array(cand_abs)
        order = np.argsort(cand_abs)
        take = min(need, len(order))
        pts = np.concatenate([pts, cand_pts[order[:take]]], axis=0)
        # src for added: 2 = filler
        src = np.concatenate([src, np.full((take,), 2, dtype=int)])
        if pts.shape[0] < n:
            # pad with random (should not happen)
            extra = rng.standard_normal((n - pts.shape[0], 4))
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            pts = np.concatenate([pts, extra], axis=0)
    # shuffle slightly to avoid ordering bias? keep as is
    return pts[:n].copy()

def _contact_deg(x, gap=1e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mc = G.max()
    adj = G > mc - gap
    deg = adj.sum(axis=1)
    return np.sort(deg.astype(float)), float(mc)

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = _norm(flat[:need].reshape(272, 4))
    x_record = x0.copy()
    rec_val = _max_cos(x_record)
    x_best = x_record.copy()
    best_val = rec_val

    # --- population of distinct local optima ---
    pop = [x_record.copy()]
    pop_vals = [rec_val]
    sigmas = [0.35, 0.25, 0.45, 0.30, 0.50, 0.22, 0.40]
    # strong kicks + anneal even if 0.905-0.93
    for si, sig in enumerate(sigmas):
        if time.time() > t0 + 55:
            break
        x = x_record + sig * rng.standard_normal((272, 4))
        x = _norm(x)
        # topology change: kill 2-3 worst into voids before anneal
        for _ in range(2):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            idx = int(G.argmax())
            ii, jj = divmod(idx, 272)
            victim = ii if rng.random() < 0.5 else jj
            mask = np.ones(272, dtype=bool); mask[victim] = False
            x[victim] = _find_void(x[mask], rng, n_samples=6000)
        x = _anneal(x, 550, 0.025, 0.002, 0.010, 0.001)
        v = _max_cos(x)
        # retain diversity by contact graph distance
        deg, _ = _contact_deg(x)
        dup = False
        for y in pop:
            d2, _ = _contact_deg(y)
            if np.linalg.norm(deg - d2) < 1e-9 and abs(v - _max_cos(y)) < 5e-4:
                dup = True
                break
        if not dup or v < min(pop_vals):
            pop.append(x.copy()); pop_vals.append(v)
        if v < best_val and v < rec_val + 0.05:
            # keep best even if slightly worse than record? only update if better
            pass
        if v < best_val:
            best_val = v; x_best = x.copy()
    # ensure record in pop
    # --- recombination ---
    nchild = 0
    # precompute degs for diversity
    attempts = 0
    while time.time() < t0 + 132 and time.time() < deadline - 35 and attempts < 14:
        attempts += 1
        if len(pop) < 2:
            break
        # pick two distinct parents: bias to best + diverse
        # order pop by val
        order = np.argsort(np.array(pop_vals))
        # choose one among top-3 and one random diverse
        i1 = int(order[rng.integers(0, min(3, len(order)))])
        i2 = int(rng.integers(0, len(pop)))
        if i2 == i1:
            i2 = (i2 + 1) % len(pop)
        A = pop[i1]; Blo = pop[i2]
        try:
            Bal, perm, R = _align(A, Blo, iters=3)
        except Exception:
            continue
        # several random great spheres per pair
        for _ in range(1):
            if time.time() > t0 + 132:
                break
            u = rng.standard_normal((4,)); u /= np.linalg.norm(u)
            try:
                C = _splice(A, Bal, u, rng)
            except Exception:
                continue
            C = _norm(C)
            # repair closest cross-boundary pairs by void reinsertion
            try:
                G = C @ C.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                proj = np.abs(C @ u)
                # worst 2 near boundary
                cand_order = np.argsort(-pw)
                repaired = 0
                for wi in cand_order[:6]:
                    if repaired >= 2:
                        break
                    if pw[wi] > 0.88 and proj[wi] < 0.35:
                        mask = np.ones(272, dtype=bool); mask[wi] = False
                        C[wi] = _find_void(C[mask], rng, n_samples=6000)
                        repaired += 1
                        # update pw/proj cheaply
                        G = C @ C.T
                        np.fill_diagonal(G, -2.0)
                        pw = G.max(axis=1)
            except Exception:
                pass
            C = _anneal(C, 450, 0.012, 0.0008, 0.008, 0.0006)
            v = _max_cos(C)
            nchild += 1
            # diversity check vs pop
            deg, _ = _contact_deg(C)
            mind = 1e99
            for y in pop:
                d2, _ = _contact_deg(y)
                mind = min(mind, float(np.linalg.norm(deg - d2)))
            # keep if good or graph-diverse
            if len(pop) < 10 and (mind > 1e-9 or v < max(pop_vals)):
                pop.append(C.copy()); pop_vals.append(v)
            else:
                # replace worst if better
                wi = int(np.argmax(np.array(pop_vals)))
                if v < pop_vals[wi] and mind > 0:
                    pop[wi] = C.copy(); pop_vals[wi] = v
            if v < best_val:
                best_val = v; x_best = C.copy()
    # final low-temp refinement of best few
    for _ in range(2):
        if time.time() > deadline - 32:
            break
        x = _anneal(x_best.copy(), 300, 0.006, 0.0007, 0.004, 0.0004)
        v = _max_cos(x)
        if v < best_val:
            best_val = v; x_best = x.copy()
    # LP polish with remaining time
    remain = deadline - time.time() - 2.0
    if remain > 4:
        try:
            xp = _lp_polish(x_best, min(remain, 28.0), deadline - 1.0)
            vp = _max_cos(xp)
            if vp <= best_val:
                x_best = xp; best_val = vp
        except Exception:
            pass
    # safe fallback to record
    if not np.all(np.isfinite(x_best)) or x_best.shape != (272, 4):
        x_best = x_record.copy()
    else:
        # if we failed to beat record, return record (guaranteed valid)
        if _max_cos(_norm(x_best)) > rec_val + 1e-12:
            x_best = x_record.copy()
    x_best = _norm(x_best)
    return x_best
