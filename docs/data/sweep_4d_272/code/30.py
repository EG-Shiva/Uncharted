import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _soft_run(y, betas, steps_list, lr=0.3, deadline=None):
    y = _normalize(y)
    for beta, steps in zip(betas, steps_list):
        for s in range(int(steps)):
            if deadline is not None and (s % 20 == 0) and time.time() > deadline:
                return _normalize(y)
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            grad = W @ y
            proj = (grad * y).sum(axis=1, keepdims=True)
            gt = grad - proj * y
            nrm = float(np.mean(np.linalg.norm(gt, axis=1)))
            if not np.isfinite(nrm) or nrm < 1e-12:
                break
            y = y - (lr / (nrm + 1e-12)) * gt
            y = _normalize(y)
    return _normalize(y)

def _lp_polish(x, deadline):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _align_procrustes_hungarian(A, B, iters=4):
    n = A.shape[0]
    Q = np.eye(4)
    perm = np.arange(n)
    for _ in range(iters):
        Bq = B @ Q
        S = A @ Bq.T
        try:
            r, c = linear_sum_assignment(-S)
        except Exception:
            break
        # r should be 0..n-1 in order; build perm array
        pi = np.empty(n, dtype=int)
        pi[r] = c
        Bperm = B[pi]
        H = Bperm.T @ A
        try:
            U, _, Vt = np.linalg.svd(H)
        except Exception:
            break
        Q = U @ Vt
        perm = pi
    B2 = B[perm] @ Q
    return B2, perm, Q

def _sig_sorted(x):
    iu = np.triu_indices(x.shape[0], 1)
    g = (x @ x.T)[iu]
    return np.sort(g)

def _best_void(others, rng, n_cand=3000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b

def _void_repair(x, rng, n_rep=3):
    x = _normalize(x)
    n = x.shape[0]
    for _ in range(n_rep):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        pw = G.max(axis=1)
        wi = int(np.argmax(pw))
        others = np.delete(x, wi, axis=0)
        b = _best_void(others, rng, n_cand=2500)
        x[wi] = b
    return _normalize(x)

def _splice(A, B2, u):
    n = A.shape[0]
    da = A @ u
    db = B2 @ u
    idxA_pos = np.where(da > 0)[0]
    idxB_neg = np.where(db <= 0)[0]
    C = []
    dots = []
    src = []  # 0 from A, 1 from B
    for i in idxA_pos:
        C.append(A[i]); dots.append(float(da[i])); src.append(0)
    for i in idxB_neg:
        C.append(B2[i]); dots.append(float(db[i])); src.append(1)
    C = np.array(C) if len(C) > 0 else np.zeros((0, 4))
    dots = np.array(dots)
    total = len(C)
    if total == n:
        return _normalize(C)
    elif total > n:
        # drop excess closest to boundary (smallest |dot|)
        ad = np.abs(dots)
        order = np.argsort(ad)  # smallest first -> drop first
        drop = total - n
        keep_mask = np.ones(total, dtype=bool)
        keep_mask[order[:drop]] = False
        C2 = C[keep_mask]
        return _normalize(C2)
    else:
        need = n - total
        # pool remaining
        idxA_neg = np.where(da <= 0)[0]
        idxB_pos = np.where(db > 0)[0]
        pool = []
        for i in idxA_neg:
            pool.append((abs(float(da[i])), A[i]))
        for i in idxB_pos:
            pool.append((abs(float(db[i])), B2[i]))
        pool.sort(key=lambda t: t[0])
        extra = np.array([p[1] for p in pool[:need]])
        if len(extra) < need:
            # fallback random (should not happen)
            pass
        C2 = np.vstack([C, extra]) if len(C) > 0 else extra
        return _normalize(C2)

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(20260707)
    try:
        x_rec = np.load("best.npy")
    except Exception:
        x_rec = np.load("record.npy")
    x_rec = _normalize(np.asarray(x_rec, dtype=np.float64))
    if x_rec.shape != (272, 4):
        x_rec = _normalize(rng.standard_normal((272, 4)))
    base_val = _max_cos(x_rec)
    best_x = x_rec.copy()
    best_val = base_val

    # --- Phase 1: diverse population by strong kicks + annealing ---
    sigmas = [0.45, 0.60, 0.35, 0.75, 0.50, 0.90, 0.40, 0.65]
    pop = []
    pop_sigs = []
    rec_sig = _sig_sorted(x_rec)
    POP_END = t0 + 42.0
    for si, sig in enumerate(sigmas):
        if time.time() > POP_END or time.time() > HARD_END - 120:
            break
        x0 = x_rec + sig * rng.standard_normal((272, 4))
        # also fully randomize a small block for stronger topology change
        k = 12 + 6 * (si % 3)
        sel = rng.choice(272, size=k, replace=False)
        x0[sel] = rng.standard_normal((k, 4)) * 1.0
        x0 = _normalize(x0)
        y = _soft_run(x0, betas=[30.0, 80.0, 160.0], steps_list=[70, 70, 70], lr=0.35, deadline=POP_END)
        v = _max_cos(y)
        s = _sig_sorted(y)
        # diversity check vs record and kept pop (permutation-invariant)
        d_rec = float(np.linalg.norm(s - rec_sig))
        dup = False
        if d_rec < 0.25:
            dup = True
        else:
            for ps in pop_sigs:
                if float(np.linalg.norm(s - ps)) < 0.35:
                    dup = True
                    break
        # keep even if "duplicate" when very good, else require diversity
        # we want 6-10 codes even if 0.905-0.93: keep all with v<0.935 unless near-dup
        if v > 0.94:
            continue
        if dup and len(pop) >= 3:
            # still keep one copy if it is better than existing? skip
            continue
        pop.append((v, y.copy()))
        pop_sigs.append(s)
        if v < best_val:
            best_val = v
            best_x = y.copy()
    # ensure at least a few parents: if pop too small, add perturbed copies without filter
    if len(pop) < 4:
        for _ in range(4 - len(pop)):
            if time.time() > POP_END:
                break
            x0 = _normalize(x_rec + 0.55 * rng.standard_normal((272, 4)))
            y = _soft_run(x0, betas=[40.0, 120.0], steps_list=[50, 50], lr=0.35, deadline=POP_END)
            pop.append((_max_cos(y), y))
    # sort pop by value
    pop.sort(key=lambda t: t[0])

    # --- Phase 2: pairwise alignment + great-sphere splicing ---
    parents = [x_rec] + [p[1] for p in pop[:7]]
    # limit parents to <=8
    parents = parents[:8]
    children = []
    SPLICE_END = t0 + 105.0
    # build pair list: record vs each + some diverse-diverse
    pairs = []
    for i in range(1, len(parents)):
        pairs.append((0, i))
    # add diverse-diverse pairs
    for i in range(1, len(parents)):
        for j in range(i + 1, len(parents)):
            if len(pairs) >= 9:
                break
            pairs.append((i, j))
        if len(pairs) >= 9:
            break
    rng.shuffle(pairs) if hasattr(rng, 'shuffle') else None
    n_sphere_per_pair = 2 if len(pairs) <= 6 else 1
    # extra spheres to reach ~12 children
    for (ai, bi) in pairs:
        if time.time() > SPLICE_END or time.time() > HARD_END - 70:
            break
        A = parents[ai]
        B = parents[bi]
        try:
            B2, _, _ = _align_procrustes_hungarian(A, B, iters=3)
        except Exception:
            B2 = B
        for sp in range(n_sphere_per_pair):
            if time.time() > SPLICE_END:
                break
            u = rng.standard_normal(4); u = u / np.linalg.norm(u)
            try:
                C0 = _splice(A, B2, u)
            except Exception:
                continue
            if C0.shape != (272, 4):
                continue
            # repair closest cross-boundary pairs by void reinsertion
            try:
                C1 = _void_repair(C0, rng, n_rep=3)
            except Exception:
                C1 = C0
            C2 = _soft_run(C1, betas=[60.0, 140.0], steps_list=[60, 60], lr=0.3, deadline=SPLICE_END)
            v = _max_cos(C2)
            children.append((v, C2.copy()))
            if v < best_val:
                best_val = v
                best_x = C2.copy()
        # second random sphere with opposite complement (swap roles) for diversity if time
        if time.time() < SPLICE_END - 10 and n_sphere_per_pair == 1 and len(children) < 14:
            try:
                u2 = rng.standard_normal(4); u2 = u2 / np.linalg.norm(u2)
                C0 = _splice(B2, A, u2)
                C1 = _void_repair(C0, rng, n_rep=2)
                C2 = _soft_run(C1, betas=[60.0, 140.0], steps_list=[50, 50], lr=0.3, deadline=SPLICE_END)
                v = _max_cos(C2)
                children.append((v, C2.copy()))
                if v < best_val:
                    best_val = v
                    best_x = C2.copy()
            except Exception:
                pass
    children.sort(key=lambda t: t[0])

    # --- Phase 3: select best + graph-diverse offspring for LP polish ---
    # diversity metric: sorted-gram distance to record
    scored = []
    for v, c in children:
        s = _sig_sorted(c)
        div = float(np.linalg.norm(s - rec_sig))
        scored.append((v, div, c))
    # rank: prefer low v, but boost high-diversity among similar v
    # take top 4 by v plus 2 most diverse with v < best+0.01
    cand_pool = []
    for v, div, c in sorted(scored, key=lambda t: t[0])[:4]:
        cand_pool.append((v, c))
    # add diverse
    for v, div, c in sorted(scored, key=lambda t: -t[1])[:6]:
        if v < base_val + 0.015 and len(cand_pool) < 6:
            if all(float(np.linalg.norm(_sig_sorted(c) - _sig_sorted(q))) > 0.2 for _, q in cand_pool):
                cand_pool.append((v, c))
    # always include record-safe fallback and current best
    cand_pool.append((base_val, x_rec.copy()))
    cand_pool.append((best_val, best_x.copy()))
    # dedup + sort
    cand_pool.sort(key=lambda t: t[0])
    # keep first 5 distinct
    uniq = []
    for v, c in cand_pool:
        ok = True
        for _, q in uniq:
            if float(np.linalg.norm(_sig_sorted(c) - _sig_sorted(q))) < 1e-6:
                ok = False
                break
        if ok:
            uniq.append((v, c))
        if len(uniq) >= 5:
            break

    polished_best_x = best_x.copy()
    polished_best_val = best_val
    # polish in order
    for v, c in uniq:
        if time.time() > HARD_END - 6:
            break
        # budget split
        remain = HARD_END - time.time()
        slots_left = len(uniq)
        budget = max(8.0, remain / max(1, slots_left) * 0.9)
        deadline = min(HARD_END, time.time() + budget)
        # short pre-anneal for non-record candidates to settle topology
        if v > base_val + 1e-9:
            try:
                c = _soft_run(c, betas=[200.0], steps_list=[40], lr=0.2, deadline=deadline - 1)
            except Exception:
                pass
        try:
            xp, vp = _lp_polish(c, deadline)
        except Exception:
            continue
        if vp < polished_best_val:
            polished_best_val = vp
            polished_best_x = xp.copy()
    # safe fallback: if nothing beat record, return record (LP polished if time allowed)
    final_x = polished_best_x if polished_best_val <= base_val + 1e-12 else x_rec.copy()
    # if we never improved, still return best found (could be record)
    if polished_best_val > base_val and base_val <= best_val:
        final_x = x_rec.copy()
    elif best_val < polished_best_val:
        final_x = best_x.copy()
        polished_best_val = best_val
    out = _normalize(final_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    # tiny shrink guard not needed; return strictly feasible
    return out
