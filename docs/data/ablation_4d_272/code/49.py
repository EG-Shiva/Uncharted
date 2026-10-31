import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts_set(x, slack=1e-6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    thr = m - slack
    I, J = np.where(np.triu(G > thr, 1))
    s = set((I.astype(np.int64) * n + J.astype(np.int64)).tolist())
    return s, m

def _jaccard_dist(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def _lp_polish(x, seconds, r0=2.5e-3, max_pairs=4500):
    t_start = time.time()
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while (time.time() - t_start) < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > max_pairs:
            part = np.argpartition(-c_all[sel_idx], max_pairs - 1)[:max_pairs]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_b[:, :d] = I[:, None] * d + ar[None, :]
        cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_b[:, 2 * d] = n * d
        cols = cols_b.ravel()
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        vals = vals_b.ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-13:
                break
        if (time.time() - t_start) >= seconds:
            break
    return x, best

def _lp_single_with_duals(x, radius=2.5e-3, max_pairs=4500):
    """One trust-region LP; return (I,J,c_sel,duals_abs or None, maxc)."""
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu0]
    m = float(c_all.max())
    thresh = m - max(20.0 * radius, 1e-9)
    sel_idx = np.where(c_all > thresh)[0]
    if sel_idx.size == 0:
        return None, None, None, None, m
    if sel_idx.size > max_pairs:
        part = np.argpartition(-c_all[sel_idx], max_pairs - 1)[:max_pairs]
        sel_idx = sel_idx[part]
    I = iu0[0][sel_idx]
    J = iu0[1][sel_idx]
    c_sel = c_all[sel_idx]
    p = sel_idx.size
    nv = n * d + 1
    ar = np.arange(d)
    cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
    cols_b[:, :d] = I[:, None] * d + ar[None, :]
    cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
    cols_b[:, 2 * d] = n * d
    cols = cols_b.ravel()
    vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
    vals_b[:, :d] = x[J]
    vals_b[:, d:2 * d] = x[I]
    vals_b[:, 2 * d] = -1.0
    vals = vals_b.ravel()
    rows = np.repeat(np.arange(p), 2 * d + 1)
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return I, J, c_sel, None, m
    if res.status != 0:
        return I, J, c_sel, None, m
    duals = None
    try:
        marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
        if marg is None:
            # try dict access
            marg = res.get("ineqlin", None)
            if isinstance(marg, dict):
                marg = marg.get("marginals", None)
            else:
                marg = getattr(marg, "marginals", None) if marg is not None else None
        if marg is not None:
            marg = np.asarray(marg, dtype=np.float64)
            if marg.size == p:
                duals = np.abs(marg)
    except Exception:
        duals = None
    return I, J, c_sel, duals, m

def _gd_forced(x, hI, hJ, steps, a0, a1, s0, s1, hinge_target, hinge_w, mu, deadline=None):
    n, d = x.shape
    x = _norm_rows(x.copy())
    vel = np.zeros_like(x)
    hI = np.asarray(hI, dtype=np.int64)
    hJ = np.asarray(hJ, dtype=np.int64)
    nh = hI.size
    for it in range(steps):
        if deadline is not None and (it % 20 == 0) and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # early exit if blown up too far (allow ~0.905, hard cap 0.92)
        if m > 0.92:
            break
        E = np.clip(alpha * (G - m), -50.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if S <= 0 or not np.isfinite(S):
            S = 1.0
            W[:] = 0.0
        grad = (W @ x) / S
        # hinge gradient: penalize dot > target (force apart to target)
        if nh > 0:
            dots = np.sum(x[hI] * x[hJ], axis=1)
            coeff = 2.0 * np.maximum(0.0, dots - hinge_target)
            # accumulate
            gh = np.zeros_like(x)
            # vectorized scatter
            np.add.at(gh, hI, coeff[:, None] * x[hJ])
            np.add.at(gh, hJ, coeff[:, None] * x[hI])
            grad = grad + hinge_w * gh
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm_rows(x)
    return x

def _anneal(x, steps, a0, a1, s0, s1, mu=0.83, deadline=None):
    n, d = x.shape
    x = _norm_rows(x.copy())
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    best_X = x.copy()
    for it in range(steps):
        if deadline is not None and it % 50 == 0 and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_X = x.copy()
        E = np.clip(alpha * (G - m), -50.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm_rows(x)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return x, fv
    return best_X, bv

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(987654321)
    n, d = 272, 4

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        cand.append(_norm_rows(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])

    # short baseline polish
    if HARD_END - time.time() > 25:
        bp, bv = _lp_polish(best, seconds=min(8.0, HARD_END - time.time() - 15.0))
        if bv < best_val:
            best, best_val = bp, bv
        else:
            # keep polished position even if equal improvement tiny? keep best pos
            if bv <= best_val + 1e-12:
                best = bp

    # archive: list of (x, val, contact_set)
    archive = []
    cs0, m0 = _contacts_set(best, slack=1e-6)
    archive.append({"x": best.copy(), "val": best_val, "contacts": cs0})

    arch_ptr = 0
    it = 0
    # surgery loop
    while time.time() < HARD_END - 7.0:
        remain = HARD_END - time.time()
        if remain < 9:
            break
        # choose base: round-robin over archive, bias to best
        if len(archive) == 1:
            base_entry = archive[0]
        else:
            # iterate surgery from diverse basins: cycle, but 50% from best
            if it % 3 == 0:
                # best
                # find best entry
                bidx = int(np.argmin([e["val"] for e in archive]))
                base_entry = archive[bidx]
            else:
                base_entry = archive[arch_ptr % len(archive)]
                arch_ptr += 1
        base_x = base_entry["x"]
        base_val = base_entry["val"]

        # LP duals on base
        radius = 2.5e-3
        # vary radius slightly to get different active sets
        if it % 4 == 1:
            radius = 1.2e-3
        elif it % 4 == 3:
            radius = 4.0e-3
        I, J, c_sel, duals, m = _lp_single_with_duals(_norm_rows(base_x), radius=radius)
        if I is None or I.size == 0:
            # fallback: random perturbation anneal
            start = _norm_rows(base_x + rng.standard_normal((n, d)) * 0.015)
            mid, mid_val = _anneal(start, steps=300, a0=40.0, a1=500.0, s0=0.04, s1=0.004, mu=0.83, deadline=HARD_END - 5.0)
            pol, pol_val = _lp_polish(mid, seconds=min(3.0, HARD_END - time.time() - 1.0))
            cand_val = pol_val
            cand_x = pol
        else:
            p = I.size
            # point stress
            if duals is not None and duals.size == p and np.any(duals > 0):
                w = duals
            else:
                # fallback: weight by closeness to max
                w = np.maximum(0.0, c_sel - (m - 5e-4)) + 1e-9
            # highest-degree points by stress
            stress = np.zeros(n)
            np.add.at(stress, I, w)
            np.add.at(stress, J, w)
            top_pts = np.argsort(-stress)[:6]
            # rank edges
            order = np.argsort(-w)
            k = int(rng.integers(5, 16))  # 5..15
            k = min(k, p)
            sel_edges = order[:k].copy()
            # ensure coverage of top stressed points: if a top point not incident, swap in its best edge
            inc = set(I[sel_edges].tolist()) | set(J[sel_edges].tolist())
            for tp in top_pts[:3]:
                if tp not in inc:
                    # find best edge incident to tp
                    mask = (I == tp) | (J == tp)
                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        # pick highest w among them
                        b = idxs[int(np.argmax(w[idxs]))]
                        if b not in sel_edges:
                            # replace weakest
                            sel_edges[-1] = b
                            inc.add(tp)
                    # only one replacement to keep k stable? allow up to 2
                    if len(inc) >= k + 2:
                        break
            hI = I[sel_edges]
            hJ = J[sel_edges]

            # small random jitter before forcing to break exact symmetry
            jitter = 0.0015 + 0.001 * rng.random()
            start = _norm_rows(base_x + rng.standard_normal((n, d)) * jitter)

            # Phase 1: forced hinge move (allow rise to ~0.905)
            # hinge target 0.885, weight ~ 25-45
            hinge_w = float(25.0 + 20.0 * rng.random())
            # moderate softmax during forcing so others still repel
            fa0, fa1 = 30.0, 150.0
            if it % 2 == 1:
                fa0, fa1 = 20.0, 80.0
            forced = _gd_forced(start, hI, hJ, steps=260,
                                a0=fa0, a1=fa1, s0=0.05, s1=0.012,
                                hinge_target=0.885, hinge_w=hinge_w,
                                mu=0.80, deadline=HARD_END - 5.5)
            # Phase 2: re-anneal at moderate temperature without hinge
            # remove hinge, anneal
            mid, mid_val = _anneal(forced, steps=380, a0=35.0, a1=550.0,
                                   s0=0.035, s1=0.004, mu=0.83,
                                   deadline=HARD_END - 4.5)
            # Phase 3: LP polish
            pol_sec = min(4.0, HARD_END - time.time() - 1.2)
            if pol_sec < 1.0:
                pol_sec = max(0.8, HARD_END - time.time() - 0.6)
            pol, pol_val = _lp_polish(mid, seconds=float(pol_sec))
            cand_x = pol
            cand_val = pol_val

        # evaluate candidate
        cand_cs, cand_m = _contacts_set(cand_x, slack=1e-6)
        # global best update
        if cand_val < best_val - 1e-15:
            best = cand_x.copy()
            best_val = cand_val
            # add to archive (keep)
            archive.append({"x": cand_x.copy(), "val": cand_val, "contacts": cand_cs})
            # cap archive
            if len(archive) > 7:
                # remove worst non-best
                bidx = int(np.argmin([e["val"] for e in archive]))
                # find worst index != bidx
                worst = int(np.argmax([e["val"] for e in archive]))
                if worst != bidx:
                    archive.pop(worst)
                else:
                    # remove second worst
                    vals_a = [e["val"] for e in archive]
                    vals_a[worst] = -1
                    worst2 = int(np.argmax(vals_a))
                    archive.pop(worst2)
        else:
            # accept into archive if diverse (Jaccard >0.15) even if up to 2e-4 worse
            if cand_val <= best_val + 2e-4:
                # compute min Jaccard distance to archive
                min_d = 1.0
                for e in archive:
                    dd = _jaccard_dist(cand_cs, e["contacts"])
                    if dd < min_d:
                        min_d = dd
                if min_d > 0.15:
                    archive.append({"x": cand_x.copy(), "val": cand_val, "contacts": cand_cs})
                    if len(archive) > 7:
                        # evict worst (but keep global best entry)
                        # locate entry with best val
                        bidx = int(np.argmin([e["val"] for e in archive]))
                        # worst among rest
                        rest_vals = [(e["val"] if i != bidx else -1) for i, e in enumerate(archive)]
                        worst = int(np.argmax(rest_vals))
                        archive.pop(worst)
                # else: check if candidate improves its nearest basin? replace nearest if better
                else:
                    # find nearest
                    best_j = -1
                    best_jd = 2.0
                    for idx_e, e in enumerate(archive):
                        dd = _jaccard_dist(cand_cs, e["contacts"])
                        if dd < best_jd:
                            best_jd = dd
                            best_j = idx_e
                    if best_j >= 0 and cand_val < archive[best_j]["val"] - 1e-12:
                        archive[best_j] = {"x": cand_x.copy(), "val": cand_val, "contacts": cand_cs}
            else:
                # much worse: only keep if extremely diverse and archive has room? skip
                pass
        it += 1
        if it > 40:
            break

    # final polish of global best
    if HARD_END - time.time() > 3.0:
        try:
            pf, pv = _lp_polish(best, seconds=min(2.5, HARD_END - time.time() - 0.5), r0=8e-4)
            if pv <= best_val:
                best, best_val = pf, pv
        except Exception:
            pass
    best = _norm_rows(best)
    best[~np.isfinite(best)] = 0.25
    best = _norm_rows(best)
    return best
