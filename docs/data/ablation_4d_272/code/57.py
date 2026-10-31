import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import os

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)

    def _normr(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def _max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def _lp_polish(x, deadline, maxpairs=5000):
        x = _normr(x.copy())
        n, d = x.shape
        try:
            best = _max_cos(x)
        except Exception:
            return x, 2.0
        radius = 1e-3
        iu0 = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            G = x @ x.T
            c_all = G[iu0]
            m = float(c_all.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c_all > thresh
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > maxpairs:
                idx = np.where(sel)[0]
                order = np.argpartition(-c_all[idx], maxpairs - 1)[:maxpairs]
                keep = np.zeros_like(sel, dtype=bool)
                keep[idx[order]] = True
                sel = keep
            c = c_all[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _normr(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, best

    # ---- load ----
    x_init = None
    for fn in ("best.npy", "record.npy"):
        try:
            if os.path.exists(fn):
                a = np.load(fn)
                a = np.asarray(a, dtype=np.float64)
                if a.shape == (272, 4):
                    # keep last loaded best as init; prefer best.npy
                    if x_init is None:
                        x_init = a
                    if fn == "best.npy":
                        x_init = a
        except Exception:
            pass
    if x_init is None:
        r = rng.standard_normal((272, 4))
        x_init = r / np.linalg.norm(r, axis=1, keepdims=True)
    x_init = _normr(np.asarray(x_init, dtype=np.float64))

    # baseline polish (short)
    x_best, best_val = _lp_polish(x_init, min(HARD_END - 95.0, time.time() + 12.0))
    # retain unmodified polished best as fallback
    fallback_x = x_best.copy()
    fallback_v = best_val

    n, d = x_best.shape
    N = n * d
    iu0_r, iu0_c = np.triu_indices(n, 1)

    def active_encode(x, tol):
        G = x @ x.T
        c = G[iu0_r, iu0_c]
        m = float(c.max())
        sel = np.where(c > m - tol)[0]
        I = iu0_r[sel]; J = iu0_c[sel]
        codes = I.astype(np.int64) * n + J.astype(np.int64)
        return set(codes.tolist()), m, c

    def jaccard(a, b):
        if len(a) == 0 and len(b) == 0:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - inter / union

    def null_basis(x, active_codes, k=15, drop_frac=0.0):
        codes = list(active_codes)
        p0 = len(codes)
        if drop_frac > 0 and p0 > 10:
            nd = int(p0 * drop_frac)
            if nd > 0:
                idx = rng.choice(p0, size=p0 - nd, replace=False)
                codes = [codes[i] for i in idx]
        p = len(codes)
        m_rows = p + n
        C = np.zeros((m_rows, N), dtype=np.float64)
        # active rows
        inv = 1.0 / np.sqrt(2.0)
        for r, code in enumerate(codes):
            i = int(code // n); j = int(code % n)
            C[r, i*d:(i+1)*d] = x[j] * inv
            C[r, j*d:(j+1)*d] = x[i] * inv
        # radial rows
        for i in range(n):
            C[p+i, i*d:(i+1)*d] = x[i]
        # M = C^T C
        try:
            M = C.T @ C
        except Exception:
            return None
        # eigh ascending
        try:
            w, V = np.linalg.eigh(M)
        except Exception:
            return None
        # V columns are eigenvectors; take k smallest
        k = min(k, N)
        Vk = V[:, :k]  # N x k
        return Vk

    # archive: list of dicts x, val, active
    archive = []
    base_active, _, _ = active_encode(x_best, 1e-7)
    archive.append({"x": x_best.copy(), "val": best_val, "active": set(base_active)})

    # helpers to evaluate walk candidates quickly
    def eval_candidates(x_cur, m_cur, Vk, n_trials=240, step=0.004):
        k = Vk.shape[1]
        coeffs = rng.standard_normal((n_trials, k))
        # normalize
        cn = np.linalg.norm(coeffs, axis=1, keepdims=True) + 1e-300
        coeffs = coeffs / cn
        # D matrix: N x trials? compute D = Vk @ coeffs.T => N x trials
        Dmat = Vk @ coeffs.T  # N x trials
        Dmat = Dmat.T.reshape(n_trials, n, d)
        # normalize each to max row norm 1
        rn = np.linalg.norm(Dmat, axis=2).max(axis=1, keepdims=True)  # trials x1? actually need per trial
        # rn shape (trials,1)? compute correctly
        # Dmat trials x n x d
        maxrn = np.sqrt((Dmat*Dmat).sum(axis=2)).max(axis=1)  # trials
        maxrn[maxrn < 1e-300] = 1.0
        Dmat = Dmat / maxrn[:, None, None]
        results = []
        for t in range(n_trials):
            if time.time() > HARD_END - 25.0:
                break
            D = Dmat[t]
            # two step sizes: alternate
            s = step * (0.5 if (t % 2 == 0) else 1.5)
            y = x_cur + s * D
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mv = float(G.max())
            results.append((mv, y, t))
        return results

    # main walk loop
    visited_start = set()
    it_outer = 0
    # precompute target slack info not needed separately; filter by max only + diversity
    while time.time() < HARD_END - 30.0 and it_outer < 40:
        it_outer += 1
        # choose start: diverse - prefer unvisited with good val but far Jaccard
        # find candidate start index
        best_idx = 0
        best_score = archive[0]["val"]
        for i, a in enumerate(archive):
            if a["val"] < best_score:
                best_score = a["val"]; best_idx = i
        # diversity: with prob alternate, pick farthest within tolerance
        cur_idx = best_idx
        if len(archive) > 1 and (it_outer % 3 == 0):
            # pick unvisited far from best
            ref = archive[best_idx]["active"]
            far_i = -1; far_d = -1
            for i, a in enumerate(archive):
                if i in visited_start:
                    continue
                if a["val"] > best_val + 8e-06:
                    continue
                jd = jaccard(ref, a["active"])
                if jd > far_d:
                    far_d = jd; far_i = i
            if far_i >= 0 and far_d > 0.003:
                cur_idx = far_i
            else:
                # pick any unvisited
                for i in range(len(archive)):
                    if i not in visited_start:
                        cur_idx = i
                        break
        if cur_idx in visited_start and len(archive) > 1:
            # still allow revisits of best occasionally
            pass
        visited_start.add(cur_idx)
        x_cur = archive[cur_idx]["x"]
        # recompute m_cur
        m_cur = _max_cos(x_cur)
        # build active set within 1e-9 (tight) for Jacobian
        tight_active, m_tmp, _ = active_encode(x_cur, 1e-9)
        if len(tight_active) < 300:
            # widen a bit if degenerate
            tight_active, _, _ = active_encode(x_cur, 1e-7)
        if len(tight_active) > 1600:
            # keep closest 1400
            G = x_cur @ x_cur.T
            c = G[iu0_r, iu0_c]
            m = float(c.max())
            order = np.argpartition(-c, 1400)[:1400]
            I = iu0_r[order]; J = iu0_c[order]
            tight_active = set((I.astype(np.int64)*n+J.astype(np.int64)).tolist())
        # vary k 5..20 and drop_frac
        k_choice = [12, 18, 8][it_outer % 3]
        drop_frac = 0.0 if (it_outer % 4 != 0) else 0.04
        Vk = null_basis(x_cur, tight_active, k=k_choice, drop_frac=drop_frac)
        if Vk is None:
            continue
        step_base = 0.0035 * (1.5 if (it_outer % 2 == 0) else 1.0)
        res = eval_candidates(x_cur, m_cur, Vk, n_trials=220, step=step_base)
        if len(res) == 0:
            continue
        res.sort(key=lambda t: t[0])
        # filter: keep those with mv <= m_cur + 1.2e-06 and diverse Jaccard
        cands = []
        for mv, y, _ in res[:30]:
            if mv > m_cur + 1.5e-06:
                continue
            # also require some topology change OR improvement
            act_y, _, _ = active_encode(y, 1e-7)
            jd = jaccard(archive[cur_idx]["active"], act_y)
            # keep if improved or jd sufficient
            if mv < m_cur - 1e-09 or jd > 0.004:
                cands.append((mv, y, act_y, jd))
            if len(cands) >= 6:
                break
        if len(cands) == 0:
            # take outright best if it improved
            mv0, y0, _ = res[0]
            if mv0 < m_cur:
                act_y, _, _ = active_encode(y0, 1e-7)
                cands.append((mv0, y0, act_y, 1.0))
        # LP-polish top 2 candidates (time permitting)
        polished_any = False
        for ci in range(min(2, len(cands))):
            if time.time() > HARD_END - 22.0:
                break
            mv, y, act_y, jd = cands[ci]
            # quick pre-check: if y much worse than best, skip polish unless diverse
            if mv > best_val + 6e-06 and jd < 0.02:
                continue
            y_p, v_p = _lp_polish(y, min(HARD_END - 12.0, time.time() + 7.0))
            act_p, _, _ = active_encode(y_p, 1e-7)
            # add to archive if diverse or better
            is_new = True
            for a in archive:
                if jaccard(act_p, a["active"]) < 0.002 and abs(a["val"] - v_p) < 5e-09:
                    is_new = False
                    break
            if is_new:
                archive.append({"x": y_p.copy(), "val": v_p, "active": set(act_p)})
                # cap archive
                if len(archive) > 24:
                    # drop worst, keep best
                    archive.sort(key=lambda a: a["val"])
                    # keep best 12 + 6 most diverse from best
                    best_a = archive[:12]
                    rest = archive[12:]
                    ref = best_a[0]["active"]
                    rest.sort(key=lambda a: jaccard(ref, a["active"]), reverse=True)
                    archive = best_a + rest[:6]
                    visited_start = set([i for i in visited_start if i < len(archive)])
            if v_p < best_val - 1e-12:
                best_val = v_p
                x_best = y_p.copy()
                polished_any = True
            elif v_p < best_val + 1e-09:
                pass
        # also if no candidate, add raw best trial as diverse point (unpolished) to continue walk
        if not polished_any and len(cands) > 0:
            # add raw walk point with best jd to allow chaining without polish
            # pick the most diverse among top
            cands.sort(key=lambda t: t[3], reverse=True)
            mv, y, act_y, jd = cands[0]
            if jd > 0.015 and mv <= best_val + 5e-06:
                # check not duplicate
                dup = False
                for a in archive:
                    if jaccard(act_y, a["active"]) < 0.002:
                        dup = True
                        break
                if not dup and len(archive) < 24:
                    archive.append({"x": y.copy(), "val": mv, "active": set(act_y)})

    # final: ensure fallback retained
    if fallback_v < best_val:
        best_val = fallback_v
        x_best = fallback_x.copy()
    # final short polish of best with remaining time
    if time.time() < HARD_END - 4.0:
        x_best, best_val = _lp_polish(x_best, HARD_END - 2.0)
        if fallback_v < best_val:
            x_best = fallback_x.copy()
    x_best = _normr(x_best)
    # tiny shrink not needed (unit rows already)
    return np.asarray(x_best, dtype=np.float64)
