import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _contact_set(X, slack=1e-6):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0 = np.triu_indices(n, 1)
    mask = G[iu0] >= (m - slack)
    ii = iu0[0][mask]
    jj = iu0[1][mask]
    s = set((int(a) * n + int(b)) for a, b in zip(ii, jj))
    return s, m


def _jaccard(A, B):
    if not A and not B:
        return 1.0
    u = len(A | B)
    if u == 0:
        return 1.0
    return len(A & B) / u


def _lp_polish(x, deadline, cap=4000):
    x = _norm(x.copy())
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
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _tangent_bases(X):
    n = X.shape[0]
    T = np.empty((n, 4, 3), dtype=np.float64)
    for i in range(n):
        try:
            _, _, Vh = np.linalg.svd(X[i:i + 1, :], full_matrices=True)
            T[i] = Vh[1:, :].T
        except Exception:
            # fallback Gram-Schmidt
            x = X[i]
            k = int(np.argmin(np.abs(x)))
            e = np.zeros(4)
            e[k] = 1.0
            v0 = e - float(e @ x) * x
            v0 = v0 / (np.linalg.norm(v0) + 1e-300)
            # build two more via cross-like
            M = np.stack([x, v0], axis=0)
            # find null via qr
            Q, _ = np.linalg.qr(np.eye(4)[:, :3] - np.outer(x, x[:3]) if False else np.random.randn(4, 3))
            T[i] = Q[:, :3] if Q.shape == (4, 3) else np.zeros((4, 3))
    return T


def _build_A(X, T, I, J):
    n, d = X.shape
    N = n * 3
    m = len(I)
    A = np.zeros((m, N), dtype=np.float64)
    for k in range(m):
        i = int(I[k]); j = int(J[k])
        vi = X[j] @ T[i]  # (3,)
        vj = X[i] @ T[j]
        A[k, i * 3:(i + 1) * 3] = vi
        A[k, j * 3:(j + 1) * 3] = vj
    return A


def _apply_step(X, T, y, tmax):
    n = X.shape[0]
    Y = y.reshape(n, 3)
    DX = np.einsum('nij,nj->ni', T, Y)
    norms = np.linalg.norm(DX, axis=1)
    mn = float(norms.max())
    if mn < 1e-15:
        return X.copy()
    scale = tmax / mn
    Xn = _norm(X + DX * scale)
    return Xn


def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    WALK_END = t0 + 148.0
    n, d = 272, 4
    rng = np.random.RandomState(20261031)
    best_X = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = _norm(rng.randn(n, d))
        best_val = _max_cos(best_X)
    init_val = float(best_val)
    fallback_X = best_X.copy()
    fallback_val = float(best_val)

    # light initial polish of fallback copy? keep fallback unmodified per spec,
    # but also try a polished incumbent
    try:
        px, pv = _lp_polish(best_X.copy(), min(HARD_END - 20.0, t0 + 12.0), cap=4000)
        if pv < best_val and np.all(np.isfinite(px)):
            best_X = px
            best_val = pv
    except Exception:
        pass

    # archive by contact Jaccard
    archive = []
    try:
        cs0, _ = _contact_set(best_X, slack=1e-6)
    except Exception:
        cs0 = set()
    archive.append((best_X.copy(), float(best_val), set(cs0)))
    try:
        csf, _ = _contact_set(fallback_X, slack=1e-6)
    except Exception:
        csf = set()
    # ensure fallback in archive
    if _jaccard(cs0, csf) < 0.999:
        archive.append((fallback_X.copy(), float(fallback_val), set(csf)))

    iu0 = np.triu_indices(n, 1)
    NK = 12  # null dims to use

    it = 0
    while time.time() < WALK_END:
        it += 1
        if time.time() > HARD_END - 25.0:
            break
        # pick base: 50% best else random archive
        if len(archive) == 1 or rng.rand() < 0.5:
            base = best_X.copy()
            base_val = float(best_val)
        else:
            pick = int(rng.randint(len(archive)))
            base = archive[pick][0].copy()
            base_val = float(archive[pick][1])
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
        except Exception:
            continue
        c_all = G[iu0]
        # active set within 1e-9
        tol_active = 1e-9
        # if too few, widen slightly to get stable Jacobian
        sel_active = c_all >= m0 - tol_active
        cnt_active = int(np.count_nonzero(sel_active))
        if cnt_active < 200:
            sel_active = c_all >= m0 - 1e-7
            cnt_active = int(np.count_nonzero(sel_active))
        if cnt_active > 2500:
            idx = np.where(sel_active)[0]
            order = np.argpartition(-c_all[idx], 2500 - 1)[:2500]
            keep = np.zeros_like(sel_active, dtype=bool)
            keep[idx[order]] = True
            sel_active = keep
            cnt_active = 2500
        Ia = iu0[0][sel_active]
        Ja = iu0[1][sel_active]
        # tangent bases
        try:
            T = _tangent_bases(base)
        except Exception:
            continue
        try:
            A = _build_A(base, T, Ia, Ja)
        except Exception:
            continue
        # null / near-null directions via M = A^T A
        try:
            M = A.T @ A
            # symmetrize
            M = 0.5 * (M + M.T)
            w, V = np.linalg.eigh(M)
            # V columns, w ascending
            k = int(min(NK, V.shape[1]))
            V0 = V[:, :k]  # N x k
            sval = w[:k]
        except Exception:
            continue
        # candidate noncontacts with slack 1e-5 .. 1e-3
        try:
            slack = m0 - c_all
            cand_mask = (slack >= 1e-5) & (slack <= 1e-3)
            cand_idx = np.where(cand_mask)[0]
            # subsample to closest ~400
            if cand_idx.size > 600:
                order = np.argsort(slack[cand_idx])[:600]
                cand_idx = cand_idx[order]
            elif cand_idx.size == 0:
                # fallback: closest 300 below active
                order = np.argsort(slack)
                # skip actives
                filt = slack[order] > 1e-9
                cand_idx = order[filt][:300]
        except Exception:
            cand_idx = np.array([], dtype=int)

        # build trial directions
        trials = []
        # 2 random null combos
        for _ in range(2):
            coeff = rng.randn(V0.shape[1])
            coeff /= (np.linalg.norm(coeff) + 1e-300)
            y = V0 @ coeff
            trials.append(('rand', y, None))
        # targeted: project candidate rows onto nullspace
        n_target = min(4, int(cand_idx.size))
        if n_target > 0:
            # pick random subset of candidates (closest biased)
            pick_c = cand_idx[rng.choice(cand_idx.size, size=n_target, replace=False)] if cand_idx.size > n_target else cand_idx
            for ci in pick_c:
                ii = int(iu0[0][ci]); jj = int(iu0[1][ci])
                try:
                    b = np.zeros((n * 3,), dtype=np.float64)
                    b[ii * 3:(ii + 1) * 3] = base[jj] @ T[ii]
                    b[jj * 3:(jj + 1) * 3] = base[ii] @ T[jj]
                    # project onto nullspace
                    proj = V0 @ (V0.T @ b)
                    pn = float(np.linalg.norm(proj))
                    if pn < 1e-12 or not np.finite(pn):
                        continue
                    proj /= pn
                    trials.append(('target', proj, (ii, jj)))
                    # also opposite? increasing dot is +proj; keep both signs as separate trial
                    trials.append(('target_neg', -proj, (ii, jj)))
                except Exception:
                    continue
        if not trials:
            continue
        # limit trials per iteration
        if len(trials) > 6:
            # keep diverse: shuffle and cut
            rng.shuffle(trials)
            trials = trials[:6]
        for tname, ydir, info in trials:
            if time.time() > WALK_END:
                break
            yn = float(np.linalg.norm(ydir))
            if yn < 1e-14 or not np.finite(yn):
                continue
            ydir = ydir / yn
            # line search over tangent max-step
            if tname.startswith('target'):
                steps = [0.012, 0.006, 0.003, 0.020]
            else:
                steps = [0.008, 0.004, 0.015, 0.002]
            for tmax in steps:
                if time.time() > WALK_END:
                    break
                try:
                    Xt = _apply_step(base, T, ydir, tmax)
                    Gt = Xt @ Xt.T
                    np.fill_diagonal(Gt, -2.0)
                    mt = float(Gt.max())
                    # active-only max: check current contacts not increasing much
                    # gather active dots on Xt
                    act_dots = Gt[Ia, Ja]
                    m_act = float(act_dots.max()) if act_dots.size else mt
                except Exception:
                    continue
                # accept if actives not much up and overall not exploded
                if m_act > m0 + 3e-5:
                    continue
                if mt > m0 + 6e-4:
                    continue
                # we want topology change: new pair reaching max or Jaccard shift
                try:
                    cst, _ = _contact_set(Xt, slack=1e-6)
                    csb, _ = _contact_set(base, slack=1e-6)
                    jac = _jaccard(csb, cst)
                except Exception:
                    jac = 0.0
                # require some change unless random small step that already improved
                # polish it
                deadline = min(time.time() + 4.0, HARD_END - 5.0)
                if deadline <= time.time() + 0.5:
                    break
                try:
                    Xp, vp = _lp_polish(Xt, deadline, cap=4000)
                except Exception:
                    continue
                # archive by diversity
                try:
                    csp, _ = _contact_set(Xp, slack=1e-6)
                except Exception:
                    csp = set()
                # update best
                if vp < best_val - 1e-12 and np.all(np.isfinite(Xp)):
                    best_val = float(vp)
                    best_X = Xp.copy()
                    archive.append((Xp.copy(), float(vp), set(csp)))
                    if len(archive) > 24:
                        # drop worst non-best
                        # keep best first element? just trim worst
                        vals = [a[1] for a in archive]
                        # keep entries with smallest vals + most diverse? simple: keep 24 best
                        order = np.argsort(vals)[:24]
                        archive = [archive[i] for i in order]
                    break  # new best: go to next outer iteration
                else:
                    # add to archive if diverse (jac distance)
                    try:
                        min_jac = 1.0
                        for (_, _, s) in archive:
                            j = _jaccard(s, csp)
                            if j < min_jac:
                                min_jac = j
                        # diverse if max similarity < 0.97 and value not terrible
                        if min_jac < 0.97 and vp < m0 + 8e-4:
                            archive.append((Xp.copy(), float(vp), set(csp)))
                            if len(archive) > 24:
                                vals = [a[1] for a in archive]
                                order = np.argsort(vals)[:24]
                                archive = [archive[i] for i in order]
                    except Exception:
                        pass
                # only try first viable step per direction unless no change at all
                if jac < 0.99:
                    break
        # end trials
    # final: ensure best is at least fallback, final short polish of best if time
    try:
        if best_val > fallback_val and fallback_val < 2.0:
            best_X = fallback_X.copy()
            best_val = float(fallback_val)
    except Exception:
        pass
    # final polish with remaining time (up to HARD_END-2)
    try:
        if time.time() < HARD_END - 8.0:
            Xf, vf = _lp_polish(best_X.copy(), HARD_END - 2.0, cap=5000)
            if vf < best_val and np.all(np.isfinite(Xf)):
                best_X, best_val = Xf, vf
    except Exception:
        pass
    best_X = _norm(np.asarray(best_X, dtype=np.float64))
    return best_X
