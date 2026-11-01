import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _contacts(x, m=None, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    if m is None:
        m = float(g.max())
    mask = g >= m - tol
    cnt = mask.sum(axis=1)
    return m, cnt, g


def _biased_polish(x_init, seconds, r=None, eps=0.0, freeze_r=False,
                   r_keep=40, force_thresh=None, use_biased_accept=True):
    """Tangent trust-region LP with bias eps on pairs involving r.
    Minimise t s.t. lin(i,j) <= t - (eps if involves r else 0).
    If freeze_r, dx[r]==0.
    Returns (x_best, best_raw_max)."""
    x = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = x.shape
    m0 = _max_cos(x)
    # biased objective initial
    def raw_and_biased(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if r is None or eps == 0.0:
            return m, m, G
        # max over r-pairs
        mr = float(np.max(np.concatenate([G[r, :r], G[r, r+1:]]))) if n > 1 else -2
        b = max(m if True else m, mr + eps)  # m already includes mr, but bias it
        # careful: m = max(m_other, mr); b = max(m_other, mr+eps)
        # compute m_other
        # to avoid O(n^2) extra, approximate: if argmax involves r then m_other may be smaller
        # compute exactly:
        G2 = G.copy()
        # mask out r row/col to get m_other
        G2[r, :] = -2.0
        G2[:, r] = -2.0
        m_other = float(G2.max())
        b = max(m_other, mr + eps)
        return m, b, G
    _, best_bias, _ = raw_and_biased(x)
    best_raw = m0
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    # precompute for speed
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        # current raw max
        # use iu for max
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        if force_thresh is not None:
            thresh = m - force_thresh
        else:
            thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        I0 = iu0[sel]; J0 = iu1[sel]
        c0 = c_all[sel]
        is_r0 = (I0 == r) | (J0 == r) if r is not None else np.zeros(len(I0), dtype=bool)
        # ensure r nearest kept
        if r is not None and r_keep > 0:
            row = G[r].copy()
            row[r] = -2.0
            # top r_keep indices
            if r_keep >= n - 1:
                top = np.where(np.arange(n) != r)[0]
            else:
                top = np.argpartition(-row, r_keep)[:r_keep]
                # argpartition may include r; filter
                top = top[top != r]
                # if we lost one, add next best
                if len(top) < r_keep:
                    order = np.argsort(-row)
                    top = np.array([j for j in order if j != r][:r_keep])
            # add missing pairs
            existing = set()
            # build set of partners already present for r
            mask_r = is_r0
            have = set()
            for ii, jj in zip(I0[mask_r], J0[mask_r]):
                have.add(jj if ii == r else ii)
            need = [j for j in top if j not in have]
            if need:
                nadd = len(need)
                Iadd = np.empty(nadd, dtype=np.int64)
                Jadd = np.empty(nadd, dtype=np.int64)
                cadd = np.empty(nadd)
                for k, j in enumerate(need):
                    a, b = (r, j) if r < j else (j, r)
                    Iadd[k] = a; Jadd[k] = b
                    cadd[k] = G[a, b]
                I0 = np.concatenate([I0, Iadd])
                J0 = np.concatenate([J0, Jadd])
                c0 = np.concatenate([c0, cadd])
                is_r0 = np.concatenate([is_r0, np.ones(nadd, dtype=bool)])
        p = len(I0)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3200 and force_thresh is None:
            # keep tightest 3200 but always keep r pairs
            if r is not None:
                idx_nr = np.where(~is_r0)[0]
                idx_r = np.where(is_r0)[0]
                keep_nr = 3200 - len(idx_r)
                if keep_nr < 0:
                    keep_nr = 0
                if len(idx_nr) > keep_nr and keep_nr > 0:
                    order = np.argsort(c0[idx_nr])[-keep_nr:]
                    keep = np.concatenate([idx_r, idx_nr[order]])
                    I0, J0, c0, is_r0 = I0[keep], J0[keep], c0[keep], is_r0[keep]
                    p = len(I0)
                elif keep_nr == 0:
                    I0, J0, c0, is_r0 = I0[idx_r], J0[idx_r], c0[idx_r], is_r0[idx_r]
                    p = len(I0)
            else:
                order = np.argsort(c0)[-3200:]
                I0, J0, c0 = I0[order], J0[order], c0[order]
                is_r0 = is_r0[order]
                p = len(I0)
        nv = n * d + 1
        cols = np.concatenate([I0[:, None] * d + np.arange(d),
                               J0[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J0], x[I0], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        b_ub = -c0.copy()
        if r is not None and eps != 0.0:
            b_ub[is_r0] -= eps
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        if freeze_r and r is not None:
            bnds = [(-radius, radius)] * (n * d) + [(None, None)]
            for k in range(d):
                bnds[r * d + k] = (0.0, 0.0)
        else:
            bnds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bnds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        m_new, b_new, _ = raw_and_biased(y)
        cmp_new = b_new if (use_biased_accept and eps != 0.0) else m_new
        cmp_best = best_bias if (use_biased_accept and eps != 0.0) else best_raw
        if cmp_new < cmp_best - 1e-15:
            x = y
            best_raw = min(best_raw, m_new) if m_new < best_raw else best_raw
            # update best trackers: keep x as current; best_x is lowest raw?
            # for biased phase we want to follow biased objective, so update best_x to current
            # but also remember lowest raw
            if m_new < _max_cos(best_x):
                best_x = y.copy()
            else:
                # still move x forward (already), keep best_x as lowest raw
                pass
            best_bias = b_new if b_new < best_bias else best_bias
            # if raw improved, record
            radius = min(radius * 1.5, 0.02)
        else:
            # check raw improvement even if biased didn't (for eps=0 they coincide)
            if m_new < best_raw - 1e-15:
                x = y
                best_x = y.copy()
                best_raw = m_new
                _, best_bias, _ = raw_and_biased(x)
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
    # return current x (biased endpoint) and best raw; caller decides.
    # We return the current iterate x (which followed biased objective) for chaining,
    # but also best_x. Prefer returning x if its biased value is better.
    return x, best_raw


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(an)):
                    v = _max_cos(an)
                    if v < best_val:
                        best_val = v
                        best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    record_val = min(best_val, 0.7585616884050)

    def contact_count(X, tol=1e-9):
        m, cnt, G = _contacts(X, tol=tol)
        return m, cnt, G

    # identify rattler
    m, cnt, G = contact_count(best)
    r = int(np.argmin(cnt))
    # if no exact zero, take min
    overall_best = best.copy()
    overall_val = best_val

    balanced_best = None
    balanced_val = 1e9

    eps_cycle = [1e-4, 3e-4, 1e-3, 3e-3]
    # starting point for exploration
    cur = overall_best.copy()

    it = 0
    while True:
        rem = deadline - time.time()
        if rem < 12.0:
            break
        eps = eps_cycle[it % len(eps_cycle)]
        # re-identify rattler on cur every few iters
        if it % 4 == 0:
            mc, cntc, _ = contact_count(cur)
            r = int(np.argmin(cntc))
        # occasionally relocate rattler to void to help feasibility
        if it % 8 == 7:
            # void fill for r: sample candidates, minimise max dot to others
            mask = np.ones(n, dtype=bool); mask[r] = False
            rem_pts = cur[mask]
            cand = rng.normal(size=(4000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            s = cand @ rem_pts.T
            smax = s.max(axis=1)
            idx = int(np.argmin(smax))
            cur[r] = cand[idx]
            cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)

        # Phase A: freeze r, bias eps, others constrained near record
        tA = 3.0
        if rem < 30:
            tA = 1.5
        tA = min(tA, rem - 9.0)
        if tA < 0.8:
            break
        xa, _ = _biased_polish(cur, seconds=tA, r=r, eps=eps, freeze_r=True,
                               r_keep=40, force_thresh=None, use_biased_accept=True)
        # Phase B: unfreeze, include nearest 15 + all within 5e-3, standard polish
        tB = 2.0
        if rem < 30:
            tB = 1.0
        # B1: short biased-unfrozen with eps to settle? then standard with force_thresh=5e-3
        xb, _ = _biased_polish(xa, seconds=tB, r=r, eps=0.0, freeze_r=False,
                               r_keep=15, force_thresh=5e-3, use_biased_accept=False)
        # evaluate
        vb = _max_cos(xb)
        mb, cntb, _ = contact_count(xb)
        rc = int(cntb[r])
        # track overall
        if vb < overall_val:
            overall_val = vb
            overall_best = xb.copy()
        # track balanced: rattler >=3 contacts
        if rc >= 3 and vb < balanced_val:
            balanced_val = vb
            balanced_best = xb.copy()
        # choose next cur: accept if balanced even if worse (topology change)
        # strategy: if rc>=3, move cur to xb to explore new basin; else keep cur or move if improved
        if rc >= 3:
            # accept balanced as new cur even if vb worse than cur's value (up to 3e-3 slack)
            cur_val = _max_cos(cur)
            if vb < cur_val + 3e-3:
                cur = xb.copy()
            else:
                # still occasionally jump
                if rng.random() < 0.2:
                    cur = xb.copy()
        else:
            # no contacts: if improved, move; else small random nudge of r + neighbours
            cur_val = _max_cos(cur)
            if vb < cur_val:
                cur = xb.copy()
            else:
                # nudge rattler a bit and continue from cur with perturbation
                if it % 2 == 0:
                    cur2 = cur.copy()
                    cur2[r] += 0.05 * rng.normal(size=(d,))
                    # also shake 3 random points slightly
                    sel = rng.choice(n, 3, replace=False)
                    cur2[sel] += 0.02 * rng.normal(size=(3, d))
                    cur2 /= np.linalg.norm(cur2, axis=1, keepdims=True)
                    cur = cur2
        it += 1
        # if we beat record meaningfully, keep polishing overall
        if overall_val < record_val - 1e-7:
            # found improvement; focus polish on overall for remaining? continue to try more
            pass

    # final selection: prefer overall lowest; if balanced is within 1e-6 and overall is rattler-free? just take overall
    final = overall_best.copy()
    # if balanced found and overall still has rattler, try to polish balanced a bit more if time
    rem = deadline - time.time()
    if balanced_best is not None and rem > 6.0:
        # polish balanced
        mb, cntb, _ = contact_count(balanced_best)
        rb = int(np.argmin(cntb))
        tpol = min(4.0, rem - 3.0)
        xp, _ = _biased_polish(balanced_best, seconds=tpol, r=None, eps=0.0,
                               freeze_r=False, r_keep=0, force_thresh=None,
                               use_biased_accept=False)
        vp = _max_cos(xp)
        if vp < overall_val:
            overall_val = vp
            final = xp.copy()
    # final long polish of final
    rem = deadline - time.time()
    if rem > 1.5:
        final, _ = _biased_polish(final, seconds=max(0.5, rem - 0.7), r=None, eps=0.0,
                                  freeze_r=False, r_keep=0, force_thresh=None,
                                  use_biased_accept=False)
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final
