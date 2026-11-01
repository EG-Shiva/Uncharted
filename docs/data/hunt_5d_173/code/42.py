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

def _contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G >= m - tol)
    cnt = C.sum(axis=1)
    return m, cnt, G

def _smooth_frozen(X0, r, to, tr, steps, beta, lr):
    X = X0.copy()
    orig = X0[r].copy()
    n, d = X.shape
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        E = G - to
        E[r, :] = G[r, :] - tr
        E[:, r] = G[:, r] - tr
        B = np.clip(beta * E, -20.0, 5.0)
        W = np.exp(B)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        Gt[r] = 0.0
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        X = X - lr * (Gt / np.maximum(1.0, gn * 0.5))
        X = _norm(X)
        X[r] = orig
    return X

def _frozen_lp_minimize(X0, r, target_other, seconds, cap=2500):
    X = _norm(X0.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    # current r max
    G0 = X @ X.T
    mask0 = np.ones(n, dtype=bool); mask0[r] = False
    best_r = float(G0[r][mask0].max())
    radius = 1e-3
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        cfull = G[iu0, iu1]
        involves = (iu0 == r) | (iu1 == r)
        sel_o = (~involves) & (cfull > target_other - max(20.0 * radius, 1e-9))
        if int(sel_o.sum()) > cap:
            idx = np.where(sel_o)[0]
            order = np.argsort(cfull[idx])[-cap:]
            m2 = np.zeros_like(sel_o); m2[idx[order]] = True
            sel_o = m2
        Io = iu0[sel_o]; Jo = iu1[sel_o]; co = cfull[sel_o]
        Jr = np.where(np.arange(n) != r)[0]
        cr = G[r, Jr]
        p_r = len(Jr); p_o = len(Io); p = p_r + p_o
        if p == 0:
            break
        nv = n * d + 1
        # build A_ub
        # r rows
        Ir_rep = np.full(p_r, r)
        # for r rows: cols for I=r and J=Jr
        # other rows: I=Io,J=Jo
        # assemble per-row cols/vals via vectorized concat
        # r part
        r_cols_I = (Ir_rep[:, None] * d + np.arange(d)).ravel()
        r_cols_J = (Jr[:, None] * d + np.arange(d)).ravel()
        r_cols_T = np.full(p_r, nv - 1)
        # interleave per row: need rows repeat
        # Build coo lists
        rows_r = np.repeat(np.arange(p_r), 2 * d + 1)
        cols_r = np.empty(p_r * (2 * d + 1), dtype=int)
        vals_r = np.empty(p_r * (2 * d + 1))
        # fill row by row vectorized using reshape
        # create blocks: [X[Jr] (p_r,d), X[r] tiled (p_r,d), -1]
        blk_r = np.concatenate([X[Jr], np.tile(X[r], (p_r, 1)), -np.ones((p_r, 1))], axis=1)
        colblk_r = np.concatenate([Ir_rep[:, None] * d + np.arange(d),
                                   Jr[:, None] * d + np.arange(d),
                                   np.full((p_r, 1), nv - 1)], axis=1)
        cols_r = colblk_r.ravel()
        vals_r = blk_r.ravel()
        if p_o > 0:
            rows_o = np.repeat(np.arange(p_r, p), 2 * d)
            colblk_o = np.concatenate([Io[:, None] * d + np.arange(d),
                                       Jo[:, None] * d + np.arange(d)], axis=1)
            blk_o = np.concatenate([X[Jo], X[Io]], axis=1)
            cols_o = colblk_o.ravel()
            vals_o = blk_o.ravel()
            rows = np.concatenate([rows_r, rows_o])
            cols = np.concatenate([cols_r, cols_o])
            vals = np.concatenate([vals_r, vals_o])
        else:
            rows, cols, vals = rows_r, cols_r, vals_r
        b_ub = np.concatenate([-cr, target_other - co] if p_o > 0 else [-cr])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        # freeze r: bounds zero
        for k in range(d):
            bounds[r * d + k] = (0.0, 0.0)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        y = _norm(X + dx)
        y[r] = X[r]  # exact freeze
        Gy = y @ y.T
        rmax_new = float(Gy[r][mask0].max())
        # other max (excluding r)
        Gy2 = Gy.copy()
        # mask out r row/col for other max: set to -2
        Gy2[r, :] = -2.0; Gy2[:, r] = -2.0
        np.fill_diagonal(Gy2, -2.0)
        omax_new = float(Gy2.max())
        # accept if r improves and others stay within cap (+ small slack)
        if rmax_new < best_r - 1e-12 and omax_new <= target_other + 1e-7:
            X = y
            best_r = rmax_new
            radius = min(radius * 1.5, 0.02)
        elif rmax_new < best_r - 1e-12 and omax_new <= target_other + 20 * radius + 1e-9:
            # allow tiny violation early
            X = y
            best_r = rmax_new
            radius = min(radius * 1.2, 0.02)
        else:
            radius *= 0.5
    return X

def _lp_focused(X0, r, seconds, cap=3500):
    X = _norm(X0.copy())
    n, d = X.shape
    best = _max_cos(X)
    iu0, iu1 = np.triu_indices(n, 1)
    radius = 1e-3
    start = time.time()
    while time.time() - start < seconds and radius > 1e-14:
        G = X @ X.T
        cf = G[iu0, iu1]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 5e-3)
        sel = cf > thr
        # ensure r nearest 15 included
        Gr = G[r].copy()
        Gr[r] = -2.0
        top15 = np.argsort(-Gr)[:15]
        # find pair indices for (r, j)
        # build set of flat triu positions for those pairs
        for j in top15:
            a = r if r < j else j
            b = j if r < j else r
            # locate: need index; use search (n=173 small, brute)
            # instead force include by expanding sel: find where (iu0==a)&(iu1==b)
            # do via loop (15 * 15k = fine)
            sel = sel | ((iu0 == a) & (iu1 == b))
        if int(sel.sum()) > cap:
            idx = np.where(sel)[0]
            # always keep r pairs: prioritize them
            isr = ((iu0[idx] == r) | (iu1[idx] == r))
            # keep all r pairs in sel, cap others
            ridx = idx[isr]
            oidx = idx[~isr]
            keep_o = cap - len(ridx)
            if keep_o < 0:
                # keep largest r pairs only
                order = np.argsort(cf[ridx])[-cap:]
                sel = np.zeros_like(sel)
                sel[ridx[order]] = True
            else:
                if len(oidx) > keep_o and keep_o >= 0:
                    order = np.argsort(cf[oidx])[-keep_o:] if keep_o > 0 else np.array([], dtype=int)
                    sel = np.zeros_like(sel)
                    sel[ridx] = True
                    if len(order) > 0:
                        sel[oidx[order]] = True
                # else keep all
        I = iu0[sel]; J = iu1[sel]; c = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), nv - 1)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def _find_void(Xr, rng, ns=12000):
    C = rng.normal(size=(ns, 5))
    C = _norm(C)
    D = C @ Xr.T
    m = D.max(axis=1)
    return C[int(np.argmin(m))]

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(7)
    n, d = 173, 5
    best_g = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = _norm(a)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best_g = an.copy()
        except Exception:
            pass
    if best_g is None:
        best_g = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best_g)
    cur = best_g.copy()
    cur_val = best_val
    # identify rattler
    def rattler_of(X):
        m, cnt, G = _contact_counts(X, tol=1e-9)
        return int(np.argmin(cnt)), m, cnt, G
    r, _, _, _ = rattler_of(cur)
    eps_list = [1e-4, 3e-4, 1e-3, 3e-3]
    # Phase A: targeted feasibility cycles (~95s)
    phaseA_end = t0 + 105.0
    cyc = 0
    while time.time() < min(phaseA_end, deadline - 30.0):
        cyc += 1
        for eps in eps_list:
            if time.time() > min(phaseA_end, deadline - 30.0):
                break
            # re-identify rattler every eps step if current r active
            _, cnt_tmp, _ = _contact_counts(cur, tol=1e-9)[1], None, None
            # (recompute properly)
            mtmp, cnttmp, _ = _contact_counts(cur, tol=1e-9)
            if cnttmp[r] >= 3:
                r = int(np.argmin(cnttmp))
            target_other = best_val
            target_r = best_val - eps
            # smooth frozen push
            try:
                cur_s = _smooth_frozen(cur, r, target_other, target_r, steps=220, beta=70.0, lr=0.02)
            except Exception:
                cur_s = cur.copy()
            rem = deadline - time.time()
            if rem < 12:
                break
            # frozen LP
            try:
                cur_f = _frozen_lp_minimize(cur_s, r, target_other, seconds=0.8)
            except Exception:
                cur_f = cur_s
            if time.time() > min(phaseA_end, deadline - 28.0):
                break
            # unfreeze focused polish
            try:
                cur_p, v_p = _lp_focused(cur_f, r, seconds=1.2)
            except Exception:
                cur_p, v_p = cur_f, _max_cos(cur_f)
            # count r contacts at global max
            Gp = cur_p @ cur_p.T
            np.fill_diagonal(Gp, -2.0)
            mp = float(Gp.max())
            row = Gp[r].copy()
            cnt_r_7 = int((row >= mp - 1e-7).sum())
            cnt_r_9 = int((row >= mp - 1e-9).sum())
            active = (cnt_r_7 >= 3)
            if active:
                if v_p < best_val - 1e-12:
                    best_g, best_val = cur_p.copy(), v_p
                    cur, cur_val = cur_p.copy(), v_p
                    r, _, _, _ = rattler_of(cur)
                elif v_p <= best_val + 5e-3:
                    # accept worse basin with active rattler
                    cur, cur_val = cur_p.copy(), v_p
                    # pick new rattler (lowest contacts)
                    m2, cnt2, _ = _contact_counts(cur, tol=1e-9)
                    r = int(np.argmin(cnt2))
                # else too worse: keep cur, perturb slightly
                else:
                    # shake cur a bit around r neighbours
                    cur = _norm(cur + 0.01 * rng.normal(size=(n, d)))
            else:
                if v_p < best_val - 1e-12:
                    best_g, best_val = cur_p.copy(), v_p
                    cur, cur_val = cur_p.copy(), v_p
                    r, _, _, _ = rattler_of(cur)
                else:
                    # no activation and no improvement: small diversification of cur
                    # move r toward largest void (keep others) to try new topology
                    if cyc % 2 == 0:
                        try:
                            mask = np.ones(n, dtype=bool); mask[r] = False
                            vd = _find_void(cur[mask], rng, ns=8000)
                            cur[r] = vd
                            cur = _norm(cur)
                            cur_val = _max_cos(cur)
                        except Exception:
                            pass
        # occasional global re-anchor to best if cur drifted too far
        if cur_val > best_val + 5e-3:
            cur = best_g.copy(); cur_val = best_val
            r, _, _, _ = rattler_of(cur)
            # kick a low-contact point to void
            try:
                m2, cnt2, _ = _contact_counts(cur, tol=1e-7)
                rl = int(np.argmin(cnt2))
                mask = np.ones(n, dtype=bool); mask[rl] = False
                cur[rl] = _find_void(cur[mask], rng, ns=8000)
                cur = _norm(cur)
            except Exception:
                pass
    # Phase B: basin-hop from cur/best with void moves + focused polish (remaining time)
    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        trial += 1
        # base: mostly cur, sometimes best_g
        base = cur if (trial % 3 != 0) else best_g
        m_b, cnt_b, Gb = _contact_counts(base, tol=1e-7)
        order_asc = np.argsort(cnt_b)
        y = base.copy()
        mode = trial % 4
        if mode == 0:
            victim = int(order_asc[0])
            mask = np.ones(n, dtype=bool); mask[victim] = False
            y[victim] = _find_void(y[mask], rng, ns=15000)
        elif mode == 1:
            victim = int(order_asc[int(rng.integers(0, 5))])
            mask = np.ones(n, dtype=bool); mask[victim] = False
            y[victim] = _find_void(y[mask], rng, ns=15000)
            y[victim] += 0.003 * rng.normal(size=(d,))
        elif mode == 2:
            k = 3
            victims = order_asc[:k].copy()
            extra = rng.choice(n, 2, replace=False)
            victims = np.unique(np.concatenate([victims, extra]))
            y[victims] += 0.05 * rng.normal(size=(len(victims), d))
        else:
            y += 0.015 * rng.normal(size=(n, d))
        y = _norm(y)
        # quick smooth (unfrozen, simple repulsion toward record)
        # use frozen-style smooth with r = rattler but unfrozen: just do few gradient steps via _smooth_frozen on random r? skip, use focused LP directly if promising
        # light LP polish focused on its own rattler
        try:
            mr, cntr, _ = rattler_of(y)
        except Exception:
            mr = r
        pt = min(1.4, rem - 8.0)
        if pt < 0.4:
            break
        try:
            y2, v2 = _lp_focused(y, mr, seconds=pt)
        except Exception:
            continue
        # evaluate rattler activation
        G2 = y2 @ y2.T
        np.fill_diagonal(G2, -2.0)
        m2 = float(G2.max())
        row2 = G2[mr].copy()
        act2 = int((row2 >= m2 - 1e-7).sum()) >= 3
        if v2 < best_val - 1e-12:
            best_g, best_val = y2.copy(), v2
            cur, cur_val = y2.copy(), v2
            r = int(np.argmin((_contact_counts(cur, tol=1e-9)[1]))
            )
        elif act2 and v2 <= best_val + 5e-3:
            cur, cur_val = y2.copy(), v2
            r = int(np.argmin((_contact_counts(cur, tol=1e-9)[1])))
    # final polish best
    try:
        rem = deadline - time.time()
        if rem > 3.0:
            # polish best_g focused on its rattler
            rf, _, _, _ = rattler_of(best_g)
            best_g, best_val = _lp_focused(best_g, rf, seconds=min(rem - 1.0, 5.0))
    except Exception:
        pass
    best_g = _norm(best_g)
    return best_g
