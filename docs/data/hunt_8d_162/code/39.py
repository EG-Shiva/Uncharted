import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def all_dots(x):
    g = x @ x.T
    return g

def lp_polish(x, seconds, r0=1e-3, rmax=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    iuI, iuJ = iu[0], iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        cfull = (x @ x.T)[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            # keep tightest 2500
            idx = np.argpartition(-cfull, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]
        c = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return x


def contact_counts(x, m, thresh=1e-4):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    cnt = np.sum(g > m - thresh, axis=1)
    return cnt


def build_active_sparse(x, thresh=1e-6, cap=1600):
    n, d = x.shape
    N = n * d
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iuI, iuJ = np.triu_indices(n, 1)
    c = G[iuI, iuJ]
    order = np.argsort(-c)
    # select slack < thresh
    sel_mask = c > m - thresh
    sel_idx = np.where(sel_mask)[0]
    # sort by tightness and cap
    if len(sel_idx) > cap:
        # keep tightest cap
        top = order[:cap]
        # ensure sorted? not needed
        I = iuI[top]; J = iuJ[top]
        cc = c[top]
    else:
        # sort sel by tightness desc
        o = np.argsort(-c[sel_idx])
        sel_idx = sel_idx[o]
        I = iuI[sel_idx]; J = iuJ[sel_idx]
        cc = c[sel_idx]
    p = len(I)
    return I, J, cc, m, p


def nullspace_direction(x, I, J, weights, rng, lsmr_iters=250):
    n, d = x.shape
    N = n * d
    p = len(I)
    m_eq = p + n
    # build sparse A (m_eq x N)
    # entries: p*2d + n*d
    rows_list = []
    cols_list = []
    data_list = []
    # active rows
    for k in range(p):
        ik = int(I[k]); jk = int(J[k])
        # block ik: x[jk]
        for dd in range(d):
            rows_list.append(k); cols_list.append(ik * d + dd); data_list.append(float(x[jk, dd]))
            rows_list.append(k); cols_list.append(jk * d + dd); data_list.append(float(x[ik, dd]))
    # equality rows (tangent): row p+i has x[i] at block i
    for i in range(n):
        for dd in range(d):
            rows_list.append(p + i); cols_list.append(i * d + dd); data_list.append(float(x[i, dd]))
    rows_a = np.array(rows_list, dtype=np.int32)
    cols_a = np.array(cols_list, dtype=np.int32)
    data_a = np.array(data_list, dtype=float)
    A = coo_matrix((data_a, (rows_a, cols_a)), shape=(m_eq, N)).tocsr()
    # random tangent field biased by weights
    R = rng.randn(n, d)
    # project to tangent
    dot = np.sum(R * x, axis=1, keepdims=True)
    R = R - dot * x
    R = R * weights[:, None]
    # normalize
    nr = np.linalg.norm(R)
    if nr < 1e-12:
        R = rng.randn(n, d)
        dot = np.sum(R * x, axis=1, keepdims=True)
        R = R - dot * x
        nr = np.linalg.norm(R)
    R = R / (nr + 1e-300)
    r = R.ravel()
    # solve min_y ||A^T y - r|| via LSMR (approximate nullspace projection)
    try:
        AT = A.T.tocsr()
        # lsmr signature varies; use maxiter param
        try:
            out = lsmr(AT, r, maxiter=lsmr_iters, atol=1e-7, btol=1e-7)
        except TypeError:
            out = lsmr(AT, r, maxiter=lsmr_iters)
        y = out[0]
        vflat = r - AT.dot(y)
    except Exception:
        # fallback dense lstsq small
        try:
            Ad = A.toarray()
            y2, *_ = np.linalg.lstsq(Ad.T, r, rcond=None)
            vflat = r - Ad.T.dot(y2)
        except Exception:
            vflat = r
    V = vflat.reshape(n, d)
    # re-tangentize
    dot2 = np.sum(V * x, axis=1, keepdims=True)
    V = V - dot2 * x
    nv = np.linalg.norm(V)
    if nv < 1e-12 or not np.isfinite(nv):
        return None, 0.0
    V = V / nv
    # null residual measure
    try:
        res_norm = float(np.linalg.norm(A.dot(vflat)))
    except Exception:
        res_norm = 0.0
    return V, res_norm


def retract(x, V, s):
    y = x + s * V
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    rng = np.random.RandomState(20261103)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(162, 8)
    xb = np.asarray(xb, dtype=float).reshape(162, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    # quick LP warm? keep best polished copy already; don't waste time

    n, d = 162, 8
    walk_id = 0

    # outer loop of nullspace crawls
    while time.time() < DEADLINE - 14.0:
        walk_id += 1
        # rebuild active set from current best (or from perturbed best to diversify)
        # choose base: mostly x_best, occasionally slightly jittered base to get different Jacobian
        if walk_id % 7 == 0:
            # start walk from lightly jittered copy to diversify basin
            sigma = 0.002
            x_cur = x_best + sigma * rng.randn(n, d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            # quick check: if much worse, reset to best
            if max_cos(x_cur) > best_val + 0.005:
                x_cur = x_best.copy()
        else:
            x_cur = x_best.copy()

        m0 = max_cos(x_cur)
        I, J, cc, m_act, p = build_active_sparse(x_cur, thresh=1e-6, cap=1500)
        if p < 10:
            # degenerate, add more
            I, J, cc, m_act, p = build_active_sparse(x_cur, thresh=1e-4, cap=1500)
        cnt = contact_counts(x_cur, m0, thresh=1e-4)
        # weights bias away from overloaded points (LP dual proxy = contact counts)
        # overloaded -> small weight
        weights = 1.0 / np.sqrt(1.0 + cnt.astype(float))
        # extra: rattlers get boost
        weights = weights / weights.mean()
        # clamp
        weights = np.clip(weights, 0.15, 2.0)

        # do several random nullspace walks
        n_walks = 4 if (walk_id % 3 != 0) else 6
        crawled = False
        for w in range(n_walks):
            if time.time() > DEADLINE - 14.0:
                break
            # vary lsmr iters / weights slightly
            # occasionally invert bias to explore
            if rng.rand() < 0.2:
                w_use = np.ones(n)
            else:
                # add noise to weights
                w_use = weights * (0.7 + 0.6 * rng.rand(n))
            V, rn = nullspace_direction(x_cur, I, J, w_use, rng, lsmr_iters=200)
            if V is None:
                continue
            # line-search for largest step whose true max rises by at most 1e-7
            # candidate steps ascending
            steps = [0.002, 0.005, 0.01, 0.02, 0.035, 0.06, 0.09, 0.13, 0.18]
            # random direction sign? V sign irrelevant; try both? test positive first, if fails try negative
            found_s = 0.0
            found_x = None
            # try both signs
            signs = [1.0, -1.0] if rng.rand() < 0.5 else [1.0]
            # To save evals, evaluate max for each step
            best_s = 0.0
            best_xs = None
            for sgn in signs:
                Vs = V * sgn
                for s in steps:
                    if time.time() > DEADLINE - 14.0:
                        break
                    y = retract(x_cur, Vs, s)
                    v = max_cos(y)
                    if v <= m0 + 1e-7:
                        if s > best_s:
                            best_s = s
                            best_xs = y
                    else:
                        # larger steps likely also violate; break for this sign
                        # but due to nonmonotonicity continue? break to save time
                        if s > 0.02:
                            break
                if best_s >= 0.09:
                    break
            if best_xs is None or best_s <= 1e-12:
                # too jammed in this direction, try new random field
                continue
            x_cur = best_xs
            m_cur = max_cos(x_cur)
            # update m0 for next crawl step (allow tiny rise)
            m0 = m_cur
            crawled = True
            # check topology change: new inactive pair entered or active separated
            # compute slacks quickly
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            m_new = float(G.max())
            # active separation: fraction of old active with slack > 1e-4?
            # map old pairs: need dots for old I,J
            old_dots = G[I, J]
            n_sep = int(np.sum(old_dots < m_new - 5e-5))
            # new entry: any pair not in old active now within 1e-6?
            # approximate: count pairs within 1e-6 minus p
            iuI, iuJ = np.triu_indices(n, 1)
            # subsample check via top values
            # to avoid full sort each time, check max of non-active? Use threshold count
            cnt_new = int(np.sum(G[iuI, iuJ] > m_new - 1e-6))
            # if topology changed materially, break to polish
            if n_sep >= 3 or cnt_new > p + 5 or best_s >= 0.06:
                break
            # else refresh active set every 2 steps and continue crawling
            if w % 2 == 1:
                I2, J2, cc2, _, p2 = build_active_sparse(x_cur, thresh=1e-6, cap=1500)
                if p2 >= 10:
                    I, J, cc, p = I2, J2, cc2, p2
                    cnt = contact_counts(x_cur, m_new, thresh=1e-4)
                    weights = 1.0 / np.sqrt(1.0 + cnt.astype(float))
                    weights = weights / weights.mean()
                    weights = np.clip(weights, 0.15, 2.0)
                    m_act = m_new
        if not crawled:
            # fallback: tiny random kick + smooth? just continue to next walk
            # occasional single-point hole move to avoid stall
            if walk_id % 5 == 0 and time.time() < DEADLINE - 20:
                # move one rattler to hole
                cnt_t = contact_counts(x_cur, max_cos(x_cur), thresh=1e-7)
                ri = int(np.argmin(cnt_t))
                cand = rng.randn(4000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                others = np.delete(x_cur, ri, axis=0)
                worst = (cand @ others.T).max(axis=1)
                bi = int(np.argmin(worst))
                x_cur[ri] = cand[bi]
                x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            else:
                continue
        # LP-polish the crawled point
        remain = DEADLINE - 14.0 - time.time()
        if remain < 0.5:
            # just evaluate
            v = max_cos(x_cur)
            if v < best_val - 1e-12:
                x_best = x_cur
                best_val = v
            continue
        budget = min(2.5, max(0.6, remain / 6.0))
        try:
            xl = lp_polish(x_cur, budget)
        except Exception:
            continue
        vl = max_cos(xl)
        if vl < best_val - 1e-12:
            x_best = xl
            best_val = vl
        # also consider unpolished if better (rare)
        else:
            vcur = max_cos(x_cur)
            if vcur < best_val - 1e-12:
                x_best = x_cur
                best_val = vcur

    # final polish
    try:
        remain = DEADLINE - time.time() - 0.8
        if remain > 1.5:
            x_best = lp_polish(x_best, min(remain, 10.0))
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
