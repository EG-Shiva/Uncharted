import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def lp_polish_budget(x, budget_sec, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    start = time.time()
    while True:
        if time.time() - start >= budget_sec:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        np.fill_diagonal(Gfull, -2.0)
        m = float(Gfull.max())
        c_full = (x @ x.T)[iu0]
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_full[sel]
        # cap size
        if len(c) > 2500:
            o = np.argpartition(c, -2500)[-2500:]
            I, J, c = I[o], J[o], c[o]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best


def constrained_push(x0, a, b, c_, d_, m_ref, steps, radius, iu0, deadline=None):
    x = x0.copy()
    n, dd = x.shape
    nv = n * dd + 1
    # ensure ordering for comparison (I<J)
    pa = (min(a, b), max(a, b))
    pb = (min(c_, d_), max(c_, d_))
    for _ in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        G = x @ x.T
        # current dots for special pairs
        c_ab_cur = float(G[a, b])
        c_cd_cur = float(G[c_, d_])
        c_all = G[iu0]
        # select active set: close to m_ref
        thresh = m_ref - 0.015
        # also include current max vicinity
        # get indices of top pairs
        # mask
        mask = c_all > thresh
        # exclude promote/ban
        # build I,J for masked
        Im = iu0[0][mask]
        Jm = iu0[1][mask]
        # filter out pa/pb
        keep = ~(((Im == pa[0]) & (Jm == pa[1])) | ((Im == pb[0]) & (Jm == pb[1])))
        Im = Im[keep]
        Jm = Jm[keep]
        cm = c_all[mask][keep]
        if len(cm) > 800:
            o = np.argpartition(cm, -800)[-800:]
            Im, Jm, cm = Im[o], Jm[o], cm[o]
        p = len(Im)
        # Build A_ub: p rows for t + 1 ban row + 1 promote row
        # ban RHS, promote RHS use m_ref and delta stored in closure? pass via args
        # caller passes delta via m_ref? we need delta; store as attribute: use extra param
        # To avoid signature change, we encode delta in radius? No.
        # Instead we will handle outside: this function expects delta in variable delta_val attached
        # Workaround: use global? Simpler: recompute inside with delta passed
        break
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    n, d = 153, 4
    rng = np.random.default_rng(0)

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a0 = np.load(fn)
            a0 = np.asarray(a0, dtype=float)
            if a0.shape == (n, d):
                a0 = a0 / np.linalg.norm(a0, axis=1, keepdims=True)
                if np.all(np.isfinite(a0)):
                    cand.append(a0)
        except Exception:
            pass
    if len(cand) == 0:
        a0 = rng.standard_normal((n, d))
        a0 /= np.linalg.norm(a0, axis=1, keepdims=True)
        cand.append(a0)
    vals = [max_cos(a0) for a0 in cand]
    bi = int(np.argmin(vals))
    best_x = cand[bi].copy()
    best_val = float(vals[bi])
    m_ref0 = best_val

    iu = np.triu_indices(n, 1)
    G0 = best_x @ best_x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    c0 = (best_x @ best_x.T)[iu]
    slack0 = m0 - c0
    order = np.argsort(slack0)
    # contact threshold: polished to ~1e-12, use 1e-7 to count
    num_contact = int(np.count_nonzero(slack0 < 1e-7))
    if num_contact < 10:
        num_contact = int(np.count_nonzero(slack0 < 1e-6))
    # candidate promoted pairs: ranks num_contact .. num_contact+120 with slack < 0.005
    Pcand = []
    for k in range(num_contact, min(len(order), num_contact + 160)):
        idx = int(order[k])
        s = float(slack0[idx])
        if s < 1e-9:
            continue
        if s > 0.006:
            break
        Pcand.append((int(iu[0][idx]), int(iu[1][idx]), s))
    # fallback: if too few, take smallest slack non-contacts
    if len(Pcand) < 50:
        for k in range(num_contact, len(order)):
            idx = int(order[k])
            s = float(slack0[idx])
            tup = (int(iu[0][idx]), int(iu[1][idx]), s)
            # avoid duplicates
            if all(not (t[0] == tup[0][0] and t[1] == tup[0][1]) for t in [(p,) for p in []]):
                pass
            if not any(p[0] == tup[0][0] and p[1] == tup[0][1] for p in Pcand):
                if s < 0.01:
                    Pcand.append((tup[0][0], tup[0][1], s))
            if len(Pcand) >= 60:
                break
    # contact edges list (tight)
    contact_mask = slack0 < 5e-7
    cI = iu[0][contact_mask]
    cJ = iu[1][contact_mask]
    contacts = list(zip([int(v) for v in cI], [int(v) for v in cJ]))
    # degrees
    deg = np.zeros(n, dtype=int)
    for (u, v) in contacts:
        deg[u] += 1
        deg[v] += 1
    hub = int(np.argmax(deg))
    hub_edges = [(u, v) for (u, v) in contacts if u == hub or v == hub]
    if len(hub_edges) == 0 and len(contacts) > 0:
        hub_edges = contacts[:5]
    # incident map
    from collections import defaultdict
    inc = defaultdict(list)
    for (u, v) in contacts:
        inc[u].append((u, v))
        inc[v].append((u, v))

    deltas = [1e-4, 3e-4, 1e-3]
    radii = [0.015, 0.02, 0.03]

    # build at least 40 trials (aim 50)
    trials = []
    K = min(len(Pcand), 50)
    if K < 40:
        # repeat with different deltas to reach 40
        reps = (40 + max(K, 1) - 1) // max(K, 1)
    else:
        reps = 1
    # expand Pcand by cycling if needed
    ext = []
    ci = 0
    while len(ext) < 50 and len(Pcand) > 0:
        ext.append(Pcand[ci % len(Pcand)])
        ci += 1
    for ti, (a, b, s) in enumerate(ext):
        # banned edge: sharing a or b, else hub
        shared = []
        for e in inc.get(a, []):
            shared.append(e)
        for e in inc.get(b, []):
            if e not in shared:
                shared.append(e)
        if len(shared) > 0:
            ban = shared[ti % len(shared)]
        else:
            ban = hub_edges[ti % len(hub_edges)] if len(hub_edges) > 0 else contacts[ti % len(contacts)]
        # every 3rd trial force hub edge to satisfy "or incident to degree-8 point"
        if ti % 3 == 2 and len(hub_edges) > 0:
            ban = hub_edges[(ti // 3) % len(hub_edges)]
        # ensure ban != promote (always true as promote non-contact)
        delta = deltas[ti % 3]
        radius = radii[ti % 3]
        trials.append((a, b, int(ban[0]), int(ban[1]), float(delta), float(radius)))
    # ensure >=40
    assert len(trials) >= 40

    changed_pool = []  # list of (val, x)
    ntrials_done = 0
    # constrained phase parameters
    CSTEPS = 22

    for (a, b, cc, dd, delta, radius) in trials:
        if time.time() >= HARD_END - 25:
            break
        # time per trial guard
        if time.time() >= HARD_END - 2:
            break
        m_ref = float(max_cos(best_x))  # current best ref (original basin value, updates if improved)
        # start from best_x (record basin)
        x = best_x.copy()
        pa = (min(a, b), max(a, b))
        pb = (min(cc, dd), max(cc, dd))
        feasible = True
        # trust-region constrained steps
        for _ in range(CSTEPS):
            if time.time() >= HARD_END - 20:
                feasible = False
                break
            G = x @ x.T
            c_ab_cur = float(G[a, b])
            c_cd_cur = float(G[cc, dd])
            c_all = G[iu]
            thresh = m_ref - 0.015
            # build active set excluding pa/pb
            # quick selection
            # use argpartition if too many
            mask_idx = np.where(c_all > thresh)[0]
            # filter pa/pb: need to map linear index? Instead filter via I,J
            Im_all = iu[0][mask_idx]
            Jm_all = iu[1][mask_idx]
            keep = ~(((Im_all == pa[0]) & (Jm_all == pa[1])) | ((Im_all == pb[0]) & (Jm_all == pb[1])))
            mask_idx = mask_idx[keep]
            Im_all = iu[0][mask_idx]
            Jm_all = iu[1][mask_idx]
            cm_all = c_all[mask_idx]
            if len(cm_all) > 800:
                o = np.argpartition(cm_all, -800)[-800:]
                Im_all, Jm_all, cm_all = Im_all[o], Jm_all[o], cm_all[o]
            p = len(Im_all)
            tot = p + 2
            nv = n * d + 1
            rows_list = []
            cols_list = []
            vals_list = []
            brow = []
            # S rows
            # vectorised construction
            if p > 0:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([Im_all[:, None] * d + np.arange(d),
                                       Jm_all[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[Jm_all], x[Im_all], -np.ones((p, 1))], 1).ravel()
                rows_list.append(rows)
                cols_list.append(cols)
                vals_list.append(vals)
                b_ub = list(-cm_all)
            else:
                b_ub = []
            # ban row index p
            r_ban = p
            # coeffs: +x[dd] at cc block, +x[cc] at dd block
            for k2, vv in enumerate(x[dd]):
                rows_list.append(np.array([r_ban]))
                cols_list.append(np.array([cc * d + k2]))
                vals_list.append(np.array([vv]))
            for k2, vv in enumerate(x[cc]):
                rows_list.append(np.array([r_ban]))
                cols_list.append(np.array([dd * d + k2]))
                vals_list.append(np.array([vv]))
            b_ub.append(m_ref - delta - c_cd_cur)
            # promote row index p+1 : -coeff
            r_pro = p + 1
            for k2, vv in enumerate(x[b]):
                rows_list.append(np.array([r_pro]))
                cols_list.append(np.array([a * d + k2]))
                vals_list.append(np.array([-vv]))
            for k2, vv in enumerate(x[a]):
                rows_list.append(np.array([r_pro]))
                cols_list.append(np.array([b * d + k2]))
                vals_list.append(np.array([-vv]))
            b_ub.append(c_ab_cur - (m_ref - 1e-7))
            rowsA = np.concatenate(rows_list)
            colsA = np.concatenate(cols_list)
            valsA = np.concatenate(vals_list)
            A_ub = coo_matrix((valsA, (rowsA, colsA)), shape=(tot, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                feasible = False
                break
            if res.status != 0:
                feasible = False
                break
            dx = res.x[:-1].reshape(n, d)
            x = x + dx
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        ntrials_done += 1
        if not feasible:
            continue
        # remove artificial constraints, LP-polish whole configuration
        remain_trial_polish = 0.9
        # shorten if many trials left
        if len(trials) - ntrials_done > 25:
            remain_trial_polish = 0.6
        xp, vp = lp_polish_budget(x, remain_trial_polish, deadline=HARD_END - 15)
        # acceptance check
        Gp = xp @ xp.T
        np.fill_diagonal(Gp, -2.0)
        mp = float(Gp.max())
        s_pro = mp - float(xp[a] @ xp[b])
        s_ban = mp - float(xp[cc] @ xp[dd])
        topo_ok = (s_pro < 1e-6) and (s_ban > 1e-6)
        if vp < best_val - 1e-12:
            best_val = float(vp)
            best_x = xp.copy()
            changed_pool.append((float(vp), xp.copy()))
        elif topo_ok:
            changed_pool.append((float(vp), xp.copy()))
        # also keep topo_ok even if worse for diversity (limit size)
        if len(changed_pool) > 12:
            changed_pool.sort(key=lambda t: t[0])
            changed_pool = changed_pool[:12]

    # further polish changed pool best few
    changed_pool.sort(key=lambda t: t[0]) if len(changed_pool) > 0 else None
    # polish top 3 changed with extra time
    for k in range(min(3, len(changed_pool))):
        if time.time() >= HARD_END - 12:
            break
        v0, x0c = changed_pool[k]
        xp, vp = lp_polish_budget(x0c, 2.0, deadline=HARD_END - 8)
        if vp < best_val:
            best_val = float(vp)
            best_x = xp.copy()

    # final polish best with remaining time
    remain = HARD_END - time.time() - 1.0
    if remain > 1.5:
        use = min(remain, 30.0)
        best_x, best_val2 = lp_polish_budget(best_x, use, deadline=HARD_END - 1.0)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
