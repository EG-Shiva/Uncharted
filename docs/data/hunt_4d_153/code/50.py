import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, deadline, max_solves=120, cap=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        if int(sel.sum()) > cap:
            idx = np.argpartition(-c, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _get_contacts_and_stress(x, radius=1e-3):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iu]
    m = float(c.max())
    # active set for stress: tight contacts
    win = 1e-7
    sel = c > m - win
    I = iu[0][sel]
    J = iu[1][sel]
    p = len(I)
    cc = c[sel]
    stress = np.zeros(p)
    ok = False
    if p > 0 and p < 3000:
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
            if res.status == 0:
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
                    if marg.size == p:
                        stress = np.abs(marg)
                        ok = True
                except Exception:
                    pass
                try:
                    if not ok:
                        # try alternate attribute
                        marg2 = np.asarray(res.get('ineqlin', None), dtype=float)
                        if marg2.size == p:
                            stress = np.abs(marg2)
                            ok = True
                except Exception:
                    pass
        except Exception:
            pass
    return I, J, cc, m, stress, ok


def _surgery_anneal(x0, forced_I, forced_J, rel_I, rel_J, t_target, lam,
                    steps, T0, T1, lrfac, rng, noise=0.12, mu=0.85):
    n, d = x0.shape
    x = x0.copy()
    # build released mask
    rel_mask = np.zeros((n, n), dtype=bool)
    if len(rel_I) > 0:
        rel_mask[rel_I, rel_J] = True
        rel_mask[rel_J, rel_I] = True
    vel = np.zeros_like(x)
    has_forced = len(forced_I) > 0
    if has_forced:
        fI = np.asarray(forced_I, dtype=int)
        fJ = np.asarray(forced_J, dtype=int)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac if T1 > 0 else T0
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        if len(rel_I) > 0:
            W[rel_I, rel_J] = 0.0
            W[rel_J, rel_I] = 0.0
        grad = W @ x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = grad / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # forced penalty gradient: lam * relu(t_target - g)
        if has_forced:
            gf_vals = G[fI, fJ]
            act = gf_vals < t_target
            if np.any(act):
                aI = fI[act]
                aJ = fJ[act]
                # residual magnitude
                resid = (t_target - gf_vals[act])  # >0
                # weight by residual (linear) capped
                wgt = np.minimum(resid * 200.0, 1.0) + 0.25
                gf = np.zeros_like(x)
                # accumulate: d/dxi = -lam*xj
                for k in range(len(aI)):
                    i = int(aI[k]); j = int(aJ[k])
                    wk = wgt[k]
                    gf[i] += -wk * x[j]
                    gf[j] += -wk * x[i]
                gfd = np.sum(gf * x, axis=1, keepdims=True)
                gft = gf - gfd * x
                gt = gt + lam * gft
        lr = lrfac * T
        # momentum
        vel = mu * vel + gt
        bc = 1.0 - mu ** (s + 1)
        step = vel / max(bc, 1e-12)
        # clip step norm to avoid blowup from forced term
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 3.0
        scale = np.minimum(1.0, cap / sn)
        step = step * scale
        x = x - lr * step
        if noise > 0:
            N = rng.standard_normal(size=x.shape)
            Nd = np.sum(N * x, axis=1, keepdims=True)
            N -= Nd * x
            x = x + noise * np.sqrt(max(lr, 1e-12)) * N
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 128.0
    rng = np.random.default_rng(1234567)
    n = 153

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])

    # initial polish to reach basin bottom
    best_x, best_val = _lp_polish(best_x, min(deadline - 5, time.time() + 5.0), max_solves=30)

    record_val = best_val
    t_target = record_val - 2e-4

    # analyse basin
    G0 = best_x @ best_x.T
    iu_all = np.triu_indices(n, 1)
    c0 = G0[iu_all]
    order = np.argsort(-c0)
    m0 = float(c0.max())
    # contact mask with 1e-7 tolerance
    contact_tol = 1e-7
    base_mask = c0 > m0 - contact_tol
    # per-point contacts
    per = np.zeros(n, dtype=int)
    Ib, Jb = iu_all[0][base_mask], iu_all[1][base_mask]
    for i, j in zip(Ib, Jb):
        per[i] += 1
        per[j] += 1
    rattlers = np.where(per == 0)[0]
    lowdeg = np.where(per <= 4)[0]

    # dual stress
    sI, sJ, scc, sm, stress, ok = _get_contacts_and_stress(best_x)
    # order release candidates: low stress first
    if ok and len(stress) == len(sI) and len(sI) > 0:
        rel_order = np.argsort(stress)  # ascending: low stress first
        rel_pool_I = sI[rel_order]
        rel_pool_J = sJ[rel_order]
    else:
        # fallback: random among base contacts, prefer high-degree? just use base
        rel_pool_I = Ib.copy()
        rel_pool_J = Jb.copy()
        # shuffle deterministically
        idx = rng.permutation(len(rel_pool_I))
        rel_pool_I = rel_pool_I[idx]
        rel_pool_J = rel_pool_J[idx]

    # second-shell pool: rank ~460-1000, slack 1e-5..2e-3
    # number of contacts ~ len(Ib) ~460
    nc = int(base_mask.sum())
    lo_rank = nc
    hi_rank = min(len(order), nc + 700)
    cand_idx = order[lo_rank:hi_rank]
    cand_I = iu_all[0][cand_idx]
    cand_J = iu_all[1][cand_idx]
    cand_c = c0[cand_idx]
    slack = m0 - cand_c
    # keep slack in [1e-5, 2e-3]
    keep = (slack >= 5e-6) & (slack <= 2e-3)
    cand_I = cand_I[keep]
    cand_J = cand_J[keep]
    cand_slack = slack[keep]
    if len(cand_I) < 10:
        # relax
        keep2 = (slack >= 1e-6) & (slack <= 5e-3)
        cand_I = iu_all[0][cand_idx][keep2]
        cand_J = iu_all[1][cand_idx][keep2]
        cand_slack = slack[keep2]
    # score forced candidates: bonus if incident to rattler or lowdeg
    ratt_set = set(rattlers.tolist())
    low_set = set(lowdeg.tolist())
    scores = np.zeros(len(cand_I))
    for k in range(len(cand_I)):
        i = int(cand_I[k]); j = int(cand_J[k])
        s = 0.0
        if i in ratt_set or j in ratt_set:
            s += 10.0
        if i in low_set or j in low_set:
            s += 3.0
        # prefer smaller slack (closer) slightly
        s += -np.log10(max(cand_slack[k], 1e-9)) * 0.1
        scores[k] += s
    # sort by score descending for weighted sampling
    if len(cand_I) == 0:
        cand_I = iu_all[0][order[nc:nc+200]]
        cand_J = iu_all[1][order[nc:nc+200]]
        scores = np.zeros(len(cand_I))

    # precompute record graph mask for diff
    record_mask = base_mask.copy()

    pool = []  # list of (val, x, diff)
    cur_x = best_x.copy()
    cur_val = best_val

    ntrials = 30
    trial = 0
    # probability weights for forced sampling
    if len(scores) > 0:
        w_forced = np.exp(scores - scores.max())
        w_forced /= w_forced.sum()
    else:
        w_forced = None

    while trial < ntrials and time.time() < explore_end:
        trial += 1
        # choose numbers
        k_forced = int(rng.integers(1, 4))  # 1..3
        k_rel = int(rng.integers(3, 9))  # 3..8
        k_forced = min(k_forced, len(cand_I))
        k_rel = min(k_rel, len(rel_pool_I))
        if k_forced <= 0 or k_rel <= 0:
            break
        # sample forced
        if w_forced is not None:
            f_choice = rng.choice(len(cand_I), size=k_forced, replace=False, p=w_forced)
        else:
            f_choice = rng.choice(len(cand_I), size=k_forced, replace=False)
        fI = cand_I[f_choice]
        fJ = cand_J[f_choice]
        # sample released: low-stress first with some randomness
        # take top 40 low-stress then sample
        topN = min(len(rel_pool_I), 60)
        perm = rng.permutation(topN)[:k_rel]
        # occasionally include random deeper
        if rng.random() < 0.3 and len(rel_pool_I) > topN:
            extra = rng.choice(len(rel_pool_I), size=k_rel, replace=False)
            perm_idx = extra
            rI = rel_pool_I[perm_idx]
            rJ = rel_pool_J[perm_idx]
        else:
            rI = rel_pool_I[perm[:k_rel]]
            rJ = rel_pool_J[perm[:k_rel]]
        # ensure forced and released disjoint (no shared pair exactly)
        # build init y with kick on involved points
        y = best_x.copy()
        involved = set()
        for i, j in zip(fI, fJ):
            involved.add(int(i)); involved.add(int(j))
        for i, j in zip(rI, rJ):
            involved.add(int(i)); involved.add(int(j))
        # also add rattlers always a bit
        for r in rattlers:
            involved.add(int(r))
        inv = np.array(sorted(involved), dtype=int)
        # kick magnitude
        amp = 0.035 + 0.02 * rng.random()
        y[inv] += amp * rng.standard_normal((len(inv), 4))
        y += 0.003 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # anneal params: beta 50->400 => T 0.02->0.0025
        lam = 2.0 + rng.random() * 2.0  # 2..4
        T0a = 0.020
        T1a = 0.0025
        steps = 520
        # vary slightly
        if trial % 3 == 0:
            steps = 650
            T0a = 0.024
        y = _surgery_anneal(y, fI, fJ, rI, rJ, t_target, lam,
                            steps=steps, T0=T0a, T1=T1a,
                            lrfac=1.25, rng=rng, noise=0.13, mu=0.86)
        # cool down without forced/release, low T
        y = _surgery_anneal(y, [], [], [], [], t_target, 0.0,
                            steps=130, T0=0.004, T1=0.0012,
                            lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
        v = _max_cos(y)
        # quick check: must be within 5e-4 to be worth polishing
        if v > best_val + 8e-4:
            # still maybe update cur for diversity? skip
            continue
        # LP polish
        y2, v2 = _lp_polish(y, min(deadline - 4.0, time.time() + 1.8), max_solves=18)
        # compute graph diff vs record
        G2 = y2 @ y2.T
        c2 = G2[iu_all]
        m2 = float(c2.max())
        mask2 = c2 > m2 - 1e-7
        diff = int(np.sum(mask2 != record_mask))
        # accept into pool if diff>=10 (5 edges) and within 5e-4
        if v2 < best_val - 1e-12:
            best_x, best_val = y2.copy(), float(v2)
            # update record mask? keep original for diff reference, but also update t_target?
            # keep original mask as reference per spec (record graph)
            pool.append((float(v2), y2.copy(), diff))
            cur_x, cur_val = y2.copy(), float(v2)
            continue
        if diff >= 10 and v2 < record_val + 5e-4:
            pool.append((float(v2), y2.copy(), diff))
            # move cur to this for exploration
            if v2 < cur_val or rng.random() < 0.4:
                cur_x, cur_val = y2.copy(), float(v2)
        else:
            # basin hopping acceptance for cur
            if v2 < cur_val:
                cur_x, cur_val = y2.copy(), float(v2)
            elif rng.random() < np.exp(-(v2 - cur_val) / 0.0006):
                cur_x, cur_val = y2.copy(), float(v2)

    # re-polish pool members with fresh LP (neighbouring topologies)
    if pool:
        # sort by val
        pool.sort(key=lambda t: t[0])
        # take top few
        for k in range(min(len(pool), 6)):
            if time.time() > deadline - 12.0:
                break
            _, px, _ = pool[k]
            # small jitter + anneal cool then LP
            if time.time() > deadline - 12.0:
                break
            q = px + 0.004 * rng.standard_normal(px.shape)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            q = _surgery_anneal(q, [], [], [], [], t_target, 0.0,
                                steps=150, T0=0.0035, T1=0.0010,
                                lrfac=1.0, rng=rng, noise=0.0, mu=0.0)
            q2, vq = _lp_polish(q, min(deadline - 4.0, time.time() + 2.5), max_solves=30)
            if vq < best_val:
                best_x, best_val = q2, float(vq)

    # final squeeze of best
    if time.time() < deadline - 3.0:
        best_x, best_val = _lp_polish(best_x, deadline - 1.0, max_solves=120)

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
