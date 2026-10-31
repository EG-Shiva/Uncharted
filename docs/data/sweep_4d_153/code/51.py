import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_full(x, seconds, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, iters, k, lr, rng):
    n, d = x.shape
    for t in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _surgery_step(x, m0, forced, etas, radius, contact_thr=1e-7, nkeep=1500):
    n, d = x.shape
    G = x @ x.T
    # contact adjacency for neighbour freeze
    iu_all = np.triu_indices(n, 1)
    vall = G[iu_all]
    slack_all = m0 - vall
    is_contact = slack_all < contact_thr
    # neighbour sets
    neigh = [set() for _ in range(n)]
    CI = iu_all[0][is_contact]; CJ = iu_all[1][is_contact]
    for a, b in zip(CI, CJ):
        neigh[int(a)].add(int(b))
        neigh[int(b)].add(int(a))
    endpoints = []
    for (a, b) in forced:
        endpoints.append(int(a)); endpoints.append(int(b))
    mov_set = set(endpoints)
    for e in endpoints:
        mov_set.update(neigh[e])
    mov = np.array(sorted(mov_set), dtype=int)
    Mm = len(mov)
    if Mm == 0:
        return x.copy(), False
    pos = -np.ones(n, dtype=int)
    pos[mov] = np.arange(Mm)
    forced_set = set()
    for (a, b) in forced:
        a = int(a); b = int(b)
        if a > b:
            a, b = b, a
        forced_set.add((a, b))
    # collect active pairs involving mov (excluding forced)
    # gather candidate indices
    cand_I = []
    cand_J = []
    cand_C = []
    # iterate over mov to limit work: for each m, pairs with all j>m etc.
    # simpler: scan all upper pairs but filter
    mask_involves = np.zeros(len(vall), dtype=bool)
    # quick vectorized: i in mov or j in mov
    in_mov = np.zeros(n, dtype=bool)
    in_mov[mov] = True
    mask_involves = in_mov[iu_all[0]] | in_mov[iu_all[1]]
    # exclude forced
    # build arrays for involved
    II = iu_all[0][mask_involves]; JJ = iu_all[1][mask_involves]; CC = vall[mask_involves]
    # remove forced pairs
    if len(II) > 0 and len(forced_set) > 0:
        keep = np.ones(len(II), dtype=bool)
        for k,(a,b) in enumerate(zip(II,JJ)):
            if (int(a),int(b)) in forced_set:
                keep[k] = False
        II, JJ, CC = II[keep], JJ[keep], CC[keep]
    # keep closest nkeep
    if len(II) > nkeep:
        order = np.argsort(-CC)[:nkeep]
        II, JJ, CC = II[order], JJ[order], CC[order]
    p1 = len(II)
    p2 = len(forced)
    p = p1 + p2
    if p == 0:
        return x.copy(), False
    # variable order: mov*d + t
    nv = Mm * d + 1
    rows_l = []; cols_l = []; vals_l = []; b_ub = []
    # active constraints: coeff*dx - t <= -cij
    for r in range(p1):
        i = int(II[r]); j = int(JJ[r]); cij = float(CC[r])
        pi = pos[i]; pj = pos[j]
        if pi >= 0:
            for dd in range(d):
                rows_l.append(r); cols_l.append(pi*d+dd); vals_l.append(float(x[j, dd]))
        if pj >= 0:
            for dd in range(d):
                rows_l.append(r); cols_l.append(pj*d+dd); vals_l.append(float(x[i, dd]))
        rows_l.append(r); cols_l.append(Mm*d); vals_l.append(-1.0)
        b_ub.append(-cij)
    # forced constraints: coeff*dx <= m0 - eta - cab
    for k2,(pair, eta) in enumerate(zip(forced, etas)):
        r = p1 + k2
        a = int(pair[0]); b = int(pair[1])
        cab = float(x[a] @ x[b])
        rhs = float(m0 - eta - cab)
        pa = pos[a]; pb = pos[b]
        # both should be movable (endpoints)
        if pa >= 0:
            for dd in range(d):
                rows_l.append(r); cols_l.append(pa*d+dd); vals_l.append(float(x[b, dd]))
        if pb >= 0:
            for dd in range(d):
                rows_l.append(r); cols_l.append(pb*d+dd); vals_l.append(float(x[a, dd]))
        b_ub.append(rhs)
    A_ub = coo_matrix((np.array(vals_l), (np.array(rows_l), np.array(cols_l))), shape=(p, nv)).tocsr()
    # equality: <x_i, dx_i>=0
    er = np.repeat(np.arange(Mm), d)
    ec = np.arange(Mm*d)
    ev = x[mov].ravel()
    A_eq = coo_matrix((ev, (er, ec)), shape=(Mm, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    bounds = [(-radius, radius)] * (Mm*d) + [(None, None)]
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.zeros(Mm),
                      bounds=bounds, method="highs")
    except Exception:
        return x.copy(), False
    if res.status != 0:
        return x.copy(), False
    dx = np.zeros((n, d))
    sol = res.x[:-1].reshape(Mm, d)
    dx[mov] = sol
    y = x + dx
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y, True

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    SURG_END = t0 + 100.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        base = np.load("best.npy")
    except Exception:
        try:
            base = np.load("record.npy")
        except Exception:
            base = rng.standard_normal((n, d))
    base = np.asarray(base, dtype=float)
    if base.shape != (n, d):
        base = rng.standard_normal((n, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    best_val = _max_cos(base)
    incumbent = base.copy()
    inc_val = best_val

    def second_shell(x, m, lo=1e-5, hi=5e-4):
        G = x @ x.T
        iu = np.triu_indices(n, 1)
        c = G[iu]
        slack = m - c
        sel = (slack >= lo) & (slack <= hi)
        I = iu[0][sel]; J = iu[1][sel]
        S = slack[sel]
        return list(zip(I.tolist(), J.tolist())), S
    pool, _ = second_shell(incumbent, inc_val)
    if len(pool) < 20:
        pool, _ = second_shell(incumbent, inc_val, 1e-5, 1e-3)
    if len(pool) < 20:
        pool, _ = second_shell(incumbent, inc_val, 5e-6, 1e-3)

    radii_choices = np.array([0.01, 0.02, 0.03, 0.05])
    trial = 0
    while time.time() < SURG_END:
        trial += 1
        if len(pool) == 0:
            pool, _ = second_shell(incumbent, inc_val, 1e-5, 1e-3)
            if len(pool) == 0:
                break
        k = int(rng.integers(1, 4))
        k = min(k, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        forced = [pool[int(ii)] for ii in idx]
        # etas log-uniform 1e-4..1e-3
        etas = [float(10.0 ** rng.uniform(-4.0, -3.0)) for _ in range(k)]
        radius = float(rng.choice(radii_choices))
        # occasionally uniform radius
        if rng.random() < 0.3:
            radius = float(rng.uniform(0.01, 0.05))
        # occasionally do 2 sequential LPs with same forced (trust iteration)
        xcur = incumbent
        m0 = inc_val
        ok_any = False
        nsteps = 1 if rng.random() < 0.7 else 2
        # for multi-step, shrink radius a bit
        y = xcur.copy()
        success = True
        for s in range(nsteps):
            rr = radius * (0.6 ** s)
            y, ok = _surgery_step(y, m0, forced, etas, rr)
            if not ok:
                success = False
                break
        if not success:
            continue
        m_try = _max_cos(y)
        # feasibility: allow degradation up to 0.001
        if m_try < inc_val + 0.001:
            # smooth-relax short
            kexp = float(rng.choice(np.array([80.0, 150.0, 300.0])))
            lr = float(rng.choice(np.array([0.005, 0.01, 0.02])))
            it = int(rng.choice(np.array([200, 400])))
            ys = _smooth(y.copy(), it, kexp, lr, rng)
            vs = _max_cos(ys)
            # only full polish if promising
            if vs < inc_val + 0.001:
                remain = SURG_END - time.time()
                if remain < 0.6:
                    break
                budget = min(1.2, remain - 0.1)
                yp = _lp_polish_full(ys, budget)
                vp = _max_cos(yp)
                if vp < inc_val - 1e-12:
                    incumbent = yp.copy()
                    inc_val = vp
                    if vp < best_val:
                        base = yp.copy()
                        best_val = vp
                    pool, _ = second_shell(incumbent, inc_val)
                    if len(pool) < 20:
                        pool, _ = second_shell(incumbent, inc_val, 1e-5, 1e-3)
                # also if equal but different topology, keep exploring from it occasionally
                elif vp < inc_val + 5e-5 and rng.random() < 0.05:
                    # random restart base from this basin briefly (don't replace incumbent)
                    pass
        # time check
        if time.time() >= SURG_END - 0.3:
            break

    # final polish best available
    rem = DEADLINE - time.time()
    target = base if best_val <= inc_val else incumbent
    tval = min(best_val, inc_val)
    if rem > 4.0:
        polished = _lp_polish_full(target, min(rem - 1.0, 45.0))
        pv = _max_cos(polished)
        if pv < tval:
            target = polished
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    return target
