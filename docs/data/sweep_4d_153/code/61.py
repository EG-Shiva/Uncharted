import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def full_dots(x):
    G = x @ x.T
    return G

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_relax(x, iters, k, lr, rng=None, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.8
    for t in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1))
        x = x - lr * vc
        if noise > 0:
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + noise * pert
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((n, d))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        x0 = tmp
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    m0 = max_cos(x0)
    incumbent = x0.copy()
    inc_val = m0

    # precompute pair info for incumbent (will refresh periodically)
    def pair_info(x):
        G = x @ x.T
        iu = np.triu_indices(n, 1)
        c = G[iu]
        m = c.max()
        slack = m - c
        return G, iu, c, m, slack

    G, iu, c, m, slack = pair_info(incumbent)
    # second shell: slack in [1e-5, 5e-4]
    ss_mask = (slack >= 1e-5) & (slack <= 5e-4)
    ss_idx = np.where(ss_mask)[0]
    # expand if too few
    if len(ss_idx) < 20:
        ss_mask = (slack >= 5e-6) & (slack <= 1e-3)
        ss_idx = np.where(ss_mask)[0]
    if len(ss_idx) < 20:
        order = np.argsort(slack)
        # take pairs ranked 450..700 excluding contacts
        cand = [o for o in order if slack[o] > 1e-9][:300]
        ss_idx = np.array(cand, dtype=int)
    # contact neighbours for freezing: slack < 1e-4 (tight + near)
    contact_tol = 1e-4
    # adjacency from current incumbent
    # rebuild adjacency quickly via threshold
    def build_adj(G_, m_, tol):
        adj = [set() for _ in range(n)]
        I, J = iu[0], iu[1]
        cfull = G_[I, J]
        sel = cfull > m_ - tol
        for a, b in zip(I[sel], J[sel]):
            adj[a].add(b); adj[b].add(a)
        return adj

    adj = build_adj(G, m, contact_tol)

    candidates = []  # (val, array)

    SEARCH_END = t0 + 100.0
    trial = 0
    # for speed, cache iu arrays
    Iall = iu[0]; Jall = iu[1]
    while time.time() < SEARCH_END:
        trial += 1
        # periodically refresh incumbent info if improved
        # choose base: mostly incumbent
        base = incumbent
        # refresh G,m,adj every 50 trials from base (base rarely changes in search)
        if trial % 60 == 1:
            G, iu, c, m, slack = pair_info(base)
            # update second shell list
            ss_mask = (slack >= 1e-5) & (slack <= 5e-4)
            tmp_idx = np.where(ss_mask)[0]
            if len(tmp_idx) >= 10:
                ss_idx = tmp_idx
            adj = build_adj(G, m, contact_tol)
            Iall = iu[0]; Jall = iu[1]
            c_all = c
        else:
            # use last G,m
            pass
        # random choice of 1-3 pairs
        kpair = int(rng.integers(1, 4))  # 1..3
        if len(ss_idx) == 0:
            break
        choice = rng.choice(ss_idx, size=min(kpair, len(ss_idx)), replace=False)
        eta = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-3))))  # log-uniform
        radius = float(rng.uniform(0.01, 0.05))
        forced = []
        endpoints = set()
        for ci in choice:
            a = int(Iall[ci]); b = int(Jall[ci])
            cab = float(G[a, b])
            # require eta larger than slack to force movement; if not, bump eta
            sl = m - cab
            e = eta
            if e <= sl:
                e = sl + float(rng.uniform(1e-4, 5e-4))
                e = min(e, 2e-3)
            forced.append((a, b, cab, e))
            endpoints.add(a); endpoints.add(b)
        # movable = endpoints + contact neighbours
        mov_set = set(endpoints)
        for ept in list(endpoints):
            for nb in adj[ept]:
                mov_set.add(nb)
        # cap movable size for speed
        mov = sorted(mov_set)
        # if too large (>45), subsample neighbours randomly
        if len(mov) > 45:
            # keep endpoints, sample rest
            rest = [v for v in mov if v not in endpoints]
            keep = rng.choice(rest, size=45 - len(endpoints), replace=False)
            mov = sorted(list(endpoints) + [int(v) for v in keep])
            mov_set = set(mov)
        loc = {g: i for i, g in enumerate(mov)}
        nmov = len(mov)
        # active window
        active_window = min(20 * radius, 0.02)
        # gather active pairs involving mov
        # c_all = G[Iall,Jall]
        # to avoid O(N^2) each time, only scan pairs where at least one endpoint in mov
        # build mask via isin
        # vectorized: create boolean for I/J in mov_set
        # use set lookup via numpy isin (fast enough: 11628 pairs)
        import numpy as _np  # local alias
        in_mov_I = _np.isin(Iall, mov)
        in_mov_J = _np.isin(Jall, mov)
        involve = in_mov_I | in_mov_J
        active = (c_all > m - active_window) & involve
        act_idx = _np.where(active)[0]
        # limit active constraints to e.g. 800 strongest
        if len(act_idx) > 800:
            order = _np.argsort(-c_all[act_idx])[:800]
            act_idx = act_idx[order]
        # build LP
        # variables: nmov*d + 1
        nv = nmov * d + 1
        nA = len(act_idx) + len(forced)
        if nA == 0:
            continue
        rows_list = []
        cols_list = []
        vals_list = []
        b_ub = []
        # active constraints
        r = 0
        xbase = base  # (n,d)
        for ci in act_idx:
            a = int(Iall[ci]); b = int(Jall[ci])
            cab = float(c_all[ci])
            a_in = a in mov_set
            b_in = b in mov_set
            # row r: coeffs
            if a_in and b_in:
                la = loc[a]; lb = loc[b]
                # dx_a coeffs = x_b
                for k_ in range(d):
                    rows_list.append(r); cols_list.append(la * d + k_); vals_list.append(float(xbase[b, k_]))
                    rows_list.append(r); cols_list.append(lb * d + k_); vals_list.append(float(xbase[a, k_]))
                rows_list.append(r); cols_list.append(nmov * d); vals_list.append(-1.0)
                b_ub.append(-cab)
            elif a_in:
                la = loc[a]
                for k_ in range(d):
                    rows_list.append(r); cols_list.append(la * d + k_); vals_list.append(float(xbase[b, k_]))
                rows_list.append(r); cols_list.append(nmov * d); vals_list.append(-1.0)
                b_ub.append(-cab)
            else:  # b_in only
                lb = loc[b]
                for k_ in range(d):
                    rows_list.append(r); cols_list.append(lb * d + k_); vals_list.append(float(xbase[a, k_]))
                rows_list.append(r); cols_list.append(nmov * d); vals_list.append(-1.0)
                b_ub.append(-cab)
            r += 1
        for (a, b, cab, e) in forced:
            target = m - e
            rhs = target - cab  # need lin <= rhs (negative typically)
            la = loc[a]; lb = loc[b]
            for k_ in range(d):
                rows_list.append(r); cols_list.append(la * d + k_); vals_list.append(float(xbase[b, k_]))
                rows_list.append(r); cols_list.append(lb * d + k_); vals_list.append(float(xbase[a, k_]))
            # no t coefficient
            b_ub.append(rhs)
            r += 1
        try:
            A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(r, nv)).tocsr()
        except Exception:
            continue
        # equality: x_k . dx_k = 0
        neq = nmov
        erows = []; ecols = []; evals = []
        for i, g in enumerate(mov):
            for k_ in range(d):
                erows.append(i); ecols.append(i * d + k_); evals.append(float(xbase[g, k_]))
        try:
            A_eq = coo_matrix((np.array(evals), (np.array(erows), np.array(ecols))), shape=(neq, nv)).tocsr()
        except Exception:
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        bounds = [(-radius, radius)] * (nmov * d) + [(None, None)]
        b_ub = np.array(b_ub)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(neq), bounds=bounds, method="highs")
        except Exception:
            continue
        if res.status != 0:
            continue
        dx = res.x[:-1].reshape(nmov, d)
        # apply
        y = base.copy()
        for i, g in enumerate(mov):
            y[g] = y[g] + dx[i]
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        # check true values
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        v = float(Gy.max())
        # check forced satisfaction (with margin)
        ok = True
        for (a, b, cab, e) in forced:
            dab = float(y[a] @ y[b])
            if dab > m - e * 0.5:  # allow linearization error, half eta
                ok = False
                break
        if not ok:
            continue
        # within 0.001 of incumbent?
        if v > inc_val + 0.001 + 1e-12:
            continue
        # feasible candidate
        candidates.append((v, y.copy()))
        # if actually better, update incumbent immediately
        if v < inc_val - 1e-12:
            incumbent = y.copy()
            inc_val = v
            # refresh info next iteration
            # also keep polishing quickly? no, continue search
            if inc_val < m0 - 1e-7:
                # big win, can break to polish
                pass
        # keep list bounded: keep 10 best
        if len(candidates) > 20:
            candidates.sort(key=lambda t: t[0])
            candidates = candidates[:10]
        # time check
        if time.time() > SEARCH_END:
            break

    # post: smooth-relax + full polish promising candidates
    # sort candidates
    candidates.sort(key=lambda t: t[0])
    # always include incumbent
    pool = []
    seen = set()
    for v, y in candidates[:8]:
        pool.append((v, y))
    # deduplicate roughly? just proceed
    best = incumbent.copy()
    best_val = inc_val
    # if no candidates, pool = incumbent
    if len(pool) == 0:
        pool = [(best_val, best.copy())]
    remaining = BUDGET - (time.time() - t0)
    # reserve 25s for final polish of best
    # polish up to 3 pools with smooth+lp
    npool = min(len(pool), 3)
    for pi in range(npool):
        if time.time() - t0 > BUDGET - 8:
            break
        v0, y0 = pool[pi]
        # only polish if within 0.001 of original? already filtered; also allow incumbent
        # smooth relax short
        # budget per candidate
        time_left = BUDGET - (time.time() - t0)
        # split
        # quick smooth
        try:
            ys = smooth_relax(y0.copy(), iters=300, k=250.0, lr=0.01, rng=rng, noise=0.0)
            vs = max_cos(ys)
            # a bit of annealing with two k phases
            if vs < best_val + 0.002:
                ys2 = smooth_relax(ys, iters=200, k=600.0, lr=0.005, rng=rng, noise=0.0)
                vs2 = max_cos(ys2)
                if vs2 < vs:
                    ys, vs = ys2, vs2
            # lp polish with remaining/ (npool-pi) but cap 20s
            alloc = min(20.0, max(5.0, time_left / (npool - pi) - 1.0))
            yp = lp_polish(ys, alloc * 0.8)
            vp = max_cos(yp)
            if vp < best_val:
                best, best_val = yp, vp
                incumbent = yp.copy()
        except Exception:
            continue
    # final full polish of best with leftover time
    time_left = BUDGET - (time.time() - t0)
    if time_left > 5:
        try:
            best = lp_polish(best, min(time_left - 1.0, 25.0))
            best_val = max_cos(best)
        except Exception:
            pass
    # ensure better or equal to original; return best found (even if slightly worse, return best)
    # compare to x0
    if best_val > m0 and True:
        # if surgery failed to beat, return original polished? return best (could be x0)
        # pick min between best and x0
        if m0 <= best_val:
            best = x0.copy()
            best_val = m0
            # light polish if time remains? already spent; just return
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
