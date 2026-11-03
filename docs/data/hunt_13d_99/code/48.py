import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _refine_point(v0, Others, rng, steps=120, alpha=70.0):
    v = v0.copy()
    v = v / (np.linalg.norm(v) + 1e-12)
    dots0 = Others @ v
    best_v = v.copy()
    best_s = float(np.max(dots0))
    for s in range(steps):
        dots = Others @ v
        m = float(np.max(dots))
        w = np.exp(alpha * (dots - m))
        sw = float(np.sum(w)) + 1e-12
        grad = (w @ Others) / sw
        gt = grad - float(grad @ v) * v
        ng = float(np.linalg.norm(gt))
        if ng < 1e-12:
            # random tangent kick
            r = rng.standard_normal(size=v.shape)
            r = r - float(r @ v) * v
            nr = float(np.linalg.norm(r))
            if nr < 1e-12:
                continue
            r = r / nr
            step = 0.005 * (1.0 - s / max(steps, 1))
            v = v + step * r
            v = v / np.linalg.norm(v)
            continue
        gt = gt / ng
        step = 0.03 * (1.0 - 0.8 * s / max(steps, 1))
        # decay if close to best
        v = v - step * gt
        v = v / np.linalg.norm(v)
        sc = float(np.max(Others @ v))
        if sc < best_s:
            best_s = sc
            best_v = v.copy()
            # increase alpha to focus
            # keep going
        # occasional random kick to escape
        if s % 40 == 39:
            r = rng.standard_normal(size=v.shape)
            r = r - float(r @ v) * v
            r = r / (np.linalg.norm(r) + 1e-12)
            vt = v + 0.005 * r
            vt = vt / np.linalg.norm(vt)
            sct = float(np.max(Others @ vt))
            if sct < float(np.max(Others @ v)):
                v = vt
    return best_v

def _small_lp_polish(X, S, seconds_left=10.0, iters=6, r0=8e-3):
    # X (99,13) normalized, S list of indices to move
    n, d = X.shape
    S = list(map(int, S))
    k = len(S)
    if k == 0:
        return X
    g2l = {g: l for l, g in enumerate(S)}
    ar = np.arange(d)
    iu0, iu1 = np.triu_indices(n, 1)
    radius = r0
    cur = X.copy()
    cur_m = _max_cos(cur)
    for _it in range(iters):
        if seconds_left is not None and seconds_left <= 0:
            break
        G = cur @ cur.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        win = max(30.0 * radius, 3e-6)
        thresh = m - win
        # only pairs touching S
        isS = np.zeros(n, dtype=bool)
        isS[S] = True
        touch = isS[iu0] | isS[iu1]
        sel = (c_all > thresh) & touch
        # ensure at least some
        if int(np.sum(sel)) < 80:
            # take top touching
            tc = np.where(touch)[0]
            order = np.argsort(-c_all[tc])
            need = min(200, len(tc))
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[tc[order[:need]]] = True
            sel = sel | sel2
        I = iu0[sel].astype(np.int64)
        J = iu1[sel].astype(np.int64)
        c_sel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        # build sparse A_ub: p x (k*d+1)
        NV = k * d + 1
        # count nnz: for each row, d (for I if in S) + d (for J if in S) +1
        rows_list = []
        cols_list = []
        vals_list = []
        # vectorize per row type
        inS_I = np.array([g in g2l for g in I])
        inS_J = np.array([g in g2l for g in J])
        # To build quickly use python loop (p~few hundred, fine)
        rids = []
        cids = []
        vls = []
        for r in range(p):
            i = int(I[r]); j = int(J[r])
            ci = c_sel[r]
            # linearised: x_j·dx_i + x_i·dx_j + c <= t  -> A=[... -1], b=-c
            # need b_ub = -c_sel? Actually constraint A*vars <= b with vars including t:
            # x_j dx_i + x_i dx_j - t <= -c
            if i in g2l:
                li = g2l[i] * d
                xj = cur[j]
                for a in range(d):
                    rids.append(r); cids.append(li + a); vls.append(float(xj[a]))
            if j in g2l:
                lj = g2l[j] * d
                xi = cur[i]
                for a in range(d):
                    rids.append(r); cids.append(lj + a); vls.append(float(xi[a]))
            rids.append(r); cids.append(NV - 1); vls.append(-1.0)
        try:
            A_ub = coo_matrix((np.array(vls, dtype=float), (np.array(rids), np.array(cids))), shape=(p, NV)).tocsr()
            # equality: x_i·dx_i=0
            er = []
            ec = []
            ev = []
            for l, g in enumerate(S):
                for a in range(d):
                    er.append(l); ec.append(l * d + a); ev.append(float(cur[g, a]))
            A_eq = coo_matrix((np.array(ev, dtype=float), (np.array(er), np.array(ec))), shape=(k, NV)).tocsr()
            obj = np.zeros(NV); obj[-1] = 1.0
            b_ub = -c_sel
            b_eq = np.zeros(k)
            bounds = [(-radius, radius)] * (k * d) + [(None, m)]
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs",
                          options={"presolve": True, "time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            if radius < 1e-9:
                break
            continue
        dx = res.x[:-1].reshape(k, d)
        nxt = cur.copy()
        for l, g in enumerate(S):
            nxt[g] = cur[g] + dx[l]
        nxt[S] = _norm_rows(nxt[S])
        v = _max_cos(nxt)
        if v < cur_m - 1e-12:
            cur = nxt
            cur_m = v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-9:
                break
    return cur

def _full_lp_short(X, radius=3e-4, iters=2, window=8e-6, kmin=700):
    n, d = X.shape
    cur = _norm_rows(X.copy())
    best = cur.copy()
    best_m = _max_cos(cur)
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    r = radius
    for _ in range(iters):
        G = cur @ cur.T
        c = G[iu]
        m = float(np.max(c))
        thresh = m - max(window, 20.0 * r)
        sel = c > thresh
        if int(np.sum(sel)) < kmin:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:kmin]] = True
            sel = sel | sel2
        I = iu[0][sel].astype(np.int64); J = iu[1][sel].astype(np.int64)
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        NV = n * d + 1
        # build sparse
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([cur[J], cur[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, NV)).tocsr()
            A_eq = coo_matrix((cur.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, NV)).tocsr()
            obj = np.zeros(NV); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-r, r)] * (n * d) + [(None, None)], method="highs",
                          options={"presolve": True, "time_limit": 3.0})
        except Exception:
            r *= 0.5
            continue
        if res.status != 0 or res.x is None:
            r *= 0.5
            continue
        y = cur + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best_m:
            best_m = v
            best = y.copy()
            cur = y
            r = min(r * 1.3, 0.005)
        else:
            r *= 0.5
            break
    return best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    n, d = 99, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        cands.append(_norm_rows(rng0.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_val = float(vals[bi])
    Xcur = best_X.copy()
    cur_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)
    # temperature for Metropolis on post-polish max
    T0 = 2.0e-6
    it = 0
    # main surgery loop ~140s
    while time.time() - t0 < 140.0:
        it += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 15.0:
            break
        # --- analyse plateau graph ---
        G = Xcur @ Xcur.T
        c_all = G[iu0, iu1]
        mcur = float(np.max(c_all))
        slack = mcur - c_all
        plateau = slack < 2e-6
        # degrees in plateau graph
        deg = np.zeros(n, dtype=int)
        # per-point max
        per_max = np.full(n, -2.0)
        # accumulate
        Ip = iu0[plateau]; Jp = iu1[plateau]
        for a, b in zip(Ip, Jp):
            deg[int(a)] += 1
            deg[int(b)] += 1
        # per point max cosine
        # fast: for each i max over j
        # use G max per row (excluding diag)
        G2 = G.copy()
        np.fill_diagonal(G2, -2.0)
        per_max = np.max(G2, axis=1)
        # strict contacts
        strict = slack < 1e-9
        sdeg = np.zeros(n, dtype=int)
        Is = iu0[strict]; Js = iu1[strict]
        for a, b in zip(Is, Js):
            sdeg[int(a)] += 1
            sdeg[int(b)] += 1
        deg1_pts = np.where(sdeg == 1)[0]
        # choose cluster size 4-8
        fsize = 4 + ((it - 1) % 5)  # 4,5,6,7,8 cycle
        # adjacency for BFS (plateau)
        adj = [[] for _ in range(n)]
        for a, b in zip(Ip, Jp):
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        # seed selection
        seed = None
        use_deg1 = (it % 2 == 0) and len(deg1_pts) > 0
        if use_deg1:
            # pick first deg1 point + its neighbour as forced members
            forced = list(map(int, deg1_pts[:2]))
            # add their plateau neighbours
            extra = set(forced)
            for f in forced:
                for nb in adj[f][:4]:
                    extra.add(int(nb))
            # seed is forced[0]
            seed = int(forced[0])
            # BFS starting from seed but ensure forced included
            visited = set([seed])
            queue = [seed]
            # pre-add forced
            for f in forced:
                if f not in visited and len(visited) < fsize + 4:
                    visited.add(f)
            # BFS expand ordered by deg desc
            qidx = 0
            while len(visited) < fsize and qidx < len(queue):
                u = queue[qidx]; qidx += 1
                nbrs = sorted(adj[u], key=lambda v: -deg[v])
                for v in nbrs:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
                        if len(visited) >= fsize:
                            break
                # if queue exhausted, add global high-deg node
                if qidx >= len(queue) and len(visited) < fsize:
                    cand_order = np.argsort(-deg)
                    for v in cand_order:
                        v = int(v)
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                            break
            F = sorted(list(visited))[:fsize]
            # ensure forced members kept if too many: keep forced + top deg
            if len(F) > fsize:
                F = F[:fsize]
            # if forced not fully inside (because visited truncated), force them
            for f in forced:
                if f not in F and len(F) == fsize:
                    # replace lowest-deg member
                    degsF = [(deg[x], x) for x in F if x not in forced]
                    if degsF:
                        degsF.sort()
                        F.remove(degsF[0][1])
                        F.append(f)
        else:
            # seed = highest per_max (tie -> highest deg)
            order = np.lexsort((-deg), -per_max)  # descending per_max? lexsort ascending, so negate
            # Actually np.lexsort keys ascending; we want largest per_max first
            # Recompute: sort by (-per_max, -deg) ascending = largest first
            # lexsort(( -deg, -per_max)) gives primary -per_max? last key primary.
            # Simpler argsort
            score = per_max + 1e-6 * deg
            seed = int(np.argmax(score + 1e-12 * np.random.default_rng(1000 + it).standard_normal(n) * 0.0))
            # BFS
            visited = set([seed])
            queue = [seed]
            qidx = 0
            while len(visited) < fsize:
                if qidx < len(queue):
                    u = queue[qidx]; qidx += 1
                    nbrs = sorted(adj[u], key=lambda v: (-deg[v], -per_max[v]))
                    added = False
                    for v in nbrs:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                            added = True
                            if len(visited) >= fsize:
                                break
                    if not added:
                        # isolated: jump to high-deg unvisited connected to visited?
                        cand_order = np.argsort(-(per_max + 1e-6 * deg))
                        found = False
                        for v in cand_order:
                            v = int(v)
                            if v not in visited:
                                # prefer one adjacent to visited
                                visited.add(v)
                                queue.append(v)
                                found = True
                                break
                        if not found:
                            break
                else:
                    break
            F = sorted(list(visited))[:fsize]
        F = list(map(int, F))
        if len(F) < 4:
            # fallback: top deg nodes
            F = list(map(int, np.argsort(-deg)[:fsize]))
        # --- block surgery: largest-hole insertion ---
        rng = np.random.default_rng(5000 + it * 17)
        # shared candidate pool
        try:
            pool = rng.standard_normal(size=(5000, d))
        except Exception:
            continue
        pool = _norm_rows(pool)
        Xprop = Xcur.copy()
        orderF = list(F)
        # random order
        perm = rng.permutation(len(orderF))
        orderF = [orderF[i] for i in perm]
        for kk in orderF:
            mask = np.ones(n, dtype=bool); mask[kk] = False
            Others = Xprop[mask]
            # scores for pool
            D = pool @ Others.T  # 5000x98
            scores = np.max(D, axis=1)
            bi2 = int(np.argmin(scores))
            v0 = pool[bi2]
            # local minimax refinement
            vref = _refine_point(v0, Others, rng, steps=100, alpha=70.0)
            Xprop[kk] = vref
        # neighbours with dot>0.15 to any freed
        Gp = Xprop @ Xprop.T
        Sset = set(F)
        for kk in F:
            row = Gp[kk]
            for j in range(n):
                if j not in Sset and row[j] > 0.15:
                    Sset.add(j)
        Slist = sorted(list(Sset))
        # cap at 20: keep F + top dots
        if len(Slist) > 20:
            # score neighbours by max dot to F
            neigh = [j for j in Slist if j not in F]
            smax = []
            for j in neigh:
                smax.append(float(np.max([Gp[j, kk] for kk in F])))
            idx = np.argsort(-np.array(smax))
            keep = [neigh[i] for i in idx[:20 - len(F)]]
            Slist = sorted(F + keep)
        # small LP polish on S
        try:
            Xpol = _small_lp_polish(Xprop, Slist, seconds_left=remain, iters=5, r0=6e-3)
        except Exception:
            Xpol = Xprop
        v_pol = _max_cos(Xpol)
        # short full polish if promising (within 6e-6 of current)
        did_full = False
        if v_pol < cur_val + 6e-6:
            try:
                Xpol2 = _full_lp_short(Xpol, radius=3e-4, iters=1, window=6e-6, kmin=600)
                v_pol2 = _max_cos(Xpol2)
                # keep better of the two
                if v_pol2 < v_pol:
                    Xpol = Xpol2
                    v_pol = v_pol2
                did_full = True
            except Exception:
                pass
        # acceptance
        if v_pol < best_val - 1e-12:
            best_val = v_pol
            best_X = Xpol.copy()
            Xcur = Xpol.copy()
            cur_val = v_pol
        elif v_pol < cur_val - 1e-12:
            Xcur = Xpol.copy()
            cur_val = v_pol
        else:
            # Metropolis with decaying temperature
            T = T0 * (0.995 ** (it // 10))
            T = max(T, 3e-7)
            delta = v_pol - cur_val
            # only allow small uphill
            if delta < 8e-6:
                try:
                    prob = float(np.exp(-delta / T)) * 0.25
                except Exception:
                    prob = 0.0
                u = float(rng.random())
                if u < prob:
                    Xcur = Xpol.copy()
                    cur_val = v_pol
            # else stay; occasionally revert to best if drifted too far
            if cur_val > best_val + 1.2e-5:
                Xcur = best_X.copy()
                cur_val = best_val
    # final polish of global best with remaining time
    remain = BUDGET - (time.time() - t0)
    if remain > 8.0:
        try:
            Xf = _full_lp_short(best_X, radius=4e-4, iters=3, window=1e-5, kmin=900)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_X = Xf
        except Exception:
            pass
        # extra small full iterations if time left
        remain2 = BUDGET - (time.time() - t0)
        if remain2 > 8.0:
            try:
                Xf2 = _small_lp_polish(best_X, list(range(99)), seconds_left=remain2, iters=2, r0=4e-4)
                vf2 = _max_cos(Xf2)
                if vf2 < best_val:
                    best_X = Xf2
                    best_val = vf2
            except Exception:
                pass
    best_X = _norm_rows(best_X)
    return best_X
