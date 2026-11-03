import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(x, m=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    mask = G > (m - 1e-9)
    deg = np.sum(mask, axis=1).astype(int)
    return deg, m, mask

def _best_hole(X, rng, n_sample=20000):
    d = X.shape[1]
    best_y = None
    best_m = 1e9
    chunk = 5000
    # generate in chunks to save time
    for s in range(0, n_sample, chunk):
        cs = min(chunk, n_sample - s)
        Y = rng.standard_normal(size=(cs, d))
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        S = Y @ X.T
        m = S.max(axis=1)
        idx = int(np.argmin(m))
        if float(m[idx]) < best_m:
            best_m = float(m[idx])
            best_y = Y[idx].copy()
    return best_y, best_m

def _refine_hole_single(X, y, steps=100, a0=60.0, a1=300.0, lr0=0.05, lr1=0.005):
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        s = X @ y
        mx = float(np.max(s))
        w = np.exp(alpha * (s - mx))
        sw = float(np.sum(w)) + 1e-30
        g = w @ X
        gd = float(np.dot(g, y))
        gt = g - gd * y
        y = y - lr * gt / sw
        nrm = float(np.linalg.norm(y))
        if not np.isfinite(nrm) or nrm == 0:
            break
        y = y / nrm
    return y

def _anneal_masked(x, active, steps, a0, a1, lr0, lr1):
    # update only active rows, forces from all rows
    act = np.asarray(active, dtype=bool)
    if not np.any(act):
        return x
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        mx = float(np.max(G))
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        direction = gt / S
        x[act] = x[act] - lr * direction[act]
        x[act] /= (np.linalg.norm(x[act], axis=1, keepdims=True) + 1e-300)
    return x

def _lp_polish(x, seconds, t_deadline):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    pcap = 1500
    while time.time() - start < seconds and time.time() < t_deadline:
        if radius < 1e-13:
            break
        if time.time() > t_deadline - 1.0:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    HOP_END = t0 + 140.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(n, d)
    xb = _norm(xb)
    gval = _max_cos(xb)
    gbest = xb.copy()
    deg0, m0, mask0 = _degrees(xb)
    base_var = float(np.var(deg0.astype(float)))
    # adjacency for BFS
    # build neighbor lists from contact mask
    adj = [np.where(mask0[i])[0].tolist() for i in range(n)]
    rattler = int(np.argmin(deg0))
    order_desc = np.argsort(-deg0, kind="stable")
    hubs4 = [int(o) for o in order_desc[:4]]

    ks = [16, 20, 14, 22, 18, 15, 21, 17]
    cands = []  # (v, var, X)

    for trial in range(8):
        if time.time() > HOP_END - 8.0:
            break
        if time.time() > HARD - 35.0:
            break
        k = int(ks[trial % len(ks)])
        k = max(14, min(22, k))
        # selection
        sel_set = set()
        sel_set.add(rattler)
        for h in hubs4:
            sel_set.add(h)
        need = k - len(sel_set)
        if trial % 2 == 0:
            # BFS cluster seeded at top hub
            seed = hubs4[0] if hubs4 else rattler
            # BFS
            visited = set(sel_set)
            queue = [seed]
            # expand using adjacency; if exhausted, add high-degree nodes
            cluster = []
            qidx = 0
            # include neighbours breadth-first
            q = [seed]
            seen = set([seed]) | set(sel_set)
            # to get BFS order, use queue
            from collections import deque
            dq = deque([seed])
            # if seed already selected, start from its neighbours
            while len(cluster) < need and dq:
                u = dq.popleft()
                for v in adj[u]:
                    if v in seen:
                        continue
                    seen.add(v)
                    cluster.append(v)
                    dq.append(v)
                    if len(cluster) >= need:
                        break
                if not dq and len(cluster) < need:
                    # disconnected: jump to next highest-degree unseen
                    for o in order_desc:
                        o = int(o)
                        if o not in seen:
                            seen.add(o)
                            dq.append(o)
                            break
                    else:
                        break
            for v in cluster[:need]:
                sel_set.add(int(v))
            # fill remainder randomly if short
            while len(sel_set) < k:
                c = int(rng.integers(0, n))
                sel_set.add(c)
        else:
            # random variant: keep rattler+hubs, add random (biased to high degree + random)
            pool = [int(o) for o in order_desc if o not in sel_set]
            # take half from top, half random
            nhalf = need // 2
            for o in pool[:nhalf + 4]:
                if len(sel_set) >= k - (need - nhalf):
                    break
                sel_set.add(int(o))
            while len(sel_set) < k:
                c = int(rng.integers(0, n))
                if c not in sel_set:
                    sel_set.add(c)
        selected = sorted(list(sel_set))[:k]
        sel_bool = np.zeros(n, dtype=bool)
        sel_bool[selected] = True
        keep = xb[~sel_bool].copy()  # frozen core
        Xcur = keep.copy()
        # sequential reinsertion
        feasible = True
        for t_ins in range(k):
            if time.time() > HOP_END - 5.0:
                feasible = False
                break
            y0, _ = _best_hole(Xcur, rng, n_sample=20000)
            y = _refine_hole_single(Xcur, y0, steps=100, a0=60.0, a1=300.0, lr0=0.05, lr1=0.005)
            Xcur = np.vstack([Xcur, y[None, :]])
            Xcur = _norm(Xcur)
        if not feasible:
            continue
        # Xcur is (184,8) but order changed; that's fine (permutation)
        # jointly refine only moved points + within-0.55 neighbours, frozen core fixed
        # last k rows are moved
        try:
            G = Xcur @ Xcur.T
        except Exception:
            continue
        moved_idx = np.arange(Xcur.shape[0] - k, Xcur.shape[0])
        # neighbours: any row with cos > 0.55 to any moved row
        neigh = np.zeros(Xcur.shape[0], dtype=bool)
        neigh[moved_idx] = True
        # G max to moved
        mx_to_moved = np.max(G[:, moved_idx], axis=1)
        neigh = np.logical_or(neigh, mx_to_moved > 0.55)
        # local refine 200 steps (frozen fixed)
        Xcur = _anneal_masked(Xcur, neigh, steps=200, a0=40.0, a1=300.0, lr0=0.02, lr1=0.003)
        if time.time() > HOP_END:
            # still evaluate
            pass
        # unfreeze short global anneal 60 steps
        all_active = np.ones(Xcur.shape[0], dtype=bool)
        Xcur = _anneal_masked(Xcur, all_active, steps=60, a0=40.0, a1=300.0, lr0=0.015, lr1=0.003)
        try:
            v = _max_cos(Xcur)
        except Exception:
            continue
        deg, mcur, _ = _degrees(Xcur, m=v)
        var = float(np.var(deg.astype(float)))
        # acceptance: variance drops and max within reach / improves
        # allow small degradation pre-LP but require balance improvement
        if var < base_var and v < gval + 0.003:
            cands.append((v, var, Xcur.copy()))
            if v < gval:
                gval = v
                gbest = Xcur.copy()
        elif v < gval and var < base_var * 1.05:
            cands.append((v, var, Xcur.copy()))
            gval = v
            gbest = Xcur.copy()
        # update hubs/rattler info from gbest occasionally? keep original for diversity
        # refresh adjacency every 4 trials from gbest to follow new topology
        if trial == 3 and len(cands) > 0:
            cands_sorted_tmp = sorted(cands, key=lambda t: t[0])
            _, _, Xb_tmp = cands_sorted_tmp[0]
            degT, _, maskT = _degrees(Xb_tmp)
            adj = [np.where(maskT[i])[0].tolist() for i in range(n)]
            rattler = int(np.argmin(degT))
            order_desc = np.argsort(-degT, kind="stable")
            hubs4 = [int(o) for o in order_desc[:4]]

    # final capped 20s LP on top candidates with variance improvement
    if cands:
        cands.sort(key=lambda t: t[0])
        budget_total = min(20.0, max(0.0, HARD - 2.0 - time.time()))
        # at most 2 LPs
        num_lp = min(2, len(cands))
        per = budget_total / max(num_lp, 1)
        per = min(per, 10.0)
        for i in range(num_lp):
            if time.time() > HARD - 2.0:
                break
            v, var, Xc = cands[i]
            if v > gval + 0.01:
                continue
            secs = min(per, HARD - 1.0 - time.time())
            if secs < 1.5:
                break
            Xp = _lp_polish(Xc, seconds=secs, t_deadline=HARD - 1.0)
            try:
                vp = _max_cos(Xp)
            except Exception:
                continue
            degp, _, _ = _degrees(Xp, m=vp)
            varp = float(np.var(degp.astype(float)))
            # accept only if variance not exploded and max improves
            if vp < gval and varp < base_var * 1.1:
                gval = vp
                gbest = _norm(Xp)
            elif vp < gval and i == 0:
                # accept best anyway if it improves record
                gval = vp
                gbest = _norm(Xp)
    gbest = _norm(gbest)
    return gbest
