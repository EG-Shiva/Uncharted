import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, tol=2e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rm = G.max(axis=1)
    cnt = (G >= m - tol).sum(axis=1)
    return m, rm, cnt, G

def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        if int(np.count_nonzero(sel)) == 0:
            break
        if int(np.count_nonzero(sel)) > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            if int(np.count_nonzero(sel)) > cap + 600:
                ids = np.where(sel)[0][::2][:cap]
                ns = np.zeros_like(sel)
                ns[ids] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None]*d+cd, J[:, None]*d+cd, np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs")
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
            x, best = y, v
            radius = min(radius*1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters-1, 1)
        T = T0*(T1/T0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G-m)/T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        rs = W.sum(axis=1, keepdims=True)
        step = gt/np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step/np.maximum(sn, 1.0)
        x = x - lr*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_points(x, n_need, rng, n_samp=4000):
    # sample sphere, keep farthest from x, refine slightly
    cand = rng.normal(size=(n_samp, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ x.T
    sc = s.max(axis=1)
    order = np.argsort(sc)[:n_need*8]
    pool = cand[order]
    out = []
    for v0 in pool:
        v = v0.copy()
        for _ in range(15):
            dots = x @ v
            mm = dots.max()
            w = np.exp(50.0*(dots-mm))
            g = w @ x
            gt = g - np.dot(g, v)*v
            nrm = np.linalg.norm(gt)
            if nrm < 1e-12:
                break
            v = v - 0.12*gt/nrm
            v /= np.linalg.norm(v)
        out.append(v)
        if len(out) >= n_need*8:
            break
    out = np.array(out)
    sc2 = (out @ x.T).max(axis=1)
    idx = np.argsort(sc2)
    # greedy diverse pick
    sel = []
    for j in idx:
        if len(sel) >= n_need:
            break
        if len(sel)==0:
            sel.append(j)
        else:
            if np.max(out[sel] @ out[j]) < 0.95:
                sel.append(j)
    while len(sel) < n_need:
        sel.append(idx[len(sel)])
    return out[np.array(sel)]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(0)
    try:
        x0 = np.load("best.npy").astype(float)
    except Exception:
        x0 = np.load("record.npy").astype(float)
    x0 = x0/np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x0.shape
    best_x = x0.copy()
    best_v = _max_cos(best_x)
    # quick contact info
    m, rm, cnt, G = _contacts(best_x)
    order_load = np.argsort(-cnt)  # overloaded first
    rattlers = np.where(cnt <= 1)[0]
    # main basin loop
    it = 0
    # reserve 22s for final polish
    while time.time()-t0 < BUDGET-22:
        it += 1
        time_left = BUDGET-22-(time.time()-t0)
        if time_left <= 2:
            break
        y = best_x.copy()
        mode = it % 5
        # choose targets
        m2, rm2, cnt2, _ = _contacts(best_x) if (it % 3 == 1) else (m, rm, cnt, G)
        overload = order_load[:12]
        if mode == 0:
            # relocate rattlers/near-rattlers to voids
            k = 3
            # pick k worst-or-rattler indices
            cand_idx = list(rattlers[:4]) if len(rattlers)>0 else []
            # add most overloaded to diversify
            while len(cand_idx) < k:
                cand_idx.append(int(overload[rng.randint(len(overload))]))
            cand_idx = np.array(cand_idx[:k])
            keep_mask = np.ones(n, bool); keep_mask[cand_idx]=False
            voids = _void_points(y[keep_mask], k, rng, n_samp=3500)
            y[cand_idx] = voids[:len(cand_idx)]
            # small shake of neighbours of overloaded
            sig = 0.015
            nb = rng.randint(0, n, size=8)
            y[nb] += sig*rng.normal(size=(len(nb), d))
        elif mode == 1:
            # shake overloaded cluster + random
            idx = list(overload[rng.choice(len(overload), size=6, replace=False)])
            idx += list(rng.randint(0, n, size=10))
            idx = np.array(idx)
            sig = 0.03 + 0.02*rng.rand()
            y[idx] += sig*rng.normal(size=(len(idx), d))
        elif mode == 2:
            # global moderate shake
            sig = 0.02
            y += sig*rng.normal(size=(n, d))
        elif mode == 3:
            # shake worst-pair endpoints
            Gc = best_x @ best_x.T
            np.fill_diagonal(Gc, -2.0)
            # find top pairs
            iu = np.triu_indices(n, 1)
            c = Gc[iu]
            top = np.argsort(-c)[:25]
            pts = np.unique(np.concatenate([iu[0][top], iu[1][top]]))
            sig = 0.035
            y[pts] += sig*rng.normal(size=(len(pts), d))
            # plus few random
            r = rng.randint(0, n, size=5)
            y[r] += 0.02*rng.normal(size=(5, d))
        else:
            # void relocate one + local shake
            k = 2
            worst = np.argsort(-rm2)[:6]
            cand_idx = rng.choice(worst, size=1, replace=False).tolist()
            if len(rattlers)>0 and rng.rand()<0.7:
                cand_idx.append(int(rattlers[rng.randint(len(rattlers))]))
            cand_idx = np.array(cand_idx[:k])
            keep_mask = np.ones(n, bool); keep_mask[cand_idx]=False
            voids = _void_points(y[keep_mask], len(cand_idx), rng, n_samp=3500)
            y[cand_idx] = voids
            y += 0.008*rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # fast smooth screen (no LP yet)
        T0 = 0.004 + 0.003*rng.rand()
        y = _smooth(y, iters=220, T0=T0, T1=0.0004, lr0=0.02, lr1=0.004)
        v_s = _max_cos(y)
        # only LP if promising (within 0.004 of best or better)
        if v_s > best_v + 0.004 and rng.rand() > 0.05:
            continue
        # short LP polish
        allow = min(2.2, max(0.6, (BUDGET-22-(time.time()-t0))/8.0))
        y2 = _lp_polish(y, seconds=allow, r0=2e-3)
        v2 = _max_cos(y2)
        if v2 < best_v - 1e-12:
            best_x, best_v = y2, v2
            m, rm, cnt, G = _contacts(best_x)
            order_load = np.argsort(-cnt)
            rattlers = np.where(cnt <= 1)[0]
        elif v_s < best_v - 1e-12 and v2 < best_v + 1e-4:
            # keep smoothed version's basin info? no
            pass
        # occasional refresh of contact info
        if it % 10 == 0:
            m, rm, cnt, G = _contacts(best_x)
            order_load = np.argsort(-cnt)
            rattlers = np.where(cnt <= 1)[0]
    # final polish of best
    remain = BUDGET - (time.time()-t0) - 1.0
    if remain > 3:
        best_x = _lp_polish(best_x, seconds=min(remain, 20.0), r0=1.5e-3)
        best_v = _max_cos(best_x)
    best_x = best_x/np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
