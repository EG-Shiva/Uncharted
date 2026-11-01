import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def relax_softmax(X, steps, T0, T1, lr, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    # geometric schedule
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        if t_deadline is not None and (s % 200 == 0) and time.time() > t_deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        # exp((G-m)/T)
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)  # per-row sum
        # avoid div by zero
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme rows
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # optional clip to 2.0
        scale = np.ones_like(gn)
        mask = (gn[:, 0] > 2.0)
        scale[mask] = 2.0 / gn[mask]
        Gt = Gt * scale
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def single_refine(s, R, steps=400, T=0.004, lr=0.15, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def find_hole_point(R, rng, n_sample=20000, n_cand=4):
    n, d = R.shape
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    # scores = max dot with R
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        # two temperatures refine
        s1 = single_refine(s0, R, steps=300, T=0.005, lr=0.15)
        s1 = single_refine(s1, R, steps=300, T=0.0015, lr=0.08)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def lp_polish(x, seconds):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 175.0
    rng = np.random.default_rng(0)
    # load starting points
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = _norm_rows(np.asarray(xb, dtype=np.float64))
        if xb.shape == (71, 12):
            cands.append(xb)
    if xr is not None:
        xr = _norm_rows(np.asarray(xr, dtype=np.float64))
        if xr.shape == (71, 12):
            cands.append(xr)
    if len(cands) == 0:
        X0 = _norm_rows(rng.standard_normal(size=(71, 12)))
        cands.append(X0)
    bestX = min(cands, key=lambda z: max_cos(z)).copy()
    bestVal = max_cos(bestX)
    curX = bestX.copy()
    curVal = bestVal
    # quick initial relax to diversify? keep best
    it = 0
    # reserve final polish
    explore_deadline = t0 + 148.0
    # temperature for basin hopping acceptance
    bh_T = 0.0015
    while time.time() < explore_deadline:
        it += 1
        # time check for final polish reserve
        if time.time() > explore_deadline:
            break
        do_hole = (it % 3 == 0)
        # vary steps/lr/T randomly for diversity, deterministic via rng
        if do_hole:
            # choose victim(s) from bestX based on contacts
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            # contact threshold near max
            thr = bestVal - 0.008
            cnt = (G > thr).sum(axis=1)
            # pick among top-8 most loaded + random
            order = np.argsort(-cnt)
            topk = order[:8]
            # alternate 1 or 2 removals
            nrem = 1 if (it % 6 == 0) else (2 if (it % 9 == 0) else 1)
            # mostly pick worst, sometimes random top
            if rng.random() < 0.7:
                victims = [int(topk[0])]
                if nrem == 2:
                    victims.append(int(topk[1]))
            else:
                victims = [int(rng.choice(topk))]
                if nrem == 2:
                    victims.append(int(rng.choice(topk)))
            victims = sorted(set(victims))
            mask = np.ones(71, dtype=bool)
            mask[victims] = False
            R = bestX[mask]  # 70 or 69 rows (use best as base)
            # if 2 removals, insert 2 holes sequentially
            Xt_base = R
            for _ in range(len(victims)):
                ns = 16000 if len(victims) == 1 else 20000
                s_new = find_hole_point(Xt_base, rng, n_sample=ns, n_cand=4)
                Xt_base = np.vstack([Xt_base, s_new[None, :]])
            Xt = Xt_base  # (71,12)
            # global relax with annealing
            T0 = float(rng.choice([0.008, 0.012, 0.006]))
            T1 = float(rng.choice([0.0008, 0.0004, 0.0012]))
            lr = float(rng.choice([0.04, 0.06, 0.03]))
            steps = int(rng.choice([1800, 2500, 3200]))
            mu = 0.9
            Xt = relax_softmax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu,
                               t_deadline=explore_deadline)
        else:
            # shake subset of best or cur
            base = bestX if rng.random() < 0.7 else curX
            Xt = base.copy()
            k = int(rng.choice([1, 2, 2, 3, 4]))
            idx = rng.choice(71, size=k, replace=False)
            # log-uniform sigma
            ls = rng.uniform(np.log(0.015), np.log(0.5))
            sigma = float(np.exp(ls))
            # occasionally replace fully random
            if rng.random() < 0.08:
                Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
            else:
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
                Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.01, 0.007, 0.015]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003]))
            lr = float(rng.choice([0.05, 0.03, 0.07]))
            steps = int(rng.choice([1500, 2200, 3000]))
            Xt = relax_softmax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9,
                               t_deadline=explore_deadline)
        v = max_cos(Xt)
        if v < bestVal:
            bestVal = v
            bestX = Xt.copy()
            curX = Xt.copy()
            curVal = v
            # squeeze promising with tiny extra low-T relax immediately
            if time.time() < explore_deadline - 2:
                Xt2 = relax_softmax(bestX.copy(), steps=800, T0=0.001, T1=0.0002, lr=0.02, mu=0.9,
                                    t_deadline=explore_deadline)
                v2 = max_cos(Xt2)
                if v2 < bestVal:
                    bestVal = v2
                    bestX = Xt2.copy()
                    curX = Xt2.copy()
                    curVal = v2
        else:
            # basin-hopping acceptance for exploration
            if v < curVal or rng.random() < np.exp(-(v - curVal) / bh_T):
                curX = Xt.copy()
                curVal = v
        # occasional restart cur to best
        if it % 25 == 0:
            curX = bestX.copy()
            curVal = bestVal
    # final polish with remaining time
    remain = TIME_LIMIT - (time.time() - t0)
    # ensure at least a bit
    if remain > 4:
        # first a final low-T relax to tighten before LP
        t1 = time.time()
        budget = min(8.0, remain - 4.0)
        if budget > 1:
            bestX = relax_softmax(bestX, steps=4000, T0=0.002, T1=0.00015, lr=0.02, mu=0.9,
                                  t_deadline=t1 + budget)
            bestVal = max_cos(bestX)
        remain2 = TIME_LIMIT - (time.time() - t0)
        if remain2 > 3:
            bestX = lp_polish(bestX, seconds=remain2 - 0.5)
    bestX = _norm_rows(bestX)
    return bestX
