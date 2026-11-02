import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, iu, tol=1e-7):
    G = x @ x.T
    gall = G[iu]
    mx = float(gall.max())
    tight = gall > mx - tol
    cnt = np.zeros(x.shape[0], dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return G, mx, cnt

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _adam_subset(x0, active, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    Iu, Ju = iu
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    act = np.zeros(n, dtype=bool)
    act[np.asarray(list(active), dtype=int)] = True
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        grad = np.zeros_like(x)
        # weighted accumulation without dense matrix
        np.add.at(grad, Iu, w[:, None] * x[Ju])
        np.add.at(grad, Ju, w[:, None] * x[Iu])
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        gt[~act] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x[act] = x[act] - lr * step[act]
        nr = np.linalg.norm(x[act], axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x[act] /= nr
    return x

def _greedy_refill(rng, kept_mat, already, n_samp=100000, batch=20000, jitter=0.005):
    best_vec = None
    best_score = np.inf
    nb = n_samp // batch
    for _ in range(nb):
        cand = rng.normal(size=(batch, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        s = cand @ kept_mat.T
        m1 = s.max(axis=1)
        if already is not None and len(already) > 0:
            a = np.asarray(already)
            s2 = cand @ a.T
            m2 = s2.max(axis=1)
            m = np.maximum(m1, m2)
        else:
            m = m1
        j = int(np.argmin(m))
        if float(m[j]) < best_score:
            best_score = float(m[j])
            best_vec = cand[j].copy()
    v = best_vec + rng.normal(0, jitter, size=4)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        v = best_vec
    else:
        v /= nrm
    return v

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    ks = [4, 6, 8, 12]
    it = 0
    # surgery loop
    while time.time() - t0 < 138.0:
        if time.time() - t0 > HARD - 32:
            break
        it += 1
        k = ks[(it - 1) % len(ks)]
        # vary sample size: 100k-200k per spec; use 100k normally, 160k for small k
        if k <= 6:
            n_samp = 120000
        elif k <= 8:
            n_samp = 100000
        else:
            n_samp = 100000
        G, mx, cnt = _contacts(x_best, iu, tol=1e-7)
        # rattler = minimal contacts
        minc = cnt.min()
        ratt_cands = np.where(cnt == minc)[0]
        ratt = int(ratt_cands[rng.integers(len(ratt_cands))]) if len(ratt_cands) > 1 else int(ratt_cands[0])
        # nearest neighbours of rattler (largest cosine)
        order = np.argsort(-G[ratt])
        order = order[order != ratt]
        hubs = np.where(cnt >= 7)[0]
        hubs = hubs[hubs != ratt]
        # build deletion set
        D = [ratt]
        # take ~half from nearest neighbours
        n_nn = min(k - 1, max(1, k // 2))
        added = 0
        for idx in order:
            if int(idx) not in D:
                D.append(int(idx))
                added += 1
                if added >= n_nn:
                    break
        # fill rest from hubs randomly
        need = k - len(D)
        if need > 0 and len(hubs) > 0:
            # exclude already in D
            hub_pool = [h for h in hubs if h not in D]
            rng.shuffle(hub_pool)
            take = min(need, len(hub_pool))
            # random subset: take first `take` after shuffle
            D.extend([int(h) for h in hub_pool[:take]])
            need = k - len(D)
        if need > 0:
            pool = [i for i in range(n) if i not in D]
            pick = rng.choice(pool, size=need, replace=False)
            D.extend([int(p) for p in pick])
        D = np.array(D[:k], dtype=int)
        kept_idx = np.array([i for i in range(n) if i not in set(D.tolist())], dtype=int)
        kept_mat = x_best[kept_idx].copy()
        # refill one at a time greedily
        new_pts = []
        # jitter magnitude alternates slightly for diversity
        jit = 0.004 + 0.004 * ((it % 3) / 2.0)
        for s in range(k):
            already = np.array(new_pts) if len(new_pts) > 0 else None
            v = _greedy_refill(rng, kept_mat if len(new_pts) == 0 else np.vstack([kept_mat, np.array(new_pts)]) if len(new_pts) > 0 else kept_mat,
                               already=None if len(new_pts) == 0 else None,
                               n_samp=n_samp, batch=20000, jitter=jit)
            # Note: kept_mat+new_pts already combined above when needed; simplify:
            # Actually _greedy_refill handles kept+already; we passed combined as kept_mat so already=None
            new_pts.append(v)
        new_pts = np.array(new_pts)
        y = x_best.copy()
        y[D] = new_pts
        # active set: D plus neighbours with cosine > 0.75 to any new point
        # compute cosines to new points
        sims = y[kept_idx] @ new_pts.T  # (n-k, k)
        neigh_flag = sims.max(axis=1) > 0.75
        neigh_idx = kept_idx[neigh_flag]
        active1 = np.concatenate([D, neigh_idx])
        # stage 1: optimise only refilled + neighbours, beta 50->300
        s1_steps = 600
        # occasionally use 900 steps
        if it % 4 == 0:
            s1_steps = 900
        y1 = _adam_subset(y, active1, s1_steps, 50.0, 300.0, 0.006, 0.0004, iu)
        v1 = _max_cos(y1)
        # keep candidate even if up to 0.005 worse pre-LP
        if v1 > best_val + 0.005:
            continue
        # stage 2: unfreeze all for ~900 steps
        s2_steps = 900
        # beta schedule slight variation
        if it % 2 == 0:
            b0, b1 = 50.0, 300.0
        else:
            b0, b1 = 40.0, 280.0
        y2 = _adam_subset(y1, np.arange(n), s2_steps, b0, b1, 0.005, 0.0003, iu)
        v2 = _max_cos(y2)
        if v2 > best_val + 0.005:
            # still allow occasional LP if close-ish and time permits? skip
            continue
        remain = HARD - (time.time() - t0) - 22.0
        if remain < 1.0:
            # no time for LP; direct accept if better
            if v2 < best_val:
                x_best, best_val = y2 / np.linalg.norm(y2, axis=1, keepdims=True), v2
            break
        b = min(3.0, max(1.2, remain / 10.0))
        # only LP if promising: v2 < best+0.005 (already) -> go
        yp, vp = _lp_polish(y2, b)
        if vp < best_val - 1e-12:
            x_best, best_val = yp, vp
        # sideways acceptance rarely to change basin even if not better (keeps diversity)
        elif vp < best_val + 1e-9 and rng.random() < 0.05:
            # adopt different graph for further surgery without losing much
            x_best = yp
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
