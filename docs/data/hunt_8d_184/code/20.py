import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _anneal(x, steps, a0, a1, lr0, lr1, rng=None, momentum=0.0):
    n, d = x.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    vel = np.zeros_like(x)
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
        gt = (grad - dot * x) / S
        if momentum > 0:
            vel = momentum * vel + gt
            step = vel
        else:
            step = gt
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy(x, steps, m0, m1, lr):
    n, d = x.shape
    for t in range(steps):
        f = t / max(steps - 1, 1)
        margin = m0 + (m1 - m0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = (G > (m - margin)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        deg = M.sum(axis=1, keepdims=True)
        grad = M @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        gt /= np.maximum(deg, 1.0) + 1e-12
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp(x, budget, t0, deadline, r0=1.5e-3, cap=2600):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - t0 < budget and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            idx = np.argpartition(-cs, cap)[:cap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = cap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-7:
                break
    return xb, best

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
    else:
        gbest = _norm(rng.standard_normal((n, d)))
        gval = _max_cos(gbest)
    # contact frequency of incumbent for targeted kicks
    def freq_of(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - 5e-4
        M = G > thr
        return M.sum(axis=1), m, G
    freq, cur_m, _ = freq_of(gbest)
    order_hub = np.argsort(freq)
    # archive of raw candidates
    elite = []  # list of (val, X)
    def push_elite(v, X):
        elite.append((v, X.copy()))
        elite.sort(key=lambda t: t[0])
        del elite[6:]

    push_elite(gval, gbest)
    trial = 0
    iu_all = np.triu_indices(n, 1)
    while time.time() < HOP_END and time.time() < DEADLINE - 40:
        trial += 1
        # kick size
        r = rng.random()
        if trial % 40 == 0:
            k = 7 + int(rng.integers(0, 4))
        elif trial % 11 == 0:
            k = 5 + int(rng.integers(0, 3))
        elif r < 0.6:
            k = 1 + int(rng.integers(0, 3))
        else:
            k = 3 + int(rng.integers(0, 2))
        k = min(k, 12)
        victims = set()
        # worst pair of current gbest-in-hopping (use elite best raw half the time, else gbest)
        ref = elite[0][1] if (elite and rng.random() < 0.5) else gbest
        Gref = ref @ ref.T
        np.fill_diagonal(Gref, -2.0)
        am = int(np.argmax(Gref.max(axis=1)))
        # max pair
        ij = int(np.argmax(Gref[iu_all]))
        wi, wj = int(iu_all[0][ij]), int(iu_all[1][ij])
        if rng.random() < 0.75:
            victims.add(wi if rng.random() < 0.5 else wj)
        if rng.random() < 0.35:
            victims.add(wi); victims.add(wj)
        # hub-biased
        if rng.random() < 0.7 and len(victims) < k:
            # pick from top-freq of gbest
            top_pool = order_hub[-25:]
            victims.add(int(top_pool[rng.integers(len(top_pool))]))
        # low-contact (rattler-like) of ref
        if rng.random() < 0.4 and len(victims) < k:
            fr = (Gref > (float(Gref.max()) - 5e-4)).sum(axis=1)
            low = np.argsort(fr)
            victims.add(int(low[rng.integers(min(30, n))]))
        while len(victims) < k:
            victims.add(int(rng.integers(n)))
        victims = list(victims)[:max(k, len(victims))]
        x = (elite[0][1].copy() if (elite and rng.random() < 0.35) else gbest.copy())
        mode = rng.random()
        if mode < 0.55:
            # void relocation for a subset, gaussian kick for rest
            nrel = 1 + int(rng.integers(0, min(len(victims), 3)))
            rel = rng.choice(victims, size=nrel, replace=False)
            keep = np.ones(n, dtype=bool); keep[np.array(list(rel))] = False
            Xf = x[keep]
            # sample voids
            NS = 3500
            Y = rng.standard_normal((NS, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            S = Y @ Xf.T
            mY = S.max(axis=1)
            # pick distinct good holes
            idx = np.argpartition(mY, nrel)[:nrel]
            for a, b in zip(rel, idx):
                x[a] = Y[b]
            # gaussian kick remaining victims
            for a in victims:
                if a in set(rel):
                    continue
                sig = 0.03 + rng.random() * 0.22
                v = rng.standard_normal(d)
                v -= np.dot(v, x[a]) * x[a]
                vn = np.linalg.norm(v) + 1e-300
                x[a] = x[a] + (sig / max(vn, 1e-12)) * v
                x[a] /= np.linalg.norm(x[a])
        else:
            for a in victims:
                sig = 0.05 + rng.random() * 0.30
                v = rng.standard_normal(d)
                v -= np.dot(v, x[a]) * x[a]
                vn = np.linalg.norm(v) + 1e-300
                x[a] = x[a] + (sig / max(vn, 1e-12)) * v
                x[a] /= np.linalg.norm(x[a])
        # anneal params randomized
        a0 = 30.0 * np.exp(rng.random() * np.log(90.0 / 30.0))
        a1 = 220.0 * np.exp(rng.random() * np.log(520.0 / 220.0))
        lr0 = 0.022 * np.exp(rng.random() * np.log(0.055 / 0.022))
        lr1 = 0.0025 * np.exp(rng.random() * np.log(0.006 / 0.0025))
        steps = 220 + int(rng.integers(0, 160))
        mom = 0.0 if rng.random() < 0.6 else 0.5 + rng.random() * 0.35
        x = _anneal(x, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1, momentum=mom)
        x = _greedy(x, steps=80, m0=0.008, m1=0.0006, lr=0.008 + rng.random() * 0.006)
        v = _max_cos(x)
        if v < elite[0][0] - 1e-12:
            push_elite(v, x)
        elif len(elite) < 6 and v < elite[-1][0]:
            push_elite(v, x)
        elif rng.random() < 0.02 and len(elite) < 6:
            push_elite(v, x)
        # opportunistic: if raw beats LP incumbent convincingly, adopt path
        # (raw is upper bound on LP basin, so only adopt if clearly better)
        if v < gval - 2e-6:
            gbest = x.copy(); gval = v
            freq, cur_m, _ = freq_of(gbest)
            order_hub = np.argsort(freq)
    # LP stage on elites + incumbent
    pool = []
    for v, X in elite:
        pool.append((v, X))
    pool.append((gval, gbest))
    # dedup-ish: keep best 4 by raw
    pool.sort(key=lambda t: t[0])
    pool = pool[:4]
    best_overall_val = _max_cos(gbest)
    best_overall = gbest.copy()
    # include original incumbent even if raw worse (it is LP-snapped)
    try:
        c0 = _norm(np.load("best.npy"))
        if c0.shape == (n, d):
            v0 = _max_cos(c0)
            if v0 < best_overall_val:
                best_overall_val, best_overall = v0, c0.copy()
    except Exception:
        pass
    for i, (v, X) in enumerate(pool):
        if time.time() > DEADLINE - 12:
            break
        # budget per candidate
        remaining = DEADLINE - time.time() - (len(pool) - i - 1) * 8.0 - 10.0
        budget = max(4.0, min(10.0, remaining))
        if time.time() + budget > DEADLINE - 6:
            budget = max(2.0, DEADLINE - 6 - time.time())
        Xp, vp = _lp(X, budget=budget, t0=time.time(), deadline=DEADLINE - 4)
        if vp < best_overall_val:
            best_overall_val, best_overall = vp, Xp.copy()
    # final snap on winner if time left
    if time.time() < DEADLINE - 8:
        Xp, vp = _lp(best_overall, budget=DEADLINE - time.time() - 2.0, t0=time.time(), deadline=DEADLINE - 1.0)
        if vp < best_overall_val:
            best_overall_val, best_overall = vp, Xp.copy()
    best_overall = _norm(best_overall)
    return best_overall
