import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=0.005, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        # cap constraints for speed
        if np.count_nonzero(sel) > 1500:
            # keep top 1500
            idx = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[:1500]] = True
            sel = sel2
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
    return x

def _smooth_mom(x, steps, s0, s1, lr, mom=0.75):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        f = it / max(1, steps - 1)
        s = s0 + (s1 - s0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        gmax = G.max()
        W = np.exp(s * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        Grad = (W @ x) / (S * 0.5)
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        # limit
        vel = mom * vel + (1.0 - mom) * Gt
        # scale step
        x = x - lr * (Gt * 0.5 + vel * 0.5)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _deep_hole(others, rng, nc=12000, refine_rounds=3):
    d = others.shape[1]
    # coarse global pool
    cand = rng.randn(nc, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # chunked eval to save memory
    best_v = np.inf
    best_c = cand[0]
    chunk = 4000
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a + chunk)
        dots = cand[a:b] @ others.T
        m_all[a:b] = dots.max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi])
    best_c = cand[bi].copy()
    # local refinement around best
    rad = 0.25
    for _ in range(refine_rounds):
        prop = rng.randn(2500, d) * (rad / 2.0)
        # project to tangent of best_c
        prop -= (prop @ best_c)[:, None] * best_c[None, :]
        prop = best_c[None, :] + prop
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        dots = prop @ others.T
        m = dots.max(axis=1)
        j = int(np.argmin(m))
        if float(m[j]) < best_v:
            best_v = float(m[j])
            best_c = prop[j].copy()
        rad *= 0.45
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    rng = np.random.RandomState(1)
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    it = 0
    POLISH_DEADLINE = HARD_END - 16.0
    while time.time() < POLISH_DEADLINE:
        it += 1
        mode = rng.rand()
        x = cur.copy()
        if mode < 0.35:
            # exploit: small global shake + 1-2 hole reinsertions
            sigma = 0.003 + rng.rand() * 0.012
            x = x + rng.randn(n, d) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax = Gtmp.max(axis=1)
            order = np.argsort(-rowmax)
            k = 1 if rng.rand() < 0.6 else 2
            used = set()
            for t in range(k):
                if time.time() > POLISH_DEADLINE:
                    break
                if rng.rand() < 0.7:
                    pool = order[:10]
                    victim = int(pool[rng.randint(0, len(pool))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                others = x[mask]
                nc = 8000 + rng.randint(0, 6000)
                hc = _deep_hole(others, rng, nc=nc, refine_rounds=3)
                x[victim] = hc
            s0 = 40.0 + rng.rand() * 30.0
            s1 = 160.0 + rng.rand() * 140.0
            steps = 150 + rng.randint(0, 120)
            lr = 0.015 + rng.rand() * 0.02
            x = _smooth_mom(x, steps, s0, s1, lr, mom=0.7 + rng.rand() * 0.2)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, seconds=0.9, r0=0.004, t_deadline=POLISH_DEADLINE)
        elif mode < 0.78:
            # cluster move: coherent shift of worst neighborhood + reinsertions
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            w = int(np.argmax(rowmax)) if rng.rand() < 0.85 else int(rng.randint(0, n))
            # neighbors of w
            msize = int(rng.choice([4, 5, 6, 7, 8]))
            neigh = np.argsort(-Gc[w])[:msize]
            # start from cur with small noise
            x = cur + rng.randn(n, d) * 0.004
            # coherent shift for cluster
            common = rng.randn(d) * (0.04 + rng.rand() * 0.07)
            x[neigh] += common[None, :]
            x[neigh] += rng.randn(len(neigh), d) * (0.01 + rng.rand() * 0.02)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # hole reinsert 2-4 victims (include cluster members biased)
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax2 = Gtmp.max(axis=1)
            order = np.argsort(-rowmax2)
            k = int(rng.choice([2, 2, 3, 3, 4]))
            used = set()
            for t in range(k):
                if time.time() > POLISH_DEADLINE:
                    break
                rr = rng.rand()
                if rr < 0.5:
                    pool = order[:10]
                    victim = int(pool[rng.randint(0, len(pool))])
                elif rr < 0.75 and len(neigh) > 0:
                    victim = int(neigh[rng.randint(0, len(neigh))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                others = x[mask]
                hc = _deep_hole(others, rng, nc=12000 + rng.randint(0, 8000), refine_rounds=4)
                x[victim] = hc
            s0 = 25.0 + rng.rand() * 25.0
            s1 = 180.0 + rng.rand() * 140.0
            steps = 200 + rng.randint(0, 150)
            lr = 0.02 + rng.rand() * 0.025
            x = _smooth_mom(x, steps, s0, s1, lr, mom=0.75 + rng.rand() * 0.15)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, seconds=1.0, r0=0.006, t_deadline=POLISH_DEADLINE)
        else:
            # big kick: large noise + many reinsertions
            sigma = 0.05 + rng.rand() * 0.06
            x = cur + rng.randn(n, d) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax = Gtmp.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(rng.choice([3, 4, 4, 5]))
            used = set()
            for t in range(k):
                if time.time() > POLISH_DEADLINE:
                    break
                if rng.rand() < 0.6:
                    pool = order[:12]
                    victim = int(pool[rng.randint(0, len(pool))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                others = x[mask]
                hc = _deep_hole(others, rng, nc=10000 + rng.randint(0, 8000), refine_rounds=3)
                x[victim] = hc
            s0 = 20.0 + rng.rand() * 20.0
            s1 = 200.0 + rng.rand() * 120.0
            steps = 250 + rng.randint(0, 150)
            lr = 0.02 + rng.rand() * 0.025
            x = _smooth_mom(x, steps, s0, s1, lr, mom=0.8)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, seconds=1.1, r0=0.007, t_deadline=POLISH_DEADLINE)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best, best_val = x.copy(), v
            cur, cur_val = x.copy(), v
            if time.time() < POLISH_DEADLINE - 2.0:
                x2 = _lp_polish(x, seconds=0.8, r0=0.002, t_deadline=POLISH_DEADLINE)
                v2 = _max_cos(x2)
                if v2 < best_val:
                    best, best_val = x2.copy(), v2
                    cur, cur_val = x2.copy(), v2
        elif v < cur_val:
            cur, cur_val = x.copy(), v
        else:
            Temp = 0.0012 * (0.996 ** (it // 2)) + 8e-5
            if rng.rand() < np.exp(-(v - cur_val) / Temp) * 0.5:
                cur, cur_val = x.copy(), v
            if rng.rand() < 0.07:
                cur, cur_val = best.copy(), best_val
    remain = HARD_END - time.time() - 0.8
    if remain > 2.0:
        half = remain * 0.55
        best = _lp_polish(best, seconds=half, r0=0.004, t_deadline=HARD_END - 0.4)
        remain2 = HARD_END - time.time() - 0.4
        if remain2 > 1.0:
            best = _lp_polish(best, seconds=remain2, r0=0.001, t_deadline=HARD_END - 0.15)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
