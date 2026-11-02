import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=0.005):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            sel[np.argmax(c)] = True
            cnt = 1
        if cnt > 1600:
            idx = np.argsort(-c)[:1600]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx] = True
            sel = sel2
        I = iu0[sel]; J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2 * d] = J[:, None] * d + ar
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.empty((p, 2 * d + 1), dtype=float)
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        vals = vals.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _annealed_gd(x0, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = active.copy()
    vel = np.zeros_like(x)
    # init best
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        lratio = 0.0
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    else:
        lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Grad = Grad - dot * x
        if not np.all(act):
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        # zero frozen velocity drift
        if not np.all(act):
            vel[~act] = 0.0
        x[act] = x[act] - lr_k * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(int(act.sum()), d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                xa = xa + nz
                xa /= np.linalg.norm(xa, axis=1, keepdims=True)
                x[act] = xa
        # renormalize active
        if np.all(act):
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            xn = np.linalg.norm(x[act], axis=1, keepdims=True)
            xn = np.maximum(xn, 1e-12)
            x[act] /= xn
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _deep_hole(others, rng, nc=12000, refine_rounds=3):
    d = others.shape[1]
    cand = rng.randn(nc, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    chunk = 4000
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a + chunk)
        dots = cand[a:b] @ others.T
        m_all[a:b] = dots.max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi])
    best_c = cand[bi].copy()
    rad = 0.25
    for _ in range(refine_rounds):
        prop = rng.randn(2500, d) * (rad / 2.0)
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

def _subgraph(cur, K, rng):
    n = cur.shape[0]
    G = cur @ cur.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    if rng.rand() < 0.85:
        w = int(np.argmax(rowmax))
    else:
        w = int(rng.randint(0, n))
    # order neighbors of w
    neigh = np.argsort(-G[w])
    S = [w]
    seen = set(S)
    # BFS expansion by worst edges
    cand_pool = list(neigh[:max(K*2, 12)])
    # add worst global points too
    order = np.argsort(-rowmax)
    for o in order[:6]:
        if int(o) not in seen:
            cand_pool.append(int(o))
    for c in cand_pool:
        if len(S) >= K:
            break
        if c not in seen:
            S.append(int(c)); seen.add(c)
    # if still short, fill random
    while len(S) < K:
        r = int(rng.randint(0, n))
        if r not in seen:
            S.append(r); seen.add(r)
    return np.array(S, dtype=int), G

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    POLISH_DEADLINE = HARD_END - 14.0
    rng = np.random.RandomState(7)
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
    # quick initial tighten
    best = _lp_polish(best, min(HARD_END - 2, time.time() + 1.5), r0=0.003)
    best_val = _max_cos(best)
    if best_val < cur_val:
        cur = best.copy(); cur_val = best_val
    it = 0
    while time.time() < POLISH_DEADLINE:
        it += 1
        mode = rng.rand()
        x = cur.copy()
        if mode < 0.45:
            # subgraph reoptimize: freeze bulk
            K = int(rng.choice([8, 10, 12, 16, 20, 24]))
            S, Gc = _subgraph(cur, K, rng)
            act = np.zeros(n, dtype=bool); act[S] = True
            # small shake of subgraph + pair-breaking inside
            sigma = 0.008 + rng.rand() * 0.025
            x[S] += rng.randn(len(S), d) * sigma
            x[S] /= np.linalg.norm(x[S], axis=1, keepdims=True)
            # break worst pairs inside S
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            # collect worst pairs with at least one end in S
            iu0, iu1 = np.triu_indices(n, 1)
            c = Gtmp[iu0, iu1]
            topk = np.argsort(-c)[:6]
            step = 0.03 + rng.rand() * 0.05
            for tk in topk:
                i = int(iu0[tk]); j = int(iu1[tk])
                if (i not in S) and (j not in S):
                    continue
                if rng.rand() < 0.5:
                    # push both apart
                    dot = float(x[i] @ x[j])
                    ti = x[j] - dot * x[i]
                    tj = x[i] - dot * x[j]
                    if np.linalg.norm(ti) > 1e-12 and act[i]:
                        x[i] -= step * ti
                    if np.linalg.norm(tj) > 1e-12 and act[j]:
                        x[j] -= step * tj
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # annealed GD on subgraph only (hot then cold)
            T0 = 0.008 + rng.rand() * 0.012
            T1 = 0.0004 + rng.rand() * 0.0008
            steps = 180 + rng.randint(0, 160)
            lr = 0.03 + rng.rand() * 0.03
            mu = 0.75 + rng.rand() * 0.15
            x, _ = _annealed_gd(x, steps, T0, T1, lr, mu, rng, active=act,
                                noise0=0.004 + rng.rand() * 0.004, deadline=POLISH_DEADLINE)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            # unfreeze global short sweep
            T0g = 0.004 + rng.rand() * 0.004
            T1g = 0.0006 + rng.rand() * 0.0006
            x, _ = _annealed_gd(x, 120 + rng.randint(0, 80), T0g, T1g, 0.02 + rng.rand() * 0.02,
                                0.8, rng, active=None, noise0=0.0, deadline=POLISH_DEADLINE)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, min(POLISH_DEADLINE, time.time() + 0.9), r0=0.004 + rng.rand() * 0.003)
        elif mode < 0.75:
            # hole reinsertion of worst points + global GD
            Gtmp = x @ x.T
            np.fill_diagonal(Gtmp, -2.0)
            rowmax = Gtmp.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(rng.choice([1, 2, 2, 3]))
            # small global shake first
            x += rng.randn(n, d) * (0.003 + rng.rand() * 0.008)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            used = set()
            for t in range(k):
                if time.time() > POLISH_DEADLINE:
                    break
                if rng.rand() < 0.75:
                    pool = order[:10]
                    victim = int(pool[rng.randint(0, len(pool))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                others = x[mask]
                nc = 9000 + rng.randint(0, 7000)
                hc = _deep_hole(others, rng, nc=nc, refine_rounds=3)
                x[victim] = hc
            s0 = 0.006 + rng.rand() * 0.008
            s1 = 0.0005 + rng.rand() * 0.0008
            # convert to temperature: use T in dot units
            steps = 180 + rng.randint(0, 140)
            lr = 0.02 + rng.rand() * 0.025
            x, _ = _annealed_gd(x, steps, s0, s1, lr, 0.75 + rng.rand() * 0.12, rng,
                                active=None, noise0=0.0, deadline=POLISH_DEADLINE)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, min(POLISH_DEADLINE, time.time() + 0.9), r0=0.004, deadline=POLISH_DEADLINE) if False else _lp_polish(x, min(POLISH_DEADLINE, time.time() + 0.9), r0=0.004)
        else:
            # big topology kick: cap shake + many reinsertions
            sigma = 0.04 + rng.rand() * 0.06
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
                hc = _deep_hole(others, rng, nc=10000 + rng.randint(0, 6000), refine_rounds=3)
                x[victim] = hc
            steps = 250 + rng.randint(0, 150)
            T0 = 0.006 + rng.rand() * 0.006
            T1 = 0.0006 + rng.rand() * 0.0006
            lr = 0.025 + rng.rand() * 0.025
            x, _ = _annealed_gd(x, steps, T0, T1, lr, 0.8, rng, active=None, noise0=0.002, deadline=POLISH_DEADLINE)
            if time.time() > POLISH_DEADLINE:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            x = _lp_polish(x, min(POLISH_DEADLINE, time.time() + 1.0), r0=0.006)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best, best_val = x.copy(), v
            cur, cur_val = x.copy(), v
            # extra tighten on improvement
            if time.time() < POLISH_DEADLINE - 2.0:
                x2 = _lp_polish(x, min(POLISH_DEADLINE, time.time() + 0.7), r0=0.002)
                v2 = _max_cos(x2)
                if v2 < best_val:
                    best, best_val = x2.copy(), v2
                    cur, cur_val = x2.copy(), v2
        elif v < cur_val:
            cur, cur_val = x.copy(), v
        else:
            Temp = 0.0010 * (0.996 ** (it // 2)) + 6e-5
            if rng.rand() < np.exp(-(v - cur_val) / Temp) * 0.5:
                cur, cur_val = x.copy(), v
            if rng.rand() < 0.06:
                cur, cur_val = best.copy(), best_val
        # occasional restart from best with tiny noise to intensify
        if it % 30 == 0 and rng.rand() < 0.5:
            cur = best + rng.randn(n, d) * 0.004
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = _max_cos(cur)
    remain = HARD_END - time.time() - 0.6
    if remain > 2.0:
        half = remain * 0.55
        best = _lp_polish(best, time.time() + half, r0=0.003)
        remain2 = HARD_END - time.time() - 0.3
        if remain2 > 1.0:
            best = _lp_polish(best, time.time() + remain2, r0=0.0008)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
