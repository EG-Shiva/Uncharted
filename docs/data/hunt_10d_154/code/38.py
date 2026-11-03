import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # keep tightest 3000 to bound time
            idx = np.argpartition(-c[sel], 3000)[:3000]
            I, J = I[idx], J[idx]
            p = 3000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != len(sel) else -c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _greedy_hole(prop, rng, C=5000, K=8):
    d = prop.shape[1]
    cand = _norm(rng.standard_normal((C, d)))
    D = cand @ prop.T
    md = D.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    # pick among K best randomly to diversify
    order = idx[np.argsort(md[idx])]
    r = rng.random()
    if r < 0.5:
        pick = order[0]
    elif r < 0.85 and K >= 4:
        pick = order[rng.integers(1, min(4, K))]
    else:
        pick = order[rng.integers(K)]
    bc = cand[pick].copy()
    bv = float(md[pick])
    # one local deepening step
    pert = _norm(bc[None, :] + rng.standard_normal((1500, d)) * 0.05)
    D2 = pert @ prop.T
    m2 = D2.max(axis=1)
    j = int(np.argmin(m2))
    if float(m2[j]) < bv:
        bc = pert[j]
    return bc

def _cover_delete(cur, k, Etop, rng):
    n = cur.shape[0]
    G = cur @ cur.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(n, 1)
    c = G[iu]
    top = np.argpartition(-c, min(Etop, len(c) - 1))[:Etop]
    I = iu[0][top]
    J = iu[1][top]
    uncovered = np.ones(len(top), dtype=bool)
    chosen = []
    for _ in range(k):
        cnt = np.zeros(n)
        for t in range(len(top)):
            if uncovered[t]:
                cnt[I[t]] += 1
                cnt[J[t]] += 1
        # tie break random
        cnt = cnt + 0.01 * rng.random(n)
        v = int(np.argmax(cnt))
        if cnt[v] <= 0.02:
            # no more coverage, pick random hub
            v = int(rng.integers(n))
            while v in chosen:
                v = int(rng.integers(n))
        chosen.append(v)
        for t in range(len(top)):
            if I[t] == v or J[t] == v:
                uncovered[t] = False
    return np.array(chosen, dtype=int)

def _ortho_init(n, d, rng):
    need = n
    out = []
    while need > 0:
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        if rng.random() < 0.5:
            Q = -Q
        take = min(d, need)
        out.append(Q[:take])
        need -= take
    X = np.concatenate(out, axis=0)
    X = X + 0.02 * rng.standard_normal(X.shape)
    return _norm(X)

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 130.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    second = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    if best is not None:
                        second = best.copy()
                    best_val = v
                    best = a.copy()
                elif second is None:
                    second = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    if second is None:
        second = best.copy()
    # short snap of loaded best to get true local value (bounded)
    try:
        if DEAD - time.time() > 30:
            best = _lp_polish(best, 1.5, deadline=HOP_END)
            best_val = _max_cos(best)
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    trial = 0
    while time.time() < HOP_END and time.time() < DEAD - 28:
        trial += 1
        elapsed = time.time() - t0
        frac = max(0.0, min(1.0, elapsed / max(HOP_END - t0, 1e-9)))
        T_acc = 0.0022 * (1.0 - frac) + 0.0002
        mode = trial % 6
        try:
            if mode == 0:
                # vertex-cover deflate -> relax reduced -> regrow
                k = int(rng.integers(7, 13))
                Etop = int(rng.integers(60, 140))
                dels = _cover_delete(cur, k, Etop, rng)
                keep = np.ones(n, dtype=bool)
                keep[dels] = False
                Xr = cur[keep].copy()
                # relax reduced set to open space (short)
                Xr = _relax(Xr, steps=350, lr=0.02, T0=0.02, T1=0.002)
                # regrow one by one
                prop = Xr
                newpts = []
                for _ in range(k):
                    h = _greedy_hole(prop, rng, C=5000, K=8)
                    newpts.append(h)
                    prop = np.concatenate([prop, h[None, :]], axis=0)
                Xtry = prop  # (n,d)
                Xtry = Xtry + 0.006 * rng.standard_normal(Xtry.shape)
                Xtry = _norm(Xtry)
                s1, T0a, T1a, lr0 = 800, 0.03, 0.0015, 0.02
            elif mode == 1:
                # core teleport to holes
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(rm)
                k_tight = int(rng.integers(4, 8))
                tight = order[-10:]
                pick = rng.choice(tight, size=k_tight, replace=False)
                Xtry = cur.copy()
                for v in pick:
                    # hole w.r.t. all except v
                    mask = np.ones(n, dtype=bool)
                    mask[v] = False
                    prop = Xtry[mask]
                    h = _greedy_hole(prop, rng, C=6000, K=10)
                    Xtry[v] = h
                Xtry = Xtry + 0.008 * rng.standard_normal(Xtry.shape)
                Xtry = _norm(Xtry)
                s1, T0a, T1a, lr0 = 850, 0.025, 0.0012, 0.02
            elif mode == 2:
                # spatial cluster resample (large)
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                c0 = int(rng.integers(0, n))
                sims = G[c0]
                K = int(rng.integers(10, 20))
                nn = np.argsort(-sims)[:K]
                Xtry = cur.copy()
                Xtry[nn] = rng.standard_normal((K, d))
                Xtry = Xtry + 0.01 * rng.standard_normal(Xtry.shape)
                Xtry = _norm(Xtry)
                s1, T0a, T1a, lr0 = 900, 0.02, 0.001, 0.02
            elif mode == 3:
                # orthogonal-block fresh structure
                Xtry = _ortho_init(n, d, rng)
                # mix a few points from cur to retain goodness
                nmix = int(rng.integers(10, 30))
                idx = rng.choice(n, size=nmix, replace=False)
                Xtry[idx] = cur[idx] + 0.05 * rng.standard_normal((nmix, d))
                Xtry = _norm(Xtry)
                s1, T0a, T1a, lr0 = 1400, 0.03, 0.0008, 0.025
            elif mode == 4:
                # crossover half-half
                A = cur
                B = second if second is not None else best
                # random direction hemisphere mix
                u = _norm(rng.standard_normal((1, d)))[0]
                sa = A @ u
                sb = B @ u
                # take top half from A and bottom half from B to make complementary
                ia = np.argsort(-sa)
                ib = np.argsort(sb)
                h = n // 2
                Xtry = np.concatenate([A[ia[:h]], B[ib[:h]]], axis=0)
                if Xtry.shape[0] != n:
                    # adjust (n=154 even so fine)
                    pass
                # deduplicate jitter
                Xtry = Xtry + 0.015 * rng.standard_normal(Xtry.shape)
                Xtry = _norm(Xtry)
                s1, T0a, T1a, lr0 = 1000, 0.025, 0.001, 0.02
            else:
                # pure random fresh with long anneal (rare)
                if trial % 12 == 0:
                    Xtry = _norm(rng.standard_normal((n, d)))
                    s1, T0a, T1a, lr0 = 1600, 0.03, 0.0008, 0.025
                else:
                    Xtry = cur + (0.12 + 0.08 * rng.random()) * rng.standard_normal((n, d))
                    Xtry = _norm(Xtry)
                    s1, T0a, T1a, lr0 = 800, 0.03, 0.0015, 0.02
            # two-stage anneal
            Xann = _relax(Xtry, steps=s1, lr=lr0, T0=T0a, T1=T1a)
            # second cold stage
            Xann = _relax(Xann, steps=250, lr=0.012, T0=0.001, T1=0.0002)
            vann = _max_cos(Xann)
            # filter: only polish if within reach (acceptance judged after polish)
            reach = cur_val + 0.015
            if vann > reach and mode not in (3,):
                # occasional polish of near miss to escape
                if rng.random() > 0.1:
                    continue
            tl = min(HOP_END - time.time(), DEAD - 26 - (time.time() - t0))
            if tl < 1.2:
                if vann < best_val:
                    best, best_val = Xann.copy(), vann
                    cur, cur_val = Xann.copy(), vann
                    second = cur.copy()
                continue
            sec = min(2.2, max(1.0, tl / 25.0))
            Xpol = _lp_polish(Xann, sec, deadline=HOP_END)
            vpol = _max_cos(Xpol)
            if vpol < best_val - 1e-12:
                best, best_val = Xpol.copy(), vpol
                cur, cur_val = Xpol.copy(), vpol
                second = cur.copy()
            elif vpol < cur_val + T_acc:
                # wandering move (Metropolis on polished value)
                if vpol <= cur_val or rng.random() < np.exp((cur_val - vpol) / max(T_acc, 1e-9)):
                    cur, cur_val = Xpol.copy(), vpol
            # update second best diversity pool
            if vpol < _max_cos(second) - 5e-4 and vpol > best_val + 1e-4:
                second = Xpol.copy()
        except Exception:
            continue
    # final polish of best
    remaining = DEAD - time.time() - 1.5
    if remaining > 3 and best is not None:
        try:
            best = _lp_polish(best, min(remaining - 1.0, 12.0), deadline=DEAD - 1.0)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
