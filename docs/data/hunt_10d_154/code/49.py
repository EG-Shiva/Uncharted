import time
import itertools
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
    return float(g.max())

def _adam_relax(x, steps, lr, T0, T1, deadline=1e18):
    x = _norm(x)
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
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
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 40:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            cvals = c[sel][order]
        else:
            cvals = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cvals, A_eq=A_eq, b_eq=np.zeros(n),
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

def _build_pool(rng):
    P = []
    for i in range(10):
        for s in (1.0, -1.0):
            v = np.zeros(10); v[i] = s; P.append(v)
    for i in range(10):
        for j in range(i + 1, 10):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(10); v[i] = si / np.sqrt(2.0); v[j] = sj / np.sqrt(2.0); P.append(v)
    for mask in range(1024):
        v = np.ones(10)
        for k in range(10):
            if (mask >> k) & 1:
                v[k] = -1.0
        v /= np.sqrt(10.0)
        P.append(v)
    for combo in itertools.combinations(range(10), 3):
        for mask in range(8):
            v = np.zeros(10)
            for t, idx in enumerate(combo):
                v[idx] = 1.0 if ((mask >> t) & 1) == 0 else -1.0
            v /= np.sqrt(3.0)
            P.append(v)
    # parameterized orbits: sampled signed permutations
    def add_pattern(vals_template, n_sample):
        k = len(vals_template)
        for _ in range(n_sample):
            perm = rng.permutation(10)
            signs = rng.choice(np.array([-1.0, 1.0]), size=k)
            v = np.zeros(10)
            v[perm[:k]] = vals_template * signs
            P.append(v)
    for th in [0.25, 0.7, 1.05, 1.35]:
        a = np.cos(th) / np.sqrt(3.0); b = np.sin(th) / np.sqrt(2.0)
        add_pattern(np.array([a, a, a, b, b]), 350)
    for th in [0.4, 0.9, 1.2]:
        r = np.sqrt(0.2)
        a = r * np.cos(th); b = r * np.sin(th)
        add_pattern(np.array([a, a, a, a, a, b, b, b, b, b]), 300)
    for th in [0.5, 1.0]:
        r2 = 1.0 / np.sqrt(8.0)
        a = np.cos(th) * r2 * np.sqrt(2); b = np.sin(th) * r2 * np.sqrt(2)
        # 4+4 with 2 zeros: 8 nonzeros
        add_pattern(np.array([a, a, a, a, b, b, b, b]), 300)
    for th in [0.6, 1.1]:
        a = np.cos(th) / 2.0; b = np.sin(th) / 2.0
        # pattern (a,a,b,b,c...) use 4+2+? simpler: 2+2+2 zeros 4
        add_pattern(np.array([a, a, b, b, 0.5, 0.5]), 250)
    P = np.array(P, dtype=float)
    P = _norm(P)
    # dedup roughly: if too large subsample deterministically
    if len(P) > 5200:
        idx = rng.choice(len(P), size=5200, replace=False)
        P = P[idx]
    return P

def _greedy_select(P, n):
    M = P.shape[0]
    cur_max = np.full(M, np.inf)
    selected = []
    sel_mask = np.zeros(M, dtype=bool)
    # start from index 0
    s0 = 0
    selected.append(s0); sel_mask[s0] = True
    d = P @ P[s0]
    cur_max = np.minimum(cur_max, d)
    cur_max[s0] = np.inf
    for _ in range(1, n):
        nxt = int(np.argmin(cur_max))
        if not np.isfinite(cur_max[nxt]):
            # fill random unselected
            cand = np.where(~sel_mask)[0]
            nxt = int(cand[0])
        selected.append(nxt); sel_mask[nxt] = True
        d = P @ P[nxt]
        cur_max = np.minimum(cur_max, d)
        cur_max[nxt] = np.inf
    return np.array(selected, dtype=int)

def _swap_anneal(P, sel, rng, iters=14000, beta=60.0):
    n = len(sel)
    d = P.shape[1]
    M = P.shape[0]
    in_set = np.zeros(M, dtype=bool); in_set[sel] = True
    S = P[sel]
    G = S @ S.T
    np.fill_diagonal(G, 0.0)
    # objective sum exp
    E = np.exp(beta * G)
    np.fill_diagonal(E, 0.0)
    total = E.sum() * 0.5
    # per-point contribution (half sum)
    contrib = E.sum(axis=1) * 0.5
    T0, T1 = 0.08 * total / max(n, 1), 0.0005 * total / max(n, 1)
    # geometric schedule
    if iters > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (iters - 1)
    else:
        logr = 0.0
    sel_list = sel.tolist()
    pos_of = np.full(M, -1, dtype=int)
    for ii, g in enumerate(sel_list):
        pos_of[g] = ii
    for it in range(iters):
        T = T0 * np.exp(logr * it)
        if T < 1e-12:
            T = 1e-12
        i_out_pos = int(rng.integers(n))
        g_out = sel_list[i_out_pos]
        g_in = int(rng.integers(M))
        if in_set[g_in]:
            continue
        cand = P[g_in]
        others = S  # includes out; will correct
        dots = others @ cand
        # exclude self (out position dot with cand is dot(out,cand), keep it; exclude nothing else)
        # sum exp for candidate vs all except out
        dots_wo = np.delete(dots, i_out_pos)
        # dots_wo corresponds to candidate vs remaining
        new_c = float(np.exp(beta * dots_wo).sum())
        old_c = float(contrib[i_out_pos])
        # correction: contrib includes exp(beta*dot(out, others)); new total = total - old_c - (contribution double counted?) Actually total = sum_{i<j} e; removing out subtracts old_c; adding in adds new_c
        delta = new_c - old_c
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # accept: update structures
            # need to update contrib for remaining points: subtract old edges, add new edges
            old_edges = np.exp(beta * (S @ S[i_out_pos]))
            old_edges[i_out_pos] = 0.0
            new_edges = np.exp(beta * (S @ cand))
            new_edges[i_out_pos] = 0.0
            # update contrib array
            contrib = contrib - old_edges * 0.5 - old_edges * 0.5 * 0  # careful
            # recompute contrib from scratch every 2000 iters to avoid drift; else incremental
            # incremental: for each remaining j: contrib[j] += (new_edges[j]-old_edges[j])/1? Since contrib[j]=sum_{k} e_{jk}/2? Actually E symmetric, sum/2. Edge contributes e/2 to each endpoint? No: E.sum(axis)*0.5 counts each edge twice? Let's define contrib[j]=sum_k E[j,k]*0.5. Then updating: contrib[j]+= (new-old)*0.5 for j != out; new contrib for in = new_c*0.5? Wait new_c defined as sum (not half). Let's use half convention.
            # fix: new_c_half = new_c*0.5, old_c_half = old_c (already half? old_c = E.sum*0.5 per row => half). new_c currently full sum. So:
            new_c_half = new_c * 0.5
            for j in range(n):
                if j == i_out_pos:
                    continue
                contrib[j] += (new_edges[j] - old_edges[j]) * 0.5
            contrib[i_out_pos] = new_c_half
            total = total + (new_c * 1.0 - old_c * 1.0) * 1.0
            # Actually old_c is half-sum? old_c = sum*0.5, new_c full sum. So delta full = new_c - old_c*2? Let's recompute consistently:
            # To avoid bug, recompute total/contrib from scratch periodically.
            S[i_out_pos] = cand
            sel_list[i_out_pos] = g_in
            in_set[g_out] = False; in_set[g_in] = True
            pos_of[g_out] = -1; pos_of[g_in] = i_out_pos
            if (it % 2000) == 1999:
                G2 = S @ S.T
                np.fill_diagonal(G2, 0.0)
                E2 = np.exp(beta * G2)
                np.fill_diagonal(E2, 0.0)
                total = float(E2.sum() * 0.5)
                contrib = E2.sum(axis=1) * 0.5
        # else reject
    return np.array(sel_list, dtype=int)

def _expand_cyclic(seeds):
    # seeds (K,10) -> (K*20,10) via 10 cyclic shifts + negations
    K = seeds.shape[0]
    outs = []
    for k in range(K):
        v = seeds[k]
        for s in range(10):
            outs.append(np.roll(v, s))
        outs.append(-np.roll(v, 0))
        for s in range(1, 10):
            outs.append(-np.roll(v, s))
    return _norm(np.array(outs))

def _optim_cyclic(rng, K=8, steps=1200, deadline=1e18):
    seeds = rng.standard_normal(size=(K, 10))
    seeds = _norm(seeds)
    m_ad = np.zeros_like(seeds); v_ad = np.zeros_like(seeds)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    lr = 0.02; T = 0.015
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        X = _expand_cyclic(seeds)  # (K*20,10)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        GradX = (E @ X) / S  # (N,10)
        # backprop to seeds: sum over shifts/negations with inverse roll
        GradS = np.zeros_like(seeds)
        idx = 0
        for k in range(K):
            g = np.zeros(10)
            for s in range(10):
                gx = GradX[idx]; idx += 1
                g += np.roll(gx, -s)
            for s in range(10):
                gx = GradX[idx]; idx += 1
                g += -np.roll(gx, -s) if s == 0 else -np.roll(gx, -s)
            GradS[k] = g
        # tangent project per seed
        dot = (GradS * seeds).sum(axis=1, keepdims=True)
        Gt = GradS - dot * seeds
        b1p *= b1; b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p); vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        seeds = seeds - lr * step
        seeds = _norm(seeds)
        T *= 0.9985
        if T < 0.002:
            T = 0.002
    X = _expand_cyclic(seeds)
    return X, seeds

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d))); best_val = _max_cos(best)
    # Branch A: algebraic pool
    try:
        P = _build_pool(rng)
        sel0 = _greedy_select(P, n)
        sel1 = _swap_anneal(P, sel0, rng, iters=14000, beta=60.0)
        XA = P[sel1].copy()
        # release symmetry
        XA = _adam_relax(XA, steps=2200, lr=0.015, T0=0.02, T1=0.001, deadline=DEAD - 45)
        vA = _max_cos(XA)
        if vA < best_val:
            best_val = vA; best = XA.copy()
        # quick LP on XA if time
        if time.time() < DEAD - 60:
            XA2 = _lp_polish(XA, seconds=min(18.0, max(4.0, (DEAD - 45 - time.time()) * 0.4)), deadline=DEAD - 40)
            vA2 = _max_cos(XA2)
            if vA2 < best_val:
                best_val = vA2; best = XA2.copy()
            else:
                # keep XA if better than incumbent even without LP gain
                if vA < best_val:
                    pass
                elif vA < _max_cos(best) and vA < 0.405:
                    # keep as candidate for final pick
                    pass
        # store XA for final comparison
        XA_best = XA2 if 'XA2' in locals() else XA
        XA_best_val = _max_cos(XA_best)
    except Exception:
        XA_best = None; XA_best_val = 1e9
    # Branch B: cyclic orbit union optimization
    try:
        if time.time() < DEAD - 50:
            Xexp, seeds = _optim_cyclic(rng, K=8, steps=1100, deadline=DEAD - 35)
            # Xexp is 160 points; drop 6 worst by rowmax
            G = Xexp @ Xexp.T; np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            drop = np.argsort(-rowmax)[:6]
            keep = np.ones(len(Xexp), dtype=bool); keep[drop] = False
            XB = Xexp[keep][:n].copy()
            XB = _adam_relax(XB, steps=1800, lr=0.015, T0=0.015, T1=0.0009, deadline=DEAD - 25)
            vB = _max_cos(XB)
            if vB < best_val:
                best_val = vB; best = XB.copy()
            if time.time() < DEAD - 30:
                XB2 = _lp_polish(XB, seconds=min(15.0, max(4.0, DEAD - 8 - time.time() - 4.0)), deadline=DEAD - 6)
                vB2 = _max_cos(XB2)
                if vB2 < best_val:
                    best_val = vB2; best = XB2.copy()
    except Exception:
        pass
    # Final safety: ensure best is set and polished lightly if incumbent still best (try short relax on incumbent copy to attempt improvement without losing best)
    try:
        if time.time() < DEAD - 12:
            # short attempt to improve current best with small Adam (helps LP tolerance artifact)
            cand = _adam_relax(best.copy(), steps=600, lr=0.008, T0=0.006, T1=0.0008, deadline=DEAD - 6)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v; best = cand
        if time.time() < DEAD - 6:
            best = _lp_polish(best, seconds=max(1.0, DEAD - 3 - time.time()), deadline=DEAD - 1.5)
    except Exception:
        pass
    best = _norm(best)
    nr = np.linalg.norm(best, axis=1)
    bad = ~(np.isfinite(nr) & (nr > 1e-12))
    if np.any(bad):
        rb = rng.standard_normal(size=(int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
