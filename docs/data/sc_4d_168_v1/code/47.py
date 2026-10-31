import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lse_polish(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.015, lr1=0.0008, mu=0.88, rng=None):
    x = x_init.copy()
    vel = np.zeros_like(x)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        x = _norm(x)
    return x

def _refine_single(s0, Xref, steps=80, beta0=45.0, beta1=280.0, lr0=0.09, lr1=0.006, rng=None):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w /= (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            s = np.array([1.0, 0.0, 0.0, 0.0]); ns = 1.0
        s = s / ns
    return s

def _lp_polish_deadline(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        iu_c = G[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = iu_c[sel]
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(20261031)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    xb = _norm(xb)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # --- prune highest stress points to N0=160 ---
    G = x_best @ x_best.T
    np.fill_diagonal(G, -2.0)
    m0 = float(np.max(G))
    deg_stress = np.sum(G > m0 - 0.004, axis=1)
    deg_tight = np.sum(G > m0 - 1e-7, axis=1)
    score_rm = deg_stress * 10.0 + deg_tight
    order_rm = np.argsort(-score_rm, kind="stable")
    N0 = 160
    n_drop = 168 - N0
    drop_idx = order_rm[:n_drop]
    keep_mask = np.ones(168, dtype=bool)
    keep_mask[drop_idx] = False
    X0 = x_best[keep_mask].copy()
    # fully polish smaller code (unjammed -> lower max cos)
    X0 = _lse_polish(X0, steps=700, beta0=25.0, beta1=550.0, lr0=0.016, lr1=0.0007, mu=0.88)
    # quick LP on the core if time allows
    try:
        X0, _ = _lp_polish_deadline(X0, min(time.time() + 2.5, SEARCH_END))
    except Exception:
        pass

    BEAM = 6
    beam = [X0]
    Ncur = N0
    # anneal params for short polish
    while Ncur < 168 and time.time() < SEARCH_END - 2.0:
        pool = []
        # candidates per parent to aim ~24 total
        Kper = max(4, int(np.ceil(24.0 / max(1, len(beam)))))
        if len(beam) == 1:
            Kper = 12
        for bidx, Xc in enumerate(beam):
            if time.time() > SEARCH_END - 1.0:
                break
            nc = Xc.shape[0]
            # 30k random sphere samples
            S = rng.randn(30000, 4)
            S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
            D = S @ Xc.T
            worst = D.max(axis=1)
            # best 300 holes
            kpre = 300
            idx = np.argpartition(worst, kpre)[:kpre]
            o = np.argsort(worst[idx])
            cand_pool = S[idx[o]]
            picked = []
            for c in cand_pool:
                if len(picked) >= Kper:
                    break
                clash = False
                for p in picked:
                    if float(c @ p) > 0.86:
                        clash = True
                        break
                if clash:
                    continue
                s_ref = _refine_single(c, Xc, steps=80, beta0=45.0, beta1=300.0)
                picked.append(s_ref)
            # fill if short
            fi = 0
            while len(picked) < Kper and fi < len(cand_pool):
                c = cand_pool[fi]; fi += 1
                dup = False
                for p in picked:
                    if float(c @ p) > 0.995:
                        dup = True; break
                if dup:
                    continue
                s_ref = _refine_single(c, Xc, steps=60, beta0=40.0, beta1=250.0)
                picked.append(s_ref)
            for p in picked:
                Xnew0 = np.vstack([Xc, p.reshape(1, 4)])
                Xnew0 = _norm(Xnew0)
                # short 200-step annealed polish (different beta per replica for diversity)
                j = len(pool) % 3
                if j == 0:
                    Xn = _lse_polish(Xnew0, steps=200, beta0=30.0, beta1=600.0, lr0=0.014, lr1=0.0007)
                elif j == 1:
                    Xn = _lse_polish(Xnew0, steps=200, beta0=20.0, beta1=450.0, lr0=0.016, lr1=0.0008)
                else:
                    Xn = _lse_polish(Xnew0, steps=200, beta0=45.0, beta1=900.0, lr0=0.012, lr1=0.0006)
                v = _max_cos(Xn)
                pool.append((v, Xn))
                if time.time() > SEARCH_END - 0.5:
                    break
        if len(pool) == 0:
            break
        pool.sort(key=lambda t: t[0])
        # keep beam best BEAM with mild diversity: skip near-duplicates (same value to 1e-6 and high correlation?)
        new_beam = []
        for v, Xn in pool:
            if len(new_beam) >= BEAM:
                break
            new_beam.append(Xn)
        beam = new_beam
        Ncur += 1
        # update global if completed size and promising? only 168 counts, but track
        if Ncur == 168:
            for Xn in beam:
                v = _max_cos(Xn)
                if v < best_val:
                    best_val = v
                    x_best = Xn.copy()

    # final LP polish best few 168 completions
    # gather candidates: beam (168) + incumbent
    cands = []
    for Xn in beam:
        if Xn.shape[0] == 168:
            cands.append(Xn)
    # if beam incomplete due to timeout, complete greedily to 168
    if len(cands) == 0 or any(c.shape[0] != 168 for c in beam):
        # fallback: pad best beam element greedily
        if len(beam) > 0:
            Xp = beam[0]
            while Xp.shape[0] < 168 and time.time() < HARD_END - 8.0:
                S = rng.randn(15000, 4)
                S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
                w = (S @ Xp.T).max(axis=1)
                c0 = S[int(np.argmin(w))]
                s_ref = _refine_single(c0, Xp, steps=80)
                Xp = _norm(np.vstack([Xp, s_ref.reshape(1, 4)]))
                Xp = _lse_polish(Xp, steps=200, beta0=30.0, beta1=600.0, lr0=0.014, lr1=0.0007)
            if Xp.shape[0] == 168:
                cands.append(Xp)
    cands.append(x_best)
    # dedupe + sort by score
    scored = []
    for c in cands:
        if c.shape != (168, 4):
            continue
        scored.append((_max_cos(_norm(c)), _norm(c)))
    scored.sort(key=lambda t: t[0])
    # LP polish top 3
    final_best = scored[0][1].copy()
    final_val = scored[0][0]
    # ensure incumbent kept
    if best_val < final_val:
        final_val = best_val
        final_best = x_best.copy()
    nlp = min(3, len(scored))
    for i in range(nlp):
        if time.time() > HARD_END - 1.5:
            break
        Xi = scored[i][1]
        # medium polish then LP
        if time.time() < HARD_END - 10.0:
            Xi = _lse_polish(Xi, steps=250, beta0=40.0, beta1=900.0, lr0=0.010, lr1=0.0005)
        try:
            dl = HARD_END - 1.0 if i == 0 else min(time.time() + 4.0, HARD_END - 1.0)
            Xs, vs = _lp_polish_deadline(Xi, dl)
        except Exception:
            continue
        if vs < final_val:
            final_val = vs
            final_best = Xs.copy()
        if time.time() > HARD_END - 2.0:
            break
    final_best = _norm(final_best)
    return final_best
