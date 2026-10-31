import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lse_polish(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88, deadline=None):
    x = x_init.copy()
    vel = np.zeros_like(x)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for s in range(steps):
        if deadline is not None and s % 50 == 0 and time.time() > deadline:
            break
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
        x = _nm(x)
    return x

def _lp_polish_deadline(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    try:
        best = _mc(x)
    except Exception:
        return x, 2.0
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
        y = _nm(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _stress_order(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg_tight = np.sum(G > m - 1e-9, axis=1)
    deg_wide = np.sum(G > m - 0.005, axis=1)
    stress = np.sum(np.clip(G - (m - 0.01), 0, None), axis=1)
    order = np.lexsort((-stress, -deg_wide, -deg_tight))
    # lexsort sorts ascending; we want descending, so negate trick:
    # recompute via stable argsort on tuple
    idx = np.argsort(-(deg_tight * 1000 + deg_wide * 10 + stress * 50), kind="stable")
    return idx, m

def _gen_holes(Xref, rng, n_sample=30000, H=30):
    # random sphere samples
    S = rng.randn(n_sample, 4)
    S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
    D = S @ Xref.T
    worst = D.max(axis=1)
    K = min(n_sample, max(H * 2, H + 10))
    idx = np.argpartition(worst, K)[:K]
    cand = S[idx].copy()
    # local hole refinement (vectorized, minimize softmax max)
    beta = 80.0
    lr = 0.08
    for _ in range(35):
        D2 = cand @ Xref.T
        m = D2.max(axis=1, keepdims=True)
        W = np.exp(np.clip(beta * (D2 - m), -20, 0))
        W /= (W.sum(axis=1, keepdims=True) + 1e-30)
        grad = W @ Xref
        dot = np.sum(grad * cand, axis=1, keepdims=True)
        gt = grad - dot * cand
        cand = cand - lr * gt
        cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    Df = cand @ Xref.T
    wf = Df.max(axis=1)
    order = np.argsort(wf, kind="stable")
    cand = cand[order]
    # diversify: keep top H but ensure separation among kept holes
    # greedy filter: accept if dot to accepted < 0.98, else skip; fill remainder
    kept = []
    for c in cand:
        ok = True
        for k in kept:
            if float(c @ k) > 0.985:
                ok = False
                break
        if ok:
            kept.append(c)
        if len(kept) >= H:
            break
    if len(kept) < H:
        for c in cand:
            if len(kept) >= H:
                break
            # add duplicates to fill
            if not any(np.allclose(c, k) for k in kept):
                kept.append(c)
    if len(kept) == 0:
        kept = [cand[0]]
    return np.array(kept[:H])

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 136.0
    FINAL_END = t0 + 169.0
    rng = np.random.RandomState(20261107)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    xb = _nm(np.asarray(xb, dtype=float))
    if xb.shape != (168, 4):
        xb = _nm(rng.randn(168, 4))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _nm(xr)
        if xr.shape == (168, 4) and _mc(xr) < _mc(xb):
            xb = xr.copy()
    except Exception:
        pass
    best168_val = _mc(xb)
    x_best168 = xb.copy()

    # ---- successive prune to N=160 with polish at each stage ----
    cur = x_best168.copy()
    targets = [166, 164, 162, 160]
    for tgt in targets:
        if time.time() > SEARCH_END - 20:
            break
        ncur = cur.shape[0]
        need = ncur - tgt
        if need <= 0:
            continue
        # recompute stress order on cur
        idx, _ = _stress_order(cur)
        victims = idx[:need]
        # occasionally swap one victim with random among top-16 for diversity
        mask = np.ones(ncur, dtype=bool)
        mask[victims] = False
        cur = cur[mask]
        # polish smaller code (unjammed -> lower max cosine)
        cur = _lse_polish(cur, steps=500, beta0=25.0, beta1=500.0,
                          lr0=0.016, lr1=0.0007, mu=0.88, deadline=SEARCH_END)
        # light LP if time allows for the smaller code at 160 only
        if tgt == 160 and time.time() < SEARCH_END - 8:
            try:
                cur, _ = _lp_polish_deadline(cur, min(time.time() + 3.0, SEARCH_END))
            except Exception:
                pass
    if cur.shape[0] != 160:
        # fallback: direct prune to 160
        idx, _ = _stress_order(x_best168)
        keep = np.ones(168, dtype=bool)
        keep[idx[:168-160]] = False
        cur = x_best168[keep].copy()
        cur = _lse_polish(cur, steps=400, beta0=25.0, beta1=500.0,
                          lr0=0.016, lr1=0.0007, mu=0.88, deadline=SEARCH_END)
    X160 = _nm(cur.copy())

    # ---- init beam of 6 at N=160 ----
    B = 6
    beam = [X160]
    # add perturbed variants
    for b_i in range(B - 1):
        if time.time() > SEARCH_END:
            break
        Xp = _nm(X160 + 0.006 * rng.randn(*X160.shape))
        Xp = _lse_polish(Xp, steps=220, beta0=30.0, beta1=550.0,
                         lr0=0.014, lr1=0.0007, mu=0.88, deadline=SEARCH_END)
        beam.append(Xp)
    # trim to B best by score
    scored = [( _mc(X), X) for X in beam]
    scored.sort(key=lambda t: t[0])
    beam = [X for _, X in scored[:B]]

    # ---- beam reinsertion 160 -> 168 ----
    H = 24  # candidates per parent
    NSAMP = 30000
    short_steps = 200
    for N in range(160, 168):
        if time.time() > SEARCH_END:
            break
        pool = []
        for bi, Xp in enumerate(beam):
            if time.time() > SEARCH_END:
                break
            holes = _gen_holes(Xp, rng, n_sample=NSAMP, H=H)
            for h in holes:
                if time.time() > SEARCH_END:
                    break
                Xc0 = np.vstack([Xp, h.reshape(1, 4)])
                Xc0 = _nm(Xc0)
                Xc = _lse_polish(Xc0, steps=short_steps, beta0=30.0, beta1=550.0,
                                 lr0=0.014, lr1=0.0007, mu=0.88, deadline=SEARCH_END)
                v = _mc(Xc)
                pool.append((v, Xc))
                if time.time() > SEARCH_END:
                    break
        if len(pool) == 0:
            break
        pool.sort(key=lambda t: t[0])
        # keep best B with diversity (avoid near-duplicates by score crowding is fine)
        # enforce diversity: greedily skip configs too close in score AND structure?
        new_beam = []
        for v, Xc in pool:
            dup = False
            for _, Y in [(0, Z) for Z in new_beam]:
                # quick check: mean alignment after optimal perm is expensive; use score proximity + Frobenius after sorting? skip
                pass
            new_beam.append(Xc)
            if len(new_beam) >= B:
                break
        beam = new_beam
        # shrink H/steps if running late
        if time.time() > SEARCH_END - 15:
            H = 14
            short_steps = 150
        elif time.time() > t0 + 90:
            H = 18

    # beam now at N=168 (or earlier if timed out -> pad by greedy void fill)
    cands168 = []
    for Xp in beam:
        if Xp.shape[0] == 168:
            cands168.append(Xp)
        elif Xp.shape[0] < 168:
            # pad to 168 greedily
            Xg = Xp.copy()
            while Xg.shape[0] < 168 and time.time() < SEARCH_END + 4:
                holes = _gen_holes(Xg, rng, n_sample=20000, H=6)
                Xg = _nm(np.vstack([Xg, holes[0].reshape(1, 4)]))
                Xg = _lse_polish(Xg, steps=150, beta0=30.0, beta1=500.0,
                                 lr0=0.013, lr1=0.0007, mu=0.88, deadline=SEARCH_END + 4)
            if Xg.shape[0] == 168:
                cands168.append(Xg)
    if len(cands168) == 0:
        cands168 = [x_best168]
    # rank and medium polish best few
    scored168 = sorted([(_mc(X), X) for X in cands168], key=lambda t: t[0])
    # medium refine top 3
    refined = []
    for k in range(min(3, len(scored168))):
        v, X = scored168[k]
        if time.time() > FINAL_END - 12:
            refined.append((v, X))
            continue
        Xr = _lse_polish(X.copy(), steps=450, beta0=30.0, beta1=900.0,
                         lr0=0.012, lr1=0.0005, mu=0.87, deadline=FINAL_END - 10)
        vr = _mc(Xr)
        refined.append((vr, Xr))
    refined.sort(key=lambda t: t[0])
    # LP polish best few
    final_best_val = best168_val
    final_best = x_best168.copy()
    # consider refined + incumbent
    allc = refined + [(best168_val, x_best168)]
    allc.sort(key=lambda t: t[0])
    for k in range(min(3, len(allc))):
        v, X = allc[k]
        if time.time() > FINAL_END - 1.5:
            if v < final_best_val:
                final_best_val, final_best = v, X.copy()
            break
        # only LP-polish if promising (within 0.002 of best to save time)
        if v > final_best_val + 0.002 and k > 0:
            continue
        try:
            Xlp, vlp = _lp_polish_deadline(X, min(time.time() + 4.0, FINAL_END - 0.5))
        except Exception:
            continue
        if vlp < final_best_val:
            final_best_val, final_best = vlp, Xlp.copy()
    final_best = _nm(final_best)
    return final_best
