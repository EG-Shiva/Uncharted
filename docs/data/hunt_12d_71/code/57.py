import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X_init, steps, T0, T1, lr, mu=0.9, deadline=None):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 200 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _lp_polish(X0, seconds, deadline=None, r0=2e-3, cap=1500):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    try:
        best = _max_cos(X)
    except Exception:
        return X
    bestX = X.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(X.shape[0], 1)
    n, d = X.shape
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (X @ X.T)[iu]
        m = float(cmat.max())
        band = max(25.0 * radius, 2e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            order = np.argsort(-cmat[sel])[:cap]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
        else:
            csel = cmat[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return bestX

def _build_pool(rng, target=3200):
    d = 12
    pool = []
    tries = 0
    while len(pool) < target and tries < target * 80:
        tries += 1
        r = rng.random()
        if r < 0.65:
            v = rng.integers(-2, 3, size=d)
            if int(v.sum()) % 2 != 0:
                continue
            n2 = int((v * v).sum())
            if n2 not in (4, 6, 8):
                continue
            pool.append(v.astype(float) / np.sqrt(n2))
        elif r < 0.85:
            v = (rng.integers(0, 4, size=d).astype(float) - 1.5)
            mask = rng.random(d) < 0.25
            v[mask] = 0.0
            n2 = float((v * v).sum())
            if n2 < 2 or n2 > 12 or n2 < 1e-9:
                continue
            pool.append(v / np.sqrt(n2))
        else:
            v = rng.integers(-1, 2, size=d).astype(float)
            n2 = float((v * v).sum())
            if n2 < 4 or n2 > 8:
                continue
            if int(v.sum()) % 2 != 0:
                continue
            pool.append(v / np.sqrt(n2))
    if len(pool) == 0:
        return _norm_rows(rng.standard_normal(size=(target, d)))
    P = np.array(pool)
    # dedup roughly
    # add some random continuous to enrich
    extra = _norm_rows(rng.standard_normal(size=(min(500, target//4), d)))
    P = np.vstack([P, extra])
    # subsample if too big
    if len(P) > target + 500:
        idx = rng.choice(len(P), target + 500, replace=False)
        P = P[idx]
    # remove near-duplicates (dot>0.999)
    return P

def _greedy_select(P, n_sel, rng):
    N = P.shape[0]
    start = int(rng.integers(0, N))
    sel = [start]
    selset = set(sel)
    # precompute P norms=1
    cur_max = np.full(N, 2.0)
    # iterative
    S = P[sel]
    for k in range(1, n_sel):
        D = P @ P[sel[-1]]
        # update best (max dot to selected)? we need max over selected
        # maintain array best_dot
        if k == 1:
            best_dot = D.copy()
            best_dot[sel[0]] = 2.0
        else:
            best_dot = np.minimum(best_dot, 2.0)  # placeholder
            # recompute fully every few steps for accuracy
            pass
        # simplest: recompute via matrix multiply each step
        Dmat = P @ np.array([P[i] for i in sel]).T
        best_dot = Dmat.max(axis=1)
        for i in sel:
            best_dot[i] = 2.0
        nxt = int(np.argmin(best_dot))
        sel.append(nxt)
    return np.array(sel, dtype=int)

def _sa_select(P, sel, rng, iters=6000, deadline=None):
    n_sel = len(sel)
    S = P[sel].copy()
    def score_of(G):
        np.fill_diagonal(G, -2.0)
        mc = float(G.max())
        cnt = int((G > 0.22).sum() // 2)
        # sum exp term on upper tri
        iu = np.triu_indices(n_sel, 1)
        gv = G[iu]
        se = float(np.exp(np.clip((gv - 0.22) / 0.002, -50, 30)).sum())
        return mc, cnt, se
    G = S @ S.T
    mc, cnt, se = score_of(G.copy())
    best_sel = sel.copy()
    best_tup = (mc, cnt, se)
    cur_sel = sel.copy()
    cur_tup = best_tup
    cur_S = S.copy()
    cur_G = G.copy()
    N = P.shape[0]
    in_set = np.zeros(N, dtype=bool)
    in_set[cur_sel] = True
    T0, T1 = 0.003, 0.0002
    for it in range(iters):
        if deadline is not None and (it % 500 == 0) and time.time() > deadline:
            break
        frac = it / max(1, iters - 1)
        T = T0 * (T1 / T0) ** frac
        # propose swap: 1 index
        a = int(rng.integers(0, n_sel))
        b = int(rng.integers(0, N))
        if in_set[b]:
            continue
        old_idx = cur_sel[a]
        # build new S quickly: replace row a
        new_row = P[b]
        # compute new Gram row/col
        # full recompute 71x71 cheap
        trial_S = cur_S.copy()
        trial_S[a] = new_row
        Gt = trial_S @ trial_S.T
        mc2, cnt2, se2 = score_of(Gt.copy())
        # lexicographic with annealing on mc + penalties
        # scalarize: mc + 1e-4*cnt/100 + 1e-7*log(se)? use hierarchy
        # accept by mc primarily
        def scalar(t):
            return t[0] + 2e-4 * t[1] + 1e-9 * t[2]
        s_cur = scalar(cur_tup)
        s_new = scalar((mc2, cnt2, se2))
        if s_new < s_cur or rng.random() < np.exp(-(s_new - s_cur) / max(T, 1e-12)):
            cur_sel[a] = b
            in_set[old_idx] = False
            in_set[b] = True
            cur_S = trial_S
            cur_tup = (mc2, cnt2, se2)
            if scalar(cur_tup) < scalar(best_tup) - 1e-15:
                best_tup = cur_tup
                best_sel = cur_sel.copy()
    return best_sel, best_tup

def _cyclic_expand(S):
    blocks = [np.roll(S, k, axis=1) for k in range(12)]
    return np.vstack(blocks)

def _cyclic_grad_back(Gt, ns):
    # Gt shape (ns*12,12), blocks of ns rows each k
    grad = np.zeros((ns, 12))
    for k in range(12):
        Gb = Gt[k*ns:(k+1)*ns, :]
        grad += np.roll(Gb, -k, axis=1)
    return grad / 12.0

def _blockrot_expand(S, angles=None):
    ns = S.shape[0]
    if angles is None:
        angles = [k * 2 * np.pi / 6 for k in range(6)]
    outs = []
    for th in angles:
        c, s = np.cos(th), np.sin(th)
        Xb = S.copy()
        for p in range(6):
            a = S[:, 2*p]; b = S[:, 2*p+1]
            Xb[:, 2*p] = c * a - s * b
            Xb[:, 2*p+1] = s * a + c * b
        outs.append(Xb)
    return np.vstack(outs)

def _blockrot_grad_back(Gt, ns, angles=None):
    if angles is None:
        angles = [k * 2 * np.pi / 6 for k in range(6)]
    grad = np.zeros((ns, 12))
    for k, th in enumerate(angles):
        c, s = np.cos(th), np.sin(th)
        Gb = Gt[k*ns:(k+1)*ns, :]
        for p in range(6):
            g0 = Gb[:, 2*p]; g1 = Gb[:, 2*p+1]
            # R^T * g
            grad[:, 2*p] += c * g0 + s * g1
            grad[:, 2*p+1] += -s * g0 + c * g1
    return grad / len(angles)

def _orbit_seed_opt(rng, ns, expand_fn, back_fn, steps=1200, T0=0.01, T1=0.0008, lr=0.25, mu=0.85, deadline=None):
    S = _norm_rows(rng.standard_normal(size=(ns, 12)))
    V = np.zeros_like(S)
    lt0, lt1 = np.log(T0), np.log(T1)
    best_S = S.copy()
    # evaluate expanded best
    X0 = expand_fn(S)
    # trim to 71 later; for seed opt use full orbit (72) score
    G0 = X0 @ X0.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    for st in range(steps):
        if deadline is not None and (st % 200 == 0) and time.time() > deadline:
            break
        frac = st / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        X = expand_fn(S)
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_S = S.copy()
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gs = back_fn(Gt)
        # project to tangent of seeds
        dotS = np.sum(Gs * S, axis=1, keepdims=True)
        GtS = Gs - dotS * S
        gn = np.linalg.norm(GtS, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            GtS[bad] *= (2.0 / gn[bad])
        V = mu * V + GtS
        S = S - lr * V
        S = _norm_rows(S)
    return best_S, best_m

def _drop_to_71(X):
    # X is (72,12) -> drop one index minimizing max
    n = X.shape[0]
    if n == 71:
        return X
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    best_i = -1
    best_v = 1e100
    for i in range(n):
        # max excluding i
        m = G.copy()
        m[i, :] = -2.0
        m[:, i] = -2.0
        v = float(m.max())
        if v < best_v:
            best_v = v
            best_i = i
    keep = [k for k in range(n) if k != best_i]
    return X[keep]

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        Xbest = np.load("best.npy")
    except Exception:
        Xbest = None
    try:
        Xrec = np.load("record.npy")
    except Exception:
        Xrec = None
    cands = []
    if Xbest is not None:
        try:
            Xbest = _norm_rows(np.asarray(Xbest, dtype=float))
            if Xbest.shape == (71, 12):
                cands.append(Xbest)
        except Exception:
            pass
    if Xrec is not None:
        try:
            Xrec = _norm_rows(np.asarray(Xrec, dtype=float))
            if Xrec.shape == (71, 12):
                cands.append(Xrec)
        except Exception:
            pass
    if len(cands) == 0:
        Xbest = _norm_rows(rng.standard_normal(size=(71, 12)))
        best = _max_cos(Xbest)
    else:
        vals = [_max_cos(c) for c in cands]
        bi = int(np.argmin(vals))
        Xbest = cands[bi]
        best = float(vals[bi])
    # ---------- (a) lattice shell route ----------
    try:
        rngL = np.random.default_rng(1)
        P = _build_pool(rngL, target=3000)
        # greedy from a few starts, pick best greedy quickly
        sel0 = _greedy_select(P, 71, rngL)
        S0 = P[sel0]
        Gg = S0 @ S0.T
        np.fill_diagonal(Gg, -2.0)
        gval = float(Gg.max())
        # SA selection
        sel_sa, tup = _sa_select(P, sel0, rngL, iters=5000, deadline=HARD_END - 100)
        Ssa = P[np.array(sel_sa)]
        Xq, vq = _quench(Ssa, steps=2200, T0=0.008, T1=0.0006, lr=0.08, mu=0.9, deadline=HARD_END - 90)
        vq = _max_cos(Xq)
        if vq < best + 0.01:
            Xq2 = _lp_polish(Xq, seconds=2.0, deadline=HARD_END - 80)
            vq2 = _max_cos(Xq2)
            if vq2 < best:
                Xbest, best = Xq2, vq2
            elif vq < best:
                Xbest, best = Xq, vq
            # keep as restart base even if slightly worse
            lat_base = Xq2 if vq2 < vq else Xq
            lat_val = min(vq, vq2)
        else:
            lat_base, lat_val = Xq, vq
    except Exception:
        lat_base, lat_val = Xbest, best
    if time.time() > HARD_END - 95:
        lat_base = Xbest
    # ---------- (b) group-orbit routes ----------
    orb_bases = []
    try:
        # cyclic 6 seeds -> 72 pts
        rngO = np.random.default_rng(2)
        ns = 6
        S_cyc, _ = _orbit_seed_opt(rngO, ns, _cyclic_expand, lambda Gt: _cyclic_grad_back(Gt, ns),
                                  steps=900, T0=0.01, T1=0.001, lr=0.3, mu=0.85, deadline=HARD_END - 75)
        Xc = _drop_to_71(_cyclic_expand(S_cyc))
        # break symmetry + quench
        Xc = _norm_rows(Xc + 0.008 * rngO.standard_normal(size=Xc.shape))
        Xc, vc = _quench(Xc, steps=1800, T0=0.006, T1=0.0005, lr=0.08, mu=0.9, deadline=HARD_END - 65)
        vc = _max_cos(Xc)
        orb_bases.append((Xc, vc))
        if vc < best - 1e-12:
            Xcp = _lp_polish(Xc, seconds=1.5, deadline=HARD_END - 60)
            vcp = _max_cos(Xcp)
            if vcp < best:
                Xbest, best = Xcp, vcp
            else:
                if vc < best:
                    Xbest, best = Xc, vc
        # block-rotation 12 seeds -> 72 pts
        rngO2 = np.random.default_rng(3)
        ns2 = 12
        S_br, _ = _orbit_seed_opt(rngO2, ns2, _blockrot_expand, lambda Gt: _blockrot_grad_back(Gt, ns2),
                                 steps=900, T0=0.01, T1=0.001, lr=0.3, mu=0.85, deadline=HARD_END - 55)
        Xb = _drop_to_71(_blockrot_expand(S_br))
        Xb = _norm_rows(Xb + 0.01 * rngO2.standard_normal(size=Xb.shape))
        Xb, vb = _quench(Xb, steps=1800, T0=0.006, T1=0.0005, lr=0.08, mu=0.9, deadline=HARD_END - 45)
        vb = _max_cos(Xb)
        orb_bases.append((Xb, vb))
        if vb < best - 1e-12:
            Xbp = _lp_polish(Xb, seconds=1.5, deadline=HARD_END - 40)
            vbp = _max_cos(Xbp)
            if vbp < best:
                Xbest, best = Xbp, vbp
            else:
                if vb < best:
                    Xbest, best = Xb, vb
    except Exception:
        pass
    # collect restart pool: Xbest + lat_base + orb bases
    pool_restart = [Xbest]
    try:
        pool_restart.append(lat_base)
    except Exception:
        pass
    for (Xo, vo) in orb_bases:
        # only keep reasonably close to avoid wasting time
        if vo < best + 0.02:
            pool_restart.append(Xo)
    # ---------- basin-hopping refinement ----------
    rngH = np.random.default_rng(4)
    n, d = (71, 12)
    trial = 0
    EXPLORE_END = HARD_END - 14.0
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.03, 0.07]
    while time.time() < EXPLORE_END and trial < 300:
        trial += 1
        base = pool_restart[trial % len(pool_restart)] if len(pool_restart) > 0 else Xbest
        # if base is far worse, mostly use Xbest
        if trial % 4 != 0:
            base = Xbest
        X0 = base.copy()
        rtype = rngH.random()
        sigma = sigmas[trial % len(sigmas)]
        if rtype < 0.55:
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            cnt = (G > best - 1e-7).sum(axis=1)
            w = np.ones(n)
            try:
                thresh = np.sort(cnt)[-8]
                w[cnt >= thresh] = 2.0
            except Exception:
                pass
            noise = rngH.standard_normal(size=X0.shape)
            X0 = X0 + sigma * noise * w[:, None]
            X0 = _norm_rows(X0)
        elif rtype < 0.85:
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            cnt = (G > best - 1e-7).sum(axis=1)
            order = np.argsort(-cnt)
            cand_v = order[:6]
            victim = int(cand_v[rngH.integers(0, len(cand_v))])
            R = np.delete(X0, victim, axis=0)
            cand = _norm_rows(rngH.standard_normal(size=(2000, d)))
            scores = (cand @ R.T).max(axis=1)
            X0[victim] = cand[np.argmin(scores)]
            X0 = _norm_rows(X0 + 0.012 * rngH.standard_normal(size=X0.shape))
        else:
            k = 4 + int(rngH.integers(0, 3))
            idx = rngH.choice(n, k, replace=False)
            keep = np.delete(np.arange(n), idx)
            R = X0[keep]
            for j in idx:
                cand = _norm_rows(rngH.standard_normal(size=(2000, d)))
                scores = (cand @ R.T).max(axis=1)
                X0[j] = cand[np.argmin(scores)]
                R = np.vstack([R, X0[j][None, :]])
            X0 = _norm_rows(X0 + 0.008 * rngH.standard_normal(size=X0.shape))
        # quench params alternate
        if trial % 3 == 0:
            T0_, T1_, lr_, steps_ = 0.008, 0.0004, 0.09, 1600
        elif trial % 3 == 1:
            T0_, T1_, lr_, steps_ = 0.004, 0.0003, 0.07, 1400
        else:
            T0_, T1_, lr_, steps_ = 0.012, 0.0008, 0.10, 1200
        Xopt, v = _quench(X0, steps=steps_, T0=T0_, T1=T1_, lr=lr_, mu=0.9, deadline=EXPLORE_END)
        if v < best - 1e-12:
            Xp = _lp_polish(Xopt, seconds=1.2, deadline=HARD_END - 10)
            vp = _max_cos(Xp)
            if vp < best:
                Xbest, best = Xp, vp
            else:
                Xbest, best = Xopt, v
            pool_restart[0] = Xbest
        elif v < best + 0.001 and (trial % 3 == 0):
            Xp = _lp_polish(Xopt, seconds=0.8, deadline=HARD_END - 10)
            vp = _max_cos(Xp)
            if vp < best:
                Xbest, best = Xp, vp
        if time.time() > EXPLORE_END:
            break
    remain = HARD_END - time.time() - 1.5
    if remain > 2.5:
        Xbest = _lp_polish(Xbest, seconds=min(remain, 11.0), deadline=HARD_END - 0.5)
    Xbest = _norm_rows(Xbest)
    return Xbest
