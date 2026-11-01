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
        if (s % 150 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _lp_polish(X0, seconds, deadline=None, r0=2e-3, band_mult=25.0, floor=2e-4, cap=1600):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    bestX = X.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (X @ X.T)[iu]
        m = float(cmat.max())
        band = max(band_mult * radius, floor)
        # shrink floor when very tight
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

def _build_pool(rng, target=3600):
    d = 12
    pool = []
    # integer D12 shells norm 4,6,8
    tries = 0
    while len(pool) < target and tries < target * 60:
        tries += 1
        r = rng.random()
        if r < 0.7:
            v = rng.integers(-2, 3, size=d)
            if int(v.sum()) % 2 != 0:
                continue
            n2 = int((v * v).sum())
            if n2 not in (4, 6, 8):
                continue
            pool.append(v.astype(float) / np.sqrt(n2))
        else:
            # half-integer shell: entries in +-0.5,+-1.5
            v = (rng.integers(0, 4, size=d).astype(float) - 1.5)
            # v in -1.5,-0.5,0.5,1.5 ; all half-odd => sum parity auto
            # random zero-out some coords to vary norm
            mask = rng.random(d) < 0.25
            v[mask] = 0.0
            n2 = float((v * v).sum())
            if n2 < 1e-9:
                continue
            # keep only mid norms to mimic shell
            if n2 > 12 or n2 < 2:
                continue
            pool.append(v / np.sqrt(n2))
    if len(pool) == 0:
        pool = [rng.standard_normal(d) for _ in range(target)]
        pool = [_norm_rows(np.array([p]))[0] for p in pool]
    P = np.array(pool, dtype=float)
    P = _norm_rows(P)
    # dedupe roughly
    # keep target
    if len(P) > target:
        P = P[:target]
    return P

def _greedy_select(P, k, rng):
    m = len(P)
    # start from random
    start = int(rng.integers(0, m))
    sel = [start]
    # max dot to selected
    best_dot = P @ P[start]
    best_dot[start] = 2.0
    in_sel = np.zeros(m, dtype=bool); in_sel[start] = True
    for _ in range(1, k):
        # choose unselected minimizing max dot to sel
        # compute max dot to current sel for all
        # incremental: maintain cur_max
        if len(sel) == 1:
            cur = P @ P[start]
        else:
            # recompute cheap: P @ sel.T max
            cur = (P @ P[sel].T).max(axis=1)
        cur[in_sel] = 2.0
        nxt = int(np.argmin(cur))
        sel.append(nxt)
        in_sel[nxt] = True
    return np.array(sel, dtype=int)

def _set_scores(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mc = float(G.max())
    cnt = int((G > 0.22).sum() // 2)
    # sum exp
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    s = float(np.exp(np.clip((c - 0.22) / 0.002, -50, 50)).sum())
    return mc, cnt, s

def _sa_swaps(P, sel0, rng, iters, T0=0.004, T1=0.00005):
    m = len(P)
    k = len(sel0)
    sel = sel0.copy()
    in_sel = np.zeros(m, dtype=bool); in_sel[sel] = True
    X = P[sel]
    mc, cnt, s = _set_scores(X)
    # scalar energy: lexicographic approximated
    def energy(mc_, cnt_, s_):
        return mc_ * 10.0 + cnt_ * 0.0008 + s_ * 0.00002
    E = energy(mc, cnt, s)
    best_sel = sel.copy(); best_tup = (mc, cnt, s); bestE = E
    lt0, lt1 = np.log(T0), np.log(T1)
    # precompute for speed? just loop
    for it in range(iters):
        frac = it / max(1, iters - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        oi = int(rng.integers(0, k))
        # random outside
        ni = int(rng.integers(0, m))
        if in_sel[ni]:
            continue
        old = int(sel[oi])
        sel[oi] = ni
        Xn = P[sel]
        mc2, cnt2, s2 = _set_scores(Xn)
        E2 = energy(mc2, cnt2, s2)
        dE = E2 - E
        # lexicographic hard accept if max improves notably
        accept = False
        if E2 < E:
            accept = True
        else:
            if rng.random() < np.exp(-dE / max(T, 1e-12)):
                accept = True
        if accept:
            E = E2; mc, cnt, s = mc2, cnt2, s2
            in_sel[old] = False; in_sel[ni] = True
            if E2 < bestE - 1e-15:
                bestE = E2; best_sel = sel.copy(); best_tup = (mc, cnt, s)
        else:
            sel[oi] = old
    return best_sel, best_tup

def _orbit_quench(seeds0, perms, inv_perms, steps, T0, T1, lr, mu, rng=None, noise=0.0):
    ns, d = seeds0.shape
    Gsz = perms.shape[0]
    seeds = seeds0.copy()
    V = np.zeros_like(seeds)
    lt0, lt1 = np.log(T0), np.log(T1)
    # build helper to get full X
    def build(S):
        Xf = np.empty((ns * Gsz, d))
        for g in range(Gsz):
            Xf[g::Gsz] = S[:, perms[g]] if False else S  # placeholder
        # correct: orbit of seed k under perm g
        for kk in range(ns):
            for g in range(Gsz):
                Xf[kk * Gsz + g] = S[kk, perms[g]]
        return _norm_rows(Xf)
    X = build(seeds)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_S = seeds.copy()
    best_X = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        X = build(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        mm = float(G.max())
        Q = (G - mm) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs  # N x d in orbit order
        # pullback to seeds
        GS = np.zeros_like(seeds)
        for kk in range(ns):
            acc = np.zeros(d)
            for g in range(Gsz):
                row = Grad[kk * Gsz + g]
                # y[j]=s[perm[g][j]] => dL/ds[p] += dL/dy[j] where perm[g][j]==p
                # so scatter
                acc[perms[g]] += row  # since row[j] corresponds to s[perm[j]]
            GS[kk] = acc
        # tangent project per seed
        dot = np.sum(GS * seeds, axis=1, keepdims=True)
        Gt = GS - dot * seeds
        # clip
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        if rng is not None and noise > 0:
            N = rng.standard_normal(seeds.shape) * noise
            N -= seeds * np.sum(N * seeds, axis=1, keepdims=True)
            Gt = Gt + N
        V = mu * V + Gt
        seeds = seeds - lr * V
        seeds = _norm_rows(seeds)
        if (s % 100 == 0) or (s == steps - 1):
            Xc = build(seeds)
            Gc = Xc @ Xc.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_S = seeds.copy()
                best_X = Xc.copy()
    return best_S, best_X, best_m

def solve():
    t0 = time.time()
    HARD = t0 + 173.0
    rng = np.random.default_rng(7)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((n, d)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    # quick LP touch on incumbent if time allows (keeps us <= lab best)
    # ---- Path A: lattice shell + greedy + SA + quench ----
    try:
        P = _build_pool(rng, target=3400)
        # two greedy starts
        for gtry in range(2):
            if time.time() > t0 + 55:
                break
            sel0 = _greedy_select(P, n, rng)
            sel_best, tup = _sa_swaps(P, sel0, rng, iters=9000, T0=0.003, T1=0.00004)
            X0 = P[sel_best].copy()
            # jitter slightly to leave lattice
            X0 = _norm_rows(X0 + rng.standard_normal(X0.shape) * 1e-4)
            Xq, mq = _quench(X0, steps=900, T0=0.008, T1=0.0006, lr=0.18, mu=0.88,
                             deadline=min(HARD - 60, t0 + 70))
            if mq < best_score:
                best_score, best_X = mq, Xq.copy()
            # light LP
            Xp = _lp_polish(Xq, seconds=3.0, deadline=min(HARD - 55, time.time() + 3.2))
            vp = _max_cos(Xp)
            if vp < best_score:
                best_score, best_X = vp, Xp.copy()
            # stronger quench from jittered lattice best
            Xj = _norm_rows(Xp + rng.standard_normal(Xp.shape) * 0.004)
            Xq2, mq2 = _quench(Xj, steps=700, T0=0.004, T1=0.0003, lr=0.15, mu=0.9,
                               deadline=min(HARD - 50, time.time() + 12))
            if mq2 < best_score:
                best_score, best_X = mq2, Xq2.copy()
    except Exception:
        pass
    # ---- Path B: group-orbit ansatze (six 2D planes) ----
    try:
        # cyclic shifts of 12 coords (order 12) + block-plane cyclic (order 6) share same perms family
        Gsz = 12
        perms = np.zeros((Gsz, d), dtype=int)
        for g in range(Gsz):
            perms[g] = np.array([(j + g) % d for j in range(d)], dtype=int)
        # also signed variant: flip sign of first half on odd g (block rotation by pi)
        signs = np.ones((Gsz, d))
        for g in range(Gsz):
            if g % 2 == 1:
                signs[g, :6] = -1.0
        # helper build with signs outside _orbit_quench: we fold sign into optimization by using perms only
        # and a second trial with signs via pre-flipped seeds? Use perms only for stability, test sign by augment.
        for trial in range(3):
            if time.time() > t0 + 105:
                break
            ns = 6  # 6*12=72 -> drop 1
            if trial == 2:
                ns = 6
            seeds0 = _norm_rows(rng.standard_normal((ns, d)))
            # init seeds from lattice pool diverse rows if available
            try:
                idx = rng.choice(len(P), size=ns, replace=False)
                if rng.random() < 0.5:
                    seeds0 = P[idx].copy() + rng.standard_normal((ns, d)) * 0.05
                    seeds0 = _norm_rows(seeds0)
            except Exception:
                pass
            inv_dummy = None
            Sbest, Xfull, mfull = _orbit_quench(seeds0, perms, inv_dummy, steps=700,
                                               T0=0.01, T1=0.0008, lr=0.12, mu=0.85,
                                               rng=rng, noise=0.0)
            # Xfull is 72; drop worst row to get 71
            Gf = Xfull @ Xfull.T
            np.fill_diagonal(Gf, -2.0)
            rowmax = Gf.max(axis=1)
            drop = int(np.argmax(rowmax))
            keep = [i for i in range(ns * Gsz) if i != drop]
            X71 = Xfull[keep]
            v71 = _max_cos(X71)
            # release symmetry: small break + full quench
            Xb = _norm_rows(X71 + rng.standard_normal(X71.shape) * (0.006 + 0.004 * trial))
            Xq, mq = _quench(Xb, steps=800, T0=0.006, T1=0.0004, lr=0.16, mu=0.88,
                             deadline=min(HARD - 30, time.time() + 12))
            if mq < best_score:
                best_score, best_X = mq, Xq.copy()
            # signed variant: apply sign flips to orbit then quench
            if trial == 0:
                Xsign = Xfull.copy()
                for g in range(Gsz):
                    Xsign[g::Gsz] = Xsign[g::Gsz] * signs[g]
                    # note strided mapping differs from block mapping; approximate: apply per full index
                # rebuild correctly with signs: Xsign[kk*G+g] *= signs[g]
                Xsign2 = Xfull.copy()
                for kk in range(ns):
                    for g in range(Gsz):
                        Xsign2[kk * Gsz + g] *= signs[g]
                Xsign2 = _norm_rows(Xsign2)
                keep2 = keep
                X71s = Xsign2[keep2]
                Xbs = _norm_rows(X71s + rng.standard_normal(X71s.shape) * 0.008)
                Xqs, mqs = _quench(Xbs, steps=600, T0=0.006, T1=0.0004, lr=0.16, mu=0.88,
                                   deadline=min(HARD - 25, time.time() + 10))
                if mqs < best_score:
                    best_score, best_X = mqs, Xqs.copy()
    except Exception:
        pass
    # ---- Final intensification on global best ----
    try:
        # a few full quenches with different temps
        for _ in range(2):
            if time.time() > HARD - 24:
                break
            Xq, mq = _quench(best_X, steps=900, T0=0.004, T1=0.00015, lr=0.14, mu=0.9,
                             deadline=HARD - 20)
            if mq < best_score:
                best_score, best_X = mq, Xq.copy()
        Xp = _lp_polish(best_X, seconds=max(2.0, (HARD - 19) - time.time()),
                        deadline=HARD - 2.0, r0=2.5e-3)
        vp = _max_cos(Xp)
        if vp < best_score:
            best_score, best_X = vp, Xp.copy()
        # last micro quench
        if time.time() < HARD - 8:
            Xq, mq = _quench(best_X, steps=500, T0=0.0015, T1=0.00008, lr=0.10, mu=0.9,
                             deadline=HARD - 3.0)
            if mq < best_score:
                best_score, best_X = mq, Xq.copy()
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    # safety: if somehow worse than record, return incumbent file content
    try:
        cur = _max_cos(best_X)
        if not np.isfinite(cur):
            best_X = _norm_rows(np.load("best.npy").astype(float))
    except Exception:
        pass
    return np.asarray(best_X, dtype=float)
