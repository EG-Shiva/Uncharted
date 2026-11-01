import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _loads(X, band=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    rm = G.max(axis=1)
    cnt = np.sum(G > (m - band), axis=1)
    return rm, cnt, m

def _quench_adam(X_init, steps, T0, T1, lr0, lr1, alpha=0.6):
    X = _norm_rows(X_init.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Grad_g = (E @ X) / S
        rm = G.max(axis=1)
        Q = (G - rm[:, None]) / T
        np.clip(Q, -50, 0, out=Q)
        Er = np.exp(Q)
        np.fill_diagonal(Er, 0.0)
        rs = Er.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = Er / rs
        Ws = 0.5 * (W + W.T)
        Grad_r = Ws @ X
        Grad = alpha * Grad_g + (1.0 - alpha) * Grad_r
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        maxn = 10.0 * np.sqrt(n * d)
        if sn > maxn:
            step *= (maxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        best_m = mf
        best_X = X.copy()
    return best_X

def _quench_riesz(X, steps, s_pow, lr, mu=0.85):
    X = _norm_rows(X.copy())
    n, d = X.shape
    V = np.zeros_like(X)
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        D2 = np.clip(2.0 - 2.0 * G, 1e-6, 4.0)
        W = np.power(D2, -s_pow * 0.5 - 1.0) * s_pow
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        med = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / med
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 3.0
        if np.any(bad):
            Gt[bad] *= (3.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return X

def _quench_hard(X_init, steps=250, d0=0.008, d1=0.0008, lr=0.02, mu=0.85):
    X = _norm_rows(X_init.copy())
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bm = float(G0.max()); bX = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        delta = d0 * (1 - frac) + d1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bm:
            bm = m; bX = X.copy()
        mask = G > (m - delta)
        np.fill_diagonal(mask, False)
        W = mask.astype(float)
        # weight by closeness
        W = W * (1.0 + (G - (m - delta)) / max(delta, 1e-12))
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize
        med = float(np.median(np.linalg.norm(Gt, axis=1)) + 1e-12)
        Gt = Gt / med
        V = mu * V + Gt
        X = X - lr * (1 - 0.5 * frac) * V
        X = _norm_rows(X)
    return bX

def _refine_single(s0, R, steps=200, T=0.004, lr=0.12, mu=0.8):
    s = s0 / (np.linalg.norm(s0) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=6000, n_cand=3):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    k = min(n_cand, n_sample)
    idx = np.argpartition(scores, k)[:k]
    best_s = None; best_v = 1e100
    for kk in idx:
        s0 = Z[kk]
        s1 = _refine_single(s0, R, steps=180, T=0.005, lr=0.14)
        s1 = _refine_single(s1, R, steps=180, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v; best_s = s1
    best_s = best_s / (np.linalg.norm(best_s) + 1e-12)
    return best_s

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cv = G[iu]
        m = float(cv.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = cv > thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        csub = cv[sel]
        p = len(I)
        if p > 1600:
            od = np.argsort(-csub)[:1600]
            I = I[od]; J = J[od]; csub = csub[od]
            p = 1600
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 30.0
    np.random.seed(0)
    rng = np.random.default_rng(7)
    n, d = 71, 12
    best_X = None
    best_score = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            Xc = np.asarray(np.load(fn), dtype=np.float64)
            if Xc.shape == (71, 12) and np.all(np.isfinite(Xc)):
                Xc = _norm_rows(Xc)
                s = _max_cos(Xc)
                if s < best_score:
                    best_score = s
                    best_X = Xc.copy()
        except Exception:
            continue
    if best_X is None:
        best_X = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(best_X)
    else:
        best_X = _norm_rows(best_X)
    # elite archive
    archive = [(best_score, best_X.copy())]
    # seed archive with quick random Riesz quenches for diversity
    def push(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda z: z[0])
        # dedup very close
        uniq = []
        for vv, XX in archive:
            ok = True
            for v2, X2 in uniq:
                if abs(vv - v2) < 1e-12:
                    ok = False
                    break
            if ok:
                uniq.append((vv, XX))
        del archive[:]
        archive.extend(uniq[:8])
    # quick initial LP squeeze on incumbent if time
    try:
        if HOP_END - time.time() > 8:
            Xp = _lp_polish(best_X, seconds=2.0, deadline=HOP_END)
            sp = _max_cos(Xp)
            if sp < best_score:
                best_score = sp; best_X = _norm_rows(Xp)
                push(sp, best_X)
    except Exception:
        pass
    curX = best_X.copy(); curVal = best_score
    it = 0
    while time.time() < HOP_END:
        it += 1
        tleft = HOP_END - time.time()
        if tleft < 1.0:
            break
        # pick base: favour best but often elites / current
        r = rng.random()
        if r < 0.45:
            base = best_X
        elif r < 0.75 and len(archive) > 1:
            base = archive[int(rng.integers(0, len(archive)))][1]
        elif r < 0.9:
            base = curX
        else:
            base = archive[int(rng.integers(0, len(archive)))][1] if archive else best_X
        u = rng.random()
        try:
            if it % 41 == 0 or (u < 0.14 and tleft > 10):
                # fresh random start (different basin)
                X0 = _norm_rows(rng.standard_normal((n, d)))
                # light Riesz spread
                s_pow = float(rng.uniform(4, 10))
                X0 = _quench_riesz(X0, steps=300, s_pow=s_pow, lr=0.03)
            elif u < 0.30:
                # large global shake
                sigma = float(10 ** rng.uniform(-1.6, -0.45))
                X0 = base + rng.standard_normal((n, d)) * sigma
                X0 = _norm_rows(X0)
            elif u < 0.52:
                # kill k worst + hole refill (k up to 8, sometimes larger)
                if rng.random() < 0.15:
                    k_rep = int(rng.integers(5, 13))
                else:
                    k_rep = int(rng.integers(1, 6))
                    if rng.random() < 0.2:
                        k_rep = int(rng.integers(6, 10))
                k_rep = min(k_rep, 14)
                try:
                    rm, cnt, _ = _loads(base, band=0.006)
                    order = np.argsort(-rm - 0.003 * cnt + rng.random(n) * 0.002)
                    idx = order[:k_rep]
                except Exception:
                    idx = rng.choice(n, size=k_rep, replace=False)
                keep = np.ones(n, dtype=bool); keep[idx] = False
                R = base[keep]
                X0 = base.copy()
                # refill sequentially into largest holes
                for j in idx:
                    try:
                        h = _find_hole(R, rng, n_sample=5000, n_cand=3)
                    except Exception:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                sigma = float(10 ** rng.uniform(-2.8, -1.1))
                X0[keep] += rng.standard_normal((n - k_rep, d)) * sigma * 0.6
                X0 = _norm_rows(X0)
            elif u < 0.66:
                # cap-cluster kill: remove points inside random cap
                cu = _norm_rows(rng.standard_normal((1, d)))[0]
                proj = base @ cu
                # cap size random 4-14
                kcap = int(rng.integers(4, 15))
                idx = np.argsort(-proj)[:kcap]
                keep = np.ones(n, dtype=bool); keep[idx] = False
                R = base[keep]
                X0 = base.copy()
                for j in idx:
                    try:
                        h = _find_hole(R, rng, n_sample=4000, n_cand=2)
                    except Exception:
                        h = _norm_rows(rng.standard_normal((1, d)))[0]
                    X0[j] = h
                    R = np.concatenate([R, h[None, :]], axis=0)
                # jitter survivors near cap boundary more
                sigma = float(10 ** rng.uniform(-2.4, -1.0))
                X0[keep] += rng.standard_normal((n - kcap, d)) * sigma * 0.5
                X0 = _norm_rows(X0)
            elif u < 0.78:
                # crossover of two elites via Procrustes + greedy mix
                if len(archive) >= 2:
                    a = archive[0][1]
                    b = archive[int(rng.integers(1, len(archive)))][1]
                    try:
                        M = a.T @ b
                        U, _, Vt = np.linalg.svd(M)
                        Rot = U @ Vt
                        bA = b @ Rot.T
                    except Exception:
                        bA = b
                    # greedy build: start from random half of a, fill from bA far points
                    X0 = np.zeros_like(a)
                    perm = rng.permutation(n)
                    na = n // 2
                    X0[:na] = a[perm[:na]]
                    # for remaining, pick bA points farthest from chosen
                    chosen = X0[:na]
                    scores_b = []
                    for k in range(n):
                        sc = float((chosen @ bA[k]).max()) if len(chosen) else -1
                        scores_b.append(sc)
                    order_b = np.argsort(scores_b)
                    # take farthest
                    take = order_b[:n - na]
                    X0[na:] = bA[take]
                    # small jitter + random swap
                    X0 += rng.standard_normal((n, d)) * float(10 ** rng.uniform(-2.6, -1.2))
                    X0 = _norm_rows(X0)
                else:
                    sigma = float(10 ** rng.uniform(-2.5, -1.0))
                    X0 = base + rng.standard_normal((n, d)) * sigma
                    X0 = _norm_rows(X0)
            else:
                # targeted small shake weighted by load
                sigma = float(10 ** rng.uniform(-2.8, -0.9))
                try:
                    rm, cnt, _ = _loads(base, band=0.005)
                    w = (rm - rm.min() + 1e-6)
                    w = w / w.max()
                    sig = sigma * (0.35 + 1.7 * w)[:, None]
                    X0 = base + rng.standard_normal((n, d)) * sig
                except Exception:
                    X0 = base + rng.standard_normal((n, d)) * sigma
                X0 = _norm_rows(X0)
        except Exception:
            continue
        # choose anneal params
        T0 = float(10 ** rng.uniform(-2.5, -1.6))
        T1 = float(10 ** rng.uniform(-4.2, -3.1))
        lr0 = float(10 ** rng.uniform(-2.9, -2.0))
        lr1 = lr0 * float(10 ** rng.uniform(-1.2, -0.4))
        if tleft < 8:
            steps = 900
        elif tleft < 30:
            steps = 1400
        else:
            steps = 1900
            if rng.random() < 0.20:
                steps = 2800
        try:
            # optional Riesz pre-spread for fresh/cap moves
            if u < 0.70 and rng.random() < 0.35 and tleft > 12:
                X0 = _quench_riesz(X0, steps=220, s_pow=float(rng.uniform(5, 11)), lr=0.025)
            X1 = _quench_adam(X0, steps, T0, T1, lr0, lr1, alpha=0.6)
            s1 = _max_cos(X1)
        except Exception:
            continue
        if not np.isfinite(s1):
            continue
        # hard focus if promising
        if s1 < best_score + 0.003 and tleft > 4:
            try:
                Xh = _quench_hard(X1, steps=220, d0=0.006, d1=0.0006, lr=0.018)
                sh = _max_cos(Xh)
                if sh < s1:
                    X1, s1 = Xh, sh
            except Exception:
                pass
        # acceptance: always track archive, move current with Metropolis-ish
        if s1 < best_score - 1e-12:
            best_score = s1; best_X = _norm_rows(X1)
            push(s1, best_X)
            curX, curVal = best_X.copy(), s1
        else:
            # allow sideways move to different basin occasionally
            push(s1, X1)
            if s1 < curVal or rng.random() < 0.12:
                curX, curVal = _norm_rows(X1), s1
    # final polish: best few elites with longer low-T + hard + LP
    try:
        archive.sort(key=lambda z: z[0])
        # ensure best_X included
        cands = []
        seen = set()
        for v, X in archive[:4]:
            cands.append((v, X.copy()))
        # also current best
        if best_score not in [c[0] for c in cands]:
            cands.append((best_score, best_X.copy()))
        for ci in range(len(cands)):
            if time.time() > HARD_END - 4.0:
                break
            v, X = cands[ci]
            X = _norm_rows(X.copy())
            # low-T tightening
            try:
                Xt = _quench_adam(X, steps=1600, T0=0.0012, T1=0.00008, lr0=0.003, lr1=0.0004, alpha=0.55)
                st = _max_cos(Xt)
            except Exception:
                Xt, st = X, v
            try:
                Xh = _quench_hard(Xt, steps=300, d0=0.004, d1=0.0004, lr=0.012)
                sh = _max_cos(Xh)
                if sh < st:
                    Xt, st = Xh, sh
            except Exception:
                pass
            # LP squeeze
            try:
                secs = max(1.0, min(6.0, (HARD_END - 3.0 - time.time()) / max(1, len(cands) - ci)))
                if secs > 0.5:
                    Xp = _lp_polish(Xt, seconds=secs, deadline=HARD_END - 2.0)
                    sp = _max_cos(Xp)
                else:
                    Xp, sp = Xt, st
            except Exception:
                Xp, sp = Xt, st
            if sp < best_score and np.all(np.isfinite(Xp)):
                best_score = sp; best_X = _norm_rows(Xp)
        # ultimate LP on winner
        if time.time() < HARD_END - 2.0:
            try:
                Xp = _lp_polish(best_X, seconds=max(1.0, HARD_END - 2.0 - time.time()), deadline=HARD_END - 1.0)
                sp = _max_cos(Xp)
                if sp < best_score:
                    best_score = sp; best_X = _norm_rows(Xp)
            except Exception:
                pass
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
