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

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = int(np.sum(sel))
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq,
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
    return x

def _unconstrained_refine(x_init, steps, rng, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007, mu=0.9):
    x = x_init.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    SEARCH_END = t0 + 138.0
    SYM_END = t0 + 70.0
    BREAK_END = t0 + 132.0
    rng = np.random.RandomState(0)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
        xb = _norm(xb)
    except Exception:
        xb = _norm(rng.randn(n, d))
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (n, d):
        xb = _norm(rng.randn(n, d))
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # ---- symmetric subspace search ----
    # cyclic Hopf group: block rotations by p,q
    Ls = [42, 28, 24, 21, 14, 12, 56, 84, 8, 7, 6]
    cand_sym = []  # (val, X)

    def make_twists(L):
        qs = []
        for q in [3, 5, 7, 11, 13]:
            if q >= L:
                continue
            if q % L == 1 or q % L == L - 1:
                continue
            qs.append(q)
        if not qs:
            qs = [1]
        # keep at most 2
        return qs[:2]

    def expand_B(B, C1, S1, C2, S2):
        k = B.shape[0]
        L = C1.shape[0]
        # B: k x 4, output k*L x 4
        X = np.empty((k * L, 4))
        b0 = B[:, 0][:, None]; b1 = B[:, 1][:, None]
        b2 = B[:, 2][:, None]; b3 = B[:, 3][:, None]
        # for each i compute rotated copies for all j vectorized
        for i in range(L):
            c1, s1, c2, s2 = C1[i], S1[i], C2[i], S2[i]
            r0 = c1 * b0 - s1 * b1
            r1 = s1 * b0 + c1 * b1
            r2 = c2 * b2 - s2 * b3
            r3 = s2 * b2 + c2 * b3
            # place: j*L + i? use block per j: index j*L+i
            # to allow contiguous per j we fill strided
            X[0 + i::L if False else 0:0] = 0  # noop
            for j in range(k):
                X[j * L + i, 0] = r0[j, 0]
                X[j * L + i, 1] = r1[j, 0]
                X[j * L + i, 2] = r2[j, 0]
                X[j * L + i, 3] = r3[j, 0]
        return X

    # faster vectorized expand using broadcasting
    def expand_B_fast(B, C1, S1, C2, S2):
        k = B.shape[0]
        L = C1.shape[0]
        b0 = B[:, 0:1, None]  # k x1 x1
        # Use broadcasting: B k x4 -> repeat
        B0 = B[:, 0][:, None]  # k x1
        B1 = B[:, 1][:, None]
        B2 = B[:, 2][:, None]
        B3 = B[:, 3][:, None]
        # C: L
        # r0[k,L] = C1[L]*B0[k] - S1[L]*B1[k]
        r0 = B0 * C1[None, :] - B1 * S1[None, :]
        r1 = B0 * S1[None, :] + B1 * C1[None, :]
        r2 = B2 * C2[None, :] - B3 * S2[None, :]
        r3 = B2 * S2[None, :] + B3 * C2[None, :]
        X = np.empty((k * L, 4))
        # ordering j*L+i
        X[:, 0] = r0.reshape(-1, order='C').ravel() if False else r0.T.ravel() if False else 0
        # r0 is k x L, we want row-major j*L+i => r0.ravel() with C order
        X[:, 0] = r0.ravel()
        X[:, 1] = r1.ravel()
        X[:, 2] = r2.ravel()
        X[:, 3] = r3.ravel()
        return X

    def sym_refine(B0, C1, S1, C2, S2, steps, beta0=18.0, beta1=350.0, lr0=0.03, lr1=0.002, mu=0.85):
        B = _norm(B0.copy())
        k = B.shape[0]; L = C1.shape[0]
        vel = np.zeros_like(B)
        for s in range(steps):
            if time.time() > SYM_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            X = expand_B_fast(B, C1, S1, C2, S2)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            gradX = (E @ X) / rs
            dotX = np.sum(gradX * X, axis=1, keepdims=True)
            gtX = gradX - dotX * X  # N x4
            # pullback: reshape to k x L x 4
            gt = gtX.reshape(k, L, 4)
            g0 = gt[:, :, 0]; g1 = gt[:, :, 1]; g2 = gt[:, :, 2]; g3 = gt[:, :, 3]
            # R^T application
            # out0 = c1*g0 + s1*g1 ; out1 = -s1*g0 + c1*g1
            o0 = np.sum(C1[None, :] * g0 + S1[None, :] * g1, axis=1) / L
            o1 = np.sum(-S1[None, :] * g0 + C1[None, :] * g1, axis=1) / L
            o2 = np.sum(C2[None, :] * g2 + S2[None, :] * g3, axis=1) / L
            o3 = np.sum(-S2[None, :] * g2 + C2[None, :] * g3, axis=1) / L
            gB = np.stack([o0, o1, o2, o3], axis=1)
            dotB = np.sum(gB * B, axis=1, keepdims=True)
            gtB = gB - dotB * B
            # stabilize scale
            sc = np.mean(np.linalg.norm(gtB, axis=1)) + 1e-12
            if sc > 1e-9:
                gtB = gtB / sc * 0.5
            vel = mu * vel + (1.0 - mu) * gtB
            B = B - lr * vel
            B = _norm(B)
        Xf = expand_B_fast(B, C1, S1, C2, S2)
        return Xf, B

    # run symmetric trials
    for L in Ls:
        if time.time() > SYM_END - 2.0:
            break
        if 168 % L != 0:
            continue
        k = 168 // L
        for q in make_twists(L):
            if time.time() > SYM_END - 1.0:
                break
            p = 1
            th = 2.0 * np.pi * np.arange(L) / L
            C1 = np.cos(p * th); S1 = np.sin(p * th)
            C2 = np.cos(q * th); S2 = np.sin(q * th)
            for rep in range(2):
                if time.time() > SYM_END - 0.5:
                    break
                B0 = _norm(rng.randn(k, 4))
                steps = 500 if k >= 8 else 700
                try:
                    Xs, _ = sym_refine(B0, C1, S1, C2, S2, steps=steps)
                except Exception:
                    continue
                v = _max_cos(Xs)
                cand_sym.append((v, Xs.copy()))
                if v < best_val:
                    # keep but will break symmetry later; don't update best yet with symmetric (still valid)
                    best_val = v
                    x_best = Xs.copy()
            # also seeded from best projected? try one projection init: fit B to current x_best
            try:
                if time.time() > SYM_END - 0.5:
                    break
                # random B init already covers; do one k-means-ish init from x_best subset
                idx = rng.choice(n, size=k, replace=False)
                B0 = xb[idx].copy()
                Xs, _ = sym_refine(B0, C1, S1, C2, S2, steps=350)
                v = _max_cos(Xs)
                cand_sym.append((v, Xs.copy()))
                if v < best_val:
                    best_val = v
                    x_best = Xs.copy()
            except Exception:
                pass

    # sort symmetric candidates
    cand_sym.sort(key=lambda t: t[0])
    top_sym = [x for _, x in cand_sym[:6]]

    # ---- break symmetry: unconstrained descent of top symmetric + kicks from best ----
    pool = []
    for x in top_sym:
        pool.append(x)
    pool.append(xb.copy())
    # also add antipodal symmetrized best as extra diversity
    try:
        # antipodalize: keep 84 + neg
        G = xb @ xb.T
        np.fill_diagonal(G, -2.0)
        # greedy pick 84 far set? just take first 84 and mirror
        xa = np.vstack([xb[:84], -xb[:84]])
        xa = _norm(xa + 0.005 * rng.randn(168, 4))
        pool.append(xa)
    except Exception:
        pass

    # evaluate broken versions
    best_broken = x_best.copy()
    # reload true best among pool after unconstrained refine
    # first re-establish baseline from file (sym may have set worse? actually sym val likely worse than file, but we updated best_val only if better)
    # ensure best_val tracks x_best
    best_val = _max_cos(x_best)
    # if sym never beat file, x_best is still near file or file; fine
    for xi in pool:
        if time.time() > BREAK_END - 5.0:
            break
        try:
            y = _unconstrained_refine(xi, steps=600, rng=rng)
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                x_best = y.copy()
                best_broken = y.copy()
            # second stage with stronger anneal
            if v < _max_cos(xi) - 0.002 or v < 0.862:
                y2 = _unconstrained_refine(y, steps=500, rng=rng, beta0=40.0, beta1=800.0, lr0=0.008, lr1=0.0004)
                v2 = _max_cos(y2)
                if v2 < best_val:
                    best_val = v2
                    x_best = y2.copy()
        except Exception:
            continue

    # basin hopping around current best with small/large kicks (ensure progress + diversity)
    it = 0
    while time.time() < BREAK_END:
        it += 1
        if time.time() > BREAK_END - 2.0:
            break
        base = x_best.copy()
        # alternate kick strengths
        if it % 3 == 0:
            sig = 0.03
            raw = base + sig * rng.randn(n, d)
            raw = _norm(raw)
            steps = 500
        elif it % 3 == 1:
            sig = 0.10
            raw = base + sig * rng.randn(n, d)
            raw = _norm(raw)
            steps = 650
        else:
            # cluster ejection: remove worst cluster, reinsert at voids
            try:
                idx = rng.randint(0, n)
                dots = base @ base[idx]
                kk = int(rng.randint(8, 16))
                order = np.argsort(-dots)[:kk]
                mask = np.ones(n, dtype=bool); mask[order] = False
                keep = base[mask]
                # find voids
                S = _norm(rng.randn(12000, 4))
                D = S @ keep.T
                worst = D.max(axis=1)
                pick = np.argpartition(worst, kk)[:kk]
                newp = S[pick]
                # local repel new points among themselves lightly
                raw = np.vstack([keep, newp])
                perm = rng.permutation(n)
                raw = raw[perm]
                raw = _norm(raw + 0.008 * rng.randn(n, d))
                steps = 600
            except Exception:
                raw = _norm(base + 0.06 * rng.randn(n, d))
                steps = 500
        try:
            y = _unconstrained_refine(raw, steps=steps, rng=rng)
        except Exception:
            continue
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            x_best = y.copy()

    # final LP squeeze on best
    try:
        x_best = _lp_polish(x_best, deadline=t0 + 170.0)
        best_val = _max_cos(x_best)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
