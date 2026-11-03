import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=0.15, mu=0.85):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum() / max(n, 1) + 1e-30
        Grad_g = (E @ X) / Sg
        bl = beta * 0.25
        if bl < 8.0:
            bl = 8.0
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + one_mu * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _hard_polish(X, steps=200, lr0=0.0032, lr1=0.00012, tol0=3e-4, tol1=2e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _refine_hole(Y, p, iters=150, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / (np.linalg.norm(p) + 1e-30)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        mx = g.max()
        w = np.exp(beta * (g - mx))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _find_voids(Y, rng, n_probe=8000, n_top=4):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 2000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    cands = []
    for idx in order[:max(n_top * 6, 24)]:
        p = P[idx].copy()
        ok = True
        for q in cands:
            if abs(float(p @ q)) > 0.90:
                ok = False
                break
        if ok:
            cands.append(p)
        if len(cands) >= n_top:
            break
    if len(cands) == 0:
        cands = [P[order[0]]]
    out = []
    for p in cands:
        q = _refine_hole(Y, p, iters=150, beta=80.0 + rng.rand() * 40.0,
                         lr0=0.10 + rng.rand() * 0.06, lr1=0.003)
        out.append(q)
    # sort by quality
    Ym = Y
    scores = [float((Ym @ q).max()) for q in out]
    o = np.argsort(scores)
    return [out[i] for i in o]

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 3000
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_E8():
    vecs = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    for mask in range(256):
        cnt = 0
        s = np.ones(8)
        for k in range(8):
            if (mask >> k) & 1:
                s[k] = -1.0
                cnt += 1
        if cnt % 2 == 0:
            vecs.append(0.5 * s)
    E = np.array(vecs, dtype=np.float64) / np.sqrt(2.0)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return E

def _greedy_e8_subset(E, rng, k=168):
    N = E.shape[0]
    start = int(rng.randint(N))
    sel = [start]
    selmask = np.zeros(N, dtype=bool)
    selmask[start] = True
    # max dot to selected
    best_dot = E @ E[start]
    best_dot[selmask] = 1e9
    for _ in range(1, k):
        # pick min best_dot among unselected (+ tiny noise for diversity)
        cand = np.where(~selmask)[0]
        vals = best_dot[cand] + rng.randn(len(cand)) * 1e-6
        nxt = int(cand[np.argmin(vals)])
        sel.append(nxt)
        selmask[nxt] = True
        d = E @ E[nxt]
        # update
        upd = d < best_dot
        # but keep selected as inf
        best_dot = np.minimum(best_dot, d)
        best_dot[selmask] = 1e9
    return E[np.array(sel)]

def _align_perm(A, B, iters=2):
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    for _ in range(iters):
        Brot = B @ R
        S = A @ Brot.T
        _, col = linear_sum_assignment(-S)
        perm = col
        Bp = B[perm, :]
        M = Bp.T @ A
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    return R, perm

def solve():
    t0 = time.time()
    BUDGET = 175.0
    n, d = 168, 8
    rng = np.random.RandomState(12345)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    archive = [{'X': best.copy(), 'val': best_val}]
    E8 = _build_E8()
    # E8 diverse subsets
    for trial in range(3):
        if time.time() - t0 > 30:
            break
        try:
            S = _greedy_e8_subset(E8, rng, k=n)
        except Exception:
            S = E8[rng.choice(E8.shape[0], n, replace=False)]
        S = S + rng.randn(n, d) * 0.02
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        Sq = _balanced_quench(S, steps=450, b0=25.0, b1=350.0,
                              lr0=0.006, lr1=0.0005, lam=0.15)
        Sq = _hard_polish(Sq, steps=150)
        vq = _max_cos(Sq)
        if vq < best_val - 1e-12:
            best = Sq.copy()
            best_val = vq
            archive.append({'X': Sq.copy(), 'val': vq})
        elif vq < best_val + 4e-5:
            archive.append({'X': Sq.copy(), 'val': vq})
        archive = sorted(archive, key=lambda e: e['val'])[:12]
    # main basin hopping
    it = 0
    while time.time() - t0 < 132.0:
        it += 1
        # source
        if len(archive) > 1 and rng.rand() < 0.30:
            src = archive[rng.randint(len(archive))]['X']
        else:
            src = best
        u = rng.rand()
        X = None
        if u < 0.45:
            sigma = [0.04, 0.08, 0.15, 0.25, 0.40][rng.randint(5)]
            X = src + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.78:
            # void relocation of worst + rattlers
            X = src.copy()
            pm = _per_max(X)
            gm = pm.max()
            order = np.argsort(pm)  # ascending: rattlers first
            k = 5 + int(rng.randint(16))  # 5..20
            # mix worst and rattlers
            idx_worst = np.argsort(pm)[-max(2, k // 2):].tolist()
            idx_ratt = order[:max(2, k - len(idx_worst))].tolist()
            idx = list(dict.fromkeys(idx_worst + idx_ratt))[:k]
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[idx] = False
            Yrest = X[rest_mask]
            try:
                voids = _find_voids(Yrest, rng, n_probe=6000, n_top=len(idx))
            except Exception:
                voids = [rng.randn(d) / np.linalg.norm(rng.randn(d)) for _ in idx]
            for ii, j in enumerate(idx):
                v = voids[ii % len(voids)]
                if rng.rand() < 0.2:
                    v = v + rng.randn(d) * 0.05
                    v /= np.linalg.norm(v)
                X[j] = v
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            X = X + rng.randn(n, d) * 0.006
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.90 and len(archive) >= 2:
            # crossover
            a_idx = rng.randint(len(archive))
            b_idx = rng.randint(len(archive))
            if a_idx == b_idx:
                b_idx = (b_idx + 1) % len(archive)
            PA = archive[a_idx]['X']
            PB = archive[b_idx]['X']
            try:
                R, perm = _align_perm(PA, PB, iters=2)
            except Exception:
                R = np.eye(d)
                perm = np.arange(n)
            PBm = (PB @ R)[perm, :]
            proj_u = rng.randn(d)
            proj_u /= np.linalg.norm(proj_u)
            proj = PA @ proj_u
            mask = np.zeros(n, dtype=bool)
            if rng.rand() < 0.5:
                q = np.median(proj) + rng.randn() * 0.12
                mask = proj > q
                if mask.sum() < 20 or mask.sum() > n - 20:
                    mask = proj > np.median(proj)
            else:
                kk = 40 + int(rng.randint(89))
                order = np.argsort(proj)
                mask[order[-kk:]] = True
                if rng.rand() < 0.5:
                    mask = ~mask
            C = np.empty_like(PA)
            C[mask] = PA[mask]
            C[~mask] = PBm[~mask]
            # repair near duplicates
            G = C @ C.T
            np.fill_diagonal(G, -2.0)
            iu0 = np.triu_indices(n, 1)
            high = np.where(G[iu0] > 0.88)[0]
            dups = set()
            for hh in high[:20]:
                i = int(iu0[0][hh]); j = int(iu0[1][hh])
                dups.add(j)
            if len(dups) > 0:
                rest = np.delete(C, list(dups), axis=0)
                try:
                    voids = _find_voids(rest, rng, n_probe=5000, n_top=len(dups))
                except Exception:
                    voids = []
                for ii, j in enumerate(list(dups)):
                    if ii < len(voids):
                        C[j] = voids[ii]
                    else:
                        v = rng.randn(d); v /= np.linalg.norm(v)
                        C[j] = v
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            X = C
        else:
            X = rng.randn(n, d)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # quench params
        b0 = 15.0 + rng.rand() * 35.0
        b1 = 220.0 + rng.rand() * 380.0
        steps = 350 + int(rng.randint(401))
        lr0 = 0.004 + rng.rand() * 0.006
        lr1 = lr0 * (0.05 + rng.rand() * 0.10)
        lam = 0.05 + rng.rand() * 0.25
        try:
            Xq = _balanced_quench(X, steps, b0, b1, lr0, lr1, lam=lam)
            Xq = _hard_polish(Xq, steps=150, lr0=0.0030, lr1=0.00012)
        except Exception:
            continue
        if not np.all(np.isfinite(Xq)):
            continue
        vq = _max_cos(Xq)
        if vq < best_val - 1e-12:
            best = Xq.copy()
            best_val = vq
            archive.append({'X': Xq.copy(), 'val': vq})
            archive = sorted(archive, key=lambda e: e['val'])[:12]
        elif vq < best_val + 3e-5 and len(archive) < 12:
            # diversity: check far in value or just add
            archive.append({'X': Xq.copy(), 'val': vq})
            archive = sorted(archive, key=lambda e: e['val'])[:12]
        if time.time() - t0 > BUDGET - 30:
            break
    # final intensify around best
    try:
        Xf = _balanced_quench(best, steps=600, b0=30.0, b1=600.0,
                              lr0=0.005, lr1=0.0003, lam=0.18)
        Xf = _hard_polish(Xf, steps=250, lr0=0.0032, lr1=0.00010)
        vf = _max_cos(Xf)
        if vf < best_val:
            best, best_val = Xf, vf
    except Exception:
        pass
    remain = (t0 + BUDGET - 1.0) - time.time()
    if remain > 5:
        # use at most 25s for LP but leave 1s margin
        lp_time = min(remain, 28.0)
        try:
            Xp = _lp_polish(best, seconds=lp_time)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
