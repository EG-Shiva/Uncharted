import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _repulse(X0, steps, a0, a1, s0, s1, mu=0.85):
    X = _norm(X0.copy())
    n = X.shape[0]
    vel = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bX = X.copy()
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bX = X.copy()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            X = X - (smax / nrm) * vel
            X = _norm(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return X, fv
    return bX, bv

def _repulse_hinge(X0, steps, a0, a1, s0, s1, pairs, target=0.885, hweight=4.0, mu=0.85):
    X = _norm(X0.copy())
    vel = np.zeros_like(X)
    pairs = [(int(a), int(b)) for a, b in pairs]
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        grad_s = W @ X
        dot_s = np.sum(grad_s * X, axis=1, keepdims=True)
        gt_s = grad_s - dot_s * X
        # hinge grad: push dot down to target: penalty max(0,dot-target)^2
        gh = np.zeros_like(X)
        for (i, j) in pairs:
            d = float(X[i] @ X[j])
            if d > target:
                c = 2.0 * (d - target)
                gh[i] += c * X[j]
                gh[j] += c * X[i]
        dot_h = np.sum(gh * X, axis=1, keepdims=True)
        gt_h = gh - dot_h * X
        ns = float(np.linalg.norm(gt_s, axis=1).max())
        nh = float(np.linalg.norm(gt_h, axis=1).max())
        if nh > 1e-14 and ns > 1e-14:
            gt = gt_s + hweight * (ns / (nh + 1e-12)) * gt_h * 0.7
        elif nh > 1e-14:
            gt = gt_h
        else:
            gt = gt_s
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            X = X - (smax / nrm) * vel
            X = _norm(X)
    return _norm(X)

def _lp_polish(x, deadline, cap=5000):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _lp_duals(x, radius=1e-3, cap=3500):
    n, d = x.shape
    G = x @ x.T
    iu0 = np.triu_indices(n, 1)
    c_all = G[iu0]
    m = float(c_all.max())
    sel = c_all > m - max(20.0 * radius, 5e-3)
    cnt = int(np.count_nonzero(sel))
    if cnt == 0:
        return None, None, None, m
    if cnt > cap:
        idx = np.where(sel)[0]
        order = np.argpartition(-c_all[idx], cap - 1)[:cap]
        keep = np.zeros_like(sel, dtype=bool)
        keep[idx[order]] = True
        sel = keep
    c = c_all[sel]
    I = iu0[0][sel]
    J = iu0[1][sel]
    p = len(I)
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
        return None, None, None, m
    obj = np.zeros(nv)
    obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None, None, None, m
    if res.status != 0:
        return None, None, None, m
    dual = None
    try:
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            mg = None
            try:
                mg = np.asarray(res.ineqlin.marginals, dtype=np.float64).ravel()
            except Exception:
                try:
                    mg = np.asarray(res.ineqlin["marginals"], dtype=np.float64).ravel()
                except Exception:
                    mg = None
            if mg is not None and mg.size == p and np.all(np.isfinite(mg)):
                dual = np.abs(mg)
    except Exception:
        dual = None
    if dual is None:
        try:
            # fallback: exponential weighting by closeness to max
            dual = np.exp(250.0 * (c - m))
        except Exception:
            dual = None
    return dual, I, J, m

def _contact_set(X, slack=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    thr = m - slack
    iu0 = np.triu_indices(X.shape[0], 1)
    mask = G[iu0] >= thr
    ii = iu0[0][mask]
    jj = iu0[1][mask]
    n = X.shape[0]
    s = set((int(a) * n + int(b)) for a, b in zip(ii, jj))
    return s, m

def _jaccard(A, B):
    if len(A) == 0 and len(B) == 0:
        return 1.0
    inter = len(A & B)
    union = len(A | B)
    if union == 0:
        return 1.0
    return inter / union

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    SURG_END = t0 + 148.0
    n, d = 272, 4
    rng = np.random.RandomState(9168723)
    best_X = None
    best_val = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        a = rng.randn(n, d)
        best_X = _norm(a)
        best_val = _max_cos(best_X)
    init_best_val = float(best_val)
    # contact archive
    try:
        cs0, _ = _contact_set(best_X, slack=1e-6)
    except Exception:
        cs0 = set()
    archive = [(best_X.copy(), float(best_val), set(cs0))]
    archive_sets = [set(cs0)]
    # surgery loop from diverse basins
    it = 0
    while time.time() < SURG_END:
        it += 1
        # pick base: 50% incumbent best, else random archive entry
        if len(archive) == 1 or rng.rand() < 0.5:
            base = best_X.copy()
        else:
            pick = rng.randint(len(archive))
            base = archive[pick][0].copy()
        # small jitter to avoid exact repeat
        # get duals
        try:
            dual, I, J, m0 = _lp_duals(base, radius=1e-3, cap=3500)
        except Exception:
            dual, I, J, m0 = None, None, None, _max_cos(base)
        if dual is None or I is None:
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                iu0 = np.triu_indices(n, 1)
                c_all = G[iu0]
                m0 = float(c_all.max())
                order = np.argsort(-c_all)[:40]
                I = iu0[0][order]
                J = iu0[1][order]
                dual = (-c_all[order] + m0 + 1e-12)
                dual = np.exp(250.0 * (c_all[order] - m0)) + 1e-12
            except Exception:
                # random pairs fallback
                I = rng.randint(0, n, size=30)
                J = rng.randint(0, n, size=30)
                dual = np.ones(30)
        # point stress
        try:
            pstress = np.zeros(n)
            np.add.at(pstress, I, dual)
            np.add.at(pstress, J, dual)
        except Exception:
            pstress = np.zeros(n)
        k = int(rng.randint(5, 16))  # 5..15
        target = 0.885
        # choose edges: top points + top duals
        try:
            order_dual = np.argsort(-dual)
            top_pts = np.argsort(-pstress)[:3]
            in_top = np.isin(I, top_pts) | np.isin(J, top_pts)
            idx_top = np.where(in_top)[0]
            # sort those by dual
            if len(idx_top) > 0:
                idx_top = idx_top[np.argsort(-dual[idx_top])]
            chosen = []
            for idx in list(idx_top[:k * 2]) + list(order_dual[:k * 2]):
                a = int(I[int(idx)]); b = int(J[int(idx)])
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a, b) not in chosen:
                    chosen.append((a, b))
                if len(chosen) >= k:
                    break
            if len(chosen) < k:
                # fill random high pairs
                G2 = base @ base.T
                np.fill_diagonal(G2, -2.0)
                for _ in range(100):
                    if len(chosen) >= k:
                        break
                    a = int(rng.randint(0, n)); b = int(rng.randint(0, n))
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) not in chosen and G2[a, b] > m0 - 0.03:
                        chosen.append((a, b))
            pairs = chosen[:k]
        except Exception:
            pairs = [(int(I[i]), int(J[i])) for i in range(min(k, len(I)))]
        hweight = float(rng.uniform(3.0, 7.0))
        # forced topology change: hinge + softmax, allow rise to ~0.905
        try:
            Xf = _repulse_hinge(base, steps=380, a0=90.0, a1=420.0, s0=0.011, s1=0.0025,
                                pairs=pairs, target=target, hweight=hweight, mu=0.82)
        except Exception:
            continue
        try:
            vf = _max_cos(Xf)
        except Exception:
            continue
        # if forced move did nothing (pairs still at max), increase push: second shove
        # re-anneal at moderate temperature (softmax only, best-tracked)
        try:
            Xa, va = _repulse(Xf, steps=520, a0=70.0, a1=520.0, s0=0.011, s1=0.0016, mu=0.85)
        except Exception:
            continue
        # short LP polish
        remain = SURG_END - time.time()
        if remain > 7:
            try:
                Xp = _lp_polish(Xa, deadline=time.time() + min(5.5, remain - 1.0), cap=4500)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xa, va
        else:
            Xp, vp = Xa, va
        # archive handling with Jaccard diversity at slack 1e-6
        try:
            cs, mm = _contact_set(Xp, slack=1e-6)
        except Exception:
            cs = set()
        diverse = True
        min_sim = 1.0
        try:
            sims = [_jaccard(cs, s) for s in archive_sets]
            min_sim = min(sims) if sims else 1.0
            # Jaccard distance >0.15  <=> similarity <0.85
            diverse = (min_sim < 0.85)
        except Exception:
            diverse = True
        if vp < best_val:
            best_val = float(vp)
            best_X = Xp.copy()
            archive.append((Xp.copy(), float(vp), set(cs)))
            archive_sets.append(set(cs))
            # cap archive size
            if len(archive) > 12:
                # keep best + most recent diverse; drop middle worst with high similarity to best
                # simple: keep best, drop worst non-best if too many
                vals = [a[1] for a in archive]
                # never drop best (index of min)
                bi = int(np.argmin(vals))
                # drop worst
                wi = int(np.argmax(vals))
                if wi != bi:
                    archive.pop(wi); archive_sets.pop(wi)
        elif diverse and vp < best_val + 2e-4 and vp < init_best_val + 5e-4:
            # accept worse but diverse basin
            archive.append((Xp.copy(), float(vp), set(cs)))
            archive_sets.append(set(cs))
            if len(archive) > 12:
                vals = [a[1] for a in archive]
                bi = int(np.argmin(vals))
                wi = int(np.argmax(vals))
                if wi != bi:
                    # don't drop the just-added if it is worst but diverse? drop oldest non-best
                    # drop worst among non-best excluding last if last is worst
                    if wi == len(archive) - 1:
                        # drop second worst
                        vals2 = list(vals); vals2[wi] = -1
                        wi2 = int(np.argmax(vals2))
                        if wi2 != bi:
                            archive.pop(wi2); archive_sets.pop(wi2)
                    else:
                        archive.pop(wi); archive_sets.pop(wi)
        # occasional random kick from best to avoid stall (basin-hop)
        if it % 5 == 0 and time.time() < SURG_END - 8:
            try:
                sig = 0.02
                Xt = best_X + rng.randn(n, d) * sig
                Xt = _norm(Xt)
                Xo, vo = _repulse(Xt, steps=300, a0=80.0, a1=500.0, s0=0.010, s1=0.0015)
                if vo < best_val:
                    best_val = float(vo)
                    best_X = Xo.copy()
                    cs2, _ = _contact_set(Xo, slack=1e-6)
                    archive.append((Xo.copy(), float(vo), set(cs2)))
                    archive_sets.append(set(cs2))
            except Exception:
                pass
    # final LP polish on best with remaining time
    try:
        remain = HARD_END - time.time()
        if remain > 4:
            Xf2 = _lp_polish(best_X, deadline=HARD_END - 1.0, cap=6000)
            vf2 = _max_cos(Xf2)
            if vf2 < best_val and np.all(np.isfinite(Xf2)):
                best_val = vf2
                best_X = Xf2.copy()
    except Exception:
        pass
    return _norm(best_X)
