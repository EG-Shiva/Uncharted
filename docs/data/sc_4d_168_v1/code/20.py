import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 3000:
            # keep tightest only to keep LP fast
            order = np.argsort(-c[sel])[:3000]
            I, J = I[order], J[order]
            p = len(I)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p == np.sum(sel) else (x[I]*x[J]).sum(1)*-1.0 + 0.0,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        # fix b_ub: recompute correctly for truncated case
        # (above trick may be wrong) -> recompute simply
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _lp_polish_clean(x, deadline):
    # clean version without truncation bug: recompute b correctly
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 3500:
            idx = np.argsort(-cfull[sel])[:3500]
            I, J = I[idx], J[idx]
            p = len(I)
        csel = np.sum(x[I] * x[J], axis=1)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 141.0
    rng = np.random.RandomState(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and getattr(c, "shape", None) == (168, 4):
            c = np.asarray(c, float)
            if np.all(np.isfinite(c)):
                nrm = np.linalg.norm(c, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                c = c / nrm
                cands.append(c)
    if len(cands) == 0:
        x_best = rng.randn(168, 4)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        vals = [_max_cos(c) for c in cands]
        x_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(x_best)
    n, d = 168, 4

    def refine_lse(x_init, steps, beta0=30.0, beta1=400.0, lr0=0.012, lr1=0.0008, mu=0.9):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s > 200 and time.time() >= SEARCH_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def squeeze(x, steps=220, s0=0.003, s1=0.00015):
        # true-max subgradient on sphere
        for s in range(steps):
            frac = s / max(1, steps - 1)
            step = s0 * (s1 / s0) ** frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            Mask = G > (m - max(12.0 * step, 2e-4))
            cnt = np.sum(Mask, axis=1, keepdims=True) + 1e-12
            Grad = (Mask.astype(float) @ x) / cnt
            dd = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dd * x
            # normalize step by typical grad norm to keep stable
            gn = np.linalg.norm(Gt) + 1e-12
            if gn > 1e-12:
                Gt = Gt * (min(gn, 5.0) / gn)
            y = x - step * Gt
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # accept if not worse by large margin (allow small uphill to slide)
            v_old = m
            Gy = y @ y.T
            np.fill_diagonal(Gy, -2.0)
            v_new = float(np.max(Gy))
            if v_new <= v_old + 1e-6:
                x = y
            else:
                # shrink step locally by skipping
                pass
            if time.time() >= SEARCH_END and s > 80:
                break
        return x

    def hole_opt(X_keep, h_init, steps=180):
        # minimize max(X_keep @ h) over sphere: projected subgradient + smoothing
        h = h_init / (np.linalg.norm(h_init) + 1e-30)
        for s in range(steps):
            frac = s / max(1, steps - 1)
            lr = 0.05 * (0.002 / 0.05) ** frac
            dots = X_keep @ h
            m = float(np.max(dots))
            # softmax weights for top to smooth
            tau = 0.004 * (0.0005 / 0.004) ** frac
            w = np.exp(np.clip((dots - m) / tau, -20, 0))
            w = w / (np.sum(w) + 1e-30)
            g = w @ X_keep  # 4-vector
            # tangent component
            gt = g - np.dot(g, h) * h
            h = h - lr * gt
            h /= (np.linalg.norm(h) + 1e-30)
        return h

    def best_hole(X_keep, n_sample=9000, refine_top=3):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        m = D.max(axis=1)
        order = np.argsort(m)
        best = S[order[0]]
        # refine top few continuously, pick best
        bv = float(np.max(X_keep @ best))
        for k in range(min(refine_top, len(order))):
            h0 = S[order[k]]
            h1 = hole_opt(X_keep, h0, steps=150)
            v = float(np.max(X_keep @ h1))
            if v < bv:
                bv = v
                best = h1
        return best

    trial = 0
    lp_snaps_used = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        x = x_best.copy()
        typ = trial % 8
        if typ == 0:
            x = x + 0.018 * rng.randn(n, d)
            x = _norm(x)
            steps = 650
            kw = dict(beta0=30., beta1=350., lr0=0.012, lr1=0.0008)
        elif typ == 1:
            x = x + 0.065 * rng.randn(n, d)
            x = _norm(x)
            steps = 950
            kw = dict(beta0=25., beta1=420., lr0=0.012, lr1=0.0007)
        elif typ == 2:
            # jammed cluster shake
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.008, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 12) % n]
            nbr = np.argsort(-G[i0], kind="stable")[:9]
            x[nbr] += 0.16 * rng.randn(len(nbr), d)
            x[i0] += 0.16 * rng.randn(d)
            x = _norm(x)
            steps = 950
            kw = dict(beta0=28., beta1=450., lr0=0.013, lr1=0.0006)
        elif typ == 3:
            # overload delete + optimized hole reinsert (main topology move)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.010, axis=1)
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 10) % n]
            kdel = 4 + rng.randint(0, 5)  # 4..8
            nbrs = np.argsort(-G[i0], kind="stable")[:kdel - 1]
            cluster = np.concatenate([[i0], nbrs])
            keep = np.ones(n, dtype=bool)
            keep[cluster] = False
            cur = x[keep]
            new_pts = []
            for _ in range(len(cluster)):
                h = best_hole(cur, n_sample=9000, refine_top=2)
                new_pts.append(h)
                cur = np.vstack([cur, h[None, :]])
            x[cluster] = np.array(new_pts)
            # small jitter of neighbours of new points to break symmetry
            x = x + 0.008 * rng.randn(n, d)
            x = _norm(x)
            steps = 1150
            kw = dict(beta0=22., beta1=480., lr0=0.014, lr1=0.0006)
        elif typ == 4:
            # rattler / loose teleport to optimized hole
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 1e-4, axis=1)
            loose = np.where(deg <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(deg)[:4]
            li = loose[rng.randint(0, len(loose))]
            X_keep = np.delete(x, li, axis=0)
            h = best_hole(X_keep, n_sample=12000, refine_top=3)
            x[li] = h
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            nbr = np.argsort(-G2[li], kind="stable")[:5]
            x[nbr] = x[nbr] + 0.05 * rng.randn(len(nbr), d)
            x = _norm(x)
            steps = 900
            kw = dict(beta0=28., beta1=450., lr0=0.012, lr1=0.0006)
        elif typ == 5:
            # random subset resample + jitter
            k = 5 + rng.randint(0, 4)
            idx = rng.choice(n, size=k, replace=False)
            # resample half via holes, half pure random
            for j, li in enumerate(idx):
                if j % 2 == 0:
                    X_keep = np.delete(x, li, axis=0)
                    x[li] = best_hole(X_keep, n_sample=6000, refine_top=1)
                else:
                    v = rng.randn(d)
                    v /= np.linalg.norm(v)
                    x[li] = v
            x = x + 0.010 * rng.randn(n, d)
            x = _norm(x)
            steps = 1250
            kw = dict(beta0=20., beta1=480., lr0=0.014, lr1=0.0006)
        elif typ == 6:
            # worst-pair split + medium shake
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            ij = np.argmax(G)
            a, b = ij // n, ij % n
            # push apart along tangent
            mid = x[a] + x[b]
            mid /= (np.linalg.norm(mid) + 1e-30)
            r1 = rng.randn(d); r1 -= np.dot(r1, x[a]) * x[a]; r1 /= (np.linalg.norm(r1) + 1e-30)
            r2 = rng.randn(d); r2 -= np.dot(r2, x[b]) * x[b]; r2 /= (np.linalg.norm(r2) + 1e-30)
            x[a] = _norm((x[a] + 0.10 * r1)[None, :])[0]
            x[b] = _norm((x[b] + 0.10 * r2)[None, :])[0]
            x = x + 0.035 * rng.randn(n, d)
            x = _norm(x)
            steps = 850
            kw = dict(beta0=26., beta1=420., lr0=0.012, lr1=0.0007)
        else:
            # occasional fresh restart, else small shake
            if trial % 24 == 7:
                x = rng.randn(n, d)
                x = _norm(x)
                steps = 1700
                kw = dict(beta0=18., beta1=500., lr0=0.014, lr1=0.0006)
            else:
                x = x + 0.040 * rng.randn(n, d)
                x = _norm(x)
                steps = 800
                kw = dict(beta0=30., beta1=400., lr0=0.012, lr1=0.0008)

        x = refine_lse(x, steps, **kw)
        # short true-max squeeze to get near LP value cheaply
        x = squeeze(x, steps=180)
        v = _max_cos(x)
        if v < best_val - 5e-13:
            # verify with quick LP snap if promising and time allows
            if v < best_val - 1e-7 and lp_snaps_used < 12 and time.time() < SEARCH_END - 8:
                try:
                    xs = _lp_polish_clean(x, min(time.time() + 1.2, SEARCH_END))
                    vs = _max_cos(xs)
                    lp_snaps_used += 1
                    if vs <= v + 1e-9:
                        x, v = xs, vs
                except Exception:
                    pass
            if v < best_val - 5e-13:
                best_val = v
                x_best = x.copy()
        elif v < best_val + 3e-4 and lp_snaps_used < 12 and time.time() < SEARCH_END - 10:
            # basin might snap lower than current best
            if rng.rand() < 0.15:
                try:
                    xs = _lp_polish_clean(x, time.time() + 1.0)
                    vs = _max_cos(xs)
                    lp_snaps_used += 1
                    if vs < best_val - 5e-13:
                        best_val = vs
                        x_best = xs.copy()
                except Exception:
                    pass

    # final tightening: high-beta anneal + squeeze + full LP
    try:
        if time.time() < HARD_END - 4:
            x_best = refine_lse(x_best, 350, beta0=150.0, beta1=800.0, lr0=0.004, lr1=0.0003)
            x_best = squeeze(x_best, steps=250, s0=0.002, s1=0.0001)
            v = _max_cos(x_best)
            if v < best_val:
                best_val = v
    except Exception:
        pass
    # try a few rattler-teleport variants of best and keep best after LP
    try:
        cand_best = x_best.copy()
        cand_val = _max_cos(cand_best)
        # generate 2 variants if time allows
        for _ in range(2):
            if time.time() > HARD_END - 22:
                break
            G = cand_best @ cand_best.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 1e-9, axis=1)
            ridx = np.where(deg == 0)[0]
            if len(ridx) == 0:
                ridx = np.argsort(deg)[:1]
            li = ridx[0]
            X_keep = np.delete(cand_best, li, axis=0)
            h = best_hole(X_keep, n_sample=15000, refine_top=3)
            trial_x = cand_best.copy()
            trial_x[li] = h
            trial_x = refine_lse(trial_x, 400, beta0=60., beta1=600., lr0=0.006, lr1=0.0004)
            trial_x = squeeze(trial_x, steps=150)
            tv = _max_cos(trial_x)
            if tv < cand_val:
                cand_val = tv
                cand_best = trial_x.copy()
        x_best, best_val = cand_best, cand_val
    except Exception:
        pass
    remain = min(HARD_END - 1.5, time.time() + 24.0)
    if remain > time.time() + 1.0:
        try:
            x_lp = _lp_polish_clean(x_best, remain)
            if _max_cos(x_lp) < _max_cos(x_best) - 1e-14:
                x_best = x_lp
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
