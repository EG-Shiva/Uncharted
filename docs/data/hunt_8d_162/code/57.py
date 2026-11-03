import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_gd(X, steps, b0, b1, lr0, lr1, n0, n1, rng, mu=0.84, topk=None):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        if b0 == b1:
            beta = b0
        else:
            beta = b0 * ((b1 / b0) ** f)
        if lr0 == lr1:
            lr = lr0
        else:
            lr = lr0 * ((lr1 / lr0) ** f)
        if n0 <= 0:
            nm = 0.0
        else:
            if n1 <= 0:
                nm = n0 * (1.0 - f)
            else:
                nm = n0 * ((n1 / n0) ** f) if n0 != n1 else n0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        if topk is None:
            W = np.exp(beta * (G - Gm))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum())
            if not np.finite(S) or S <= 0:
                break
            H = W @ X
            WG = (W * G).sum(axis=1)
            Grad = (H - WG[:, None] * X) / S
        else:
            iu0 = np.triu_indices(n, 1)
            cf = G[iu0]
            k = min(topk, cf.size)
            thr = float(np.partition(cf, -k)[-k])
            M = (G >= thr).astype(float)
            np.fill_diagonal(M, 0.0)
            S = float(M.sum())
            if S <= 0:
                break
            H = M @ X
            WG = (M * G).sum(axis=1)
            Grad = (H - WG[:, None] * X) / S
        V = mu * V + Grad
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N = N / nn * np.sqrt(d - 1)
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None, r0=0.004):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3500:
            idx = np.argpartition(-cfull[sel], 3500)[:3500]
            I = I[idx]; J = J[idx]
            c = cfull[sel][idx]
            p = 3500
        else:
            c = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _largest_holes(X_keep, k, rng, n_cand=16000, sep=0.55):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    order = np.argsort(m)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v) + 1e-12
        picked.append(v)
    return np.array(picked)

def _e8_roots():
    roots = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(8); v[i] = si; v[j] = sj
                    roots.append(v)
    for mask in range(256):
        v = np.empty(8)
        cnt = 0
        for kk in range(8):
            if (mask >> kk) & 1:
                v[kk] = -0.5; cnt += 1
            else:
                v[kk] = 0.5
        if cnt % 2 == 0:
            roots.append(v.copy())
    R = np.array(roots, dtype=float)
    R = R / np.linalg.norm(R[0])
    return R

def _e8_subset(rng):
    try:
        R = _e8_roots()
    except Exception:
        return None
    N = R.shape[0]
    idx = rng.choice(N, 162, replace=False)
    G = R @ R.T
    # quick swap to reduce number of 0.5 contacts
    cur_set = idx.copy()
    def bad_count(s):
        Gs = G[np.ix_(s, s)]
        np.fill_diagonal(Gs, -2.0)
        return int((Gs > 0.49).sum() // 2)
    try:
        cur_bad = bad_count(cur_set)
        in_set = set(cur_set.tolist())
        for _ in range(2500):
            o = int(rng.randint(0, 162))
            cand_pool = rng.choice(N, 8, replace=False)
            improved = False
            for cp in cand_pool:
                if int(cp) in in_set:
                    continue
                trial = cur_set.copy(); trial[o] = int(cp)
                b = bad_count(trial)
                if b < cur_bad:
                    in_set.remove(int(cur_set[o])); in_set.add(int(cp))
                    cur_set = trial; cur_bad = b; improved = True
                    break
            if cur_bad == 0:
                break
    except Exception:
        pass
    X = R[cur_set]
    return _norm(X)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(12345)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape
    elites = [(s_best, X_best.copy())]
    # precompute one E8 start to have diversity available
    e8_cache = None
    try:
        e8_cache = _e8_subset(np.random.RandomState(999))
    except Exception:
        e8_cache = None

    trial = 0
    # exploration
    while time.time() < t0 + 108 and trial < 400:
        if time.time() > deadline - 55:
            break
        trial += 1
        mode = trial % 8
        try:
            Xt = X_best.copy()
            if mode == 0:
                # full random restart
                Xt = _norm(rng.randn(n, d))
                Xt = _smooth_gd(Xt, 420, 25, 130, 0.014, 0.0015, 0.0032, 0.0, rng)
                Xt = _smooth_gd(Xt, 340, 120, 850, 0.006, 0.0006, 0.0, 0.0, rng)
            elif mode == 1:
                # large shake of best
                sig = 0.16 + 0.08 * rng.rand()
                Xt = _norm(Xt + rng.randn(n, d) * sig)
                Xt = _smooth_gd(Xt, 450, 25, 150, 0.014, 0.0013, 0.0028, 0.0, rng)
                Xt = _smooth_gd(Xt, 340, 120, 900, 0.006, 0.0006, 0.0, 0.0, rng)
            elif mode == 2:
                # E8-seeded or random elite big shake
                if e8_cache is not None and trial < 40 and (trial % 2 == 0):
                    Xt = _norm(e8_cache + rng.randn(n, d) * (0.015 + 0.02 * rng.rand()))
                    # shuffle order to avoid index bias
                    perm = rng.permutation(n); Xt = Xt[perm]
                else:
                    # pick random elite
                    ei = int(rng.randint(0, len(elites)))
                    Xt = elites[ei][1].copy()
                    Xt = _norm(Xt + rng.randn(n, d) * 0.30)
                Xt = _smooth_gd(Xt, 450, 22, 130, 0.014, 0.0013, 0.0030, 0.0, rng)
                Xt = _smooth_gd(Xt, 340, 130, 900, 0.006, 0.0006, 0.0, 0.0, rng)
            elif mode == 3:
                # overload deletion
                G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
                gm = float(G.max())
                cnt = (G > gm - 1e-4).sum(axis=1)
                cnt2 = (G > gm - 1e-9).sum(axis=1)
                k = 6 + int(rng.randint(0, 5))
                order = np.argsort(-(cnt + 0.5 * cnt2 + rng.rand(n) * 0.3))
                victims = order[:k]
                keep = np.ones(n, dtype=bool); keep[victims] = False
                holes = _largest_holes(Xt[keep], k, rng, n_cand=16000, sep=0.5)
                Xt[victims] = holes
                Xt = _norm(Xt + rng.randn(n, d) * 0.004)
                Xt = _smooth_gd(Xt, 450, 28, 160, 0.013, 0.0012, 0.0015, 0.0, rng)
                Xt = _smooth_gd(Xt, 320, 130, 900, 0.006, 0.0006, 0.0, 0.0, rng)
            elif mode == 4:
                # random subset deletion
                k = 8 + int(rng.randint(0, 7))
                victims = rng.choice(n, k, replace=False)
                keep = np.ones(n, dtype=bool); keep[victims] = False
                holes = _largest_holes(Xt[keep], k, rng, n_cand=15000, sep=0.5)
                Xt[victims] = holes
                Xt = _norm(Xt + rng.randn(n, d) * 0.005)
                Xt = _smooth_gd(Xt, 450, 25, 150, 0.013, 0.0012, 0.0020, 0.0, rng)
                Xt = _smooth_gd(Xt, 320, 130, 850, 0.006, 0.0006, 0.0, 0.0, rng)
            elif mode == 5:
                # different loss: top-K uniform + hard softmax
                Xt = _norm(Xt + rng.randn(n, d) * 0.03)
                Xt = _smooth_gd(Xt, 380, 30, 140, 0.013, 0.0014, 0.0022, 0.0, rng)
                Xt = _smooth_gd(Xt, 300, 0, 0, 0.004, 0.0008, 0.0, 0.0, rng, mu=0.80, topk=450)
                Xt = _smooth_gd(Xt, 260, 150, 900, 0.005, 0.0006, 0.0, 0.0, rng)
            elif mode == 6:
                # cap rotation shake
                u = rng.randn(d); u /= np.linalg.norm(u) + 1e-12
                proj = Xt @ u
                thr = np.quantile(proj, 0.80)
                sel = np.where(proj >= thr)[0]
                Rsk = rng.randn(d, d) * 0.09; Rsk = Rsk - Rsk.T
                for i in sel:
                    Xt[i] = Xt[i] + (Rsk @ Xt[i]) + rng.randn(d) * 0.010
                # also move worst point to hole
                G = _norm(Xt) @ _norm(Xt).T; np.fill_diagonal(G, -2.0)
                wrow = int(np.argmax(G.max(axis=1)))
                mask = np.ones(n, dtype=bool); mask[wrow] = False
                try:
                    Xt[wrow] = _largest_holes(_norm(Xt[mask]), 1, rng, n_cand=12000)[0]
                except Exception:
                    pass
                Xt = _norm(Xt)
                Xt = _smooth_gd(Xt, 450, 25, 150, 0.013, 0.0012, 0.0022, 0.0, rng)
                Xt = _smooth_gd(Xt, 320, 130, 900, 0.006, 0.0006, 0.0, 0.0, rng)
            else:
                # crossover
                if len(elites) >= 2:
                    a = elites[int(rng.randint(0, len(elites)))][1]
                    b = elites[int(rng.randint(0, len(elites)))][1]
                    msk = rng.rand(n) < 0.5
                    Xt = np.where(msk[:, None], a, b)
                    Xt = _norm(Xt + rng.randn(n, d) * 0.02)
                else:
                    Xt = _norm(Xt + rng.randn(n, d) * 0.08)
                Xt = _smooth_gd(Xt, 450, 25, 150, 0.013, 0.0012, 0.0024, 0.0, rng)
                Xt = _smooth_gd(Xt, 320, 130, 900, 0.006, 0.0006, 0.0, 0.0, rng)
        except Exception:
            continue
        try:
            s_try = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(s_try):
            continue
        if s_try < s_best + 0.0045:
            elites.append((s_try, Xt.copy()))
            if len(elites) > 30:
                elites = sorted(elites, key=lambda z: z[0])[:14]
        if s_try < s_best - 1e-12:
            # lock with short LP if time allows
            budget = 2.2
            if time.time() + budget < deadline - 8:
                try:
                    Xt2 = _lp_polish(Xt, budget, t_deadline=deadline - 5, r0=0.004)
                    s2 = _max_cos(Xt2)
                    if s2 < s_try:
                        Xt, s_try = Xt2, s2
                except Exception:
                    pass
            if s_try < s_best:
                X_best, s_best = Xt.copy(), float(s_try)
                elites.append((s_best, X_best.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:14]
    # phase 2: polish elites
    try:
        elites = sorted(elites, key=lambda z: z[0])[:7]
    except Exception:
        elites = [(s_best, X_best.copy())]
    for _, Xc in elites:
        if time.time() > deadline - 16:
            break
        remain = deadline - time.time() - 3
        if remain < 3:
            break
        try:
            Xc2 = np.array(Xc, copy=True)
            # extra hard anneal
            Xc2 = _smooth_gd(Xc2, 220, 200, 1000, 0.004, 0.0005, 0.0, 0.0, rng)
            s_c2 = _max_cos(Xc2)
            if s_c2 < _max_cos(Xc) - 1e-15:
                Xc = Xc2
        except Exception:
            pass
        budget = min(6.0, max(2.0, remain / 3.0))
        try:
            Xp = _lp_polish(np.array(Xc, copy=True), budget, t_deadline=deadline - 2, r0=0.003)
            sp = _max_cos(Xp)
            if sp < s_best - 1e-15:
                X_best, s_best = Xp.copy(), float(sp)
        except Exception:
            pass
    # final lock on best
    if time.time() < deadline - 3:
        budget = min(11.0, deadline - time.time() - 1.0)
        if budget > 2:
            try:
                Xf = _lp_polish(X_best, budget, t_deadline=deadline - 0.5, r0=0.0025)
                sf = _max_cos(Xf)
                if sf < s_best:
                    X_best, s_best = Xf, sf
            except Exception:
                pass
    X_best = _norm(np.array(X_best, dtype=float))
    return X_best
