import time
import collections
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _fingerprint(X, k=28, scale=1e6):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    kk = min(k, c.size)
    idx = np.argpartition(-c, kk - 1)[:kk]
    top = np.sort(c[idx])
    return tuple((np.round(top * scale)).astype(int).tolist())

def _subset_quench(X_init, movable, steps, T0, T1, lr, mu, rng=None, noise=0.0):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    mov = np.asarray(movable, dtype=int)
    is_frozen = np.ones(n, dtype=bool)
    is_frozen[mov] = False
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
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
        if len(mov) < n:
            Gt[is_frozen] = 0.0
            V[is_frozen] = 0.0
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            if len(mov) < n:
                N[is_frozen] = 0.0
            Gt = Gt + N
        V = mu * V + Gt
        X[mov] = X[mov] - lr * V[mov]
        X[mov] = _norm_rows(X[mov])
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _full_quench_track(X_init, steps, T0, T1, lr, mu=0.9):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
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
        if (s % 120 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _lp_polish_wide(x, seconds, deadline=None, r0=3e-3):
    x = _norm_rows(np.array(x, dtype=float, copy=True))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        band = max(30.0 * radius, 3e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2000:
            order = np.argsort(-cmat[sel])[:2000]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 2000
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _single_refine(s, R, steps=180, T=0.004, lr=0.12, mu=0.8):
    s = np.asarray(s, dtype=float).copy()
    ss = float(np.linalg.norm(s))
    if not np.isfinite(ss) or ss == 0:
        s = R[0] * 0.0
        s[0] = 1.0
    else:
        s = s / ss
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

def _find_hole(R, rng, n_sample=5000, n_cand=3):
    Z = rng.standard_normal(size=(n_sample, R.shape[1]))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s1 = _single_refine(Z[k], R, steps=160, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=160, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    best_s = best_s / (float(np.linalg.norm(best_s)) + 1e-12)
    return best_s, best_v

def _pair_split_move(X, i, j, delta):
    Y = X.copy()
    a = Y[i].copy(); b = Y[j].copy()
    u = a - b
    ta = u - float(u @ a) * a
    tb = -u + float(u @ b) * b if False else (-u - float((-u) @ b) * b)
    na = float(np.linalg.norm(ta)); nb = float(np.linalg.norm(tb))
    if na < 1e-12 or nb < 1e-12:
        return Y
    ta /= na; tb /= nb
    Y[i] = _norm_rows((a + delta * ta).reshape(1, -1))[0]
    Y[j] = _norm_rows((b + delta * tb).reshape(1, -1))[0]
    return Y

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_POLISH = 16.0
    HOP_END = HARD_END - FINAL_POLISH - 1.0
    rng = np.random.default_rng(711212)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    cur_X = best_X.copy()
    cur_score = best_score
    tabu_sets = collections.deque(maxlen=100)
    tabu_lookup = set()
    archive_fp = set()
    archive_fp.add(_fingerprint(best_X))
    iu_all = np.triu_indices(n, 1)
    sigmas = [0.02, 0.05, 0.09, 0.15, 0.03, 0.07, 0.12]
    it = 0
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        frac_time = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        base = cur_X if rng.random() < 0.7 else best_X
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        pmx = Gb.max(axis=1)
        mglob = float(Gb.max())
        band_c = 5e-4
        cnt = (Gb > mglob - band_c).sum(axis=1)
        kflat = int(np.argmax(Gb[iu_all]))
        wi, wj = int(iu_all[0][kflat]), int(iu_all[1][kflat])
        r = rng.random()
        Xprop = None
        if r < 0.50:
            k_r = int(rng.integers(6, 19))
            cand_idx = [i for i in range(n) if i != wi and i != wj]
            mode = rng.random()
            if mode < 0.55:
                order = sorted(cand_idx, key=lambda i: (-pmx[i], -cnt[i]))
                chosen = order[:k_r]
            elif mode < 0.80:
                order = sorted(cand_idx, key=lambda i: (-cnt[i], -pmx[i]))
                chosen = order[:k_r]
                if rng.random() < 0.5 and k_r >= 3:
                    lows = sorted(cand_idx, key=lambda i: (pmx[i]))[:20]
                    for _ in range(int(rng.integers(1, 3))):
                        chosen[int(rng.integers(0, k_r))] = int(lows[int(rng.integers(0, len(lows)))])
            else:
                order_hi = sorted(cand_idx, key=lambda i: (-pmx[i]))[:30]
                order_lo = sorted(cand_idx, key=lambda i: (pmx[i]))[:30]
                h = k_r // 2
                chosen = list(rng.choice(order_hi, size=h, replace=False)) + list(rng.choice(order_lo, size=k_r - h, replace=False))
            movable = np.unique(np.array([wi, wj] + list(map(int, chosen)), dtype=int))
            key = frozenset(movable.tolist())
            tries = 0
            while key in tabu_lookup and tries < 6:
                pool = [i for i in range(n) if i not in set(movable.tolist())]
                if not pool:
                    break
                nonpair = [v for v in movable if v != wi and v != wj]
                if len(nonpair) == 0:
                    break
                out = rng.choice(nonpair, size=min(2, len(nonpair)), replace=False)
                inn = rng.choice(pool, size=len(out), replace=False)
                sset = set(movable.tolist())
                for o, nn_ in zip(out, inn):
                    sset.remove(int(o)); sset.add(int(nn_))
                movable = np.array(sorted(sset), dtype=int)
                key = frozenset(movable.tolist())
                tries += 1
            nstarts = 3 if (time.time() < HOP_END - 15) else 2
            best_local_X = None
            best_local_s = 1e100
            for si in range(nstarts):
                sigma = sigmas[(it + si) % len(sigmas)]
                if si == 2:
                    sigma = float(rng.uniform(0.08, 0.20))
                X0 = base.copy()
                N = rng.standard_normal((len(movable), d)) * sigma
                Xm = X0[movable]
                N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
                X0[movable] = _norm_rows(Xm + N)
                T0_ = float(rng.uniform(0.003, 0.010))
                T1_ = float(rng.uniform(0.00008, 0.0006))
                lr_ = float(rng.uniform(0.10, 0.30))
                steps_ = int(rng.integers(350, 650))
                Xq, sq = _subset_quench(X0, movable, steps=steps_, T0=T0_, T1=T1_, lr=lr_, mu=0.88, rng=None, noise=0.0)
                if sq < best_local_s:
                    best_local_s = sq
                    best_local_X = Xq
                if time.time() > HOP_END - 1.0:
                    break
            Xprop = best_local_X
            tabu_sets.append(key); tabu_lookup.add(key)
            if len(tabu_sets) > 90:
                pass
            # keep lookup bounded
            if len(tabu_lookup) > 400:
                # drop oldest
                try:
                    old = tabu_sets.popleft()
                    tabu_lookup.discard(old)
                except Exception:
                    pass
        elif r < 0.66:
            # relocate chosen rattler to farthest hole of config with itself removed
            # choose victim: high row-max (loaded) or band count, sometimes random low
            if rng.random() < 0.6:
                # loaded victim
                order = np.argsort(-pmx)
                # pick among top 15 randomly to diversify
                pool = order[:15].tolist()
                # avoid always picking same: random choice weighted
                victim = int(pool[int(rng.integers(0, len(pool)))])
            else:
                # random rattler (low row-max)
                order_lo = np.argsort(pmx)
                victim = int(order_lo[int(rng.integers(0, 20))])
            R = np.delete(base, victim, axis=0)
            hole, _hv = _find_hole(R, rng, n_sample=4000, n_cand=3)
            Xprop = base.copy()
            Xprop[victim] = hole
            # small joint quench around victim + its new neighbours
            Gp = Xprop @ Xprop.T
            np.fill_diagonal(Gp, -2.0)
            nbr = np.argsort(-Gp[victim])[:8]
            movable2 = np.unique(np.array([victim] + nbr.tolist(), dtype=int))
            Xprop, _ = _subset_quench(Xprop, movable2, steps=350, T0=0.004, T1=0.0002, lr=0.18, mu=0.88)
        elif r < 0.82:
            # symmetric pair-apart along tangent bisector
            delta = float(rng.uniform(0.008, 0.06))
            Xprop = _pair_split_move(base, wi, wj, delta)
            # optionally also jiggle 2-4 neighbours
            if rng.random() < 0.5:
                k2 = int(rng.integers(2, 5))
                order = sorted([i for i in range(n) if i != wi and i != wj], key=lambda i: (-pmx[i]))[:25]
                extra = list(rng.choice(order, size=min(k2, len(order)), replace=False))
                movable3 = np.unique(np.array([wi, wj] + list(map(int, extra)), dtype=int))
            else:
                movable3 = np.array([wi, wj], dtype=int)
            # add small tangent noise before quench
            N = rng.standard_normal((len(movable3), d)) * 0.02
            Xm = Xprop[movable3]
            N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
            Xprop[movable3] = _norm_rows(Xm + N)
            Xprop, _ = _subset_quench(Xprop, movable3, steps=400, T0=0.005, T1=0.0002, lr=0.20, mu=0.88)
        else:
            # swap high-load and low-load positions
            hi = int(np.argmax(pmx))
            lo = int(np.argmin(pmx))
            if hi == lo:
                continue
            Xprop = base.copy()
            tmp = Xprop[hi].copy()
            Xprop[hi] = Xprop[lo]
            Xprop[lo] = tmp
            # jiggle both + few rattlers
            k3 = int(rng.integers(4, 10))
            cand_idx = [i for i in range(n) if i != hi and i != lo]
            order_lo = sorted(cand_idx, key=lambda i: (pmx[i]))[:30]
            extra = list(rng.choice(order_lo, size=min(k3, len(order_lo)), replace=False))
            movable4 = np.unique(np.array([hi, lo] + list(map(int, extra)), dtype=int))
            N = rng.standard_normal((len(movable4), d)) * float(rng.uniform(0.02, 0.08))
            Xm = Xprop[movable4]
            N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
            Xprop[movable4] = _norm_rows(Xm + N)
            Xprop, _ = _subset_quench(Xprop, movable4, steps=400, T0=0.006, T1=0.0002, lr=0.18, mu=0.88)
        if Xprop is None:
            continue
        # fingerprint tabu: skip if recently visited topology (unless improving best)
        try:
            fp = _fingerprint(Xprop)
        except Exception:
            fp = None
        pre_score = _max_cos(Xprop)
        # short wide-band LP acceptance polish (budget ~0.6s)
        if time.time() < HOP_END - 1.0:
            Xprop = _lp_polish_wide(Xprop, seconds=0.6, deadline=HOP_END - 0.5, r0=2.5e-3)
        v = _max_cos(Xprop)
        if fp is not None:
            is_new = fp not in archive_fp
        else:
            is_new = True
        improved_cur = v < cur_score - 1e-12
        improved_best = v < best_score - 1e-12
        if improved_best:
            best_score, best_X = v, Xprop.copy()
            cur_score, cur_X = v, Xprop.copy()
            if fp is not None:
                archive_fp.add(fp)
        elif improved_cur or (is_new and v < cur_score + 5e-6 and rng.random() < 0.25):
            cur_score, cur_X = v, Xprop.copy()
            if fp is not None:
                archive_fp.add(fp)
        else:
            # occasional sideways to escape basin even without new fp
            if rng.random() < 0.02 and v < cur_score + 1e-5:
                cur_score, cur_X = v, Xprop.copy()
        # also track best even without LP improvement
        if pre_score < best_score - 1e-12 and pre_score < v:
            pass
    # final polish on best
    try:
        Xf = best_X.copy()
        # two-stage full quench
        remain = max(2.0, HARD_END - time.time() - 8.0)
        if remain > 3.0:
            Xf, _ = _full_quench_track(Xf, steps=900, T0=0.003, T1=0.00015, lr=0.22, mu=0.9)
            v = _max_cos(Xf)
            if v < best_score:
                best_score, best_X = v, Xf.copy()
            else:
                Xf = best_X.copy()
            Xf, _ = _full_quench_track(Xf, steps=700, T0=0.001, T1=0.00005, lr=0.15, mu=0.9)
            v = _max_cos(Xf)
            if v < best_score:
                best_score, best_X = v, Xf.copy()
                Xf = best_X.copy()
        # LP squeeze with remaining time
        t_left = HARD_END - time.time() - 0.5
        if t_left > 1.0:
            Xp = _lp_polish_wide(best_X, seconds=min(t_left, 10.0), r0=2e-3)
            vp = _max_cos(Xp)
            if vp < best_score:
                best_score, best_X = vp, Xp
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
