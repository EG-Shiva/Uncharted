import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())


def _anneal(X, steps, a0, a1, s0, s1, mu=0.85):
    X = _norm_rows(X.copy())
    n, d = X.shape
    vel = np.zeros_like(X)
    best_X = X.copy()
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    la0 = np.log(a0) if a0 > 0 else 0.0
    la1 = np.log(a1) if a1 > 0 else 0.0
    ls0 = np.log(s0) if s0 > 0 else 0.0
    ls1 = np.log(s1) if s1 > 0 else 0.0
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        smax = float(np.exp(ls0 + (ls1 - ls0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_X = X.copy()
        # softmax weights, G-m <= 0 so no overflow
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ X) / S
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm < 1e-15:
            continue
        X = X - (smax / nrm) * vel
        X = _norm_rows(X)
    Gf = best_X @ best_X.T
    np.fill_diagonal(Gf, -2.0)
    # re-check last X too (best_X already tracks)
    Gc = X @ X.T
    np.fill_diagonal(Gc, -2.0)
    fc = float(Gc.max())
    if fc < bv:
        return X, fc
    return best_X, bv


def _subset_anneal(X, active, steps, alpha, smax, mu=0.5):
    X = _norm_rows(X.copy())
    active = np.asarray(active, dtype=int)
    k = len(active)
    if k == 0:
        return X
    vel = np.zeros((k, 4))
    for _ in range(steps):
        Xa = X[active]
        G = Xa @ X.T
        # mask self
        # G[p, active[p]] = -2
        G[np.arange(k), active] = -2.0
        m = float(G.max())
        W = np.exp(np.clip(alpha * (G - m), -50, 0.0))
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ X) / S
        dot = np.sum(grad * Xa, axis=1, keepdims=True)
        gt = grad - dot * Xa
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm < 1e-15:
            break
        Xa2 = Xa - (smax / nrm) * vel
        Xa2 = _norm_rows(Xa2)
        X[active] = Xa2
    return _norm_rows(X)


def _refine_hole(Y, c0, steps=80, s=0.05):
    c = c0 / (np.linalg.norm(c0) + 1e-300)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = float(dots.max())
        sel = dots > m - 5e-3
        if np.count_nonzero(sel):
            g = Y[sel].sum(axis=0)
        else:
            g = Y[int(np.argmax(dots))]
        gt = g - float(np.dot(g, c)) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.5 * vel + gt
        vn = float(np.linalg.norm(vel))
        if vn < 1e-14:
            break
        c = c - (s / vn) * vel
        nc = float(np.linalg.norm(c))
        if nc == 0:
            break
        c /= nc
        s *= 0.995
    return c / (np.linalg.norm(c) + 1e-300)


def _largest_holes(Y, rng, k, n_samples=6000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C = _norm_rows(C)
    # block to save memory
    sm_all = np.empty(n_samples)
    bs = 1500
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm_all[a:a + Cb.shape[0]] = S.max(axis=1)
    out = []
    tmp = sm_all.copy()
    for _ in range(k):
        idx = int(np.argmin(tmp))
        tmp[idx] = 2.0
        c = _refine_hole(Y, C[idx])
        out.append(c)
    return np.stack(out, axis=0)


def _lp_polish(x, deadline):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 6000 - 1)[:6000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
            cnt = 6000
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _hopf_init(rng, n_base=17, n_fib=16, phase_jitter=0.3):
    # 17*16 = 272
    pts = []
    # fibonacci on S2
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_base):
        z = 1.0 - 2.0 * (i + 0.5) / n_base
        z = max(min(z, 1.0), -1.0)
        th = float(np.arccos(z))
        ph = float(ga * i)
        ct2 = np.cos(th / 2.0)
        st2 = np.sin(th / 2.0)
        off = float(rng.uniform(0, 2 * np.pi))
        for j in range(n_fib):
            psi = 2 * np.pi * j / n_fib + off + rng.standard_normal() * phase_jitter * 0.1
            a1 = (psi + ph) / 2.0
            a2 = (psi - ph) / 2.0
            pts.append([ct2 * np.cos(a1), ct2 * np.sin(a1), st2 * np.cos(a2), st2 * np.sin(a2)])
    X = np.array(pts, dtype=np.float64)
    # random global rotation
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    X = X @ Q.T
    return _norm_rows(X)


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    ANNEAL_END = t0 + 138.0
    LP_END = HARD_END - 1.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        cand.append(_norm_rows(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])

    def consider(X):
        nonlocal best, best_val
        X = _norm_rows(X)
        v = _max_cos(X)
        if v < best_val:
            best_val = v
            best = X.copy()
            return True
        return False

    # Phase 1: diverse restarts
    n_restart = 14
    for r in range(n_restart):
        if time.time() > ANNEAL_END - 5:
            break
        kind = r % 4
        if kind == 0:
            X0 = _norm_rows(rng.standard_normal((n, d)))
            steps, a0, a1, s0, s1, mu = 500, 25.0, 400.0, 0.06, 0.004, 0.85
        elif kind == 1:
            X0 = _hopf_init(rng, 17, 16, phase_jitter=0.5)
            # add noise to break perfect fibers
            X0 = _norm_rows(X0 + 0.02 * rng.standard_normal((n, d)))
            steps, a0, a1, s0, s1, mu = 500, 30.0, 500.0, 0.05, 0.003, 0.88
        elif kind == 2:
            # antipodally-augmented random: random + jittered best mix
            X0 = _norm_rows(rng.standard_normal((n, d)))
            steps, a0, a1, s0, s1, mu = 400, 12.0, 250.0, 0.08, 0.005, 0.80
        else:
            sig = float(rng.choice([0.05, 0.15, 0.3]))
            X0 = _norm_rows(best + sig * rng.standard_normal((n, d)))
            steps, a0, a1, s0, s1, mu = 450, 20.0, 350.0, 0.05, 0.003, 0.85
        Xa, va = _anneal(X0, steps, a0, a1, s0, s1, mu)
        consider(Xa)
        # quick hole-regrow on the result sometimes
        if r % 3 == 0 and time.time() < ANNEAL_END - 5:
            try:
                rm, _ = _row_max(Xa)
                kdel = 4
                vict = np.argsort(-rm)[:kdel]
                holes = _largest_holes(np.delete(Xa, vict, axis=0), rng, kdel, n_samples=4000)
                Xb = Xa.copy()
                Xb[vict] = holes
                Xb = _subset_anneal(Xb, vict, 150, 80.0, 0.05, mu=0.5)
                Xb, vb = _anneal(Xb, 250, 40.0, 400.0, 0.02, 0.002, mu=0.85)
                consider(Xb)
            except Exception:
                pass

    # Phase 2: basin hopping + delete-regrow from global best
    it = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.25, 0.4]
    while time.time() < ANNEAL_END:
        it += 1
        if it % 5 == 0:
            # delete-and-regrow move
            try:
                rm, _ = _row_max(best)
                # victims: worst + random among top-30 to diversify
                order = np.argsort(-rm)
                kdel = int(rng.choice([3, 4, 6, 8]))
                pool = order[:30]
                vict = np.sort(rng.choice(pool, size=min(kdel, len(pool)), replace=False))
                # ensure at least the worst is included sometimes
                if rng.random() < 0.5 and order[0] not in vict:
                    vict[0] = order[0]
                Y = np.delete(best, vict, axis=0)
                holes = _largest_holes(Y, rng, len(vict), n_samples=5000)
                X0 = best.copy()
                X0[vict] = holes
                # local relax of newcomers vs all
                X0 = _subset_anneal(X0, vict, 120, 60.0, 0.06, mu=0.5)
                a0 = float(rng.choice([25.0, 40.0]))
                a1 = float(rng.choice([350.0, 600.0]))
                X0, v0 = _anneal(X0, 350, a0, a1, 0.03, 0.002, mu=0.85)
                consider(X0)
            except Exception:
                pass
        else:
            sig = sigmas[int(rng.integers(0, len(sigmas)))]
            X0 = _norm_rows(best + sig * rng.standard_normal((n, d)))
            # random schedule
            a0 = float(np.exp(rng.uniform(np.log(10), np.log(50))))
            a1 = float(np.exp(rng.uniform(np.log(200), np.log(800))))
            s0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.09))))
            s1 = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.006))))
            mu = float(rng.uniform(0.75, 0.92))
            steps = int(rng.choice([300, 400, 500]))
            Xa, va = _anneal(X0, steps, a0, a1, s0, s1, mu)
            consider(Xa)
        # opportunistic extra Kick: occasionally try antipodal flip of rattlers
        if it % 11 == 0 and time.time() < ANNEAL_END:
            try:
                rm, _ = _row_max(best)
                # rattlers = lowest row max
                order = np.argsort(rm)
                ratt = order[:3]
                X0 = best.copy()
                # reinsert rattlers at holes far from core
                Y = np.delete(best, ratt, axis=0)
                holes = _largest_holes(Y, rng, len(ratt), n_samples=5000)
                X0[ratt] = holes
                X0, v0 = _anneal(X0, 300, 30.0, 500.0, 0.03, 0.002, mu=0.85)
                consider(X0)
            except Exception:
                pass

    # Phase 3: LP polish best
    try:
        best = _lp_polish(best, LP_END)
        best = _norm_rows(best)
        best_val = _max_cos(best)
    except Exception:
        pass
    # safety: if polish somehow worsened vs stored file, reload best file? keep ours if finite
    try:
        # ensure no NaN / zero rows
        best = _norm_rows(np.asarray(best, dtype=np.float64))
        assert best.shape == (272, 4)
        assert np.all(np.isfinite(best))
        nr = np.linalg.norm(best, axis=1)
        assert np.all(nr > 1e-9)
    except Exception:
        try:
            best = _norm_rows(np.load("best.npy").astype(np.float64))
        except Exception:
            best = _norm_rows(rng.standard_normal((272, 4)))
    return best
