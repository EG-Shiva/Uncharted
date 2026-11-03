import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _lp_polish(x, seconds, t0=None, limit=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > limit:
            break
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                              np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
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
        v = _max_cos(y)
        if v < best:
            best = float(v)
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx, best


def _anneal(x, alpha0, alpha1, steps, lr, mom, noise, rng, deadline=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * (alpha1 / max(alpha0, 1e-9)) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = alpha * (G - m)
        # E <= 0 by construction
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gw = Gw / rs
        dn = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dn * x
        # normalize gradient scale to keep lr stable across alpha
        gn = np.linalg.norm(Gt) + 1e-12
        # cap huge gradients
        cur_lr = lr * (1.0 - 0.5 * frac)
        v = mom * v - (cur_lr / max(1.0, gn * 2.0)) * Gt * max(1.0, 1.0) - cur_lr * Gt * 0.0
        # simpler: use raw Gt but scaled lr; do heavy-ball:
        # recompute correctly: v = mom*v - cur_lr*Gt (Gt already averaged)
        # above line mixed; fix to standard:
        # (we already updated v, undo weird scaling by resetting)
        # Actually just do standard step: v = mom*v_old - cur_lr*Gt ; use v_old
        # To avoid confusion, recompute with stored previous? Simplify: use standard form below
        # We will correct by using momentum properly: v = mom*v_prev - cur_lr*Gt
        # Since we overwrote, approximate is fine; add noise
        if noise > 0:
            nz = rng.standard_normal((n, d)) * noise
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            v = v + nz
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        cap = 0.08
        over = vn > cap
        if np.any(over):
            v = np.where(over, v * (cap / np.maximum(vn, 1e-12)), v)
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _anneal2(x, alpha0, alpha1, steps, lr, mom, noise, rng, deadline=None):
    # clean standard heavy-ball on averaged LSE gradient
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * (alpha1 / max(alpha0, 1e-9)) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ x
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gw = Gw / rs
        dn = np.sum(Gw * x, axis=1, keepdims=True)
        Gt = Gw - dn * x
        cur_lr = lr * (1.0 - 0.5 * frac)
        v = mom * v - cur_lr * Gt
        if noise > 0:
            nz = rng.standard_normal((n, d)) * noise
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            v = v + nz
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        cap = 0.08
        over = vn > cap
        if np.any(over):
            v = np.where(over, v * (cap / np.maximum(vn, 1e-12)), v)
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _local_repel_newbie(c, Xo, alpha=120.0, iters=250, lr=0.03):
    for _ in range(iters):
        dots = Xo @ c
        m = float(dots.max())
        w = np.exp(alpha * (dots - m))
        sw = float(w.sum()) + 1e-12
        g = (w @ Xo) / sw
        gt = g - float(np.dot(g, c)) * c
        c = c - lr * gt
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            break
        c /= n
    return c


def _hole_surgery(base, rng, k=None):
    n, d = base.shape
    pm = _per_max(base)
    if k is None:
        k = int(rng.choice([1, 1, 1, 2, 2, 3]))
    # kill worst k (add small random to avoid always same)
    jitter = rng.standard_normal(n) * 1e-4
    kill = np.argsort(pm + jitter)[-k:]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[kill] = False
    Xo = base[keep_mask]
    x_new = base.copy()
    for kk in kill:
        cand = rng.standard_normal((6000, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        md = (cand @ Xo.T).max(axis=1)
        order = np.argsort(md)[:8]
        best_c = None
        best_m = 1e9
        for b in order[:4]:
            c = _local_repel_newbie(cand[int(b)].copy(), Xo)
            mm = float((Xo @ c).max())
            if mm < best_m:
                best_m = mm
                best_c = c
        x_new[kk] = best_c
    # small shake to break symmetry
    x_new = x_new + rng.standard_normal((n, d)) * 0.003
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new


def _cap_shake(base, rng):
    n, d = base.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    dots = base @ u
    # random cap size 4..22
    cap = int(rng.integers(4, 23))
    idx = np.argsort(dots)[-cap:]
    x_new = base.copy()
    sig = float(rng.uniform(0.15, 0.5))
    x_new[idx] = x_new[idx] + rng.standard_normal((cap, d)) * sig
    x_new[idx] /= np.linalg.norm(x_new[idx], axis=1, keepdims=True)
    # also tiny global jitter
    x_new = x_new + rng.standard_normal((n, d)) * 0.002
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new


def _align_and_mix(A, B, rng):
    n, d = A.shape
    # PCA alignment: match right singular vectors
    try:
        _, _, Va = np.linalg.svd(A, full_matrices=False)
        _, _, Vb = np.linalg.svd(B, full_matrices=False)
        Q = Vb.T @ Va
        # fix determinant to +1 (proper rotation) if needed
        if np.linalg.det(Q) < 0:
            Vt = Va.copy()
            Vt[-1] *= -1
            Q = Vb.T @ Vt
        Br = B @ Q
    except Exception:
        Br = B
    m = int(rng.integers(n // 3, 2 * n // 3))
    ia = rng.choice(n, m, replace=False)
    ib = rng.choice(n, n - m, replace=False)
    C = np.empty_like(A)
    C[:m] = A[ia]
    C[m:] = Br[ib]
    # resolve exact duplicates by jitter
    C = C + rng.standard_normal((n, d)) * 0.005
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def solve():
    t0 = time.time()
    HARD = 172.0
    EXPLORE = 148.0
    rng = np.random.RandomState(20261103)
    n, d = 99, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            m = min(xb.size, n * d)
            tmp.ravel()[:m] = xb.ravel()[:m]
        except Exception:
            pass
        xb = tmp
    xb = np.where(np.isfinite(xb), xb, 0.0)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = np.where(np.isfinite(xr), xr, 0.0)
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass

    best_x = xb.copy()
    best_v = _max_cos(best_x)
    # elite pool: list of (val, x)
    elites = [(best_v, best_x.copy())]

    # quick diversity: two short random descents to seed pool
    for _ in range(2):
        if time.time() - t0 > 20:
            break
        xr0 = rng.standard_normal((n, d))
        xr0 /= np.linalg.norm(xr0, axis=1, keepdims=True)
        xr0 = _anneal2(xr0, 30.0, 200.0, 400, 0.08, 0.85, 0.0, rng,
                       deadline=t0 + EXPLORE)
        vv = _max_cos(xr0)
        if vv < best_v + 0.05:
            xp, vp = _lp_polish(xr0, seconds=1.0, t0=t0, limit=EXPLORE)
            if vp < best_v:
                best_v, best_x = float(vp), xp.copy()
            elites.append((float(vp), xp.copy()))
    elites = sorted(elites, key=lambda z: z[0])[:5]

    sigmas = [0.006, 0.012, 0.025, 0.05, 0.09, 0.16, 0.30]
    trial = 0
    while time.time() - t0 < EXPLORE:
        trial += 1
        if time.time() - t0 > EXPLORE - 1.0:
            break
        # pick base: mostly best, sometimes random elite
        if len(elites) > 1 and rng.rand() < 0.3:
            bi = int(rng.integers(0, len(elites)))
            base = elites[bi][1]
        else:
            base = best_x
        r = rng.rand()
        if r < 0.34:
            sig = sigmas[trial % len(sigmas)]
            if rng.rand() < 0.2:
                sig *= float(rng.uniform(0.5, 1.5))
            x_try = base + rng.standard_normal((n, d)) * sig
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.60:
            x_try = _hole_surgery(base, rng)
        elif r < 0.78:
            x_try = _cap_shake(base, rng)
        else:
            if len(elites) >= 2:
                i, j = rng.choice(len(elites), 2, replace=False)
                x_try = _align_and_mix(elites[i][1], elites[j][1], rng)
            else:
                x_try = _hole_surgery(base, rng)

        deadline = min(t0 + EXPLORE, t0 + HARD - 8.0)
        # two-stage anneal: broad then focused
        a0 = float(rng.choice([20.0, 30.0, 50.0]))
        a1 = float(rng.choice([150.0, 250.0, 400.0]))
        steps1 = int(rng.integers(350, 700))
        lr = float(rng.uniform(0.05, 0.12))
        mom = float(rng.uniform(0.80, 0.90))
        noise = 0.0
        if rng.rand() < 0.25:
            noise = float(rng.uniform(0.0005, 0.003))
        x_try = _anneal2(x_try, a0, a1, steps1, lr, mom, noise, rng, deadline=deadline)
        # short high-alpha focusing
        if time.time() < deadline:
            x_try = _anneal2(x_try, 250.0, 500.0, 150, lr * 0.5, 0.88, 0.0, rng,
                             deadline=deadline)
        v_s = _max_cos(x_try)
        # decide LP budget
        do_lp = False
        lp_time = 0.6
        if v_s < best_v + 0.004:
            do_lp = True
            lp_time = 2.2 if v_s < best_v + 0.001 else 1.0
        elif trial % 7 == 0 and v_s < best_v + 0.012:
            do_lp = True
            lp_time = 0.8
        if do_lp:
            lp_dead = min(deadline, time.time() + lp_time)
            # cap seconds
            x_pol, v_pol = _lp_polish(x_try, seconds=max(0.3, lp_dead - time.time()),
                                      t0=t0, limit=EXPLORE)
            # extra smoothing after LP sometimes helps
            if v_pol < best_v + 0.002 and time.time() < deadline:
                x_alt = _anneal2(x_pol, 300.0, 500.0, 120, 0.03, 0.9, 0.0, rng,
                                 deadline=deadline)
                v_alt = _max_cos(x_alt)
                if v_alt < v_pol:
                    # quick re-polish
                    x_alt2, v_alt2 = _lp_polish(x_alt, seconds=0.6, t0=t0, limit=EXPLORE)
                    if v_alt2 < v_alt:
                        x_pol, v_pol = x_alt2, v_alt2
                    else:
                        x_pol, v_pol = x_alt, v_alt
            if v_pol < best_v:
                best_v = float(v_pol)
                best_x = x_pol.copy()
                elites.append((float(v_pol), x_pol.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:5]
                if best_v < 0.2430:
                    break
            else:
                # keep diversity: if good enough, add to pool
                if v_pol < elites[-1][0] - 1e-6 and len(elites) < 5:
                    elites.append((float(v_pol), x_pol.copy()))
                    elites = sorted(elites, key=lambda z: z[0])[:5]
                elif v_s < best_v and not do_lp:
                    pass
        else:
            if v_s < best_v:
                # polish immediately
                x_pol, v_pol = _lp_polish(x_try, seconds=1.2, t0=t0, limit=EXPLORE)
                if v_pol < best_v:
                    best_v = float(v_pol)
                    best_x = x_pol.copy()
                    elites.append((float(v_pol), x_pol.copy()))
                    elites = sorted(elites, key=lambda z: z[0])[:5]
        if trial > 500:
            break

    rem = HARD - (time.time() - t0) - 1.0
    if rem > 0.5:
        rem = min(rem, 18.0)
        # alternate smooth + LP for final squeeze
        try:
            best_x = _anneal2(best_x, 300.0, 600.0, 300, 0.02, 0.9, 0.0, rng, deadline=None)
        except Exception:
            pass
        best_x, best_v = _lp_polish(best_x, seconds=rem)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
