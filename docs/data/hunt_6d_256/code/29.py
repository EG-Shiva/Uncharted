import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_sm(x0, steps, K0, K1, lr, track_best=True):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        # clip extreme steps for stability
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # adaptive clip at 0.1
        sc = np.minimum(1.0, 0.1 / (sn + 1e-12))
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if track_best and (t % 200 == 0 or t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    if track_best:
        return best_x, best_val
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, _max_cos(x)

def _lp_polish(x, deadline, r0=3e-3, cap=6000, mfac=30.0, base=1e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mm = float(c.max())
        margin = max(mfac * radius, base)
        sel = c > (mm - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _void_reinsert_points(x, idx_list, rng, n_global=12000, n_local=2048, local_sig=0.05):
    n, d = x.shape
    for k in idx_list:
        # global candidates
        Z = rng.normal(size=(n_global, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ x.T
        S[:, k] = -2.0
        cm = S.max(axis=1)
        bi = int(np.argmin(cm))
        best_cand = Z[bi]
        best_v = float(cm[bi])
        # local refinement around global best
        R = rng.normal(size=(n_local, d))
        C = best_cand + local_sig * R
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S2 = C @ x.T
        S2[:, k] = -2.0
        cm2 = S2.max(axis=1)
        bi2 = int(np.argmin(cm2))
        if float(cm2[bi2]) < best_v:
            best_cand = C[bi2]
            best_v = float(cm2[bi2])
        # compare to current
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur = float(G[k].max())
        # global other max
        Gm = G.copy()
        Gm[k, :] = -2.0
        Gm[:, k] = -2.0
        other = float(Gm.max())
        new_global = best_v if best_v > other else other
        old_global = float(G.max())
        if new_global < old_global - 1e-14 or (best_v < cur - 1e-12 and new_global <= old_global + 1e-12):
            x[k] = best_cand
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 148.0
    rng = _np.random.default_rng(7)
    n, d = 256, 6
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    # quick diverse pool to pick a good basin start
    try:
        pool_best = x_best.copy()
        pool_val = best
        # 4 fast random anneals
        for s in range(4):
            if _time.time() > t0 + 22:
                break
            y = rng.normal(size=(n, d))
            y /= _np.linalg.norm(y, axis=1, keepdims=True)
            # antipodal symmetrize one of them
            if s == 1:
                h = n // 2
                y[h:] = -y[:h]
                y += 0.01 * rng.normal(size=(n, d))
                y /= _np.linalg.norm(y, axis=1, keepdims=True)
            yb, vb = _adam_sm(y, 350, 80.0, 1200.0, 0.012)
            if vb < pool_val - 5e-6:
                pool_val = vb
                pool_best = yb.copy()
        # keep incumbent if still best; else keep incumbent as x_best but remember pool
        # we keep the better of the two as hopping base, but never discard incumbent
        alt_base = pool_best if pool_val < best + 2e-4 else None
        alt_val = pool_val
    except Exception:
        alt_base = None
    # short equalize of incumbent
    try:
        if _time.time() < t0 + 25:
            x_best = _lp_polish(x_best, min(_time.time() + 4.0, HOP_END), r0=2e-3, cap=5000)
            best = _max_cos(x_best)
    except Exception:
        pass

    cycle = 0
    bases_tried = 0
    while _time.time() < HOP_END:
        cycle += 1
        rem = HOP_END - _time.time()
        if rem < 3:
            break
        # choose starting point: mostly incumbent, occasionally alt base / fresh
        if alt_base is not None and (cycle % 9 == 0) and bases_tried < 6:
            x = alt_base.copy()
            # shake it
            x += 0.04 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, K0, K1, lr = 700, 60.0, 1500.0, 0.010
            bases_tried += 1
        elif cycle % 13 == 0:
            # fresh random basin
            x = rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, K0, K1, lr = 800, 80.0, 1800.0, 0.012
        else:
            x = x_best.copy()
            mode = cycle % 6
            if mode == 0:
                # subset gaussian shake
                k = 24
                pick = rng.choice(n, size=k, replace=False)
                sig = 0.10 if (cycle % 12 == 0) else 0.06
                x[pick] += sig * rng.normal(size=(k, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 650, 60.0, 1600.0, 0.010
            elif mode == 1:
                # coherent cap rotation: common shift for a cap + jitter
                u = rng.normal(size=d); u /= _np.linalg.norm(u)
                proj = x @ u
                thr = _np.quantile(proj, 0.88)
                cap = _np.where(proj > thr)[0]
                if len(cap) < 5:
                    cap = _np.argsort(-proj)[:20]
                # random tangent direction at cap center
                tdir = rng.normal(size=d); tdir -= float(tdir @ u) * u
                tn = float(_np.linalg.norm(tdir))
                if tn < 1e-12:
                    tdir = rng.normal(size=d)
                    tn = float(_np.linalg.norm(tdir))
                tdir /= tn
                amp = 0.06 + 0.04 * rng.random()
                x[cap] += amp * tdir[None, :] + 0.015 * rng.normal(size=(len(cap), d))
                # also jitter worst few
                G = x @ x.T; _np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                w = _np.argsort(-prow)[:6]
                x[w] += 0.03 * rng.normal(size=(6, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 700, 50.0, 1400.0, 0.011
            elif mode == 2:
                # worst-neighbourhood shake
                G = x @ x.T; _np.fill_diagonal(G, -2.0)
                fi = int(_np.argmax(G)); ci = fi // n
                dists = x @ x[ci]
                neigh = _np.argsort(-dists)[:18]
                x[neigh] += 0.07 * rng.normal(size=(len(neigh), d))
                pick = rng.choice(n, size=8, replace=False)
                x[pick] += 0.05 * rng.normal(size=(8, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 600, 70.0, 1600.0, 0.009
            elif mode == 3:
                # void reinsert worst points then anneal
                G = x @ x.T; _np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                worst = _np.argsort(-prow)[:3]
                try:
                    x = _void_reinsert_points(x, worst[:2], rng, n_global=9000, n_local=1024)
                except Exception:
                    pass
                x += 0.015 * rng.normal(size=(n, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 600, 90.0, 1800.0, 0.009
            elif mode == 4:
                # global small shake all
                x += 0.035 * rng.normal(size=(n, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 550, 100.0, 2000.0, 0.008
            else:
                # large kick subset
                pick = rng.choice(n, size=14, replace=False)
                x[pick] += 0.22 * rng.normal(size=(14, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = 750, 40.0, 1200.0, 0.012
            if rem < 25:
                steps = min(steps, 400)
        try:
            xb, vb = _adam_sm(x, steps, K0, K1, lr)
        except Exception:
            continue
        # one round of exact void micro on worst point if promising
        if vb < best + 3e-4:
            try:
                G = xb @ xb.T; _np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                w1 = int(_np.argmax(prow))
                # fast void check with 6000 samples
                Z = rng.normal(size=(6000, d))
                Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ xb.T; S[:, w1] = -2.0
                cm = S.max(axis=1)
                bi = int(_np.argmin(cm))
                # evaluate
                Gm = G.copy(); Gm[w1, :] = -2.0; Gm[:, w1] = -2.0
                other = float(Gm.max())
                nv = float(cm[bi])
                ng = nv if nv > other else other
                if ng < vb - 1e-12:
                    xb[w1] = Z[bi]
                    vb = ng
            except Exception:
                pass
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
            # squeeze new basin immediately with short LP
            if _time.time() < HOP_END - 5:
                try:
                    xp = _lp_polish(x_best, _time.time() + 2.0, r0=2.5e-3, cap=6000)
                    vp = _max_cos(xp)
                    if vp < best:
                        x_best, best = xp, vp
                except Exception:
                    pass
            # refresh alt base if we beat it a lot
            if alt_base is not None and best < alt_val - 1e-6:
                pass
        # occasional LP equalize of incumbent to spread contacts
        if cycle % 10 == 0 and _time.time() < HOP_END - 6:
            try:
                xp = _lp_polish(x_best, _time.time() + 2.0, r0=2.0e-3, cap=6000)
                vp = _max_cos(xp)
                if vp < best:
                    x_best, best = xp, vp
            except Exception:
                pass
    # final intensive polish: void sweep on worst points + LP + fine anneal
    try:
        if _time.time() < T_END - 20:
            G = x_best @ x_best.T; _np.fill_diagonal(G, -2.0)
            prow = G.max(axis=1)
            worst = _np.argsort(-prow)[:8]
            x_best = _void_reinsert_points(x_best, worst[:4], rng, n_global=16000, n_local=2048, local_sig=0.03)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 14:
            xb, vb = _adam_sm(x_best, 500, 120.0, 2500.0, 0.006)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 2:
            x_best = _lp_polish(x_best, T_END - 1.0, r0=3e-3, cap=8000, mfac=30.0, base=5e-5)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
