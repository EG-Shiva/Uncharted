import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), float(np.max(g))

def solve():
    import time as _tm
    import numpy as _np
    t0 = _tm.time()
    HARD = t0 + 170.0
    rng = _np.random.default_rng(0)
    _np.random.seed(0)
    n, d = 249, 6

    def normed(a):
        a = _np.asarray(a, dtype=_np.float64)
        nr = _np.linalg.norm(a, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return a / nr

    # loads
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=_np.float64)
            if a.shape == (n, d):
                cands.append(normed(a))
        except Exception:
            pass
    if not cands:
        cands.append(normed(rng.standard_normal(size=(n, d))))
    best = None
    best_val = 1e100
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    # ---------- smooth LSE Adam ----------
    def smooth_refine(x, steps, K0, K1, lr0, lr1, w0, w1):
        x = x.copy()
        m_ad = _np.zeros_like(x)
        v_ad = _np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for s in range(steps):
            if _tm.time() > HARD - 45.0:
                break
            frac = s / (steps - 1) if steps > 1 else 0.0
            K = K0 * (K1 / K0) ** frac
            lr = lr0 * (lr1 / lr0) ** frac
            win = w0 * (w1 / w0) ** frac
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            m = float(_np.max(G))
            G -= m
            G *= K
            thr = -K * win
            # mask far pairs
            G[G < thr] = -_np.inf
            _np.exp(G, out=G)
            _np.fill_diagonal(G, 0.0)
            S = float(_np.sum(G))
            if not _np.isfinite(S) or S <= 0:
                continue
            G /= S
            grad = G @ x
            proj = _np.sum(grad * x, axis=1, keepdims=True)
            grad -= proj * x
            m_ad *= b1
            m_ad += (1.0 - b1) * grad
            _np.add(v_ad * b2, (1.0 - b2) * (grad * grad), out=v_ad)
            bc1 = 1.0 - b1 ** (s + 1)
            bc2 = 1.0 - b2 ** (s + 1)
            mhat = m_ad / bc1
            vhat = v_ad / bc2
            step = mhat / (_np.sqrt(vhat) + eps)
            x = x - lr * step
            nr = _np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    # ---------- hole surgery ----------
    def hole_improve(x, n_cand=4000, n_worst=8):
        x = x.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        row_max = _np.max(G, axis=1)
        glob = float(_np.max(G))
        order = _np.argsort(-row_max)
        for k in range(min(n_worst, n)):
            if _tm.time() > HARD - 40.0:
                break
            idx = int(order[k])
            mask = _np.ones(n, dtype=bool)
            mask[idx] = False
            Xo = x[mask]
            G2 = Xo @ Xo.T
            _np.fill_diagonal(G2, -2.0)
            glob_excl = float(_np.max(G2))
            C = rng.standard_normal(size=(n_cand, d))
            C /= _np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Xo.T
            cmax = _np.max(D, axis=1)
            bi = int(_np.argmin(cmax))
            bv = float(cmax[bi])
            new_glob = glob_excl if glob_excl > bv else bv
            if new_glob < glob - 5e-8:
                x[idx] = C[bi]
                # local push of inserted point: 150 tangent steps away from nearest
                v = C[bi]
                for _ in range(150):
                    dots = Xo @ v
                    # softmax weights to nearest
                    mm = float(_np.max(dots))
                    w = _np.exp(60.0 * (dots - mm))
                    w /= float(_np.sum(w))
                    gvec = w @ Xo
                    gvec = gvec - float(gvec @ v) * v
                    nn = float(_np.linalg.norm(gvec))
                    if nn < 1e-12:
                        break
                    gvec /= nn
                    v = v - 0.02 * gvec
                    v /= float(_np.linalg.norm(v))
                x[idx] = v
                glob = new_glob
                # refresh
                G = x @ x.T
                _np.fill_diagonal(G, -2.0)
                row_max = _np.max(G, axis=1)
                order = _np.argsort(-row_max)
        return x

    # stage 1: deep polish incumbent
    try:
        xp = smooth_refine(best, 2600, 150.0, 3500.0, 0.006, 0.0004, 0.03, 0.002)
        vp = _max_cos(xp)
        if vp < best_val:
            best, best_val = xp, vp
    except Exception:
        pass
    try:
        best = hole_improve(best, n_cand=5000, n_worst=10)
        best_val = _max_cos(best)
        xp2 = smooth_refine(best, 1500, 400.0, 4000.0, 0.003, 0.0003, 0.01, 0.0015)
        vp2 = _max_cos(xp2)
        if vp2 < best_val:
            best, best_val = xp2, vp2
    except Exception:
        pass

    # stage 2: basin hops guided by contacts
    sigmas = [0.025, 0.012, 0.04, 0.018, 0.008, 0.03, 0.015, 0.05]
    hop = 0
    while _tm.time() < HARD - 42.0 and hop < 10:
        sigma = sigmas[hop % len(sigmas)]
        hop += 1
        try:
            rm, gv = _row_max(best)
            worst = _np.argsort(-rm)[:35]
            Xp = best.copy()
            Xp += sigma * 0.5 * rng.standard_normal(size=Xp.shape)
            # extra kick to worst cluster
            Xp[worst] += sigma * rng.standard_normal(size=(len(worst), d))
            Xp = normed(Xp)
            Xs = smooth_refine(Xp, 900, 300.0, 2500.0, 0.0025, 0.0004, 0.012, 0.002)
            Xs = hole_improve(Xs, n_cand=3000, n_worst=5)
            Xs2 = smooth_refine(Xs, 700, 800.0, 4000.0, 0.0012, 0.00025, 0.006, 0.0012)
            v = _max_cos(Xs2)
            if v < best_val - 1e-12:
                best, best_val = Xs2, v
            elif v < best_val + 0.002 and (hop % 3 == 0):
                # sideways move to different basin: keep if not much worse to diversify
                # only keep for next hops if it enables later gain: do short LP-free check
                # we do not replace best, but continue hops from this point once
                Xp3 = smooth_refine(Xs2, 500, 200.0, 1500.0, 0.002, 0.0004, 0.015, 0.003)
                v3 = _max_cos(Xp3)
                if v3 < best_val - 1e-12:
                    best, best_val = Xp3, v3
        except Exception:
            continue

    # stage 3: trust-region LP polish
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
        x = best.copy()
        x = normed(x)
        cur = _max_cos(x)
        best, best_val = x, cur
        radius = 1.5e-3
        lp_start = _tm.time()
        iu0 = _np.triu_indices(n, 1)
        # allow ~30s for LP
        while _tm.time() - lp_start < 28.0 and _tm.time() < HARD - 8.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0]
            m = float(_np.max(c))
            win = max(20.0 * radius, 2e-7)
            sel = c > m - win
            I = iu0[0][sel]; J = iu0[1][sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            # cap p to avoid huge LP: keep tightest 2500
            if p > 2500:
                idx = _np.argsort(-c[sel])[:2500]
                I = I[idx]; J = J[idx]
                p = 2500
                # recompute sel values accordingly for b
                csel = c[sel][idx]
            else:
                csel = c[sel]
            nv = n * d + 1
            rows = _np.repeat(_np.arange(p), 2 * d + 1)
            cols = _np.concatenate([I[:, None] * d + _np.arange(d),
                                    J[:, None] * d + _np.arange(d),
                                    _np.full((p, 1), n * d)], 1).ravel()
            vals = _np.concatenate([x[J], x[I], -_np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (_np.repeat(_np.arange(n), d), _np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = _np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                              b_eq=_np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = normed(y)
            v = _max_cos(y)
            if v < best_val:
                best, best_val = y, v
                x = y
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
                # keep x as is
        best = normed(best)
    except Exception:
        pass

    # stage 4: greedy worst-pair micro polish (tiny time left)
    try:
        G = best @ best.T
        _np.fill_diagonal(G, -2.0)
        best_val = float(G.max())
        g_dead = HARD - 1.0
        radius = 8e-5
        flat = int(_np.argmax(G)); a, b = divmod(flat, n)
        while _tm.time() < g_dead and radius > 1e-13:
            improved = False
            per = G.max(axis=1)
            order = _np.argsort(per)[-30:]
            flat = int(_np.argmax(G)); a, b = divmod(flat, n)
            for _ in range(3000):
                if _tm.time() >= g_dead:
                    break
                r = rng.random()
                if r < 0.7:
                    ii = a if rng.random() < 0.5 else b
                else:
                    ii = int(order[rng.integers(len(order))])
                t = rng.standard_normal(d)
                t -= float(t @ best[ii]) * best[ii]
                nm = float(_np.linalg.norm(t))
                if nm < 1e-14:
                    continue
                t /= nm
                if rng.random() < 0.5:
                    rival = b if ii == a else (a if ii == b else int(_np.argmax(G[ii])))
                    dv = -best[rival].copy()
                    dv -= float(dv @ best[ii]) * best[ii]
                    n2 = float(_np.linalg.norm(dv))
                    if n2 > 1e-12:
                        dv /= n2
                        t = 0.7 * dv + 0.3 * t
                        t -= float(t @ best[ii]) * best[ii]
                        nn = float(_np.linalg.norm(t))
                        if nn < 1e-14:
                            continue
                        t /= nn
                sl = radius * (0.2 + 0.8 * rng.random())
                row = best[ii] + sl * t
                nr = float(_np.linalg.norm(row))
                if nr == 0 or not _np.isfinite(nr):
                    continue
                row /= nr
                dots = best @ row
                dots[ii] = -2.0
                if float(dots.max()) >= best_val:
                    continue
                Xt = best.copy(); Xt[ii] = row
                Gt = Xt @ Xt.T
                _np.fill_diagonal(Gt, -2.0)
                v = float(Gt.max())
                if v < best_val - 1e-12:
                    best = Xt; G = Gt; best_val = v
                    improved = True
                    flat = int(_np.argmax(G)); a, b = divmod(flat, n)
            if not improved:
                radius *= 0.5
    except Exception:
        pass

    best = normed(best)
    best[~_np.isfinite(best)] = 0.0
    nr = _np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
