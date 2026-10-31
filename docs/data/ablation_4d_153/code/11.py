import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad * np.where(gn > 2.0, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _riesz_opt(x, steps, s0, s1, lr0, lr1, mom=0.88, rng=None, noise=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        s = s0 + (s1 - s0) * f
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        # squared chordal distance
        R2 = 2.0 - 2.0 * G
        np.fill_diagonal(R2, np.inf)
        # clip to avoid overflow
        np.clip(R2, 1e-6, 4.0, out=R2)
        # W = R2^{-s/2-1}, stable via log
        # row-normalized so scale cancels; use max-shift per row
        L = (-s * 0.5 - 1.0) * np.log(R2)  # log W
        m = np.max(L, axis=1, keepdims=True)
        W = np.exp(L - m)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad * np.where(gn > 2.0, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        if noise > 0 and rng is not None:
            step = step + noise * rng.standard_normal(size=(n, d))
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
        if p0 > 2500:
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
            p0 = 2500
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def _find_holes(X, k, rng, n_sample=25000):
    n, d = X.shape
    S = rng.standard_normal(size=(n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    scores = np.max(S @ X.T, axis=1)
    order = np.argsort(scores)
    holes = []
    for ii in order:
        cand = S[ii]
        ok = True
        for h in holes:
            if float(cand @ h) > 0.9:
                ok = False
                break
        if ok:
            holes.append(cand.copy())
            if len(holes) >= k:
                break
    if len(holes) == 0:
        return np.empty((0, d))
    H = np.stack(holes, axis=0)
    # refine holes: minimize softmax max-dot
    tau0, tau1 = 0.02, 0.002
    lr0, lr1 = 0.15, 0.01
    steps = 250
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = H @ X.T
        m = np.max(G, axis=1, keepdims=True)
        W = np.exp((G - m) / tau)
        row = np.sum(W, axis=1, keepdims=True)
        grad = (W @ X) / np.maximum(row, 1e-12)
        dot = np.sum(grad * H, axis=1, keepdims=True)
        grad = grad - dot * H
        H = H - lr * grad
        H /= np.linalg.norm(H, axis=1, keepdims=True)
    return H

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    it = 0
    lp_count = 0
    # exploration until reserve for final polish
    while _time.time() < HARD_END - 48.0:
        it += 1
        u = rng.random()
        base_choice = rng.random()
        base = best_x if (base_choice < 0.35) else cur_x
        x0 = base.copy()
        move = ""
        if u < 0.38:
            # global gaussian
            # log-uniform sigma
            ls = rng.uniform(np.log(0.004), np.log(0.55))
            sigma = float(np.exp(ls))
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            if sigma < 0.05:
                steps = 1400
                kind = "soft_small"
            elif sigma < 0.2:
                steps = 1800
                kind = "soft_med" if rng.random() < 0.7 else "riesz"
            else:
                steps = 2200
                kind = "soft_large" if rng.random() < 0.6 else "riesz"
            move = f"g{sigma:.3f}"
        elif u < 0.62:
            # rattler -> hole surgery
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                delta = 0.008
                deg = np.sum(G > m - delta, axis=1)
                # candidates: lowest degree
                k = int(rng.integers(2, 9))
                idx_low = np.argsort(deg)[:k]
                # sometimes include worst nodes too to break contacts
                if rng.random() < 0.3:
                    # replace one worst node
                    cnt_worst = np.sum(G > m - 1e-7, axis=1)
                    iw = int(np.argmax(cnt_worst))
                    idx_low[0] = iw
                nh = k
                H = _find_holes(base, nh, rng, n_sample=22000)
                if H.shape[0] >= 1:
                    for t_, ii in enumerate(idx_low[:H.shape[0]]):
                        x0[ii] = H[t_] + 1e-3 * rng.standard_normal(size=(d,))
                    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                    # small noise to neighbours to allow topology change
                    if rng.random() < 0.5:
                        x0 = x0 + 0.01 * rng.standard_normal(size=(n, d))
                        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                else:
                    x0 = base + 0.02 * rng.standard_normal(size=(n, d))
                    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            except Exception:
                x0 = base + 0.02 * rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = 1800
            kind = "soft_med" if rng.random() < 0.6 else "riesz"
            move = "hole"
        elif u < 0.82:
            # patch randomize
            uu = rng.standard_normal(size=(d,))
            uu /= np.linalg.norm(uu)
            # random angular radius 20-45 deg
            th = rng.uniform(20.0, 45.0) * np.pi / 180.0
            cth = float(np.cos(th))
            dots = base @ uu
            inside = np.where(dots > cth)[0]
            if len(inside) < 3 or len(inside) > 35:
                # pick center as random point + noise to get reasonable patch
                ci = int(rng.integers(0, n))
                uu = base[ci] + 0.3 * rng.standard_normal(size=(d,))
                uu /= np.linalg.norm(uu)
                dots = base @ uu
                inside = np.where(dots > cth)[0]
            if len(inside) >= 2 and len(inside) <= 40:
                # randomize inside points within cap: gaussian around uu
                sig = float(np.tan(th) * 0.6 + 0.05)
                for ii in inside:
                    v = uu + sig * rng.standard_normal(size=(d,))
                    v /= np.linalg.norm(v)
                    # ensure inside cap else project
                    if float(v @ uu) < cth:
                        v = (uu + 0.1 * rng.standard_normal(size=(d,)))
                        v /= np.linalg.norm(v)
                    x0[ii] = v
                # slight noise outside
                x0 = x0 + 0.005 * rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            else:
                x0 = base + 0.05 * rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = 2000
            kind = "soft_med" if rng.random() < 0.5 else "riesz"
            move = "patch"
        elif u < 0.93:
            # subset reinit
            k = int(rng.integers(3, 16))
            idx = rng.choice(n, size=k, replace=False)
            rp = rng.standard_normal(size=(k, d))
            rp /= np.linalg.norm(rp, axis=1, keepdims=True)
            x0[idx] = rp
            # tiny noise rest
            x0 = x0 + 0.003 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = 2000
            kind = "soft_med" if rng.random() < 0.5 else "riesz"
            move = "subset"
        else:
            # full random restart
            x0 = _random_sphere(n, d, rng)
            steps = 2600
            kind = "soft_large" if rng.random() < 0.5 else "riesz"
            move = "rand"
        # run optimizer
        try:
            if kind.startswith("soft"):
                if "small" in kind:
                    x1 = _smooth_opt(x0, steps, 0.025, 0.0022, 0.06, 0.003, mom=0.88)
                elif "large" in kind:
                    x1 = _smooth_opt(x0, steps, 0.055, 0.0032, 0.09, 0.003, mom=0.88)
                else:
                    x1 = _smooth_opt(x0, steps, 0.035, 0.0028, 0.08, 0.003, mom=0.88)
            else:
                # riesz: anneal s
                if rng.random() < 0.5:
                    x1 = _riesz_opt(x0, steps, 8.0, 32.0, 0.08, 0.004, mom=0.88)
                else:
                    x1 = _riesz_opt(x0, steps, 18.0, 40.0, 0.06, 0.003, mom=0.88)
                # short softmax tail to directly target max
                x1 = _smooth_opt(x1, 400, 0.012, 0.002, 0.03, 0.003, mom=0.88)
        except Exception:
            continue
        v1 = _max_cos(x1)
        # quick LP if promising
        if v1 < best_v + 0.0025 and _time.time() < HARD_END - 55.0 and lp_count < 30:
            # only polish top candidates to save time
            if v1 < cur_v + 0.001 or v1 < best_v + 0.0008:
                dl = min(_time.time() + 0.9, HARD_END - 48.0)
                if dl > _time.time() + 0.15:
                    try:
                        x1p, v1p = _lp_polish(x1, dl)
                        lp_count += 1
                        if v1p < v1:
                            x1, v1 = x1p, v1p
                    except Exception:
                        pass
        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        # walker acceptance with decaying tolerance
        prog = min(1.0, it / 250.0)
        tol = 0.004 * (1.0 - prog) + 0.0004
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.012:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            if rng.random() < 0.12:
                cur_x = best_x.copy()
                cur_v = best_v
        if _time.time() >= HARD_END - 48.0:
            break
    # final thorough polish
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.5)
    except Exception:
        pass
    # extra micro-perturb + repolish if time remains (escape shallow LP trap)
    while _time.time() < HARD_END - 6.0:
        try:
            cand = best_x + 1e-4 * rng.standard_normal(size=(n, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand, v = _lp_polish(cand, min(_time.time() + 3.0, HARD_END - 1.0))
            if v < best_v:
                best_x, best_v = cand, v
            else:
                break
        except Exception:
            break
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
