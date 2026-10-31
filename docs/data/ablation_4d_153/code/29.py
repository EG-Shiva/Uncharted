import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G))

def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _softmax_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
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
        np.maximum(row, 1e-12, out=row)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _riesz_opt(x, steps, s, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, 1.0 - 2.0)  # dummy, will zero out
        d2 = 2.0 - 2.0 * G
        # diagonal d2 = 0 -> clip then zero
        np.fill_diagonal(d2, 1.0)
        d2 = np.clip(d2, 1e-4, 4.0)
        W = d2 ** (-s / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        # clip extreme
        np.clip(W, 0.0, 1e3, out=W)
        row = np.sum(W, axis=1, keepdims=True)
        np.maximum(row, 1e-12, out=row)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        x = x - lr * (vel / bc)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _hybrid(x, steps_r, s, steps_s, tau0, tau1, lr0, lr1):
    # long range first, then max-focused
    if steps_r > 0:
        # use larger lr for riesz phase
        x = _riesz_opt(x, steps_r, s, lr0 * 1.2, lr0 * 0.4)
    if steps_s > 0:
        x = _softmax_opt(x, steps_s, tau0, tau1, lr0, lr1)
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
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 2800:
            order = np.argpartition(c[idx], -2800)[-2800:]
            idx = idx[order]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _hopf_points(thetas, phis, Ks, offsets):
    pts = []
    for th, ph, K, off in zip(thetas, phis, Ks, offsets):
        c = np.cos(th / 2.0); s = np.sin(th / 2.0)
        for k in range(int(K)):
            psi = 2.0 * np.pi * k / float(K) + float(off)
            a1 = (psi + ph) / 2.0; a2 = (psi - ph) / 2.0
            pts.append([c * np.cos(a1), c * np.sin(a1), s * np.cos(a2), s * np.sin(a2)])
    return np.array(pts, dtype=float)

def _find_hole(x, rng, nsample=15000, refine=60):
    n, d = x.shape
    Y = _random_sphere(nsample, d, rng)
    G = Y @ x.T
    rm = np.max(G, axis=1)
    bi = int(np.argmin(rm))
    y = Y[bi].copy()
    # hill climb to deepen hole: minimize max_j <y,x_j>
    for t in range(refine):
        dots = x @ y
        j = int(np.argmax(dots))
        g = x[j] - dots[j] * y
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        g /= gn
        step = 0.05 / (1.0 + t * 0.15)
        y = y - step * g
        y /= np.linalg.norm(y)
    return y

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    EXPLORE_END = HARD_END - 42.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)

    # ---- Phase A: Hopf symmetric search (diverse basin) ----
    try:
        hopf_best = []
        # two factorizations: 9x17 and 17x9
        for (F, K) in ((9, 17), (17, 9)):
            cands = []
            for trial in range(1200):
                if _time.time() > t0 + 14.0:
                    break
                thetas = np.arccos(np.clip(rng.uniform(-1, 1, size=F), -1, 1))
                phis = rng.uniform(0, 2 * np.pi, size=F)
                offs = rng.uniform(0, 2 * np.pi, size=F)
                Ks = [K] * F
                X = _hopf_points(thetas, phis, Ks, offs)
                # X has F*K =153 rows
                v = _max_cos(X)
                cands.append((v, thetas, phis, offs))
            if not cands:
                continue
            cands.sort(key=lambda t: t[0])
            for rank in range(min(2, len(cands))):
                v0, th, ph, off = cands[rank]
                X = _hopf_points(th, ph, [K] * F, off)
                # break symmetry slightly + hybrid relax
                X = X + 0.01 * rng.standard_normal(size=X.shape)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                s = float(rng.uniform(2.0, 3.5))
                X = _hybrid(X, 700, s, 1400, 0.05, 0.003, 0.08, 0.003)
                v = _max_cos(X)
                if v < best_v + 0.006 and _time.time() < EXPLORE_END:
                    dl = min(_time.time() + 1.0, EXPLORE_END)
                    Xp, vp = _lp_polish(X, dl)
                    if vp < v:
                        X, v = Xp, vp
                if v < best_v:
                    best_v = v; best_x = X.copy()
                hopf_best.append((v, X.copy()))
                if _time.time() > t0 + 16.0:
                    break
            if _time.time() > t0 + 16.0:
                break
    except Exception:
        pass

    # ---- Phase B: diverse basin hopping ----
    it = 0
    while _time.time() < EXPLORE_END:
        it += 1
        r = float(rng.random())
        base = best_x
        try:
            if r < 0.20:
                # pure random Riesz+softmax restart (new basin)
                x0 = _random_sphere(n, d, rng)
                s = float(rng.uniform(1.5, 4.0))
                sr = int(rng.integers(600, 1100)); ss = int(rng.integers(1200, 2000))
                tau0 = float(rng.uniform(0.03, 0.07)); tau1 = float(rng.uniform(0.002, 0.005))
                x1 = _hybrid(x0, sr, s, ss, tau0, tau1, 0.09, 0.003)
            elif r < 0.32:
                # fresh Hopf sample + relax (structured basin)
                F, K = (9, 17) if (it % 2 == 0) else (17, 9)
                thetas = np.arccos(np.clip(rng.uniform(-1, 1, size=F), -1, 1))
                phis = rng.uniform(0, 2 * np.pi, size=F)
                offs = rng.uniform(0, 2 * np.pi, size=F)
                x0 = _hopf_points(thetas, phis, [K] * F, offs)
                x0 = x0 + 0.02 * rng.standard_normal(size=x0.shape)
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                s = float(rng.uniform(2.0, 3.5))
                x1 = _hybrid(x0, 600, s, 1300, 0.05, 0.003, 0.08, 0.003)
            elif r < 0.55:
                # coherent cluster rigid-body kick (coordinated move)
                x0 = base.copy()
                k = int(rng.integers(6, 26))
                cc = rng.standard_normal(size=4); cc /= np.linalg.norm(cc)
                dots = x0 @ cc
                idx = np.argsort(-dots)[:k]
                A = rng.standard_normal(size=(4, 4)); A = A - A.T
                an = float(np.linalg.norm(A))
                if an > 0:
                    A /= an
                eps = float(rng.uniform(0.25, 0.65))
                disp = x0[idx] @ A.T
                x0[idx] = x0[idx] + eps * disp
                # small global jitter
                x0 = x0 + 0.008 * rng.standard_normal(size=x0.shape)
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                s = float(rng.uniform(2.0, 4.0))
                x1 = _hybrid(x0, 500, s, 1300, 0.045, 0.0028, 0.08, 0.003)
            elif r < 0.70:
                # hole relocation of worst points
                x0 = base.copy()
                rm, _ = _row_max(x0)
                order = np.argsort(-rm)
                nmove = 1 if rng.random() < 0.6 else 2
                for q in range(nmove):
                    hole = _find_hole(x0, rng, nsample=12000, refine=40)
                    # add jitter so not exactly at hole
                    hole = hole + 0.02 * rng.standard_normal(size=4)
                    hole /= np.linalg.norm(hole)
                    # replace q-th worst (recompute order once is fine)
                    x0[order[q]] = hole
                # jitter neighbors slightly to accommodate
                x0 = x0 + 0.01 * rng.standard_normal(size=x0.shape)
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                x1 = _hybrid(x0, 400, 3.0, 1400, 0.04, 0.0025, 0.08, 0.003)
            elif r < 0.85:
                # large isotropic kick + softmax (topology change)
                sigma = float(rng.uniform(0.22, 0.50))
                x0 = base + sigma * rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                # re-randomize a few points sometimes
                if rng.random() < 0.35:
                    kk = int(rng.integers(2, 6))
                    ridx = rng.choice(n, size=kk, replace=False)
                    rp = rng.standard_normal(size=(kk, d))
                    rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                    x0[ridx] = rp
                x1 = _softmax_opt(x0, 2200, 0.05, 0.003, 0.09, 0.003)
            else:
                # small local refine
                sigma = float(rng.uniform(0.015, 0.06))
                x0 = base + sigma * rng.standard_normal(size=(n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                x1 = _softmax_opt(x0, 1600, 0.025, 0.0025, 0.06, 0.003)
        except Exception:
            continue
        try:
            v1 = _max_cos(x1)
        except Exception:
            continue
        # short LP if promising (allow slightly worse to tunnel)
        if v1 < best_v + 0.006 and _time.time() < EXPLORE_END - 0.5:
            try:
                dl = min(_time.time() + 0.9, EXPLORE_END)
                if dl > _time.time() + 0.25:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
            except Exception:
                pass
        if v1 < best_v:
            best_x = x1.copy(); best_v = v1
        # time guard
        if _time.time() >= EXPLORE_END:
            break

    # ---- final thorough polish ----
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 0.5)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
