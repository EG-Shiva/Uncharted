import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(100, 100, 150)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _hopf_build(thetas, counts, pattern, twist, ph1, ph2, rng, jitter=0.0):
    pts = []
    for th, nk, t_, p1, p2 in zip(thetas, counts, twist if isinstance(twist, list) else [twist]*len(thetas),
                                  ph1 if isinstance(ph1, list) else [ph1]*len(thetas),
                                  ph2 if isinstance(ph2, list) else [ph2]*len(thetas)):
        if nk <= 0:
            continue
        if pattern == 'square':
            n1 = int(np.ceil(np.sqrt(nk)))
            n2 = int(np.ceil(nk / n1))
            us, vs = [], []
            for a in range(n1):
                for b in range(n2):
                    if len(us) >= nk:
                        break
                    us.append((a + 0.5) / n1)
                    vs.append((b + 0.5) / n2)
            us = np.array(us); vs = np.array(vs)
        elif pattern == 'tri':
            nrow = int(np.ceil(np.sqrt(nk * 0.8660254)))
            if nrow < 1:
                nrow = 1
            us, vs = [], []
            r = 0
            while len(us) < nk:
                ncol = int(np.ceil(nk / nrow)) + 1
                for c in range(ncol):
                    if len(us) >= nk:
                        break
                    u = (c + 0.5 * (r % 2)) / ncol
                    vv = (r + 0.5) / max(nrow, 1)
                    us.append(u % 1.0); vs.append(vv % 1.0)
                r += 1
                if r > 2 * nrow + 5:
                    break
            us = np.array(us[:nk]); vs = np.array(vs[:nk])
        else:  # twisted
            n1 = int(np.ceil(np.sqrt(nk)))
            n2 = int(np.ceil(nk / n1))
            us, vs = [], []
            for a in range(n1):
                shift = t_ * a / max(n1, 1)
                for b in range(n2):
                    if len(us) >= nk:
                        break
                    us.append(((a + 0.5) / n1 + shift) % 1.0)
                    vs.append((b + 0.5) / n2)
            us = np.array(us); vs = np.array(vs)
        if jitter > 0:
            us = (us + rng.randn(len(us)) * jitter) % 1.0
            vs = (vs + rng.randn(len(vs)) * jitter) % 1.0
        f1 = 2 * np.pi * us + p1
        f2 = 2 * np.pi * vs + p2
        ct, st = np.cos(th), np.sin(th)
        x = np.stack([ct * np.cos(f1), ct * np.sin(f1), st * np.cos(f2), st * np.sin(f2)], axis=1)
        pts.append(x)
    if len(pts) == 0:
        return np.zeros((0, 4))
    return np.vstack(pts)

def _alloc_counts(n, K, rng):
    w = rng.rand(K) + 0.5
    w = w / w.sum() * n
    c = np.floor(w).astype(int)
    c[c < 1] = 1
    while c.sum() > n:
        i = int(np.argmax(c))
        c[i] -= 1
    while c.sum() < n:
        i = int(rng.randint(K))
        c[i] += 1
    return list(c)

def _gen_pool(rng, cap=6000):
    pts = []
    R = 4
    rng_vals = np.arange(-R, R + 1)
    # integer D4-ish: sum even? include all but emphasize D4
    for a in rng_vals:
        for b in rng_vals:
            for cc in rng_vals:
                for dd in rng_vals:
                    if a == 0 and b == 0 and cc == 0 and dd == 0:
                        continue
                    s = a + b + cc + dd
                    # keep D4 (even sum) plus some others sparsely
                    if (s % 2 != 0) and rng.rand() < 0.7:
                        continue
                    n2 = a*a + b*b + cc*cc + dd*dd
                    if n2 > 32 or n2 == 0:
                        continue
                    pts.append([a, b, cc, dd])
    pts = np.array(pts, dtype=float)
    # half-integer Hurwitz
    h = []
    hr = np.arange(-R+0.5, R+1, 1.0)
    # subsample half-integer shell to limit size
    for _ in range(4000):
        v = rng.randn(4)
        v = np.round(v * 2) / 2.0
        # force half-integer (all .5)
        v = np.floor(v) + 0.5
        if np.sum(v*v) > 32 or np.sum(v*v) == 0:
            continue
        h.append(v)
    if len(h) > 0:
        h = np.array(h, dtype=float)
        pts = np.vstack([pts, h])
    # normalize and deduplicate
    pts /= np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
    # dedup by rounding
    r = np.round(pts, decimals=5)
    _, idx = np.unique(r, axis=0, return_index=True)
    pts = pts[idx]
    # add random jittered shells for diversity
    jr = rng.randn(min(1500, len(pts)), 4) * 0.03 + pts[rng.choice(len(pts), size=min(1500, len(pts)), replace=True)]
    jr /= np.linalg.norm(jr, axis=1, keepdims=True)
    pts = np.vstack([pts, jr])
    r2 = np.round(pts, decimals=5)
    _, idx2 = np.unique(r2, axis=0, return_index=True)
    pts = pts[idx2]
    if len(pts) > cap:
        sel = rng.choice(len(pts), size=cap, replace=False)
        pts = pts[sel]
    return pts

def _greedy_select(pool, n, rng):
    M = pool.shape[0]
    # start from pair with minimal dot
    i0 = int(rng.randint(M))
    d0 = pool @ pool[i0]
    i1 = int(np.argmin(d0))
    selected = [i0, i1]
    in_sel = np.zeros(M, dtype=bool)
    in_sel[i0] = True; in_sel[i1] = True
    best_dot = np.maximum(pool @ pool[i0], pool @ pool[i1])
    for _ in range(n - 2):
        # pick unselected with minimal best_dot
        cand = np.where(~in_sel)[0]
        j = cand[int(np.argmin(best_dot[cand]))]
        selected.append(int(j))
        in_sel[j] = True
        nd = pool @ pool[j]
        best_dot = np.maximum(best_dot, nd)
        best_dot[in_sel] = 1e9
    return pool[np.array(selected)]

def _swap_improve(pool, cur_sel, rng, iters=400):
    n = cur_sel.shape[0]
    G = cur_sel @ cur_sel.T
    np.fill_diagonal(G, -2.0)
    cur = float(G.max())
    # precompute pool dots to sel
    for it in range(iters):
        # worst pair
        iu = np.argmax(G)
        i = iu // n; j = iu % n
        pick = i if rng.rand() < 0.5 else j
        # mask remaining max without pick
        mask = np.ones(n, dtype=bool); mask[pick] = False
        Grest = (cur_sel[mask] @ cur_sel[mask].T)
        np.fill_diagonal(Grest, -2.0)
        rest_max = float(Grest.max()) if Grest.size > 0 else -2.0
        if rest_max >= cur:
            pass
        # try random pool candidates
        best_c = None; best_v = cur
        for _ in range(60):
            ci = int(rng.randint(pool.shape[0]))
            pc = pool[ci]
            # skip if duplicate-ish
            d = float(np.max(cur_sel[mask] @ pc))
            nm = d if d > rest_max else rest_max
            if nm < best_v:
                best_v = nm
                best_c = pc.copy()
                if best_v < cur - 1e-4:
                    break
        if best_c is not None and best_v < cur:
            cur_sel[pick] = best_c
            G = cur_sel @ cur_sel.T
            np.fill_diagonal(G, -2.0)
            cur = float(G.max())
    return cur_sel, cur

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_HOPF_END = t0 + 55.0
    SEARCH_LAT_END = t0 + 85.0
    FINAL_START = t0 + 85.0
    rng = np.random.RandomState(777)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            vv = _max_cos(a)
            if vv < best_val:
                best_val = vv
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    elite_x = best_x.copy()
    elite_v = best_val
    # ---- Phase A: Hopf / Clifford torus structured search ----
    patterns = ['square', 'tri', 'twisted']
    trial = 0
    while time.time() < SEARCH_HOPF_END:
        trial += 1
        K = int([3, 4, 5, 6, 7, 8, 2, 5][trial % 8])
        pat = patterns[trial % 3]
        counts = _alloc_counts(n, K, rng)
        # torus positions: symmetric around pi/4 with jitter
        base = np.linspace(0.18, np.pi/2 - 0.18, K)
        thetas = base + rng.randn(K) * 0.06
        thetas = np.clip(thetas, 0.08, np.pi/2 - 0.08)
        # with prob, force one torus at pi/4
        if rng.rand() < 0.4:
            thetas[int(rng.randint(K))] = np.pi/4 + rng.randn() * 0.02
        twist = list(rng.rand(K) * 1.0)
        ph1 = list(rng.rand(K) * 2 * np.pi)
        ph2 = list(rng.rand(K) * 2 * np.pi)
        try:
            x0 = _hopf_build(thetas, counts, pat, twist, ph1, ph2, rng, jitter=0.01)
            if x0.shape[0] != n:
                continue
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        except Exception:
            continue
        try:
            xb, vb = _gd(x0, steps=1400, T0=0.012, T1=0.0009, lr0=0.04, mu=0.85, rng=rng, noise0=0.002, deadline=SEARCH_HOPF_END)
        except Exception:
            continue
        if vb < elite_v:
            # medium polish to confirm basin
            try:
                xb2, vb2 = _gd(xb, steps=1200, T0=0.004, T1=0.0004, lr0=0.03, mu=0.88, rng=rng, noise0=0.0, deadline=SEARCH_HOPF_END)
                if vb2 < vb:
                    xb, vb = xb2, vb2
            except Exception:
                pass
            if vb < elite_v - 1e-12:
                elite_v, elite_x = vb, xb.copy()
                if vb < best_val - 1e-12:
                    best_val, best_x = vb, xb.copy()
        # early exit if big win
        if elite_v < 0.8555:
            break
    # ---- Phase B: Hurwitz / D4 lattice pool + greedy maximin ----
    try:
        pool = _gen_pool(rng, cap=5000)
        tries = 0
        while time.time() < SEARCH_LAT_END and tries < 6:
            tries += 1
            try:
                sel = _greedy_select(pool, n, rng)
                sel, _ = _swap_improve(pool, sel, rng, iters=120)
                sel = sel / np.linalg.norm(sel, axis=1, keepdims=True)
                # release to continuous opt
                xb, vb = _gd(sel + rng.randn(n, 4) * 0.004, steps=1600, T0=0.010, T1=0.0007, lr0=0.038, mu=0.86, rng=rng, noise0=0.001, deadline=SEARCH_LAT_END)
                # renormalize inside gd already
                if vb < elite_v:
                    elite_v, elite_x = vb, xb.copy()
                if vb < best_val - 1e-12:
                    best_val, best_x = vb, xb.copy()
            except Exception:
                continue
    except Exception:
        pass
    # choose starting point for final polish: best of elite and incumbent
    start = elite_x if elite_v <= best_val else best_x
    start = start / np.maximum(np.linalg.norm(start, axis=1, keepdims=True), 1e-12)
    # ---- Phase C: intensive polish ----
    try:
        # annealed heavy-ball
        if time.time() < DEADLINE - 45:
            start, sv = _gd(start, steps=5000, T0=0.009, T1=0.00045, lr0=0.035, mu=0.88, rng=rng, noise0=0.001, deadline=DEADLINE - 35)
            if sv < best_val:
                best_val, best_x = sv, start.copy()
        if time.time() < DEADLINE - 30:
            try:
                sm = _lbfgs_smooth(start, deadline=min(time.time() + 12.0, DEADLINE - 18), taus=(0.003, 0.001, 0.0005), iters=(120, 120, 160))
                vv = _max_cos(sm)
                if vv < best_val:
                    best_val, best_x = vv, sm.copy()
                start = sm
            except Exception:
                pass
        if time.time() < DEADLINE - 12:
            try:
                pol, pv = _lp_polish(start, deadline=min(time.time() + 10.0, DEADLINE - 2.0))
                if pv < best_val:
                    best_val, best_x = pv, pol.copy()
            except Exception:
                pass
        # final short gd if LP moved
        if time.time() < DEADLINE - 4:
            try:
                f2 = _lbfgs_smooth(best_x, deadline=DEADLINE - 1.0, taus=(0.0005,), iters=(120,))
                vv2 = _max_cos(f2)
                if vv2 < best_val:
                    best_val, best_x = vv2, f2.copy()
            except Exception:
                pass
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # strict finiteness guard
    best_x[~np.isfinite(best_x)] = 0.25
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
