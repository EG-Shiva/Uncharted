import time
import numpy as np
from scipy.optimize import linprog
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
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
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

def _hopf_points(theta, p1, p2):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * np.cos(p1), c * np.sin(p1), s * np.cos(p2), s * np.sin(p2)])

def _make_torus_code(rng, n=163):
    K = int(rng.randint(3, 8))  # 3..7 tori
    # torus positions: symmetric around pi/4 plus jitter
    base = np.linspace(0.22, 1.35, K)
    thetas = base + rng.randn(K) * 0.08
    thetas = np.clip(thetas, 0.12, 1.45)
    # random allocations summing to n, at least 6 each
    w = rng.rand(K) + 0.4
    w = w / w.sum()
    alloc = np.maximum(6, np.round(w * n).astype(int))
    # fix sum
    while alloc.sum() != n:
        if alloc.sum() > n:
            i = int(np.argmax(alloc))
            alloc[i] -= 1
        else:
            i = int(rng.randint(0, K))
            alloc[i] += 1
    pts = []
    for k in range(K):
        th = float(thetas[k])
        nk = int(alloc[k])
        c, s = np.cos(th), np.sin(th)
        c = max(c, 0.08); s = max(s, 0.08)
        # grid dims matched to torus aspect
        a = max(1, int(round(np.sqrt(nk * c / max(s, 1e-6)))))
        b = max(1, int(round(np.sqrt(nk * s / max(c, 1e-6)))))
        a = max(a, 1); b = max(b, 1)
        while a * b < nk:
            if a * c < b * s:
                a += 1
            else:
                b += 1
        pat = rng.choice(['square', 'tri', 'twist'])
        twist = float(rng.rand() * 2 * np.pi) if pat == 'twist' else 0.0
        shear = float(rng.randn() * 0.6) if pat == 'twist' else 0.0
        ph1 = float(rng.rand() * 2 * np.pi)
        ph2 = float(rng.rand() * 2 * np.pi)
        grid = []
        for i in range(a):
            for j in range(b):
                if pat == 'tri' and (j % 2 == 1):
                    f1 = (i + 0.5) / a
                else:
                    f1 = i / a
                f2 = j / b
                p1 = 2 * np.pi * f1 + ph1 + shear * f2 * 2 * np.pi + twist * 0.0
                p2 = 2 * np.pi * f2 + ph2 + twist * (i / max(a, 1))
                grid.append((p1, p2))
        grid = np.array(grid)
        # if too many, greedy thin to nk
        if len(grid) > nk:
            # take evenly spaced subset
            idx = np.linspace(0, len(grid) - 1, nk).round().astype(int)
            grid = grid[idx]
        # jitter
        grid += rng.randn(*grid.shape) * 0.02
        for (p1, p2) in grid[:nk]:
            pts.append(_hopf_points(th + rng.randn() * 0.008, float(p1), float(p2)))
    pts = np.array(pts[:n])
    # global random rotation to avoid axis bias
    A = rng.randn(4, 4)
    Q, _ = np.linalg.qr(A)
    pts = pts @ Q.T
    pts /= np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
    return pts

def _build_hurwitz_pool():
    vals_i = np.arange(-3, 4, dtype=float)
    vals_h = np.arange(-3, 4, dtype=float) + 0.5
    pts = []
    # integer quaternions
    for a in vals_i:
        for b in vals_i:
            for c in vals_i:
                for d in vals_i:
                    n2 = a * a + b * b + c * c + d * d
                    if n2 < 0.5 or n2 > 20:
                        continue
                    pts.append((a, b, c, d))
    for a in vals_h:
        for b in vals_h:
            for c in vals_h:
                for d in vals_h:
                    n2 = a * a + b * b + c * c + d * d
                    if n2 < 0.5 or n2 > 20:
                        continue
                    pts.append((a, b, c, d))
    P = np.array(pts, dtype=float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # dedup by rounding
    r = np.round(P, 6)
    _, idx = np.unique(r, axis=0, return_index=True)
    P = P[np.sort(idx)]
    return P

def _greedy_select(pool, rng, n=163, n_try_seed=1):
    P = pool.shape[0]
    best_sel = None
    best_val = 1e9
    seeds = [int(rng.randint(0, P)) for _ in range(n_try_seed)]
    for seed in seeds:
        selected = [seed]
        sel_mask = np.zeros(P, dtype=bool)
        sel_mask[seed] = True
        cur = pool @ pool[seed]
        cur[seed] = 2.0
        for _ in range(1, n):
            # pick unselected with minimal cur
            # argmin over unselected
            masked = np.where(~sel_mask, cur, np.inf)
            j = int(np.argmin(masked))
            selected.append(j)
            sel_mask[j] = True
            dj = pool @ pool[j]
            cur = np.maximum(cur, dj)
            cur[j] = 2.0
        sel = np.array(selected, dtype=int)
        X = pool[sel]
        v = _max_cos(X / np.linalg.norm(X, axis=1, keepdims=True))
        if v < best_val:
            best_val = v
            best_sel = sel
    return best_sel, best_val

def _swap_improve(pool, sel, iters=30, rng=None):
    sel = sel.copy()
    n = len(sel)
    P = pool.shape[0]
    X = pool[sel].copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cur_val = _max_cos(X)
    in_set = np.zeros(P, dtype=bool)
    in_set[sel] = True
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # worst point = involved in max pair most? use max per row
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        improved = False
        for oi in order[:8]:
            # score each outside candidate vs X without oi
            rest = np.delete(X, oi, axis=0)
            # dots of candidates to rest: max dot
            D = pool @ rest.T
            cand_score = D.max(axis=1)
            cand_score[in_set] = np.inf
            j = int(np.argmin(cand_score))
            if cand_score[j] < rowmax[oi] - 1e-12:
                # do swap
                in_set[sel[oi]] = False
                sel[oi] = j
                in_set[j] = True
                X[oi] = pool[j] / max(np.linalg.norm(pool[j]), 1e-12)
                cur_val = _max_cos(X)
                improved = True
                break
        if not improved:
            break
    return sel, cur_val

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    elites = [(best_val, best_x.copy())]

    # Build Hurwitz pool once
    try:
        pool = _build_hurwitz_pool()
        # add random uniform points to pool for diversity
        R = np.random.RandomState(777)
        rnd = R.randn(1500, 4)
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        pool = np.vstack([pool, rnd])
        # dedup roughly
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        have_pool = True
    except Exception:
        have_pool = False
        pool = None

    trial = 0
    # alternate torus-from-scratch and lattice-greedy trials
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 45:
            break
        trial += 1
        mode = trial % 4
        try:
            if mode in (0, 1, 3) or (not have_pool):
                cur0 = _make_torus_code(rng, n=n)
                # short annealed softmax to evaluate basin
                cur_b, cur_v = _gd(cur0, steps=2200, T0=0.014, T1=0.0007, lr0=0.040, mu=0.86,
                                   rng=rng, noise0=0.002, deadline=SEARCH_END)
            else:
                # Hurwitz greedy trial
                sel, _gv = _greedy_select(pool, rng, n=n, n_try_seed=1)
                try:
                    sel2, _sv = _swap_improve(pool, sel, iters=12, rng=rng)
                    sel = sel2
                except Exception:
                    pass
                cur0 = pool[sel].copy()
                cur0 /= np.linalg.norm(cur0, axis=1, keepdims=True)
                cur0 += rng.randn(n, 4) * 0.003
                cur0 /= np.linalg.norm(cur0, axis=1, keepdims=True)
                cur_b, cur_v = _gd(cur0, steps=2200, T0=0.012, T1=0.0006, lr0=0.038, mu=0.87,
                                   rng=rng, noise0=0.0015, deadline=SEARCH_END)
        except Exception:
            continue
        # keep elites
        if cur_v < best_val + 0.006:
            # quick LP confirm for promising
            if cur_v < best_val + 0.001 and time.time() < SEARCH_END - 0.5:
                try:
                    cur_b2, cur_v2 = _lp_polish(cur_b, deadline=min(SEARCH_END, time.time() + 1.2), r0=8e-4)
                    if cur_v2 < cur_v:
                        cur_b, cur_v = cur_b2, cur_v2
                except Exception:
                    pass
            elites.append((cur_v, cur_b.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:6]
            if cur_v < best_val:
                best_val, best_x = cur_v, cur_b.copy()
        # occasional shake of incumbent to jump basins (torus twist / cap)
        if trial % 6 == 0 and time.time() < SEARCH_END - 2:
            try:
                base = elites[0][1]
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = base @ u
                q = 0.80 + 0.12 * rng.rand()
                thr = np.quantile(dots, q)
                idx = np.where(dots > thr)[0]
                if len(idx) < 4:
                    idx = np.argsort(dots)[-10:]
                A = rng.randn(4, 4) * 0.25
                A = A - A.T
                Q, _ = np.linalg.qr(np.eye(4) + A)
                cur = base.copy()
                cur[idx] = (Q @ cur[idx].T).T
                cur += rng.randn(n, 4) * 0.005
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                cb, cv = _gd(cur, steps=2000, T0=0.012, T1=0.0006, lr0=0.038, mu=0.86,
                             rng=rng, noise0=0.001, deadline=SEARCH_END)
                if cv < best_val + 0.006:
                    elites.append((cv, cb.copy()))
                    elites = sorted(elites, key=lambda e: e[0])[:6]
                    if cv < best_val:
                        best_val, best_x = cv, cb.copy()
            except Exception:
                pass

    # Final intensive polish of top elites + incumbent
    elites = sorted(elites, key=lambda e: e[0])[:4]
    # longer GD on each elite
    for ei in range(len(elites)):
        if time.time() > DEADLINE - 14:
            break
        v, X = elites[ei]
        try:
            Xb, Xv = _gd(X, steps=4500, T0=0.008, T1=0.00025, lr0=0.030, mu=0.89,
                         rng=rng, noise0=0.0, deadline=DEADLINE - 12)
            if Xv < v:
                v, X = Xv, Xb
                elites[ei] = (v, X.copy())
                if v < best_val:
                    best_val, best_x = v, X.copy()
        except Exception:
            continue
    # trust-region LP on the best
    try:
        Xb, Xv = _lp_polish(best_x, deadline=DEADLINE - 0.5, r0=1e-3)
        if Xv < best_val:
            best_val, best_x = Xv, Xb
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # strict feasibility: tiny shrink not needed (unit rows already), ensure finite
    best_x = np.where(np.isfinite(best_x), best_x, rng.randn(n, 4))
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
