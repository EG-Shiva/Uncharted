import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    cnt = C.sum(axis=1)
    imax = G.max(axis=1)
    return cnt, imax, m, G

def _adam_quench_mask(X, movable, steps, beta0, beta1, lr0, lr1, per_row=True, t_lim=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    mov = movable.astype(bool)
    any_frozen = (not np.all(mov))
    for t in range(1, steps + 1):
        if t_lim is not None and (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if per_row:
            S = W.sum(axis=1, keepdims=True) + 1e-18
            Grad = (W @ X) / S
        else:
            Z = W.sum() + 1e-18
            Grad = (W @ X) / Z * n
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        if any_frozen:
            X[mov] = X[mov] - lr * step[mov]
            nn = np.sqrt((X[mov]*X[mov]).sum(axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            X[mov] /= nn
        else:
            X = X - lr * step
            X /= np.sqrt((X * X).sum(axis=1, keepdims=True))
    return X

def _power_quench_mask(X, movable, steps, p0, p1, lr0, lr1, t_lim=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    mov = movable.astype(bool)
    for t in range(1, steps + 1):
        if t_lim is not None and (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        p = p0 + (p1 - p0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, 0.0)
        np.maximum(G, 0.0, out=G)
        with np.errstate(over='ignore', under='ignore'):
            W = np.power(G, p - 1.0, out=np.zeros_like(G), where=G > 1e-12)
        W *= p
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) / max(n * 0.5, 1.0)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        if np.any(mov):
            X[mov] = X[mov] - lr * step[mov]
            nn = np.sqrt((X[mov]*X[mov]).sum(axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            X[mov] /= nn
    return X

def _refine_hole(Y, p, iters=200, beta=90.0, lr0=0.15, lr1=0.004):
    p = p / (np.linalg.norm(p) + 1e-18)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-18
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _deepest_hole_insert(Cur, rng, n_probe=50000, beta_ref=90.0):
    d = Cur.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 5000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Cur.T
        bs[a:b] = D.max(axis=1)
    idx = int(np.argmin(bs))
    p0 = P[idx]
    h = _refine_hole(Cur, p0, iters=200, beta=beta_ref, lr0=0.15, lr1=0.004)
    return h, float(bs[idx])

def solve():
    t0 = time.time()
    BUDGET = 174.0
    t_hop_end = t0 + 148.0
    n, d = 168, 8
    rng = np.random.RandomState(20261103)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    base_best = best.copy()

    it = 0
    k_choices = [32, 48, 64]
    while time.time() < t_hop_end:
        if time.time() + 4.0 > t0 + BUDGET:
            break
        it += 1
        rmode = rng.rand()
        # 75% block rattler+hub re-solve, 25% global kick
        if rmode < 0.75:
            X0 = best.copy()
            cnt, imax, mcur, Gcur = _contacts(X0, tol=2e-7)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt)[:16]
            # hub pool degree>=8 else top
            hub_pool = np.where(cnt >= 8)[0]
            if len(hub_pool) < 8:
                hub_pool = np.argsort(cnt)[-16:]
            n_pick = int(8 + rng.randint(9))  # 8..16
            n_pick = min(n_pick, len(hub_pool))
            pick = rng.choice(hub_pool, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
            k_target = k_choices[rng.randint(len(k_choices))]
            victims = set(int(v) for v in ratt) | set(int(v) for v in pick)
            # expand with nearest neighbours of hubs to reach k_target
            if len(victims) < k_target:
                # order candidates by max dot to picked hubs
                cand_dots = []
                Gfull = X0 @ X0.T
                for h in list(pick):
                    row = Gfull[h].copy()
                    row[list(victims)] = -2.0
                    row[h] = -2.0
                    order = np.argsort(row)[::-1]
                    for c in order[:12]:
                        cand_dots.append((row[c], int(c)))
                cand_dots.sort(reverse=True)
                for _, c in cand_dots:
                    if len(victims) >= k_target:
                        break
                    victims.add(int(c))
                # if still short, add worst points
                if len(victims) < k_target:
                    order_w = np.argsort(imax)[::-1]
                    for idx in order_w:
                        if int(idx) not in victims:
                            victims.add(int(idx))
                        if len(victims) >= k_target:
                            break
            victims = sorted(victims)
            k = len(victims)
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            core_idx = np.where(keep_mask)[0]
            Ycore = X0[keep_mask].copy()
            # sequential reinsertion from 50000 probes
            Cur = Ycore.copy()
            new_pts = []
            beta_ref = 70.0 + rng.rand() * 60.0
            # random insertion order to diversify
            order_ins = rng.permutation(k)
            tmp_new = [None] * k
            for ii in range(k):
                # adapt n_probe slightly for speed on large k: keep 50000 as specified
                h, _ = _deepest_hole_insert(Cur, rng, n_probe=50000, beta_ref=beta_ref)
                tmp_new[ii] = h
                Cur = np.vstack([Cur, h[None, :]])
            # assign to victim positions in permuted order
            X = X0.copy()
            for vi, h in zip([victims[o] for o in order_ins], tmp_new):
                # order_ins already perm; simpler: map sorted victims to tmp in order
                X[vi] = h
            # actually above zip mismatched; redo consistently:
            # victims sorted, tmp_new in insertion order; assign sequentially
            for vi, h in zip(victims, tmp_new):
                X[vi] = h
            # boundary: core points with dot>0.3 to any inserted
            Ins = X[victims]
            Dcross = X0[keep_mask] @ Ins.T
            bmax = Dcross.max(axis=1)
            boundary_core = core_idx[bmax > 0.3]
            movable = np.zeros(n, dtype=bool)
            movable[victims] = True
            movable[boundary_core] = True
            # stage 1: frozen deep core for 1000 steps (boundary+inserted only)
            per_row = bool(rng.rand() < 0.7)
            beta0 = 25.0 + rng.rand() * 20.0
            beta1 = 250.0 + rng.rand() * 250.0
            lr0 = 0.004 + rng.rand() * 0.003
            lr1 = lr0 * 0.12
            # small power pre-relax on movable only (200 steps)
            if rng.rand() < 0.5:
                X = _power_quench_mask(X, movable, 200, 6.0, 22.0, 0.012, 0.003, t_lim=t_hop_end)
            X = _adam_quench_mask(X, movable, 1000, beta0, beta1, lr0, lr1, per_row=per_row, t_lim=t_hop_end)
            # stage 2: full thaw
            allmov = np.ones(n, dtype=bool)
            beta0b = 60.0 + rng.rand() * 40.0
            beta1b = 400.0 + rng.rand() * 300.0
            lr0b = 0.003 + rng.rand() * 0.003
            X = _adam_quench_mask(X, allmov, 700 + int(rng.randint(400)), beta0b, beta1b, lr0b, lr0b * 0.12, per_row=True, t_lim=t_hop_end)
            if not np.all(np.isfinite(X)):
                continue
            vq = _max_cos(X)
            # LP polish only if promising
            if vq < best_val + 2e-4 and time.time() + 6.0 < t0 + BUDGET:
                Xp = _lp_polish(X, seconds=1.2)
                vp = _max_cos(Xp)
                if vp < vq:
                    X, vq = Xp, vp
            if vq + 1e-12 < best_val:
                best = X / np.linalg.norm(X, axis=1, keepdims=True)
                best_val = float(vq)
        else:
            # global kick + power then softmax (diversity)
            X = best.copy()
            sigma = [0.008, 0.02, 0.04][rng.randint(3)]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            allmov = np.ones(n, dtype=bool)
            X = _power_quench_mask(X, allmov, 300 + int(rng.randint(250)), 5.0, 24.0, 0.015, 0.003, t_lim=t_hop_end)
            X = _adam_quench_mask(X, allmov, 900 + int(rng.randint(500)), 30.0 + rng.rand()*30.0, 300.0 + rng.rand()*300.0, 0.005, 0.0005, per_row=True, t_lim=t_hop_end)
            if not np.all(np.isfinite(X)):
                continue
            vq = _max_cos(X)
            if vq < best_val + 2e-4 and time.time() + 6.0 < t0 + BUDGET:
                Xp = _lp_polish(X, seconds=1.0)
                vp = _max_cos(Xp)
                if vp < vq:
                    X, vq = Xp, vp
            if vq + 1e-12 < best_val:
                best = X / np.linalg.norm(X, axis=1, keepdims=True)
                best_val = float(vq)
    # final LP polish on best with remaining time
    remain = (t0 + BUDGET) - time.time() - 1.0
    if remain > 2.0:
        try:
            best = _lp_polish(best, seconds=min(remain, 22.0))
            best_val = _max_cos(best)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
