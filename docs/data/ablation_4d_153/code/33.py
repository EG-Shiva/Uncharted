import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    r = np.max(G, axis=1)
    return r, float(np.max(r))


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
        rmax = np.max(G, axis=1, keepdims=True)
        # per-row softmax: each point repelled by its own neighbours
        W = np.exp((G - rmax) / tau)
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
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _find_hole(X, rng, S=6000):
    # largest empty cap direction w.r.t set X (m,4)
    R = rng.standard_normal(size=(S, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ X.T
    s = np.max(D, axis=1)
    return R[int(np.argmin(s))]


def _peel_reinsert(base, rng):
    n, d = base.shape
    rmax, _ = _row_max(base)
    order = np.argsort(rmax)  # ascending: loose first, worst last
    k = int(rng.integers(4, 9))  # 4..8
    # remove 2 worst + k-2 others (mix of loose and random)
    worst = order[-2:].tolist()
    cands = order[: n // 2].tolist()
    extra = []
    while len(extra) < k - 2:
        if rng.random() < 0.7 and len(cands) > 0:
            j = int(rng.integers(len(cands)))
            extra.append(cands.pop(j))
        else:
            j = int(rng.integers(n))
            if j not in worst and j not in extra:
                extra.append(j)
    remove = np.array(worst + extra[: max(0, k - 2)], dtype=int)
    remove = np.unique(remove)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove] = False
    core = base[keep_mask].copy()
    # relax core briefly with more room
    core = _smooth_opt(core, 450, 0.03, 0.004, 0.07, 0.005, mom=0.88)
    # rebuild full set, reinsert holes one by one
    y = base.copy()
    y[keep_mask] = core
    # for hole computation use current full set without the slot
    for idx in rng.permutation(remove):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        h = _find_hole(y[mask], rng, S=6000)
        # small jitter to break symmetry
        h = h + 0.01 * rng.standard_normal(size=(4,))
        h /= np.linalg.norm(h)
        y[idx] = h
    return y


def _cap_shift(base, rng):
    n, d = base.shape
    u = rng.standard_normal(size=(4,))
    u /= np.linalg.norm(u)
    dots = base @ u
    s = int(rng.integers(8, 26))  # cap size
    s = min(s, n - 1)
    thresh = np.sort(dots)[n - s]
    idx = np.where(dots >= thresh)[0]
    sigma = float(rng.choice([0.08, 0.15, 0.25, 0.35]))
    shift = rng.standard_normal(size=(4,)) * sigma
    # make shift roughly tangent to cap centre
    shift = shift - np.dot(shift, u) * u
    y = base.copy()
    y[idx] = y[idx] + shift
    # small individual jitter preserves disorder at boundary
    y[idx] += (sigma * 0.2) * rng.standard_normal(size=(len(idx), 4))
    y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
    return y


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
        I = iu[0][idx]
        J = iu[1][idx]
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


def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    EXPLORE_END = t0 + 148.0
    rng = np.random.default_rng(0)
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

    sigmas = [0.02, 0.05, 0.12, 0.25, 0.5]
    it = 0
    while _time.time() < EXPLORE_END:
        it += 1
        frac = it / 300.0
        Touter = 0.004 * (1.0 - min(1.0, frac)) + 0.0003
        # choose base: mostly walker, sometimes best
        if it % 7 == 0:
            base = best_x
        else:
            base = cur_x
        r = rng.random()
        if r < 0.32:
            x0 = _peel_reinsert(base, rng)
            steps, tau0, tau1, lr0, lr1 = 1100, 0.035, 0.003, 0.08, 0.003
        elif r < 0.58:
            x0 = _cap_shift(base, rng)
            # add tiny global jitter too
            x0 = x0 + 0.008 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, tau0, tau1, lr0, lr1 = 1000, 0.03, 0.0025, 0.07, 0.003
        elif r < 0.76:
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            if rng.random() < 0.3:
                kk = int(rng.integers(2, 8))
                idx = rng.choice(n, size=kk, replace=False)
                rp = rng.standard_normal(size=(kk, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x0[idx] = rp
            steps, tau0, tau1, lr0, lr1 = 1200, 0.04, 0.003, 0.08, 0.003
        elif r < 0.90:
            x0 = base.copy()
            kk = int(rng.integers(3, 11))
            idx = rng.choice(n, size=kk, replace=False)
            rp = rng.standard_normal(size=(kk, d))
            rp /= np.linalg.norm(rp, axis=1, keepdims=True)
            x0[idx] = rp
            # move rattlers/loose to holes as well
            if rng.random() < 0.5:
                rm, _ = _row_max(x0)
                loose = int(np.argmin(rm))
                # place loosest into deep hole
                mask = np.ones(n, dtype=bool)
                mask[loose] = False
                h = _find_hole(x0[mask], rng, S=4000)
                x0[loose] = h
            steps, tau0, tau1, lr0, lr1 = 1100, 0.04, 0.003, 0.08, 0.003
        else:
            x0 = _random_sphere(n, d, rng)
            steps, tau0, tau1, lr0, lr1 = 1600, 0.06, 0.004, 0.10, 0.004

        x1 = _smooth_opt(x0, steps, tau0, tau1, lr0, lr1, mom=0.88)
        v1 = _max_cos(x1)

        # short LP if promising (looser threshold to allow topology change to mature)
        if v1 < best_v + 0.003 and _time.time() < EXPLORE_END - 2.0:
            dl = min(_time.time() + 0.9, EXPLORE_END)
            if dl > _time.time() + 0.15:
                try:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
                except Exception:
                    pass

        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
            cur_x = x1.copy()
            cur_v = v1
        else:
            if v1 < cur_v:
                cur_x = x1.copy()
                cur_v = v1
            else:
                # Metropolis on basin level
                if rng.random() < np.exp(-(v1 - cur_v) / max(Touter, 1e-9)):
                    cur_x = x1.copy()
                    cur_v = v1
                elif rng.random() < 0.05:
                    cur_x = x1.copy()
                    cur_v = v1
            if cur_v > best_v + 0.025 and rng.random() < 0.35:
                cur_x = best_x.copy()
                cur_v = best_v
        if _time.time() >= EXPLORE_END:
            break

    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
