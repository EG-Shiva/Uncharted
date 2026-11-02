import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_DEADLINE = 165.0

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=5e-3, t0=None):
    if t0 is None:
        t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        now = time.time()
        rem = seconds - (now - t0)
        # need time for at least one linprog
        if rem < 0.5 or radius < 1e-14:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3500
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1)
        vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar
        cr[:, d:2 * d] = J[:, None] * d + ar
        cr[:, 2 * d] = n * d
        vr[:, :d] = x[J]
        vr[:, d:2 * d] = x[I]
        vr[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = float(min(8.0, max(0.5, rem - 0.2)))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_pool(rng):
    # all {-2,-1,0,1,2}^7
    N = 5 ** 7
    idx = np.arange(N, dtype=np.int64)
    V = np.empty((N, 7), dtype=np.float64)
    tmp = idx.copy()
    for k in range(7):
        V[:, k] = (tmp % 5) - 2
        tmp //= 5
    mask = np.any(V != 0, axis=1)
    V = V[mask]
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / nrm
    # half-integer odd numerators {-3,-1,1,3}^7 / norm
    M = 4 ** 7
    idx2 = np.arange(M, dtype=np.int64)
    W = np.empty((M, 7), dtype=np.float64)
    tmp2 = idx2.copy()
    lut = np.array([-3.0, -1.0, 1.0, 3.0])
    for k in range(7):
        W[:, k] = lut[tmp2 % 4]
        tmp2 //= 4
    nrm2 = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / nrm2
    P = np.vstack([V, W])
    # deduplicate roughly to keep size manageable: round to 1e-8 and unique
    Pr = np.round(P, 8)
    # use unique on canonical integer codes via void view for speed
    # fallback to np.unique axis=0 (may be heavy but M~94k is ok)
    try:
        _, uni = np.unique(Pr, axis=0, return_index=True)
        P = P[np.sort(uni)]
    except Exception:
        pass
    # cap size if too large
    if P.shape[0] > 95000:
        sel = rng.choice(P.shape[0], 95000, replace=False)
        P = P[sel]
    return P

def _greedy_select(P, n_sel, rng, t0, time_limit):
    M = P.shape[0]
    start = int(rng.integers(M))
    selected = [start]
    sel_mask = np.zeros(M, dtype=bool)
    sel_mask[start] = True
    best_dot = P @ P[start]
    best_dot[start] = np.inf  # don't reselect
    for k in range(1, n_sel):
        if time.time() - t0 > time_limit:
            break
        # argmin over unselected
        # use masked array trick
        i = int(np.argmin(np.where(sel_mask, np.inf, best_dot)))
        selected.append(i)
        sel_mask[i] = True
        # update (check time every 32)
        d = P @ P[i]
        # keep inf for selected
        best_dot = np.maximum(best_dot, d)
        best_dot[sel_mask] = np.inf
        if k % 64 == 0 and time.time() - t0 > time_limit:
            break
    return np.array(selected, dtype=np.int64)

def _swap_search(x_sel, P, rng, t0, time_limit):
    n = x_sel.shape[0]
    # map pool index set for quick exclusion (approx: skip exact check, allow duplicates rarely)
    for it in range(300):
        if time.time() - t0 > time_limit:
            break
        G = x_sel @ x_sel.T
        np.fill_diagonal(G, -2.0)
        worst = G.max(axis=1)
        order = np.argsort(-worst)
        # pick a bad index (top 10 random)
        cand_bad = order[rng.integers(0, min(20, n))]
        cur_w = float(worst[cand_bad])
        # sample pool batch
        B = 4000
        bi = rng.integers(0, P.shape[0], size=B)
        Cb = P[bi]
        # dots vs all except cand_bad
        # compute max dot to rest
        dots = Cb @ x_sel.T  # B x n
        # ignore column cand_bad by setting to -2
        dots[:, cand_bad] = -2.0
        bw = dots.max(axis=1)
        j = int(np.argmin(bw))
        if bw[j] < cur_w - 1e-12:
            x_sel[cand_bad] = Cb[j]
        elif rng.random() < 0.02:
            # occasional random kick to diversify
            x_sel[cand_bad] = Cb[j]
    return x_sel

def _adam_hinge(x, t0, time_limit, rng):
    n, d = x.shape
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = _max_cos(x)
    best_x = x.copy()
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    lr0 = 0.008
    # estimate total iters for schedule
    while True:
        if time.time() - t0 > time_limit:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur = float(G.max())
        if cur < best:
            best = cur
            best_x = x.copy()
        # hinge below current max
        hinge = cur - 0.025
        # also never above best-1e-4 to keep pressure
        if hinge > best - 0.002:
            hinge = best - 0.002
        E = G - hinge
        E[E < 0] = 0.0
        np.fill_diagonal(E, 0.0)
        # gradient
        grad = 2.0 * (E @ x)
        # tangent projection
        nr = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - nr * x
        # adam
        t += 1
        # linear decay of lr
        elapsed = (time.time() - t0) / max(time_limit, 1e-6)
        lr = lr0 * (1.0 - 0.8 * min(1.0, elapsed))
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        c1 = 1 - b1 ** t
        c2 = 1 - b2 ** t
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # clip step
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.02
        too = sn > cap
        step = np.where(too, step * (cap / np.maximum(sn, 1e-12)), step)
        x = x - step
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        # occasional time check only (every iter cheap enough)
    return best_x

def solve():
    t_start = time.time()
    rng = np.random.default_rng(0)
    # load incumbent
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(323, 7))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)
    incumbent = best_x.copy()
    incumbent_v = best_v

    def remain():
        return _DEADLINE - (time.time() - t_start)

    # --- lattice pool construction (time-boxed) ---
    try:
        if remain() > 40:
            P = _build_pool(rng)
            # greedy selection, max ~12s
            g_lim = time.time() + min(12.0, remain() - 30)
            sel = _greedy_select(P, 323, rng, time.time(), 12.0)
            # if greedy aborted early, pad randomly
            x_lat = P[sel]
            if x_lat.shape[0] < 323:
                need = 323 - x_lat.shape[0]
                extra = rng.standard_normal(size=(need, 7))
                extra /= np.linalg.norm(extra, axis=1, keepdims=True)
                x_lat = np.vstack([x_lat, extra])
            # swap search, max ~12s
            if remain() > 90:
                s_t0 = time.time()
                x_lat = _swap_search(x_lat, P, rng, s_t0, min(12.0, remain() - 75))
            # adam hinge relax, max ~60s
            if remain() > 60:
                a_t0 = time.time()
                a_budget = min(55.0, remain() - 35)
                x_lat = _adam_hinge(x_lat, a_t0, a_budget, rng)
                lv = _max_cos(x_lat)
                if lv < incumbent_v:
                    incumbent = x_lat.copy(); incumbent_v = lv
                # short LP polish of lattice result
                if remain() > 25:
                    lp_t0 = time.time()
                    x_lat2 = _lp_polish(x_lat, min(15.0, remain() - 12), init_radius=5e-3, t0=lp_t0)
                    lv2 = _max_cos(x_lat2)
                    if lv2 < incumbent_v:
                        incumbent = x_lat2.copy(); incumbent_v = lv2
    except Exception:
        pass

    # --- always spend leftover on incumbent polish ---
    try:
        if remain() > 5:
            lp_t0 = time.time()
            budget = min(20.0, remain() - 1.5)
            if budget > 1.0:
                y = _lp_polish(incumbent, budget, init_radius=3e-3, t0=lp_t0)
                yv = _max_cos(y)
                if yv < incumbent_v:
                    incumbent = y; incumbent_v = yv
    except Exception:
        pass
    # final safety normalise
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
