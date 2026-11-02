import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end=None):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        if int(np.sum(sel)) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]; J = iu[1][sel]
        cc = cfull[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        tmp_c = np.empty((p, 2 * d + 1), dtype=float)
        tmp_c[:, :d] = x[J]
        tmp_c[:, d:2 * d] = x[I]
        tmp_c[:, 2 * d] = -1.0
        tmp_i = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        tmp_i[:, :d] = I[:, None] * d + ar
        tmp_i[:, d:2 * d] = J[:, None] * d + ar
        tmp_i[:, 2 * d] = n * d
        try:
            A_ub = coo_matrix((tmp_c.ravel(), (rows, tmp_i.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _softmax_full(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, t_lim=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    best = x.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for k in range(steps):
        if t_lim is not None and (k % 400 == 0) and time.time() > t_lim:
            break
        frac = k / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (np.sum(grad * x, axis=1, keepdims=True)) * x
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (np.sum(nz * x, axis=1, keepdims=True)) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            best = x.copy()
    return best, bv

def _free_softmax(x0, free_idx, steps, T0, T1, lr0, mu, rng, t_lim=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    best = x.copy()
    bbest = bv
    lT0 = np.log(T0); lT1 = np.log(T1)
    for k in range(steps):
        if t_lim is not None and (k % 400 == 0) and time.time() > t_lim:
            break
        frac = k / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        grad -= (np.sum(grad * x, axis=1, keepdims=True)) * x
        # freeze non-free
        mask = np.zeros((n, 1), dtype=float)
        mask[free_idx] = 1.0
        grad = grad * mask
        vel = mu * vel + (1.0 - mu) * grad
        # only move free
        vel_f = vel[free_idx]
        x[free_idx] = x[free_idx] - lr * vel_f
        x[free_idx] /= np.maximum(np.linalg.norm(x[free_idx], axis=1, keepdims=True), 1e-12)
        if m < bbest:
            bbest = m
            best = x.copy()
    return best, bbest

def _target_full(x0, target, steps, lr0, mu, rng, noise0=0.0, t_lim=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    best = x.copy()
    for k in range(steps):
        if t_lim is not None and (k % 400 == 0) and time.time() > t_lim:
            break
        frac = k / max(1, steps - 1)
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        V = G - target
        V[V < 0] = 0.0
        np.fill_diagonal(V, 0.0)
        # gradient of sum V^2 : 2*V @ x
        grad = (V @ x) * 2.0
        # scale to avoid huge steps
        gn = float(np.sqrt(np.sum(grad * grad) / n))
        if gn > 5.0:
            grad *= (5.0 / gn)
        grad -= (np.sum(grad * x, axis=1, keepdims=True)) * x
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (np.sum(nz * x, axis=1, keepdims=True)) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            best = x.copy()
            if bv <= target:
                break
    return best, bv

def _seq_holes(fixed, mrep, rng, nsamp=8000):
    holes = []
    aug = fixed.copy()
    for _ in range(mrep):
        Y = rng.randn(nsamp, 4)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        D = Y @ aug.T
        vals = D.max(axis=1)
        idx = int(np.argmin(vals))
        h = Y[idx].copy()
        # quick tangential refine away from closest
        vy = float(vals[idx])
        for __ in range(120):
            dots = aug @ h
            j = int(np.argmax(dots))
            g = aug[j] - dots[j] * h
            ng = float(np.linalg.norm(g))
            if ng < 1e-12:
                break
            st = 0.02
            hn = h - st * g / ng
            hn /= np.linalg.norm(hn)
            vn = float((aug @ hn).max())
            if vn < vy:
                h, vy = hn, vn
            else:
                st = 0.005
                hn2 = h - st * g / ng
                hn2 /= np.linalg.norm(hn2)
                vn2 = float((aug @ hn2).max())
                if vn2 < vy:
                    h, vy = hn2, vn2
                else:
                    break
        holes.append(h)
        aug = np.vstack([aug, h.reshape(1, 4)])
    return np.array(holes)

def _worst_cluster(x, mrep, eps=0.0025):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # contact adjacency
    adj = G > (m - eps)
    deg = adj.sum(axis=1)
    # start from highest degree node among max-pair endpoints
    iu = np.where(G >= m - 1e-12)
    # pick endpoint with larger deg
    if len(iu[0]) > 0:
        cand = np.concatenate([iu[0], iu[1]])
        # most frequent
        vals, cnts = np.unique(cand, return_counts=True)
        start = int(vals[np.argmax(cnts)])
    else:
        start = int(np.argmax(deg))
    # BFS expand by degree order
    chosen = [start]
    seen = set(chosen)
    frontier = set(np.where(adj[start])[0].tolist())
    # add hub neighbours first
    while len(chosen) < mrep and frontier:
        # pick frontier node with largest deg
        fl = list(frontier)
        fl.sort(key=lambda v: -deg[v])
        v = fl[0]
        frontier.remove(v)
        if v in seen:
            continue
        seen.add(v)
        chosen.append(int(v))
        for w in np.where(adj[v])[0]:
            w = int(w)
            if w not in seen:
                frontier.add(w)
    # if still short, add global worst deg
    if len(chosen) < mrep:
        order = np.argsort(-deg)
        for v in order:
            v = int(v)
            if v not in seen:
                seen.add(v)
                chosen.append(v)
            if len(chosen) >= mrep:
                break
    return np.array(chosen[:mrep], dtype=int)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 130.0
    rng = np.random.RandomState(917631)
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
    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 38:
            break
        trial += 1
        r = rng.rand()
        cur_b = None
        cur_v = 1e9
        if r < 0.38:
            # jitter + full softmax
            su = rng.rand()
            sigma = float(np.exp(np.log(0.0015) + su * (np.log(0.45) - np.log(0.0015))))
            cur = best_x + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.choice([0.008, 0.012, 0.018]))
            T1 = float(rng.choice([0.0004, 0.0006, 0.0009]))
            lr0 = float(rng.choice([0.030, 0.040, 0.050]))
            mu = float(rng.choice([0.84, 0.87, 0.90]))
            steps = int(rng.choice([3500, 4500, 5500]))
            nz = float(rng.choice([0.0, 0.001, 0.003]))
            cur_b, cur_v = _softmax_full(cur, steps, T0, T1, lr0, mu, rng, noise0=nz, t_lim=SEARCH_END)
        elif r < 0.60:
            # cluster surgery
            mrep = int(rng.choice([4, 5, 6, 8, 10, 12]))
            try:
                worst = _worst_cluster(best_x, mrep)
            except Exception:
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.0025).sum(axis=1)
                worst = np.argsort(-deg)[:mrep]
            keep = np.ones(n, dtype=bool)
            keep[worst] = False
            fixed = best_x[keep].copy()
            try:
                holes = _seq_holes(fixed, mrep, rng, nsamp=7000)
            except Exception:
                holes = rng.randn(mrep, 4)
                holes /= np.linalg.norm(holes, axis=1, keepdims=True)
            cur = best_x.copy()
            cur[worst] = holes
            cur += rng.randn(n, 4) * 0.003
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            # free warm-up
            try:
                w1, v1 = _free_softmax(cur, worst, 1400, 0.008, 0.0006, 0.035, 0.87, rng, t_lim=SEARCH_END)
            except Exception:
                w1, v1 = cur, _max_cos(cur)
            # full relax
            cur_b, cur_v = _softmax_full(w1, 3500, 0.010, 0.0005, 0.038, 0.87, rng, noise0=0.001, t_lim=SEARCH_END)
            if v1 < cur_v and _max_cos(w1) < cur_v:
                # keep better of the two (already best-tracked inside, but ensure)
                pass
        elif r < 0.74:
            # cap shake
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = best_x @ u
            q = float(rng.choice([0.80, 0.85, 0.90]))
            thr = float(np.quantile(dots, q))
            idx = np.where(dots > thr)[0]
            if len(idx) < 5:
                idx = np.argsort(-dots)[:15]
            cur = best_x.copy()
            sig = float(rng.choice([0.20, 0.30, 0.45]))
            cur[idx] += rng.randn(len(idx), 4) * sig
            # occasionally fully randomize cap points
            if rng.rand() < 0.3:
                rr = rng.randn(len(idx), 4)
                rr /= np.linalg.norm(rr, axis=1, keepdims=True)
                # mix
                cur[idx] = cur[idx] + rr * 0.8
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _softmax_full(cur, 4000, 0.014, 0.0007, 0.042, 0.85, rng, noise0=0.002, t_lim=SEARCH_END)
        elif r < 0.89:
            # target feasibility attack
            delta = float(rng.choice([0.00015, 0.00035, 0.0007, 0.0012]))
            target = best_val - delta
            sigma = float(rng.choice([0.02, 0.05, 0.09]))
            cur = best_x + rng.randn(n, 4) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            tmp_b, tmp_v = _target_full(cur, target, 2500, 0.06, 0.88, rng, noise0=0.001, t_lim=SEARCH_END)
            # polish whatever we got with softmax
            cur_b, cur_v = _softmax_full(tmp_b, 3000, 0.008, 0.0004, 0.035, 0.88, rng, noise0=0.0, t_lim=SEARCH_END)
        else:
            # random restart exploration
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_b, cur_v = _softmax_full(cur, 5000, 0.018, 0.0006, 0.045, 0.85, rng, noise0=0.003, t_lim=SEARCH_END)
        if cur_b is not None and np.all(np.isfinite(cur_b)):
            # normalize check
            cur_b = cur_b / np.maximum(np.linalg.norm(cur_b, axis=1, keepdims=True), 1e-12)
            # re-evaluate to avoid stale
            try:
                cur_v = _max_cos(cur_b)
            except Exception:
                cur_v = 1e9
            if cur_v < best_val - 1e-12:
                if cur_v < best_val - 3e-5 and time.time() < DEADLINE - 42:
                    try:
                        pol, pv = _lp_polish(cur_b, seconds=2.0, t_end=DEADLINE - 32)
                        if pv < cur_v:
                            cur_b, cur_v = pol, pv
                    except Exception:
                        pass
                if cur_v < best_val:
                    best_x, best_val = cur_b, cur_v
        if best_val < 0.84:
            break
    remain = DEADLINE - time.time() - 1.5
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, seconds=min(remain, 30.0), t_end=DEADLINE - 0.8)
            if pv < best_val:
                best_x, best_val = px, pv
            else:
                # keep polished geometry even if tie (fewer near-ties)
                best_x = px
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
