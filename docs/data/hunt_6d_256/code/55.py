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

def _adam_mixed(x0, steps, K0, K1, lam0, lam1, topM, lr, deadline=1e18, log_every=100):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        frac = t / steps
        K = float(np.exp(la0 + (la1 - la0) * frac))
        lam = float(lam0 + (lam1 - lam0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if lam > 0 and topM > 0:
            c = G[iu0, iu1]
            if topM < c.size:
                thr = float(np.partition(c, c.size - topM)[c.size - topM])
            else:
                thr = float(c.min())
            Mask = (G >= thr).astype(float)
            np.fill_diagonal(Mask, 0.0)
            W = W + lam * Mask
            np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % log_every == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _adam_riesz(x0, steps, p, lr, deadline=1e18):
    x = x0.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    expc = float(p * 0.5 - 1.0)
    best_val = _max_cos(x); best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        umax = (1.0 + gmax) * 0.5
        if umax <= 1e-12:
            umax = 0.5
        logUmax = np.log(umax)
        U = (1.0 + G) * 0.5
        np.fill_diagonal(U, 1.0)
        logU = np.log(np.maximum(U, 1e-300))
        E = expc * (logU - logUmax)
        np.fill_diagonal(E, -100.0)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (mval - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            pnum = cap
        rows = np.repeat(np.arange(pnum), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((pnum, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pnum, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pnum, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _void_refill(x, kill_idx, rng, n_cand=16000):
    # greedily replace kill_idx points by farthest points from remaining set
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[kill_idx] = False
    Xk = x[keep_mask]  # (nk, d)
    nk = Xk.shape[0]
    nkill = len(kill_idx)
    # generate candidates
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # score by max cos to kept set
    # chunk to save memory
    best_vals = np.empty(n_cand)
    chunk = 4000
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        Gc = C[s:e] @ Xk.T
        best_vals[s:e] = Gc.max(axis=1)
    # greedy farthest insertion with mutual repulsion among new points
    chosen = []
    order = np.argsort(best_vals)  # smallest first
    # take greedily: iterate order, accept if good vs already chosen
    # threshold-free: pick sequentially minimizing max to kept+chosen
    # for speed, precompute candidate-candidate on the fly for top pool
    pool_size = min(2000, n_cand)
    pool = order[:pool_size]
    Cp = C[pool]
    # pairwise within pool
    Gpp = Cp @ Cp.T
    sel = []
    # first pick absolute best
    sel.append(0)
    # order pool by best_vals
    # iteratively add point minimizing max(keep_val, max to sel)
    keep_pool_vals = best_vals[pool]
    for _ in range(1, nkill):
        # max to selected
        msel = Gpp[:, sel].max(axis=1)
        tot = np.maximum(keep_pool_vals, msel)
        tot[sel] = 2.0
        sel.append(int(np.argmin(tot)))
    chosen_pts = Cp[sel]
    y = x.copy()
    y[kill_idx] = chosen_pts
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    DE_NOVO_END = t0 + 45.0
    HOP_END = t0 + 148.0
    rng0 = np.random.default_rng(0)
    x_best = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng0.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    incumbent = x_best.copy()

    # Phase 1: a few de novo balanced runs
    cands = [(best, incumbent.copy())]
    for b in range(4):
        if time.time() > DE_NOVO_END - 5:
            break
        try:
            rng = np.random.default_rng(1000 + b)
            x = rng.normal(size=(256, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x, _ = _adam_riesz(x, 120, 16 if b % 2 == 0 else 24, 0.012, deadline=DE_NOVO_END - 2)
            if time.time() > DE_NOVO_END - 2:
                cands.append((_max_cos(x), x.copy())); break
            x, _ = _adam_mixed(x, 350, 150.0, 2500.0, 0.7, 0.05, 2000 if b % 2 == 0 else 1400, 0.008, deadline=DE_NOVO_END - 1)
            if time.time() > DE_NOVO_END - 1:
                cands.append((_max_cos(x), x.copy())); break
            x, vv = _adam_mixed(x, 150, 2500.0, 5000.0, 0.0, 0.0, 0, 0.005, deadline=DE_NOVO_END - 0.5)
            cands.append((vv, x.copy()))
            if vv < best:
                best = vv; x_best = x.copy()
        except Exception:
            continue

    # Phase 2: worst-point void ejection hopping from best pools
    # sort cands
    cands_sorted = sorted(cands, key=lambda t: t[0])
    bases = [c[1] for c in cands_sorted[:2]]
    # ensure incumbent included
    hop_seed = x_best.copy()
    rng_hop = np.random.default_rng(777)
    n_hop = 0
    # varying kill sizes
    kill_schedule = [6, 10, 4, 12, 8, 5, 14, 7, 10, 6, 4, 12, 8, 16, 5, 10]
    base_cycle = bases + [hop_seed]
    hi = 0
    while time.time() < HOP_END and n_hop < len(kill_schedule) * 2:
        try:
            ks = kill_schedule[n_hop % len(kill_schedule)]
            base = base_cycle[hi % len(base_cycle)]
            hi += 1; n_hop += 1
            rm, _ = _row_max(base)
            kill_idx = np.argpartition(-rm, ks)[:ks]
            rng = np.random.default_rng(9000 + n_hop)
            y = _void_refill(base, kill_idx, rng, n_cand=16000)
            # short joint re-optimization: equalized then sharp
            y, _ = _adam_mixed(y, 260, 120.0, 2200.0, 0.8, 0.05, 2000, 0.008, deadline=HOP_END)
            if time.time() > HOP_END:
                vv = _max_cos(y)
                cands.append((vv, y.copy()))
                if vv < best:
                    best = vv; x_best = y.copy()
                break
            y, vv = _adam_mixed(y, 150, 2500.0, 7000.0, 0.0, 0.0, 0, 0.004, deadline=HOP_END)
            cands.append((vv, y.copy()))
            if vv < best and np.all(np.isfinite(y)):
                best = vv; x_best = y.copy()
                # restart cycle around new best
                base_cycle = [x_best.copy()] + base_cycle[:2]
            # also try tangent shake of new point occasionally
            if n_hop % 4 == 3 and time.time() < HOP_END - 10:
                z = rng_hop.normal(size=base.shape)
                z -= np.sum(z * x_best, axis=1, keepdims=True) * x_best
                xs = x_best + 0.08 * z
                xs /= np.linalg.norm(xs, axis=1, keepdims=True)
                xs, vvs = _adam_mixed(xs, 200, 200.0, 4000.0, 0.3, 0.0, 1500, 0.006, deadline=HOP_END)
                cands.append((vvs, xs.copy()))
                if vvs < best:
                    best = vvs; x_best = xs.copy()
        except Exception:
            continue

    # Phase 3: polish top pools with sharp LSE + LP
    cands_sorted = sorted(cands, key=lambda t: t[0])[:4]
    pool = []
    if x_best is not None:
        pool.append(x_best.copy())
    for vv, xx in cands_sorted:
        if len(pool) >= 3:
            break
        if vv < best + 8e-4:
            # avoid duplicates
            dup = False
            for yy in pool:
                if abs(_max_cos(yy) - vv) < 1e-12:
                    dup = True; break
            if not dup:
                pool.append(xx.copy())
    for xx in pool:
        if time.time() > T_END - 6:
            break
        try:
            xx2, vv2 = _adam_mixed(xx, 220, 800.0, 8000.0, 0.0, 0.0, 0, 0.0035, deadline=T_END - 6)
            if vv2 < best:
                best = vv2; x_best = xx2.copy()
                xx = xx2
            if time.time() < T_END - 3:
                xxp = _lp_polish(xx, deadline=min(time.time() + 10, T_END - 1))
                vvp = _max_cos(xxp)
                if vvp < best and np.all(np.isfinite(xxp)):
                    best = vvp; x_best = xxp.copy()
        except Exception:
            continue
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
