import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _top_stats(x):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    # top1, top100, top1000 (1-indexed largest)
    # use partition for speed
    p1 = float(c.max())
    k100 = np.partition(c, -100)[-100] if c.size >= 100 else p1
    k1000 = np.partition(c, -1000)[-1000] if c.size >= 1000 else float(np.sort(c)[0])
    return p1, float(k100), float(k1000)

def _adam_lse(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.5e-3, cap=4000):
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
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _refine_point(p0, Y, steps=75, lr=0.03, K=250.0):
    p = p0.copy()
    nrm = np.linalg.norm(p)
    if nrm == 0 or not np.isfinite(nrm):
        p = Y[0] * 0.0
        p[0] = 1.0
    else:
        p /= nrm
    m = np.zeros(6)
    v = np.zeros(6)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        dots = Y @ p
        mx = float(dots.max())
        W = np.exp(np.clip(K * (dots - mx), -60.0, 0.0))
        S = float(W.sum())
        if not np.finite(S) or S <= 0:
            break
        grad = (W @ Y) / S
        gt = grad - float(grad @ p) * p
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = step - float(step @ p) * p
        p = p - step
        nrm = np.linalg.norm(p)
        if nrm == 0 or not np.isfinite(nrm):
            break
        p /= nrm
    return p

def _surgery_once(Xbase, k, rng, n_cand=200000, refine_steps=75):
    n, d = Xbase.shape
    G = Xbase @ Xbase.T
    np.fill_diagonal(G, -2.0)
    row_max = G.max(axis=1)
    gmax = float(G.max())
    iuI, iuJ = np.triu_indices(n, 1)
    c = G[iuI, iuJ]
    TOP = min(1500, c.size)
    part = np.argpartition(-c, TOP)[:TOP]
    I = iuI[part]; J = iuJ[part]
    particip = np.bincount(np.concatenate([I, J]), minlength=n)
    # half hubs by participation (tie-break row_max)
    n_hub = k // 2
    n_rest = k - n_hub
    order_hub = np.argsort(-particip * 10.0 - row_max)
    hub_idx = order_hub[:n_hub]
    hub_set = set(int(z) for z in hub_idx)
    # remaining high-row-max pool
    order_row = np.argsort(-row_max)
    pool = [int(z) for z in order_row if int(z) not in hub_set][:max(64, n_rest * 4)]
    if len(pool) < n_rest:
        pool = [int(z) for z in order_row if int(z) not in hub_set]
    sel_rest = rng.choice(np.array(pool), size=n_rest, replace=False)
    del_idx = np.array(sorted(list(hub_set) + [int(z) for z in sel_rest]), dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[del_idx] = False
    Yfrozen = Xbase[mask]
    # candidates
    Cand = rng.normal(size=(n_cand, d))
    Cand /= np.linalg.norm(Cand, axis=1, keepdims=True)
    S0 = Cand @ Yfrozen.T
    cur_max = S0.max(axis=1)
    inserted = []
    Ycur = Yfrozen.copy()
    for s in range(k):
        bi = int(np.argmin(cur_max))
        p0 = Cand[bi].copy()
        p_ref = _refine_point(p0, Ycur, steps=refine_steps, lr=0.03, K=250.0)
        inserted.append(p_ref)
        Ycur = np.vstack([Ycur, p_ref[None, :]])
        dv = Cand @ p_ref
        cur_max = np.maximum(cur_max, dv)
        # forbid exact reuse: set picked to large
        cur_max[bi] = 1.0
    Xnew = Xbase.copy()
    for ii, idx in enumerate(del_idx):
        Xnew[int(idx)] = inserted[ii]
    Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
    return Xnew

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    ANNEAL_END = t0 + 130.0
    LP_END = t0 + 170.0
    rng = _np.random.default_rng(0)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    base_t1, base_t100, base_t1000 = _top_stats(x_best)
    # quick warm LP to equalize plateau
    try:
        if _time.time() < ANNEAL_END:
            xt = _lp_polish(x_best, min(_time.time() + 4.0, ANNEAL_END), r0=1.5e-3)
            vt = _max_cos(xt)
            if vt < best:
                x_best, best = xt, vt
                base_t1, base_t100, base_t1000 = _top_stats(x_best)
    except Exception:
        pass
    ks = [16, 32, 8, 48]
    trial = 0
    # separate trials for each k, cycling with new seeds
    k_cycle = ks * 4
    for k in k_cycle:
        if _time.time() > ANNEAL_END - 6.0:
            break
        trial += 1
        try:
            srng = _np.random.default_rng(1000 + trial * 7919 + k * 131)
            # at least 200k candidates
            Xnew = _surgery_once(x_best, int(k), srng, n_cand=200000, refine_steps=75)
        except Exception:
            continue
        # unfreeze: short LogSumExp Adam
        try:
            rem = ANNEAL_END - _time.time()
            steps = 600 if rem > 30 else (350 if rem > 12 else 200)
            Xp, vp = _adam_lse(Xnew, steps, 80.0, 1500.0, 0.009)
        except Exception:
            continue
        try:
            nt1, nt100, nt1000 = _top_stats(Xp)
        except Exception:
            continue
        # retain only if top-1, top-100, top-1000 all improve (strict)
        eps = 1e-12
        cond = (nt1 < base_t1 - eps) and (nt100 < base_t100 - eps) and (nt1000 < base_t1000 - eps)
        # relax slightly if top1 improves a lot: allow equality on others within 1e-7 after big gain
        # but primary gate is the triple improvement; also accept pure top1 gain if large
        big_gain = (nt1 < base_t1 - 5e-5) and (nt100 <= base_t100 + 1e-7) and (nt1000 <= base_t1000 + 1e-7)
        if not (cond or big_gain):
            # also try a second polish with different K before discarding (cheap)
            try:
                Xp2, vp2 = _adam_lse(Xp, 250, 40.0, 900.0, 0.012)
                nt1b, nt100b, nt1000b = _top_stats(Xp2)
                cond2 = (nt1b < base_t1 - eps) and (nt100b < base_t100 - eps) and (nt1000b < base_t1000 - eps)
                big2 = (nt1b < base_t1 - 5e-5) and (nt100b <= base_t100 + 1e-7) and (nt1000b <= base_t1000 + 1e-7)
                if cond2 or big2:
                    Xp, nt1, nt100, nt1000 = Xp2, nt1b, nt100b, nt1000b
                else:
                    continue
            except Exception:
                continue
        # accepted: plateau-wide LP then keep
        try:
            Xq = _lp_polish(Xp, min(_time.time() + 3.0, LP_END), r0=2.0e-3, cap=4000)
            vq = _max_cos(Xq)
            nq1, nq100, nq1000 = _top_stats(Xq)
            # keep best by max cosine
            if vq < best - 1e-12:
                x_best, best = Xq, vq
                base_t1, base_t100, base_t1000 = nq1, nq100, nq1000
            elif _max_cos(Xp) < best - 1e-12:
                x_best, best = Xp, _max_cos(Xp)
                base_t1, base_t100, base_t1000 = nt1, nt100, nt1000
        except Exception:
            vxp = _max_cos(Xp)
            if vxp < best:
                x_best, best = Xp, vxp
                base_t1, base_t100, base_t1000 = nt1, nt100, nt1000
    # final repolish
    try:
        if _time.time() < LP_END:
            x_best = _adam_lse(x_best, 500, 60.0, 2000.0, 0.007)[0]
    except Exception:
        pass
    try:
        if _time.time() < LP_END:
            x_best = _lp_polish(x_best, LP_END, r0=2.5e-3, cap=4000)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
