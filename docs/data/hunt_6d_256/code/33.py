import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _sorted_cos(x):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu[0], iu[1]]
    c = np.sort(c)[::-1]
    return c

def _top3(x):
    c = _sorted_cos(x)
    return float(c[0]), float(c[99]), float(c[999]), c

def _adam_lse(x0, steps, K0, K1, lr, verbose=False):
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
        x /= np.linalg.norm(x, axis=1, keepdims=True)
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
        mv = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (mv - margin)
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

def _refine_single(y0, F, steps=75, K=2500.0, lr=0.02, rng=None):
    # minimize LSE max dot to frozen set F (m,d)
    y = y0.copy()
    y = y / np.linalg.norm(y)
    m1 = np.zeros_like(y)
    v1 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_y = y.copy()
    g = F @ y
    best_val = float(g.max())
    for t in range(1, steps + 1):
        g = F @ y
        gmax = float(g.max())
        w = np.exp(K * (g - gmax))
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        w = w / s
        grad = w @ F  # (d,)
        # tangent
        gt = grad - float(grad @ y) * y
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1 - b1) * gt
        v1 *= b2; v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = step - float(step @ y) * y
        y = y - step
        y = y / np.linalg.norm(y)
        if t % 25 == 0 or t == steps:
            gv = float((F @ y).max())
            if gv < best_val:
                best_val = gv
                best_y = y.copy()
    return best_y, best_val

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    DEADLINE = t0 + 178.0
    rng = _np.random.default_rng(777)
    n, d = 256, 6
    # load incumbent
    base = None
    base_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                if _np.all(_np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < base_v:
                        base_v = vv
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        rs = _np.random.default_rng(0)
        base = rs.normal(size=(n, d))
        base /= _np.linalg.norm(base, axis=1, keepdims=True)
        base_v = _max_cos(base)
    inc_x = base.copy()
    inc_c0, inc_c99, inc_c999, _ = _top3(inc_x)
    best_x = inc_x.copy()
    best_c0, best_c99, best_c999 = inc_c0, inc_c99, inc_c999

    # pre-generate >=200k spherical candidates
    NCAND = 200000
    Z = rng.normal(size=(NCAND, d))
    Z /= _np.linalg.norm(Z, axis=1, keepdims=True)

    # compute participation in top 1500 pairs + row max for incumbent
    def _participation(x, topT=1500):
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        iu0 = _np.triu_indices(n, 1)
        c = G[iu0[0], iu0[1]]
        T = min(topT, c.size)
        idx = _np.argpartition(-c, T - 1)[:T]
        I = iu0[0][idx]; J = iu0[1][idx]
        part = _np.zeros(n, dtype=int)
        _np.add.at(part, I, 1)
        _np.add.at(part, J, 1)
        rowmax = G.max(axis=1)
        return part, rowmax, float(c.max())

    # surgery trials
    k_list = [16, 32, 8, 48]
    CHUNK = 20000
    for ki, k in enumerate(k_list):
        if _time.time() > DEADLINE - 35:
            break
        try:
            part, rowmax, _ = _participation(best_x if False else inc_x, 1500)
        except Exception:
            continue
        order_part = _np.argsort(-part, kind='stable')
        # half hubs
        nhub = k // 2
        hub_idx = order_part[:nhub].tolist()
        hub_set = set(hub_idx)
        # remaining high-row-max points: top 80 by rowmax excluding hubs
        order_row = _np.argsort(-rowmax, kind='stable')
        cand_pool = [i for i in order_row if i not in hub_set][:80]
        need = k - nhub
        # deterministic sample from cand_pool
        rsamp = _np.random.default_rng(5000 + k)
        if len(cand_pool) < need:
            need = len(cand_pool)
        samp = rsamp.choice(_np.array(cand_pool), size=need, replace=False).tolist() if need > 0 else []
        del_idx = _np.array(hub_idx + samp, dtype=int)
        if del_idx.size != k:
            # pad if needed
            extra = [i for i in order_row if i not in set(del_idx.tolist())]
            del_idx = _np.array(del_idx.tolist() + extra[:k - del_idx.size], dtype=int)
        keep_mask = _np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        survivors = inc_x[keep_mask].copy()  # (n-k,d) frozen
        # greedy insert one at a time
        F = survivors.copy()
        inserted = []
        # for speed, precompute norms? Z already normalized
        for s in range(k):
            if _time.time() > DEADLINE - 30:
                break
            # find best candidate minimizing max dot to F
            gbest = 1e9
            gbi = -1
            # chunked matmul
            for a0 in range(0, NCAND, CHUNK):
                Zb = Z[a0:a0 + CHUNK]
                S = Zb @ F.T
                cm = S.max(axis=1)
                bi = int(_np.argmin(cm))
                bv = float(cm[bi])
                if bv < gbest:
                    gbest = bv
                    gbi = a0 + bi
            y0 = Z[gbi].copy()
            # local refine 50-100 Adam steps (use 75, fewer for large k to save time)
            steps_ref = 75 if k <= 16 else 60
            try:
                yref, _ = _refine_single(y0, F, steps=steps_ref, K=2500.0, lr=0.02)
            except Exception:
                yref = y0
            inserted.append(yref)
            F = _np.vstack([F, yref[None, :]])
        if len(inserted) != k:
            continue
        x_trial = _np.vstack([survivors, _np.array(inserted)])
        # after all replacements, unfreeze + repolish LSE then LP
        try:
            # LogSumExp Adam
            steps_lse = 500 if k <= 16 else 400
            x_trial, _ = _adam_lse(x_trial, steps_lse, 300.0, 5000.0, 0.008)
        except Exception:
            pass
        if _time.time() < DEADLINE - 12:
            try:
                x_trial = _lp_polish(x_trial, min(_time.time() + 4.0, DEADLINE - 8), r0=2.5e-3, cap=4000)
            except Exception:
                pass
        try:
            nc0, nc99, nc999, _ = _top3(x_trial)
        except Exception:
            continue
        # retention: require top-1 improves and top-100/top-1000 not degraded beyond tolerance,
        # and overall best tracking
        improves_top1 = nc0 < best_c0 - 1e-12
        # also compare to inc for topology criterion
        ok_plateau = (nc99 <= inc_c99 + 3e-6) and (nc999 <= inc_c999 + 3e-6)
        # allow retain if strictly better max and plateau roughly not worse
        if improves_top1 and ok_plateau:
            best_x = x_trial.copy()
            best_c0, best_c99, best_c999 = nc0, nc99, nc999
            # also update inc reference? keep original inc for criterion? update to best for next surgery base?
            # use best as new frozen base for subsequent k? spec says targeted at near-contact network of incumbent
            # we keep inc_x as best for next iteration to compound gains
            inc_x = best_x.copy()
        elif nc0 < best_c0 - 1e-12 and nc99 < best_c99 + 1e-5:
            # fallback lenient: still take if max clearly better
            best_x = x_trial.copy()
            best_c0, best_c99, best_c999 = nc0, nc99, nc999
            inc_x = best_x.copy()
    # final repolish of best
    try:
        if _time.time() < DEADLINE - 10:
            bx, _ = _adam_lse(best_x, 400, 500.0, 8000.0, 0.006)
            c0x = _max_cos(bx)
            if c0x < best_c0:
                best_x = bx
                best_c0 = c0x
    except Exception:
        pass
    try:
        if _time.time() < DEADLINE - 4:
            bx2 = _lp_polish(best_x, DEADLINE - 1.0, r0=2.0e-3, cap=4000)
            c0x2 = _max_cos(bx2)
            if c0x2 < best_c0:
                best_x = bx2
                best_c0 = c0x2
    except Exception:
        pass
    # no-regression fallback
    try:
        b = _np.load("best.npy")
        b = _np.asarray(b, dtype=float)
        if b.shape == (256, 6):
            b = b / _np.linalg.norm(b, axis=1, keepdims=True)
            if _max_cos(b) < _max_cos(best_x):
                best_x = b
    except Exception:
        pass
    best_x = best_x / _np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
