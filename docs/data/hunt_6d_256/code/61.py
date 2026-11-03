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
    c = G[iu[0], iu[1]]
    cs = np.sort(c)[::-1]
    c0 = float(cs[0])
    c99 = float(cs[99]) if cs.size > 99 else c0
    c999 = float(cs[999]) if cs.size > 999 else c0
    rowmax = np.max(G, axis=1)
    return c0, c99, c999, rowmax, cs


def _adam_lse(x0, steps, K0, K1, lr, track_best=True):
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
        nr[~np.isfinite(nr)] = 1.0
        x /= nr
        if track_best and (t % 150 == 0 or t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    if track_best:
        return best_x, best_val
    return x, _max_cos(x)


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
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]
            J = J[part]
            csel = csel[part]
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


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    DEADLINE = t0 + 176.0
    rng = _np.random.default_rng(20261103)
    n, d = 256, 6

    base = None
    base_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
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

    best_x = base.copy()
    best_c0, best_c99, best_c999, _, _ = _top_stats(best_x)

    pool = [(best_c0, best_x.copy())]

    def _kick(xref, mode, trial):
        x0 = xref.copy()
        if mode == 0:  # large gaussian kick
            sig = 0.30 if (trial % 2 == 0) else 0.22
            x0 = x0 + sig * rng.normal(size=(n, d))
            x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 1:  # medium kick
            sig = 0.12
            x0 = x0 + sig * rng.normal(size=(n, d))
            x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 2:  # small kick
            sig = 0.05
            x0 = x0 + sig * rng.normal(size=(n, d))
            x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 3:  # reseed worst k + jitter rest
            _, _, _, rowmax, _ = _top_stats(xref)
            k = 24 if (trial % 2 == 0) else 16
            idx = _np.argsort(-rowmax, kind="stable")[:k]
            x0[idx] = rng.normal(size=(k, d))
            x0[idx] /= _np.linalg.norm(x0[idx], axis=1, keepdims=True)
            x0 = x0 + 0.03 * rng.normal(size=(n, d))
            x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        else:  # reseed worst 8 + medium jitter
            _, _, _, rowmax, _ = _top_stats(xref)
            k = 8
            idx = _np.argsort(-rowmax, kind="stable")[:k]
            x0[idx] = rng.normal(size=(k, d))
            x0[idx] /= _np.linalg.norm(x0[idx], axis=1, keepdims=True)
            x0 = x0 + 0.06 * rng.normal(size=(n, d))
            x0 /= _np.linalg.norm(x0, axis=1, keepdims=True)
        return x0

    cands_for_lp = []
    trial = 0
    # basin hopping census
    while _time.time() < DEADLINE - 28:
        # pick reference: mostly global best, sometimes pool diversity
        if len(pool) > 1 and (trial % 7 == 6):
            # use second best for diversity
            xref = pool[1][1]
        else:
            xref = best_x
        mode = trial % 5
        try:
            x0 = _kick(xref, mode, trial)
        except Exception:
            trial += 1
            continue
        # annealing params vary by mode
        if mode == 0:
            steps, K0, K1, lr = 450, 200.0, 7000.0, 0.012
        elif mode == 1:
            steps, K0, K1, lr = 400, 300.0, 6000.0, 0.010
        elif mode == 2:
            steps, K0, K1, lr = 350, 400.0, 8000.0, 0.008
        elif mode == 3:
            steps, K0, K1, lr = 500, 150.0, 5000.0, 0.012
        else:
            steps, K0, K1, lr = 400, 250.0, 6500.0, 0.010
        try:
            xa, va = _adam_lse(x0, steps, K0, K1, lr, track_best=True)
        except Exception:
            trial += 1
            continue
        try:
            c0, c99, c999, _, _ = _top_stats(xa)
        except Exception:
            trial += 1
            continue
        # immediate LP if clear max improvement
        if c0 < best_c0 - 3e-6 and _time.time() < DEADLINE - 20:
            try:
                xl = _lp_polish(xa, min(_time.time() + 3.5, DEADLINE - 14), r0=2.5e-3, cap=4000)
                cl0, cl99, cl999, _, _ = _top_stats(xl)
                if cl0 < best_c0:
                    best_x = xl.copy()
                    best_c0, best_c99, best_c999 = cl0, cl99, cl999
                    pool = [(best_c0, best_x.copy())] + pool[:2]
                elif cl0 < c0:
                    c0, c99, c999, xa = cl0, cl99, cl999, xl
                    if c0 < best_c0:
                        best_x = xa.copy()
                        best_c0, best_c99, best_c999 = c0, c99, c999
            except Exception:
                pass
            trial += 1
            continue
        # record marginal improvements for later LP
        if c0 < best_c0 - 1e-12:
            # keep, will LP later
            cands_for_lp.append((c0, xa.copy()))
            cands_for_lp.sort(key=lambda z: z[0])
            cands_for_lp = cands_for_lp[:4]
            # also update best immediately (pre-LP) if better
            if c0 < best_c0:
                best_x = xa.copy()
                best_c0, best_c99, best_c999 = c0, c99, c999
                pool = [(best_c0, best_x.copy())] + [p for p in pool if p[0] < best_c0 + 5e-4][:2]
        else:
            # plateau improvement dive: keep diverse base if plateau much better
            if c999 < best_c999 - 1.5e-4 and c0 < best_c0 + 8e-4:
                pool.append((c0, xa.copy()))
                pool.sort(key=lambda z: z[0])
                # keep at most 3, always keep global best first
                uniq = [(best_c0, best_x.copy())]
                for vv, xx in pool:
                    if len(uniq) >= 3:
                        break
                    if vv > best_c0 + 1e-3:
                        continue
                    uniq.append((vv, xx))
                pool = uniq
        trial += 1
        if trial > 80:
            break

    # LP-polish deferred marginal candidates
    for _, xc in list(cands_for_lp):
        if _time.time() > DEADLINE - 16:
            break
        try:
            c_cur = _max_cos(xc)
            if c_cur > best_c0 + 1e-9:
                continue
            xl = _lp_polish(xc, min(_time.time() + 3.0, DEADLINE - 10), r0=2.5e-3, cap=4000)
            cl0, cl99, cl999, _, _ = _top_stats(xl)
            if cl0 < best_c0:
                best_x = xl.copy()
                best_c0, best_c99, best_c999 = cl0, cl99, cl999
        except Exception:
            continue

    # final joint polish of global best
    try:
        if _time.time() < DEADLINE - 10:
            xb, vb = _adam_lse(best_x, 400, 500.0, 9000.0, 0.006, track_best=True)
            if vb < best_c0:
                best_x = xb.copy()
                best_c0 = vb
                _, _, _, _, _ = _top_stats(best_x)
    except Exception:
        pass
    try:
        if _time.time() < DEADLINE - 3:
            xl = _lp_polish(best_x, DEADLINE - 1.0, r0=2.0e-3, cap=4000)
            if _max_cos(xl) < best_c0:
                best_x = xl.copy()
                best_c0 = _max_cos(best_x)
    except Exception:
        pass

    # no-regression fallback
    try:
        b = _np.load("best.npy")
        b = _np.asarray(b, dtype=float)
        if b.shape == (n, d):
            b = b / _np.linalg.norm(b, axis=1, keepdims=True)
            if _max_cos(b) < _max_cos(best_x):
                best_x = b
    except Exception:
        pass
    best_x = best_x / _np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
