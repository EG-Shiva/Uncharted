import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr):
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
        m = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (m - margin)
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

def _split_worst_pair(x, rng, batch=2048):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    gmax = float(G.max())
    fi = int(np.argmax(G))
    i = fi // n; j = fi % n
    # alternate optimization of i and j
    for _ in range(4):
        for k in (i, j):
            other = j if k == i else i
            cur = float(G[k].max())
            # candidates: local tangent + a few global random
            nloc = batch
            R = rng.normal(size=(nloc, d))
            # mix sigmas
            sig = np.empty((nloc, 1))
            sig[:nloc//2] = 0.05
            sig[nloc//2:] = 0.015
            Cand = x[k] + sig * R
            Cand /= np.linalg.norm(Cand, axis=1, keepdims=True)
            # add some global void samples
            Z = rng.normal(size=(512, d))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            Cand = np.vstack([Cand, Z])
            S = Cand @ x.T
            S[:, k] = -2.0
            # forbid using old position? keep other column as is (pair term included)
            cm = S.max(axis=1)
            bi = int(np.argmin(cm))
            bv = float(cm[bi])
            if bv < cur - 1e-12:
                x[k] = Cand[bi]
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
    return x

def _greedy_sweep(x, rng, deadline, top_k=22, batch=1024):
    n, d = x.shape
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        order = np.argsort(-prow)[:top_k]
        improved = False
        for ii in order:
            if time.time() >= deadline:
                break
            k = int(ii)
            cur = float(prow[k])
            if cur < gmax - 3e-4:
                continue
            # other max excluding k
            Gtmp = G.copy()
            Gtmp[k, :] = -2.0
            Gtmp[:, k] = -2.0
            other = float(Gtmp.max())
            R = rng.normal(size=(batch, d))
            sig = 0.025
            Cand = x[k] + sig * R
            Cand /= np.linalg.norm(Cand, axis=1, keepdims=True)
            S = Cand @ x.T
            S[:, k] = -2.0
            cm = S.max(axis=1)
            bi = int(np.argmin(cm))
            bv = float(cm[bi])
            new_global = bv if bv > other else other
            if new_global < gmax - 1e-12:
                x[k] = Cand[bi]
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                improved = True
                break
            elif bv < cur - 2e-4 and new_global <= gmax + 1e-12:
                x[k] = Cand[bi]
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                improved = True
        if not improved:
            break
    return x

def _void_reinsert(x, rng, n_void=2):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    prow = G.max(axis=1)
    # loosest points (smallest row max) are movable without hurting max
    loosest = np.argsort(prow)[:n_void]
    for k in loosest:
        Z = rng.normal(size=(8000, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ x.T
        S[:, k] = -2.0
        cm = S.max(axis=1)
        bi = int(np.argmin(cm))
        x[k] = Z[bi]
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    ANNEAL_END = t0 + 132.0
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
    n, d = 256, 6
    # quick warm LP
    try:
        if _time.time() < ANNEAL_END:
            x_best = _lp_polish(x_best, min(_time.time() + 5.0, ANNEAL_END), r0=1.5e-3)
            best = _max_cos(x_best)
    except Exception:
        pass
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 2.0:
            break
        x = x_best.copy()
        mode = cycle % 4
        if mode == 1:
            # pair split + short anneal
            try:
                x = _split_worst_pair(x, rng)
            except Exception:
                pass
            # small hub shake
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            prow = G.max(axis=1)
            over = _np.argsort(-prow)[:10]
            x[over] += 0.02 * rng.normal(size=(10, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 600, 60.0, 1500.0, 0.009
        elif mode == 2:
            # greedy sweep then anneal
            try:
                x = _greedy_sweep(x, rng, _time.time() + min(2.5, rem * 0.3))
            except Exception:
                pass
            steps, a0, a1, lr = 500, 80.0, 1500.0, 0.008
        elif mode == 3:
            # void reinsert + hub shake
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            cnt = _np.sum(G > gmax - 5e-4, axis=1)
            ratt = _np.where(cnt == 0)[0]
            if len(ratt) > 0:
                # reinsert up to 2 rattlers into voids
                for r in ratt[:2]:
                    Z = rng.normal(size=(6000, 6))
                    Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                    S = Z @ x.T
                    S[:, r] = -2.0
                    m2 = S.max(axis=1)
                    x[r] = Z[int(_np.argmin(m2))]
            else:
                try:
                    x = _void_reinsert(x, rng, n_void=1)
                except Exception:
                    pass
            prow = (x @ x.T).max(axis=1) if False else None
            x[:0] = x[:0]
            sig = 0.03 if (cycle % 8 == 3) else 0.06
            over_n = 12
            G2 = x @ x.T
            _np.fill_diagonal(G2, -2.0)
            prow2 = G2.max(axis=1)
            over = _np.argsort(-prow2)[:over_n]
            x[over] += sig * rng.normal(size=(over_n, 6))
            rnd = rng.choice(n, size=6, replace=False)
            x[rnd] += (sig * 0.6) * rng.normal(size=(6, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 800, 40.0, 1200.0, 0.012
        else:
            # large diverse kick
            pick = rng.choice(n, size=16, replace=False)
            x[pick] += 0.25 * rng.normal(size=(16, 6))
            # cap rotation near worst pair
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            fi = int(_np.argmax(G))
            ci = fi // n
            dists = x @ x[ci]
            cap = _np.argsort(-dists)[:12]
            # random small rotation in tangent: add orthogonal noise
            x[cap] += 0.08 * rng.normal(size=(12, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, lr = 700, 50.0, 1000.0, 0.012
        if rem < 20:
            steps = min(steps, 400)
        try:
            xb, vb = _adam_sm(x, steps, a0, a1, lr)
        except Exception:
            continue
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
        # occasional LP to equalize new basin
        if cycle % 6 == 0 and _time.time() < ANNEAL_END - 4:
            try:
                xp = _lp_polish(x_best, _time.time() + 2.5, r0=2e-3)
                vp = _max_cos(xp)
                if vp < best:
                    x_best, best = xp, vp
            except Exception:
                pass
    # final greedy + polish
    try:
        if _time.time() < LP_END - 6:
            x_best = _greedy_sweep(x_best, rng, _time.time() + 3.0)
    except Exception:
        pass
    try:
        if _time.time() < LP_END:
            x_best = _lp_polish(x_best, LP_END, r0=2.5e-3)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
