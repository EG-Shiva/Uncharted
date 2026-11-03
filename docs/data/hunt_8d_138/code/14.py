import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g


def _adam_smooth(x, steps, beta0, beta1, lr, log_beta=False):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        if log_beta:
            beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1 * f
        else:
            beta = beta0 + (beta1 - beta0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        nd = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        upd = mhat / (np.sqrt(vhat) + eps)
        nd2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - nd2 * x
        x = x - lr * upd
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x


def _sgd_smooth(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tan = grad - dot * x
        # normalize per-point to keep step stable
        nr = np.sqrt(np.sum(tan * tan, axis=1, keepdims=True)) + 1e-12
        # mix normalized and raw for jammed emphasis
        x = x - lr * tan / np.maximum(nr * 0.5, 1.0)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
    return x


def _lp_polish(x, deadline, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        mv = c.max()
        thr = mv - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _void_fill(x, keep_mask, rng, n_samples=4000):
    # replace points where keep_mask False by greedy maximal voids
    n, d = x.shape
    idx_bad = np.where(~keep_mask)[0]
    # random candidates on sphere
    cand = rng.normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    for j in idx_bad:
        # dots to all kept + already placed
        others = x[keep_mask]
        # keep_mask updated as we place
        dots = cand @ others.T
        worst = dots.max(axis=1)
        pick = int(np.argmin(worst))
        x[j] = cand[pick]
        keep_mask[j] = True
        # remove nearby candidates to diversify (optional): replace picked
        # mutate candidate set slightly for next point
        if j != idx_bad[-1]:
            # refresh a chunk
            r = rng.normal(size=(512, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            cand[rng.integers(0, n_samples, size=512)] = r
    return x


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv /= np.linalg.norm(vv, axis=1, keepdims=True)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    cur = best.copy()
    cur_val = best_val

    it = 0
    # temperature for wandering acceptance
    while time.time() < deadline - 8.0 and it < 500:
        it += 1
        now = time.time()
        # choose source: mostly best, sometimes cur
        use_cur = (it % 3 == 2)
        src = cur if use_cur else best
        src_val = cur_val if use_cur else best_val
        mode = it % 6

        y = src.copy()
        if mode == 0:
            # global kick medium + adam
            sigma = 0.12 if (it % 12 != 0) else 0.30
            y = y + sigma * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_smooth(y, steps=500, beta0=15.0, beta1=150.0, lr=0.006)
        elif mode == 1:
            # kill worst k into voids then adam
            ppm, _ = _per_point_max(y)
            k = 4 if (it % 2 == 0) else 7
            worst = np.argsort(-ppm)[:k]
            keep = np.ones(n, dtype=bool)
            keep[worst] = False
            # temporarily remove
            y = _void_fill(y, keep, rng, n_samples=4000)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_smooth(y, steps=600, beta0=12.0, beta1=180.0, lr=0.006)
        elif mode == 2:
            # shake jammed core only
            ppm, G = _per_point_max(y)
            mglob = ppm.max()
            cnt = np.sum(G > mglob - 0.01, axis=1)
            order = np.argsort(-cnt)
            pool = order[:12]
            y[pool] += 0.35 * rng.normal(size=(len(pool), d))
            y += 0.015 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _sgd_smooth(y, steps=700, beta0=20.0, beta1=250.0, lr0=0.02, lr1=0.002)
        elif mode == 3:
            # partial restart: keep 100 good, resample 38
            ppm, _ = _per_point_max(y)
            order = np.argsort(ppm)  # ascending: good first
            keep_n = 100
            keep = np.zeros(n, dtype=bool)
            keep[order[:keep_n]] = True
            # also keep some random to diversify
            y = _void_fill(y, keep, rng, n_samples=5000)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_smooth(y, steps=700, beta0=10.0, beta1=220.0, lr=0.007)
        elif mode == 4:
            # small kick + high-beta squeeze
            sigma = 0.03
            y = y + sigma * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_smooth(y, steps=400, beta0=40.0, beta1=300.0, lr=0.004)
        else:
            # fresh random basin occasionally
            if it % 18 == 0:
                y = rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _adam_smooth(y, steps=900, beta0=8.0, beta1=200.0, lr=0.007)
            else:
                sigma = 0.06
                y = y + sigma * rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _sgd_smooth(y, steps=600, beta0=15.0, beta1=200.0, lr0=0.025, lr1=0.003)

        v_smooth = _max_cos(y)
        # decide whether LP polish is worthwhile
        # always polish if close to best, else polish occasionally
        need_lp = False
        if v_smooth < best_val + 0.008:
            need_lp = True
        elif v_smooth < src_val + 0.002 and (it % 4 == 0):
            need_lp = True
        if need_lp:
            lp_dead = min(deadline - 8.0, time.time() + 1.6)
            if lp_dead > time.time() + 0.2:
                try:
                    y = _lp_polish(y, lp_dead)
                except Exception:
                    pass
            v = _max_cos(y)
        else:
            v = v_smooth

        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        else:
            # wandering acceptance
            if v < cur_val:
                cur = y.copy()
                cur_val = v
            else:
                T = 0.002 * max(0.0, (deadline - 8.0 - time.time()) / 160.0) + 0.0002
                if rng.random() < np.exp(-(v - cur_val) / T):
                    cur = y.copy()
                    cur_val = v
                # periodically reset wandering chain to best
                if it % 25 == 0:
                    cur = best.copy()
                    cur_val = best_val
        # early exit if time low
        if time.time() > deadline - 8.0:
            break

    # final tight polish of incumbent with remaining time
    try:
        fin_dead = min(deadline, time.time() + 7.0)
        best = _adam_smooth(best, steps=300, beta0=60.0, beta1=400.0, lr=0.003)
        best = _lp_polish(best, fin_dead, cap=3000)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
