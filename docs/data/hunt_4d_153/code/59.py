import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def _contact_counts(x, thresh=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    gall = G[iu]
    mx = float(gall.max())
    tight = gall > mx - thresh
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return mx, cnt, G


def _anneal(x0, steps, beta0, beta1, lr0, lr1):
    n, d = x0.shape
    x = x0.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    if steps <= 1:
        lb = 0.0
        bb = 0.0
    else:
        lb = np.log(lr1 / lr0) / (steps - 1)
        bb = np.log(beta1 / beta0) / (steps - 1)
    for s in range(steps):
        beta = beta0 * np.exp(bb * s)
        lr = lr0 * np.exp(lb * s)
        G = x @ x.T
        g = G[iu0, iu1]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        if not np.isfinite(Z) or Z == 0:
            e = np.ones_like(e) / len(e)
        else:
            e = e / Z
        W = np.zeros((n, n))
        W[iu0, iu1] = e
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x


def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _void_point(x, rng, n_sample=30000, refine=250):
    n = x.shape[0]
    cand = rng.normal(size=(n_sample, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = x @ cand.T
    worst = dots.max(axis=0)
    idx = int(np.argmin(worst))
    c = cand[idx].copy()
    best_w = float(worst[idx])
    sig = 0.06
    for it in range(refine):
        sig_it = sig * (1.0 - it / refine) + 0.004 * (it / refine)
        prop = c + rng.normal(0, sig_it, size=4)
        prop /= np.linalg.norm(prop)
        w = float((x @ prop).max())
        if w < best_w:
            c = prop
            best_w = w
    return c


def _best_removal(x):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    iu0, iu1 = np.triu_indices(n, 1)
    vals = G[iu0, iu1]
    order = np.argsort(-vals)
    so0 = iu0[order]
    so1 = iu1[order]
    rem = np.empty(n)
    for i in range(n):
        for k in range(len(so0)):
            a = int(so0[k]); b = int(so1[k])
            if a != i and b != i:
                rem[i] = vals[order[k]]
                break
    return rem


def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    gval = best_val
    gx = x_best.copy()

    def remaining():
        return HARD - (time.time() - t0)

    def try_update_153(y):
        nonlocal gx, gval
        v = _max_cos(y)
        if v < gval:
            gx = y / np.linalg.norm(y, axis=1, keepdims=True)
            gval = float(v)
            return True
        return False

    # ---------- Path A: down to 148 then regrow to 153 ----------
    try:
        if remaining() > 40:
            mx, cnt, G = _contact_counts(gx, 1e-7)
            ppm = _per_point_max(gx)
            ratt = int(np.argmin(cnt))
            order_hi = np.argsort(-ppm)
            victims = [ratt]
            for o in order_hi:
                if int(o) not in victims:
                    victims.append(int(o))
                if len(victims) >= 5:
                    break
            victims = np.array(sorted(victims[:5]))
            mask = np.ones(153, dtype=bool)
            mask[victims] = False
            xa = gx[mask].copy()  # 148
            xa = _anneal(xa, 1500, 80.0, 5000.0, 0.009, 0.0008)
            if remaining() > 10:
                xa, _ = _lp_polish(xa, min(3.0, max(0.5, remaining() - 30)))
            # regrow 149..153 via voids
            for N in range(149, 154):
                if remaining() < 12:
                    break
                c = _void_point(xa, rng, n_sample=30000, refine=250)
                xa = np.vstack([xa, c.reshape(1, 4)])
                xa /= np.linalg.norm(xa, axis=1, keepdims=True)
                if N < 153:
                    xa = _anneal(xa, 1000, 120.0, 5000.0, 0.009, 0.001)
                    xa, _ = _lp_polish(xa, min(1.5, max(0.5, remaining() - 20)))
                else:
                    xa = _anneal(xa, 1400, 150.0, 7000.0, 0.009, 0.0008)
                    xa, va = _lp_polish(xa, min(4.0, max(1.0, remaining() - 18)))
                    try_update_153(xa)
    except Exception:
        pass

    # ---------- Path B: up to ~158 then greedy delete to 153 ----------
    try:
        if remaining() > 35:
            xb2 = gx.copy()
            topN = 158
            for N in range(154, topN + 1):
                if remaining() < 15:
                    break
                c = _void_point(xb2, rng, n_sample=25000, refine=200)
                xb2 = np.vstack([xb2, c.reshape(1, 4)])
                xb2 /= np.linalg.norm(xb2, axis=1, keepdims=True)
                xb2 = _anneal(xb2, 600, 100.0, 4000.0, 0.009, 0.001)
            if remaining() > 12:
                xb2, _ = _lp_polish(xb2, min(2.0, max(0.5, remaining() - 25)))
            # delete down to 153
            while xb2.shape[0] > 153 and remaining() > 10:
                ncur = xb2.shape[0]
                rem = _best_removal(xb2)
                # tie-break: among near-minimal, remove highest contact count
                thr = rem.min() + 1e-12
                cands = np.where(rem <= thr)[0]
                if len(cands) > 1:
                    _, cntc, _ = _contact_counts(xb2, 1e-7)
                    pick = int(cands[np.argmax(cntc[cands])])
                else:
                    pick = int(np.argmin(rem))
                mask = np.ones(ncur, dtype=bool)
                mask[pick] = False
                xb2 = xb2[mask].copy()
                xb2 = _anneal(xb2, 900, 120.0, 6000.0, 0.009, 0.001)
                if xb2.shape[0] == 153:
                    xb2, _ = _lp_polish(xb2, min(3.0, max(1.0, remaining() - 12)))
                    try_update_153(xb2)
                else:
                    if remaining() > 12:
                        xb2, _ = _lp_polish(xb2, min(1.2, max(0.5, remaining() - 12)))
    except Exception:
        pass

    # ---------- Extra topology hops: void-reinsertion on 153 ----------
    try:
        it = 0
        while remaining() > 22:
            it += 1
            if remaining() < 10:
                break
            y = gx.copy()
            _, cnt, _ = _contact_counts(y, 1e-7)
            ppm = _per_point_max(y)
            # choose 1-3 victims: hubs + worst per-point-max
            k = 1 + (it % 3)
            hubs = np.argsort(-cnt)[:6]
            worst = np.argsort(-ppm)[:6]
            pool = list(dict.fromkeys(list(hubs[:3]) + list(worst[:3]) + [int(np.argmax(cnt))]))
            rng.shuffle(pool)
            victims = np.array(sorted(pool[:k]))
            keep_mask = np.ones(153, dtype=bool)
            keep_mask[victims] = False
            kept = y[keep_mask]
            # sequentially insert voids w.r.t. kept+already chosen
            chosen = []
            base = kept.copy()
            for _ in range(k):
                c = _void_point(base, rng, n_sample=20000, refine=150)
                chosen.append(c)
                base = np.vstack([base, c.reshape(1, 4)])
            y[victims] = np.array(chosen)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 1000, 100.0, 6000.0, 0.009, 0.001)
            v = _max_cos(y)
            if v < gval - 1e-12:
                gx = y
                gval = v
                gx, gval2 = _lp_polish(gx, min(2.0, max(0.8, remaining() - 10)))
                gval = min(gval, float(gval2))
            elif v < gval + 0.002 and (it % 2 == 0) and remaining() > 14:
                yp, vp = _lp_polish(y, 1.0)
                if vp < gval:
                    gx = yp
                    gval = float(vp)
    except Exception:
        pass

    # ---------- final polish ----------
    try:
        rem = remaining() - 1.0
        if rem > 3.0:
            try:
                ya = _anneal(gx, 500, 2000.0, 12000.0, 0.003, 0.0005)
                va = _max_cos(ya)
                cand = ya if va < gval else gx
                xp, vp = _lp_polish(cand, min(rem, 20.0))
                if vp < gval:
                    gx, gval = xp, float(vp)
            except Exception:
                xp, vp = _lp_polish(gx, min(rem, 15.0))
                if vp < gval:
                    gx, gval = xp, float(vp)
    except Exception:
        pass

    gx = gx / np.linalg.norm(gx, axis=1, keepdims=True)
    return gx
