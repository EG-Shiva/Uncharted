import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            kth = np.partition(c_all, -3000)[-3000]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_anneal(x, iters=220, p_start=30, p_end=140, lr=0.04, moving=None):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = Gp.max()
        if m <= 1e-12:
            break
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        if moving is not None:
            mask = np.zeros(n, dtype=bool); mask[moving] = True
            gt = np.zeros_like(grad_t); gt[mask] = grad_t[mask]
            grad_t = gt
            if np.allclose(grad_t, 0):
                continue
        nrm = np.mean(np.linalg.norm(grad_t, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nrm
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def refine_hole(v, B, steps=60, p=40):
    # maximise minimal angle: minimise max dot to background B
    v = v / np.linalg.norm(v)
    for s in range(steps):
        dots = B @ v
        m = dots.max()
        # softmax over top neighbours
        w = np.exp(p * (dots - m))
        w = w / (w.sum() + 1e-300)
        g = w @ B  # weighted worst direction
        gt = g - np.dot(g, v) * v
        nrm = np.linalg.norm(gt) + 1e-12
        alpha = 0.05 * (1.0 - 0.7 * s / max(1, steps - 1))
        v = v - (alpha / (nrm + 1e-12)) * gt * nrm
        # simpler: fixed tangential step
        # recompute correctly:
        # actually step along -gt
        # redo with stable magnitude
        # (above line already did) just renormalize
        v = v / (np.linalg.norm(v) + 1e-300)
        # re-project: small correction iterations
        # adjust: use gradient descent with normalized step
        # To avoid bug, do explicit:
        # (we already moved; continue)
    return v / np.linalg.norm(v)

def best_hole(B, rng, n_try=16000, n_refine=4):
    d = B.shape[1]
    R = rng.normal(size=(n_try, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ B.T
    sc = S.max(axis=1)
    order = np.argsort(sc)[:n_refine]
    best_v = R[order[0]].copy()
    best_s = sc[order[0]]
    for oi in order:
        v = refine_hole(R[oi], B, steps=50)
        s = float((B @ v).max())
        if s < best_s:
            best_s = s; best_v = v
    return best_v

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        tmp = rng.normal(size=(181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy(); bb = scores[bi]
    rem = BUDGET - (time.time() - t0)
    if rem > 10:
        xb = lp_polish(xb, seconds=min(3.0, rem - 6.0), init_radius=2e-3)
        bb = max_cos(xb)
    xc = xb.copy(); bc = bb
    n, d = xb.shape
    T = 3e-05
    it = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 3.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 8
        if mode == 0:
            y = y + rng.normal(size=y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008; sm_p1 = 140; sm_iters = 150; do_local = False
        elif mode == 1:
            # single overloaded/rattler hole fill (topology change)
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            # pick worst overloaded or rattler
            if rng.random() < 0.6:
                r = int(np.argmax(contacts))
            else:
                # random low-contact point
                order = np.argsort(contacts)
                r = int(order[rng.integers(0, min(12, n))])
            B = np.delete(y, r, axis=0)
            v = best_hole(B, rng, n_try=16000, n_refine=4)
            y[r] = v
            y = y + rng.normal(size=y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; sm_p1 = 140; sm_iters = 220; do_local = False
        elif mode == 2:
            # double hole fill for two worst
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            order = np.argsort(-contacts)
            k = 2
            victims = order[:k].copy()
            # sequential greedy
            for r in victims:
                B = y.copy()
                # exclude other victims by masking: use remaining points only
                mask = np.ones(n, dtype=bool); mask[r] = False
                # also exclude not-yet-placed victims? keep them, they will move anyway
                Bg = y[mask]
                v = best_hole(Bg, rng, n_try=12000, n_refine=3)
                y[r] = v
            y = y + rng.normal(size=y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.02; sm_p1 = 140; sm_iters = 220; do_local = False
        elif mode == 3:
            # cap excision: remove k points in random cap, repack with local anneal
            vdir = rng.normal(size=d); vdir /= np.linalg.norm(vdir)
            dots = y @ vdir
            k = int(rng.integers(6, 13))
            victims = np.argsort(-dots)[:k]
            # reinit victims randomly biased to cap + greedy spread
            R = rng.normal(size=(k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            # push toward cap centre partially
            R = R + 0.7 * vdir[None, :]
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[victims] = R
            # local anneal moving only victims (spread + avoid background)
            y = smooth_anneal(y, iters=120, p_start=20, p_end=60, lr=0.06, moving=victims)
            init_r = 0.025; sm_p1 = 140; sm_iters = 200; do_local = False
        elif mode == 4:
            # overloaded neighbourhood shake
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            top = int(np.argmax(contacts))
            sims = G[top].copy(); sims[top] = -10
            nn = np.argsort(-sims)[:10]
            grp = np.concatenate([[top], nn])
            y[grp] = y[grp] + rng.normal(size=(len(grp), d)) * 0.12
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025; sm_p1 = 130; sm_iters = 220; do_local = False
        elif mode == 5:
            # contact-pair break
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            K = 60
            o = np.argsort(-c)[:K]
            pick = o[rng.integers(0, K)]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            diff = y[i0] - y[j0]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[i0] = y[i0] + diff * 0.06 + rng.normal(size=d) * 0.02
            y[j0] = y[j0] - diff * 0.06 + rng.normal(size=d) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; sm_p1 = 150; sm_iters = 200; do_local = False
        elif mode == 6:
            y = y + rng.normal(size=y.shape) * 0.03
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015; sm_p1 = 120; sm_iters = 220; do_local = False
        else:
            y = y + rng.normal(size=y.shape) * 0.09
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.03; sm_p1 = 110; sm_iters = 250; do_local = False
        # smoothing with higher terminal p to mimic max
        y = smooth_anneal(y, iters=sm_iters, p_start=25, p_end=sm_p1, lr=0.04)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 2.5:
            break
        budg = min(2.6, rem2 - 1.0)
        if budg < 0.8:
            budg = max(0.6, rem2 - 0.5)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-14:
            xb = y2.copy(); bb = v
            xc = y2.copy(); bc = v
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 8:
                xb2 = lp_polish(xb, seconds=1.2, init_radius=8e-4)
                vv = max_cos(xb2)
                if vv < bb:
                    xb = xb2; bb = vv; xc = xb2.copy(); bc = vv
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-9)):
                xc = y2.copy(); bc = v
        T *= 0.9994
        if T < 7e-06:
            T = 7e-06
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = lp_polish(xb, seconds=min(remf - 0.5, 5.0), init_radius=8e-4)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
