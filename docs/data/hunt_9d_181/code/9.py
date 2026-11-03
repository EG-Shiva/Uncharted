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
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 2800:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:2800]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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


def power_smooth(x, iters=400, p0=30, p1=110, lr=0.05, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p0 + (p1 - p0) * frac
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
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        # bias correction-ish scale
        nrm = np.mean(np.linalg.norm(vel, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nrm
        x = x - step * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def best_void(y, exclude, rng, nt=16000):
    n, d = y.shape
    R = rng.standard_normal((nt, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ y.T
    if len(exclude) > 0:
        S[:, list(exclude)] = -10.0
    sc = S.max(axis=1)
    return R[int(np.argmin(sc))]


def contacts_of(y, thresh=1e-7):
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    np.fill_diagonal(G, 1.0)
    c = np.sum(G > m - thresh, axis=1) - 1
    return G, m, c


def solve():
    t0 = time.time()
    BUDGET = 169.0
    rng = np.random.default_rng(42)
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
        tmp = rng.standard_normal((181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = scores[bi]
    n, d = xb.shape
    rem = BUDGET - (time.time() - t0)
    if rem > 12:
        xb = lp_polish(xb, seconds=min(3.5, rem - 8.0), init_radius=2e-3)
        bb = max_cos(xb)
    xc = xb.copy()
    bc = bb
    T = 6e-05
    it = 0
    iu = np.triu_indices(n, 1)
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 8
        if mode == 0:
            y = y + rng.standard_normal(y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008
            do_smooth = False
        elif mode == 1:
            y = y + rng.standard_normal(y.shape) * 0.045
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.016
            do_smooth = True
            sit, sp1 = 350, 90
        elif mode == 2:
            y = y + rng.standard_normal(y.shape) * 0.11
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.028
            do_smooth = True
            sit, sp1 = 450, 110
        elif mode == 3:
            _, _, cc = contacts_of(y, thresh=1e-7)
            order = np.argsort(cc + rng.random(n) * 0.5)
            nr = 2 if (it % 2 == 0) else 3
            ex = []
            for r in order[:nr]:
                ex.append(int(r))
                v = best_void(y, ex, rng, nt=16000)
                # small offset so duplicates don't coincide
                y[int(r)] = v
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.016
            do_smooth = True
            sit, sp1 = 400, 100
        elif mode == 4:
            _, _, cc = contacts_of(y, thresh=1e-7)
            top = int(np.argmax(cc + rng.random(n) * 0.3))
            G2 = y @ y.T
            np.fill_diagonal(G2, -10.0)
            sims = G2[top]
            nn = np.argsort(-sims)[:12]
            grp = np.concatenate([[top], nn])
            y[grp] = y[grp] + rng.standard_normal((len(grp), d)) * 0.16
            # also jitter rest lightly
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.024
            do_smooth = True
            sit, sp1 = 400, 100
        elif mode == 5:
            k = int(rng.integers(3, 6))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
            init_r = 0.030
            do_smooth = True
            sit, sp1 = 450, 110
        elif mode == 6:
            G3 = y @ y.T
            np.fill_diagonal(G3, -2.0)
            c = G3[iu]
            K = 80
            o = np.argsort(-c)[:K]
            pick = o[int(rng.integers(0, K))]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            # teleport one endpoint to void, push other slightly
            ex = [i0]
            v = best_void(y, ex, rng, nt=16000)
            y[i0] = v
            y[j0] = y[j0] + rng.standard_normal(d) * 0.03
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.020
            do_smooth = True
            sit, sp1 = 400, 100
        else:
            # double-void: move two low-contact points near the same big hole with separation
            _, _, cc = contacts_of(y, thresh=1e-7)
            order = np.argsort(cc)
            r1, r2 = int(order[0]), int(order[1])
            v = best_void(y, [r1, r2], rng, nt=20000)
            # create second vector orthogonal-ish to v
            w = rng.standard_normal(d)
            w = w - np.dot(w, v) * v
            w /= (np.linalg.norm(w) + 1e-12)
            ang = 0.35  # ~ separation angle proxy
            y[r1] = (v + w * ang * 0.5)
            y[r2] = (v - w * ang * 0.5)
            y[r1] /= np.linalg.norm(y[r1]); y[r2] /= np.linalg.norm(y[r2])
            y = y + rng.standard_normal(y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.018
            do_smooth = True
            sit, sp1 = 400, 110
        if do_smooth:
            y = power_smooth(y, iters=sit, p0=28, p1=sp1, lr=0.055, mu=0.85)
            vs = max_cos(y)
            # skip expensive LP if far from current
            if vs > bc + 0.009 and vs > bb + 0.004:
                # still allow SA move without LP occasionally
                if rng.random() < 0.9:
                    T *= 0.9995
                    continue
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 3.0:
            break
        budg = min(2.4, max(0.9, rem2 - 2.0))
        # smaller budget for exploit mode
        if mode == 0:
            budg = min(budg, 1.2)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-14:
            xb = y2.copy(); bb = v
            xc = y2.copy(); bc = v
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 9:
                xb2 = lp_polish(xb, seconds=1.2, init_radius=8e-4)
                vv = max_cos(xb2)
                if vv < bb:
                    xb = xb2; bb = vv
                    xc = xb2.copy(); bc = vv
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-9)):
                xc = y2.copy(); bc = v
        T *= 0.9994
        if T < 7e-06:
            T = 7e-06
        # occasional reset of current to best if wandering too high
        if bc > bb + 0.004 and rng.random() < 0.25:
            xc = xb.copy(); bc = bb
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = power_smooth(xb, iters=300, p0=40, p1=160, lr=0.03, mu=0.9)
            if max_cos(yf) < bb + 0.001:
                yf = lp_polish(yf, seconds=min(remf - 0.6, 6.0), init_radius=1e-3)
                vf = max_cos(yf)
                if vf < bb:
                    xb = yf; bb = vf
            else:
                yf = lp_polish(xb, seconds=min(remf - 0.6, 6.0), init_radius=8e-4)
                vf = max_cos(yf)
                if vf < bb:
                    xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
