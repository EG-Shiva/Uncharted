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
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            # trim to 2500 tightest deterministically
            idx = np.where(sel)[0]
            if len(idx) > 2600:
                order = np.argsort(-c_all[idx])[:2500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
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

def smooth_exp(x, steps, alpha, s0, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        rms = np.sqrt(np.mean(vel * vel)) + 1e-12
        step = s0 * (1.0 - 0.5 * k / max(1, steps - 1))
        x = x - (step / rms) * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def best_hole(x, exclude, rng, nt=4000):
    n, d = x.shape
    R = rng.standard_normal((nt, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x.T
    if exclude is not None:
        S[:, exclude] = 10.0  # don't count self to maximise freedom? we want min over others
        # actually for replacement we want max over others only; self column irrelevant
        # set self col to -inf so it is ignored
        S[:, exclude] = -10.0
    sc = S.max(axis=1)
    return R[int(np.argmin(sc))]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(1)
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
    rem = BUDGET - (time.time() - t0)
    if rem > 12:
        xb = lp_polish(xb, seconds=min(3.0, rem - 8.0), init_radius=2e-3)
        bb = max_cos(xb)
    xc = xb.copy()
    bc = bb
    n, d = xb.shape
    T = 3e-05
    it = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 7
        if mode == 0:
            y = y + rng.standard_normal(y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008
            do_smooth = False
            s_steps, s_alpha, s_s0 = 0, 80.0, 0.005
        elif mode == 1:
            y = y + rng.standard_normal(y.shape) * 0.035
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            do_smooth = True
            s_steps, s_alpha, s_s0 = 250, 90.0, 0.006
        elif mode == 2:
            k = 2 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
            init_r = 0.03
            do_smooth = True
            s_steps, s_alpha, s_s0 = 350, 70.0, 0.006
        elif mode == 3:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            order = np.argsort(contacts)
            for r in order[:3]:
                y[r] = best_hole(y, int(r), rng, nt=4000)
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            do_smooth = True
            s_steps, s_alpha, s_s0 = 300, 100.0, 0.005
        elif mode == 4:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            top = int(np.argmax(contacts))
            sims = G[top].copy()
            sims[top] = -10
            nn = np.argsort(-sims)[:10]
            grp = np.concatenate([[top], nn])
            y[grp] = y[grp] + rng.standard_normal((len(grp), d)) * 0.12
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025
            do_smooth = True
            s_steps, s_alpha, s_s0 = 300, 80.0, 0.006
        elif mode == 5:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            K = 50
            o = np.argsort(-c)[:K]
            pick = o[int(rng.integers(0, K))]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            diff = y[i0] - y[j0]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[i0] = y[i0] + diff * 0.05 + rng.standard_normal(d) * 0.02
            y[j0] = y[j0] - diff * 0.05 + rng.standard_normal(d) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            do_smooth = True
            s_steps, s_alpha, s_s0 = 250, 120.0, 0.005
        else:
            y = y + rng.standard_normal(y.shape) * 0.09
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.03
            do_smooth = True
            s_steps, s_alpha, s_s0 = 350, 60.0, 0.007
        if do_smooth:
            # two-stage alpha sharpening if budget allows
            y = smooth_exp(y, steps=s_steps, alpha=s_alpha, s0=s_s0)
            # light second pass at higher alpha every other iter
            if it % 2 == 0:
                y = smooth_exp(y, steps=150, alpha=min(300.0, s_alpha * 2.5), s0=0.003)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 3.0:
            break
        budg = min(2.2, rem2 - 1.2)
        if budg < 0.7:
            budg = max(0.6, rem2 - 0.8)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-14:
            xb = y2.copy()
            bb = v
            xc = y2.copy()
            bc = v
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 9:
                xb2 = lp_polish(xb, seconds=1.2, init_radius=8e-4)
                vv = max_cos(xb2)
                if vv < bb:
                    xb = xb2
                    bb = vv
                    xc = xb2.copy()
                    bc = vv
        else:
            if v < bc or rng.random() < np.exp((bc - v) / T):
                xc = y2.copy()
                bc = v
        T *= 0.9994
        if T < 7e-06:
            T = 7e-06
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = lp_polish(xb, seconds=min(remf - 0.6, 4.0), init_radius=8e-4)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf
                bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
