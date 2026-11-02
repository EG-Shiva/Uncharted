import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=0.004, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1600:
            idx = np.argpartition(-c, 1600)[:1600]
            sel2 = np.zeros_like(sel)
            sel2[idx] = True
            sel = sel2
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _softmax_descent(x0, steps, T0, T1, lr0, mu=0.85):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 * (1.0 - 0.7 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        vel = mu * vel + (1.0 - mu) * grad
        x -= lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _riesz_descent(x0, s0, s1, steps, lr0, mu=0.9):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        frac = k / max(1, steps - 1)
        s = s0 + (s1 - s0) * frac
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        D = 2.0 - 2.0 * G
        np.fill_diagonal(D, 1.0)
        np.maximum(D, 1e-12, out=D)
        dmin = float(D[~np.eye(n, dtype=bool)].min())
        pw = s / 2.0 + 1.0
        W = (dmin / D) ** pw
        np.fill_diagonal(W, 0.0)
        Gx = W @ x
        dot = (Gx * x).sum(axis=1, keepdims=True)
        grad = Gx - dot * x
        gn = float(np.sqrt((grad * grad).sum() / n))
        if gn > 1e-12:
            grad /= max(1.0, gn * 2.0)
        vel = mu * vel + (1.0 - mu) * grad
        x -= lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _deep_hole(others, rng, nc=14000):
    d = others.shape[1]
    cand = rng.randn(nc, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    chunk = 3500
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a + chunk)
        dots = cand[a:b] @ others.T
        m_all[a:b] = dots.max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi])
    best_c = cand[bi].copy()
    rad = 0.3
    for _ in range(3):
        prop = rng.randn(2500, d) * (rad * 0.5)
        prop -= (prop @ best_c)[:, None] * best_c[None, :]
        prop = best_c[None, :] + prop
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        dots = prop @ others.T
        mm = dots.max(axis=1)
        j = int(np.argmin(mm))
        if float(mm[j]) < best_v:
            best_v = float(mm[j])
            best_c = prop[j].copy()
        rad *= 0.45
    return best_c

def _refine_free(x_fixed, y_init, steps, T0, T1, lr0, mu=0.85):
    y = y_init.copy()
    k = y.shape[0]
    vel = np.zeros_like(y)
    def eval_max(yy):
        if k > 1:
            g1 = yy @ yy.T
            np.fill_diagonal(g1, -2.0)
            m1 = float(g1.max())
        else:
            m1 = -2.0
        m2 = float((yy @ x_fixed.T).max())
        return m1 if m1 > m2 else m2
    best_val = eval_max(y)
    best = y.copy()
    lT0 = np.log(T0); lT1 = np.log(T1)
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 * (1.0 - 0.7 * frac)
        Gyy = y @ y.T if k > 1 else None
        Gyx = y @ x_fixed.T
        if k > 1:
            np.fill_diagonal(Gyy, -2.0)
            m = float(Gyy.max())
        else:
            m = -2.0
        m2 = float(Gyx.max())
        if m2 > m:
            m = m2
        if k > 1:
            Eyy = np.exp((Gyy - m) / T)
            np.fill_diagonal(Eyy, 0.0)
            Syy = float(Eyy.sum() * 0.5)
        else:
            Eyy = None
            Syy = 0.0
        Eyx = np.exp((Gyx - m) / T)
        Syx = float(Eyx.sum())
        S = Syy + Syx
        if S < 1e-300:
            S = 1e-300
        grad = np.zeros_like(y)
        if k > 1:
            grad += (Eyy @ y) / S
        grad += (Eyx @ x_fixed) / S
        dot = (grad * y).sum(axis=1, keepdims=True)
        grad -= dot * y
        vel = mu * vel + (1.0 - mu) * grad
        y -= lr * vel
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        cm = eval_max(y)
        if cm < best_val:
            best_val = cm
            best = y.copy()
    return best, best_val

def solve():
    t0 = time.time()
    BUDGET = 173.0
    rng = np.random.RandomState(12345)
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands = []
    for z in (bx, rx):
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.shape == (163, 4) and np.all(np.isfinite(z)):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                cands.append(z)
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best_x = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best_x)
    n = 163
    cur = best_x.copy()
    cur_val = best_val
    SEARCH_END = t0 + 132.0
    FINAL_START = t0 + BUDGET - 38.0
    trial = 0
    s_pairs = [(15, 60), (40, 150), (80, 300), (10, 30), (25, 120)]
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.008, 0.035]
    while time.time() < SEARCH_END:
        remain = SEARCH_END - time.time()
        if remain < 1.5:
            break
        if time.time() > FINAL_START:
            break
        frac = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = 2.5e-4 * (1e-6 / 2.5e-4) ** frac
        sigma = sigmas[trial % len(sigmas)]
        r = rng.rand()
        cc = cur.copy()
        use_riesz = False
        rs0, rs1 = s_pairs[(trial // 2) % len(s_pairs)]
        if r < 0.05 and trial > 10:
            cc = rng.randn(n, 4)
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
            use_riesz = True
        elif r < 0.28:
            cc = cc + rng.randn(*cc.shape) * sigma
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
            use_riesz = ((trial % 3) == 0)
        elif r < 0.45:
            G = cc @ cc.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            idx = np.argsort(deg)[-10:]
            noise = rng.randn(*cc.shape) * (sigma * 0.25)
            noise[idx] += rng.randn(len(idx), 4) * (sigma * 2.0)
            cc = cc + noise
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        elif r < 0.60:
            G = cc @ cc.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.003).sum(axis=1)
            w = int(np.argmax(deg + rng.rand(n) * 0.5))
            neigh = np.where(G[w] > m - 0.006)[0]
            cluster = np.unique(np.concatenate([[w], neigh]))[:12]
            shift = rng.randn(4) * (sigma * 1.2)
            for ii in cluster:
                cc[ii] = cc[ii] + shift * 0.6 + rng.randn(4) * (sigma * 0.5)
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
            use_riesz = ((trial % 2) == 0)
        elif r < 0.80:
            G = cc @ cc.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            k = 2 + (trial % 4)
            worst = np.argsort(deg + rng.rand(n) * 0.3)[-k:]
            keep = np.delete(np.arange(n), worst)
            xf = cc[keep].copy()
            yinit = np.zeros((k, 4))
            xt = xf.copy()
            for ii in range(k):
                h = _deep_hole(xt, rng, nc=12000)
                h = h + rng.randn(4) * 0.01
                h /= np.linalg.norm(h)
                yinit[ii] = h
                xt = np.vstack([xt, h[None, :]])
            yopt, _ = _refine_free(xf, yinit, steps=1800, T0=0.015, T1=0.0012, lr0=0.04)
            for ii, w in enumerate(worst):
                cc[w] = yopt[ii]
            cc = cc + rng.randn(*cc.shape) * (sigma * 0.12)
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        elif r < 0.90:
            G = cc @ cc.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.0015).sum(axis=1)
            w = int(np.argmin(deg + rng.rand(n) * 0.2))
            keep = np.delete(np.arange(n), w)
            xf = cc[keep]
            h = _deep_hole(xf, rng, nc=16000)
            h = h + rng.randn(4) * 0.005
            h /= np.linalg.norm(h)
            cc[w] = h
            cc = cc + rng.randn(*cc.shape) * (sigma * 0.3)
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        else:
            k = 20 + (trial % 12)
            idx = rng.choice(n, k, replace=False)
            cc[idx] = cc[idx] + rng.randn(k, 4) * (sigma * 1.5)
            cc /= np.linalg.norm(cc, axis=1, keepdims=True)
            use_riesz = True
        if use_riesz:
            cur_r, v_r = _riesz_descent(cc, rs0, rs1, steps=2600, lr0=0.03)
            cur_r2, v2 = _softmax_descent(cur_r, steps=2200, T0=0.008, T1=0.0012, lr0=0.030)
            if v2 < v_r:
                cur_f, v = cur_r2, v2
            else:
                cur_f, v = cur_r, v_r
        else:
            if r >= 0.45 and r < 0.80:
                cur_f, v = _softmax_descent(cc, steps=3600, T0=0.012, T1=0.0010, lr0=0.032)
            else:
                cur_f, v = _softmax_descent(cc, steps=3800, T0=0.015, T1=0.0011, lr0=0.030)
        if v < best_val - 1e-12:
            pol = _lp_polish(cur_f, seconds=2.0, t_deadline=t0 + BUDGET - 30.0)
            vp = _max_cos(pol)
            if vp < best_val:
                best_x, best_val = pol, vp
                cur, cur_val = pol.copy(), vp
            elif v < best_val:
                best_x, best_val = cur_f.copy(), v
                cur, cur_val = cur_f.copy(), v
            else:
                if v < cur_val:
                    cur, cur_val = cur_f.copy(), v
        elif v < cur_val:
            cur, cur_val = cur_f.copy(), v
            if v < best_val + 1.2e-4 and (trial % 6 == 2) and time.time() < t0 + BUDGET - 34.0:
                pol = _lp_polish(cur_f, seconds=1.0, t_deadline=t0 + BUDGET - 30.0)
                vp = _max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
                    cur, cur_val = pol.copy(), vp
        else:
            if rng.rand() < np.exp((cur_val - v) / max(1e-12, Temp)):
                cur, cur_val = cur_f.copy(), v
            elif v < best_val + 4e-4 and (trial % 9 == 5) and time.time() < t0 + BUDGET - 36.0:
                pol = _lp_polish(cur_f, seconds=0.9, t_deadline=t0 + BUDGET - 30.0)
                vp = _max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
                    cur, cur_val = pol.copy(), vp
        trial += 1
        if trial % 15 == 0 and time.time() < t0 + BUDGET - 40.0:
            if rng.rand() < 0.5:
                cur = best_x.copy()
                cur_val = best_val
                cur = cur + rng.randn(*cur.shape) * 0.03
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    # fine anneal of best
    if time.time() < t0 + BUDGET - 30.0:
        fb, fv = _softmax_descent(best_x, steps=6000, T0=0.004, T1=0.0004, lr0=0.018)
        if fv < best_val:
            best_x, best_val = fb, fv
    # final LP polish with remaining budget
    rem = (t0 + BUDGET - 2.0) - time.time()
    if rem > 3.0:
        best_x = _lp_polish(best_x, seconds=rem, r0=0.003, t_deadline=t0 + BUDGET - 1.0)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
