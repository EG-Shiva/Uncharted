import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 > 0 and T1 > 0:
        lratio = np.log(T1 / T0)
    else:
        lratio = 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.003, 0.001, 0.0004), iters=(80, 80, 120)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            Yt /= np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
            y = Yt.ravel().copy()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _refine_h(h0, Xf, steps=200, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(np.linalg.norm(h0), 1e-12)
    if Xf.shape[0] == 0:
        return h
    lratio = np.log(T1 / T0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = np.linalg.norm(h)
        if nh < 1e-12:
            break
        h /= nh
    return h

def _best_hole(Xf, rng, n_samples=8000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:4]
    best_h = Y[idx[0]]
    best_v = float(vals[idx[0]])
    for ii in idx:
        h = _refine_h(Y[ii], Xf)
        v = float((Xf @ h).max()) if Xf.shape[0] > 0 else -2.0
        if v < best_v:
            best_v = v
            best_h = h
    return best_h, best_v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(42)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=float)
        if bx.shape == (163, 4):
            cands.append(bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12))
    if rec is not None:
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (163, 4):
            cands.append(rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12))
    if len(cands) == 0:
        c = rng.randn(n, 4)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = best_x.copy()
    cur_val = best_val
    Tb = 8e-05
    trial = 0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 38:
            break
        trial += 1
        r = rng.rand()
        base_use_cur = rng.rand() < 0.65
        base = cur_x if base_use_cur else best_x
        cur_init = base.copy()
        # choose surgery
        if trial % 23 == 0:
            # full random restart, short anneal
            cur_init = rng.randn(n, 4)
            cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            dl = min(time.time() + 4.5, SEARCH_END)
            tb, vb = _gd(cur_init, steps=6500, T0=0.025, T1=0.0008, lr0=0.04, mu=0.88, rng=rng, noise0=0.002, deadline=dl)
            # basin accept
            if vb < cur_val or rng.rand() < np.exp(-max(0.0, vb - cur_val) / Tb):
                cur_x, cur_val = tb, vb
            if vb < best_val:
                # quick smooth
                tb2, vb2 = _lbfgs_smooth(tb, deadline=min(time.time() + 2.0, t0 + BUDGET - 32))
                if vb2 < vb:
                    tb, vb = tb2, vb2
                    if vb < cur_val:
                        cur_x, cur_val = tb, vb
                if vb < best_val:
                    best_x, best_val = tb.copy(), vb
            continue
        elif r < 0.28:
            # worst shake
            try:
                G = cur_init @ cur_init.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowm = G.max(axis=1)
                deg = (G > m - 0.002).sum(axis=1)
                order = np.argsort(rowm + 0.0005 * deg)
                nw = 10 + int(rng.rand() * 6)
                worst = order[-nw:]
                sigma = [0.05, 0.12, 0.25][trial % 3]
                noise = rng.randn(n, 4) * (sigma * 0.2)
                noise[worst] += rng.randn(len(worst), 4) * (sigma * 1.8)
                cur_init = cur_init + noise
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            except Exception:
                cur_init = cur_init + rng.randn(*cur_init.shape) * 0.05
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
        elif r < 0.52:
            # cluster cavity surgery: remove connected overloaded cluster, refill holes
            try:
                k = int(rng.choice([5, 6, 8, 10, 12]))
                G = cur_init @ cur_init.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                thr = m - 0.008
                deg = (G > thr).sum(axis=1)
                seed = int(np.argmax(deg + rng.randn(n) * 0.7))
                visited = set([seed])
                order = [seed]
                frontier = [seed]
                while len(order) < k:
                    nxt = []
                    for u in frontier:
                        nbrs = np.where(G[u] > thr)[0]
                        if len(nbrs) == 0:
                            continue
                        srt = nbrs[np.argsort(-G[u, nbrs])]
                        for vv in srt:
                            vv = int(vv)
                            if vv not in visited:
                                visited.add(vv)
                                order.append(vv)
                                nxt.append(vv)
                                if len(order) >= k:
                                    break
                        if len(order) >= k:
                            break
                    if len(order) >= k:
                        break
                    if len(nxt) == 0:
                        rem = [i for i in range(n) if i not in visited]
                        if len(rem) == 0:
                            break
                        sub = G[np.ix_(rem, order)].max(axis=1)
                        pick = rem[int(np.argmax(sub))]
                        visited.add(pick)
                        order.append(pick)
                        nxt = [pick]
                    frontier = nxt
                worst = np.array(order[:k], dtype=int)
                keep = np.ones(n, dtype=bool)
                keep[worst] = False
                Xf = cur_init[keep].copy()
                newpts = []
                for _ in range(k):
                    h, _ = _best_hole(Xf, rng, n_samples=7000)
                    h = h + rng.randn(4) * 0.006
                    h /= max(np.linalg.norm(h), 1e-12)
                    h = _refine_h(h, Xf, steps=120)
                    Xf = np.vstack([Xf, h[None, :]])
                    newpts.append(h)
                for wi, h in zip(worst, newpts):
                    cur_init[wi] = h
                cur_init = cur_init + rng.randn(n, 4) * 0.008
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            except Exception:
                cur_init = cur_init + rng.randn(*cur_init.shape) * 0.08
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
        elif r < 0.74:
            # rigid cap shift
            try:
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = cur_init @ u
                kcap = int(rng.choice([10, 14, 20, 28]))
                thr = float(np.sort(dots)[n - kcap])
                mask = dots >= thr
                cnt = int(mask.sum())
                tdir = rng.randn(4); tdir -= float(tdir @ u) * u
                tn = np.linalg.norm(tdir)
                if tn < 1e-9:
                    tdir = rng.randn(4); tn = np.linalg.norm(tdir)
                tdir /= tn
                amp = 0.06 + 0.16 * rng.rand()
                cur_init[mask] += tdir[None, :] * amp
                # slight random rotation inside cap
                K = rng.randn(4, 4); K = K - K.T
                s = 0.05 + 0.1 * rng.rand()
                rot = (K @ cur_init[mask].T).T * s
                cur_init[mask] += rot
                cur_init += rng.randn(n, 4) * 0.006
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            except Exception:
                cur_init = cur_init + rng.randn(*cur_init.shape) * 0.06
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
        elif r < 0.90:
            # small hole relocation 2-3 worst to big holes
            try:
                G = cur_init @ cur_init.T
                np.fill_diagonal(G, -2.0)
                rowm = G.max(axis=1)
                m = float(G.max())
                deg = (G > m - 0.002).sum(axis=1)
                score = rowm + 0.0004 * deg
                k = 2 + (trial % 2)
                worst = np.argsort(score)[-k:]
                keep = np.ones(n, dtype=bool); keep[worst] = False
                Xf = cur_init[keep].copy()
                for w in worst:
                    h, _ = _best_hole(Xf, rng, n_samples=9000)
                    h = h + rng.randn(4) * 0.005
                    h /= max(np.linalg.norm(h), 1e-12)
                    cur_init[w] = h
                    Xf = np.vstack([Xf, h[None, :]])
                cur_init = cur_init + rng.randn(n, 4) * 0.005
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            except Exception:
                cur_init = cur_init + rng.randn(*cur_init.shape) * 0.05
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
        else:
            # pair-break push
            try:
                G = cur_init @ cur_init.T
                np.fill_diagonal(G, -2.0)
                c = G[iu0, iu1]
                top = np.argsort(c)[-10:]
                for tt in top[:6]:
                    i = int(iu0[tt]); j = int(iu1[tt])
                    if rng.rand() < 0.7:
                        d = cur_init[i] - cur_init[j]
                        dn = np.linalg.norm(d)
                        if dn < 1e-9:
                            continue
                        d /= dn
                        step = 0.03 + 0.08 * rng.rand()
                        cur_init[i] += d * step
                        cur_init[j] -= d * step
                    else:
                        cur_init[i] += rng.randn(4) * 0.08
                        cur_init[j] += rng.randn(4) * 0.08
                cur_init += rng.randn(n, 4) * 0.008
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
            except Exception:
                cur_init = cur_init + rng.randn(*cur_init.shape) * 0.06
                cur_init /= np.maximum(np.linalg.norm(cur_init, axis=1, keepdims=True), 1e-12)
        # anneal
        T0 = 0.018 + 0.012 * rng.rand()
        T1 = 0.0006 + 0.0006 * rng.rand()
        lr0 = 0.032 + 0.014 * rng.rand()
        mu = 0.86 + 0.06 * rng.rand()
        steps = 3800 + int(rng.rand() * 2500)
        dl = min(time.time() + 3.5, SEARCH_END)
        tb, vb = _gd(cur_init, steps=steps, T0=T0, T1=T1, lr0=lr0, mu=mu, rng=rng, noise0=0.0015, deadline=dl)
        # accept to walk
        if vb < cur_val or rng.rand() < np.exp(-max(0.0, vb - cur_val) / Tb):
            cur_x, cur_val = tb.copy(), vb
            # cool slightly when stuck? keep constant
        if vb < best_val - 1e-12:
            # promising: smooth
            if time.time() < t0 + BUDGET - 36:
                tb2, vb2 = _lbfgs_smooth(tb, deadline=min(time.time() + 1.8, t0 + BUDGET - 32))
                if vb2 < vb:
                    tb, vb = tb2, vb2
                    if vb < cur_val:
                        cur_x, cur_val = tb.copy(), vb
            if vb < best_val:
                best_x, best_val = tb.copy(), vb
                cur_x, cur_val = tb.copy(), vb
        # slow anneal of basin temp
        Tb *= 0.9995
        if Tb < 2e-05:
            Tb = 2e-05
    # final polish
    try:
        if time.time() < t0 + BUDGET - 30:
            bx2, bv2 = _lbfgs_smooth(best_x, deadline=t0 + BUDGET - 22)
            if bv2 < best_val:
                best_x, best_val = bx2, bv2
    except Exception:
        pass
    try:
        best_x, best_val = _lp_polish(best_x, deadline=t0 + BUDGET - 1.5, r0=1e-3)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
