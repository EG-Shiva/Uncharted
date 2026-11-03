import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _contacts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G >= m - 1e-9, axis=1)
    pmax = G.max(axis=1)
    return m, cnt, pmax


def _void_dir(x, rng, exclude=-1, s=15000):
    n, d = x.shape
    Z = rng.standard_normal((s, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ x.T
    if 0 <= exclude < n:
        S[:, exclude] = -2.0
    rowmax = S.max(axis=1)
    return Z[int(np.argmin(rowmax))]


def _anneal(x0, steps, a0, a1, lr):
    x = x0.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(x)
    best_x = x.copy()
    la0, la1 = np.log(a0), np.log(a1)
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        alpha = np.exp(la0 + (la1 - la0) * t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp((G - gmax) * alpha)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if (t % 200 == 0) or (t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val


def _lp_polish(x, deadline, r_init=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r_init)
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[0][sel]
        J = iu0[1][sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            k = np.argpartition(-cc, 2800)[:2800]
            I = I[k]
            J = J[k]
            cc = cc[k]
            p = 2800
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        base = np.empty((p, 2 * d + 1), dtype=np.int64)
        ar = np.arange(d)
        base[:, :d] = I[:, None] * d + ar
        base[:, d:2 * d] = J[:, None] * d + ar
        base[:, 2 * d] = n * d
        cols = base.ravel()
        vals = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    LOOP_END = t0 + 146.0
    rng = np.random.default_rng(12345)
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            xb = np.load(fn)
            break
        except Exception:
            continue
    if xb is None:
        xb = rng.standard_normal((256, 6))
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (256, 6):
        tmp = rng.standard_normal((256, 6))
        try:
            r = min(xb.shape[0], 256)
            cc = min(xb.shape[1], 6)
            tmp[:r, :cc] = xb[:r, :cc]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    n, d = 256, 6
    cyc = 0
    while time.time() < LOOP_END:
        rem = HARD_END - 22.0 - time.time()
        if rem < 1.0:
            break
        cyc += 1
        mode = cyc % 7
        try:
            m0, cnt0, pmax0 = _contacts(xb)
        except Exception:
            m0, cnt0, pmax0 = best, np.zeros(n), np.zeros(n)
        y = xb.copy()
        steps = 800
        a0, a1, lr = 40.0, 400.0, 0.010
        if mode == 0:
            # global small noise
            y += rng.standard_normal((n, d)) * 0.022
            steps = 700
        elif mode == 1:
            # overloaded hub shake
            order = np.argsort(-cnt0)
            top = order[:24]
            sel = rng.choice(top, size=8, replace=False)
            y[sel] += rng.standard_normal((8, d)) * 0.12
            # light global jitter
            y += rng.standard_normal((n, d)) * 0.005
            steps = 900
            a0, a1, lr = 30.0, 350.0, 0.012
        elif mode == 2:
            # rattler + worst void reinsertion
            ratt = np.where(cnt0 <= 0.5)[0]
            worst_order = np.argsort(-(pmax0 + 1e-6 * cnt0))
            cands = []
            if len(ratt) > 0:
                cands.extend(list(ratt[:3]))
            cands.extend([w for w in worst_order[:4] if w not in cands][:2])
            if len(cands) == 0:
                cands = [int(worst_order[0])]
            for w in cands:
                vd = _void_dir(xb, rng, exclude=int(w), s=15000)
                y[int(w)] = vd + rng.standard_normal(d) * 0.01
            y += rng.standard_normal((n, d)) * 0.004
            steps = 1000
        elif mode == 3:
            # worst-pair split: replace one endpoint by void, shake other
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = flat // n, flat % n
            vd = _void_dir(xb, rng, exclude=int(ii), s=15000)
            y[int(jj)] = vd + rng.standard_normal(d) * 0.012
            y[int(ii)] += rng.standard_normal(d) * 0.03
            # shake its close neighbours a bit
            nbrs = np.argsort(-G[int(ii)])[:8]
            y[nbrs] += rng.standard_normal((len(nbrs), d)) * 0.02
            steps = 1100
            a0, a1, lr = 25.0, 450.0, 0.012
        elif mode == 4:
            # random subset shake
            k = 20
            sel = rng.choice(n, size=k, replace=False)
            y[sel] += rng.standard_normal((k, d)) * 0.06
            steps = 800
        elif mode == 5:
            # coherent patch rotation
            cdir = rng.standard_normal(d)
            cdir /= np.linalg.norm(cdir)
            dots = xb @ cdir
            cap = np.argsort(-dots)[:26]
            A = rng.standard_normal((d, d))
            A = A - A.T
            theta = 0.18 + 0.12 * rng.random()
            y[cap] = y[cap] + theta * (y[cap] @ A.T)
            y[cap] += rng.standard_normal((len(cap), d)) * 0.01
            steps = 950
        else:
            # large kick of few points
            k = 5
            sel = rng.choice(n, size=k, replace=False)
            y[sel] += rng.standard_normal((k, d)) * 0.30
            steps = 1200
            a0, a1, lr = 20.0, 300.0, 0.014
        # renormalize
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        # fast anneal to change basin
        xa, va = _anneal(y, steps=steps, a0=a0, a1=a1, lr=lr)
        v = _max_cos(xa)
        # decide LP budget
        now = time.time()
        if v < best - 1e-12:
            dl = min(now + 3.0, HARD_END - 20.0)
            if dl - now > 0.4:
                xp = _lp_polish(xa, dl, r_init=0.008)
                vp = _max_cos(xp)
                if vp < best - 1e-12:
                    xb, best = xp, vp
                    continue
            # even without further polish accept
            if v < best - 1e-12:
                xb, best = xa, v
        elif v < best + 0.004:
            dl = min(now + 1.6, HARD_END - 20.0)
            if dl - now > 0.4:
                xp = _lp_polish(xa, dl, r_init=0.006)
                vp = _max_cos(xp)
                if vp < best - 1e-12:
                    xb, best = xp, vp
        # else discard
    # final long polish
    try:
        xb = _lp_polish(xb, HARD_END - 1.0, r_init=0.005)
    except Exception:
        pass
    xb = np.where(np.isfinite(xb), xb, 0.0)
    nn = np.linalg.norm(xb, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    xb = xb / nn
    return xb
