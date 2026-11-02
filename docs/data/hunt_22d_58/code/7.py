import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, limit=1e9):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > limit:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_quench(x, steps=700, beta0=30.0, beta1=550.0,
                   lr0=0.015, lr1=0.0012, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    lbr = np.log(beta1 / beta0) / max(steps - 1, 1)
    llr = np.log(lr1 / lr0) / max(steps - 1, 1)
    for it in range(steps):
        beta = beta0 * np.exp(lbr * it)
        lr = lr0 * np.exp(llr * it)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # softmax weights
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        s = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - s * x
        gn = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-12
        ghat = gt / gn
        row = W.sum(axis=1, keepdims=True)
        scale = row / (row.mean() + 1e-12)
        scale = np.clip(scale, 0.2, 3.0)
        v = mu * v - lr * scale * ghat
        x = x + v
        x = _norm(x)
    return x


def _worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def solve():
    t0 = time.time()
    HARD = 173.0
    rng = np.random.default_rng(0)
    # load seeds
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (58, 22):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands = [_norm(rng.standard_normal((58, 22)))]
    scores = [_max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    gbest = cands[bi].copy()
    gscore = float(scores[bi])
    n, d = gbest.shape

    # archive of promising basins (score, matrix)
    archive = [(gscore, gbest.copy())]

    def push_arch(x, s):
        archive.append((s, x.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[6:]

    # exploration until ~120s
    EXPLORE_END = 120.0
    it = 0
    sig_cycle = [0.04, 0.09, 0.16, 0.26, 0.06, 0.12]
    while time.time() - t0 < EXPLORE_END and time.time() - t0 < HARD - 50:
        it += 1
        mode = it % 6
        y = gbest.copy()
        if mode == 0 or mode == 1:
            sigma = sig_cycle[(it // 6) % len(sig_cycle)]
            y = _norm(y + sigma * rng.standard_normal(y.shape))
        elif mode == 2 or mode == 3:
            w, _ = _worst_per_point(y)
            k = 5 if mode == 2 else 10
            idx = np.argsort(-w)[:k]
            noise = 0.015 * rng.standard_normal(y.shape)
            noise[idx] *= (0.28 / 0.015) if mode == 2 else (0.20 / 0.015)
            # extra global jitter
            noise += 0.008 * rng.standard_normal(y.shape)
            y = _norm(y + noise)
        else:
            # hole reinsertion of 1 or 2 worst points
            nrep = 1 if mode == 4 else 2
            w, G = _worst_per_point(y)
            order = np.argsort(-w)[:nrep]
            cur = y.copy()
            for oi in order:
                # sample holes, keep farthest from rest
                ns = 2500
                samp = _norm(rng.standard_normal((ns, d)))
                # dots to all except oi
                mask = np.ones(n, dtype=bool)
                mask[oi] = False
                rest = cur[mask]
                dots = samp @ rest.T
                md = dots.max(axis=1)
                best_s = int(np.argmin(md))
                cur[oi] = samp[best_s]
                # small jitter around hole
                cur[oi] = _norm(cur[oi:oi+1] + 0.01 * rng.standard_normal((1, d)))[0]
            y = _norm(cur + 0.005 * rng.standard_normal(cur.shape))

        # quench with annealed softmax; vary length slightly for diversity
        steps = 650 if (it % 2 == 0) else 850
        # occasionally hotter start
        b0 = 25.0 if (it % 3 == 0) else 35.0
        yq = _smooth_quench(y, steps=steps, beta0=b0, beta1=600.0,
                            lr0=0.016, lr1=0.0010, mu=0.86)
        s = _max_cos(yq)
        push_arch(yq, s)
        if s < gscore:
            gbest, gscore = yq.copy(), float(s)

    # final selection: polish top few archive entries with LP
    archive.sort(key=lambda t: t[0])
    # unique-ify not needed; take top 3
    top = archive[:3]
    # always include current gbest first
    # allocate remaining time
    remain = HARD - (time.time() - t0) - 1.0
    if remain < 5:
        return _norm(gbest)
    per = remain / len(top)
    per = min(max(per, 5.0), 25.0)
    best_x, best_s = gbest, gscore
    for _, xc in top:
        if time.time() - t0 > HARD - 2:
            break
        # light re-quench before LP to tighten with cheap method
        # (short high-beta run)
        try:
            xw = _smooth_quench(xc.copy(), steps=250, beta0=120.0,
                                beta1=900.0, lr0=0.006, lr1=0.0008, mu=0.88)
        except Exception:
            xw = xc.copy()
        xp = _lp_polish(xw, seconds=per, t0=t0, limit=HARD - 2)
        sp = _max_cos(xp)
        if sp < best_s:
            best_x, best_s = xp, sp
    return _norm(best_x)
