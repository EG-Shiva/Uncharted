import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    # fix any bad rows
    nn = np.linalg.norm(x, axis=1)
    bad = ~(np.isfinite(nn)) | (nn < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
    return x, best

def _adam_phase(x, kind, steps, p0, p1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(int(steps)):
        frac = s / max(int(steps) - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        if kind == 'riesz':
            ss = p0 + (p1 - p0) * frac
            G = x @ x.T
            np.fill_diagonal(G, 1.0)  # avoid div0 on diag; will zero after
            D2 = np.clip(2.0 - 2.0 * G, 1e-12, 4.0)
            W = np.power(D2, -ss / 2.0 - 1.0)
            np.fill_diagonal(W, 0.0)
            # normalize scale to keep adam stable
            sc = float(np.median(W[W > 0])) if np.any(W > 0) else 1.0
            if sc > 0 and np.isfinite(sc):
                W = W / sc
            grad = W @ x
        else:  # softmax
            tau = p0 * ((p1 / p0) ** frac) if p1 != p0 else p0
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            W = np.exp((G - gm) / tau)
            np.fill_diagonal(W, 0.0)
            ssum = float(W.sum())
            if ssum <= 0 or not np.isfinite(ssum):
                continue
            W = W / ssum
            grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        # clip extreme grads
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _hopf_points(nb, nf, offset_mode, rng):
    # Fibonacci base on S2
    idx = np.arange(nb)
    z = 1.0 - 2.0 * (idx + 0.5) / nb
    z = np.clip(z, -1.0, 1.0)
    theta = np.arccos(z)  # 0..pi
    golden = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden * idx
    X = np.zeros((nb * nf, 4))
    r = 0
    for k in range(nb):
        th = theta[k]; ph = phi[k]
        ct = np.cos(th / 2.0); st = np.sin(th / 2.0)
        if offset_mode == 'zero':
            off = 0.0
        elif offset_mode == 'golden':
            off = (k * 2.399963229728653) % (2 * np.pi)
        elif offset_mode == 'half':
            off = (k % 2) * (np.pi / nf)
        else:  # random
            off = float(rng.uniform(0, 2 * np.pi))
        for j in range(nf):
            psi = 2 * np.pi * j / nf + off
            a1 = (psi + ph) / 2.0
            a2 = (psi - ph) / 2.0
            X[r, 0] = ct * np.cos(a1)
            X[r, 1] = ct * np.sin(a1)
            X[r, 2] = st * np.cos(a2)
            X[r, 3] = st * np.sin(a2)
            r += 1
    return _norm(X)

def _torus_layers(nlist, rlist, rng):
    # nlist: list of (m,n) per layer, rlist: radii
    X = []
    for (mm, nn), rv in zip(nlist, rlist):
        r = float(np.clip(rv, 0.05, 0.95))
        s = np.sqrt(max(1 - r * r, 1e-6))
        sh1 = float(rng.uniform(0, 2 * np.pi))
        sh2 = float(rng.uniform(0, 2 * np.pi))
        for i in range(mm):
            for j in range(nn):
                t = 2 * np.pi * i / mm + sh1
                p = 2 * np.pi * j / nn + sh2
                X.append([r * np.cos(t), r * np.sin(t), s * np.cos(p), s * np.sin(p)])
    X = np.array(X)
    # adjust to exactly 272 by trimming/padding randomly
    if len(X) > 272:
        sel = rng.choice(len(X), 272, replace=False)
        X = X[sel]
    elif len(X) < 272:
        extra = rng.standard_normal((272 - len(X), 4))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        X = np.vstack([X, extra])
    return _norm(X)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((272, 4))
    xb = _norm(np.asarray(xb, dtype=np.float64))
    if xb.shape != (272, 4):
        xb = _norm(rng.standard_normal((272, 4)))
    base_val = _max_cos(xb)
    best_x = xb.copy()
    best_val = base_val

    starts = []
    # Hopf families covering 272 = nb*nf
    for nb, nf in [(34, 8), (17, 16), (68, 4)]:
        for om in (['zero', 'golden', 'half'] if nb in (34, 17) else ['golden', 'random']):
            try:
                starts.append(_hopf_points(nb, nf, om, rng))
            except Exception:
                pass
    # a couple torus variants
    try:
        starts.append(_torus_layers([(4, 17)] * 4, [0.35, 0.5, 0.65, 0.8], rng))
        starts.append(_torus_layers([(8, 17)] * 2, [0.45, 0.75], rng))
    except Exception:
        pass
    # random starts
    for _ in range(6):
        starts.append(_norm(rng.standard_normal((272, 4))))

    cands = []
    # broad hybrid anneal: fast
    t_broad_end = t0 + 95.0
    for si, s0 in enumerate(starts):
        if time.time() > t_broad_end or time.time() > HARD - 70:
            break
        x = s0.copy()
        # tiny jitter to break exact symmetry
        x = _norm(x + 0.003 * rng.standard_normal(x.shape))
        # phases: riesz balance then softmax sharpen
        x = _adam_phase(x, 'riesz', 220, 6.0, 10.0, 0.02, 0.008)
        if time.time() > t_broad_end:
            v = _max_cos(x)
            cands.append((v, x.copy()))
            if v < best_val:
                best_val, best_x = v, x.copy()
            break
        x = _adam_phase(x, 'soft', 220, 0.02, 0.005, 0.012, 0.006)
        if time.time() > t_broad_end:
            v = _max_cos(x)
            cands.append((v, x.copy()))
            if v < best_val:
                best_val, best_x = v, x.copy()
            break
        x = _adam_phase(x, 'soft', 160, 0.005, 0.0018, 0.006, 0.003)
        v = _max_cos(x)
        cands.append((v, x.copy()))
        if v < best_val:
            best_val, best_x = v, x.copy()

    # deep refine top few with extra low-tau steps
    cands.sort(key=lambda kv: kv[0])
    refined = []
    for v, xc in cands[:4]:
        if time.time() > HARD - 45:
            break
        x = _adam_phase(xc.copy(), 'soft', 250, 0.003, 0.001, 0.004, 0.002)
        v2 = _max_cos(x)
        refined.append((v2, x))
    for v2, x in refined:
        if v2 < best_val:
            best_val, best_x = v2, x.copy()
    pool = sorted(cands[:4] + refined + [(base_val, xb)], key=lambda kv: kv[0])

    # LP polish best distinct
    polished_x, polished_v = best_x, best_val
    for v, xc in pool[:3]:
        if time.time() > HARD - 6:
            break
        budget = max(10.0, (HARD - time.time()) / 2.0)
        deadline = min(HARD, time.time() + budget)
        try:
            xp, vp = _lp_polish(xc, deadline)
        except Exception:
            continue
        if vp < polished_v:
            polished_v, polished_x = vp, xp.copy()
        if vp < best_val:
            best_val, best_x = vp, xp.copy()

    out = _norm(polished_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = _norm(out)
    return out
