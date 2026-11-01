import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def anneal(x0, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        Tn = Tg / rms
        vel = mu * vel + (1.0 - mu) * Tn
        # bias correction for early steps
        vc = vel / (1.0 - mu ** (s + 1))
        x = x - lr * vc
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def greedy_select(pool, k, rng):
    # pool assumed normalized (M,d)
    M = pool.shape[0]
    first = int(rng.integers(0, M))
    sel = [first]
    # scores = max dot to selected set
    scores = pool @ pool[first]
    scores[first] = 2.0  # never pick again
    for _ in range(1, k):
        nxt = int(np.argmin(scores))
        sel.append(nxt)
        d = pool @ pool[nxt]
        scores = np.maximum(scores, d)
        scores[nxt] = 2.0
    return pool[np.array(sel)].copy()

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0 / np.linalg.norm(x0, axis=1, keepdims=True)

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.25:
            break
        if deadline is not None and time.time() > deadline - 0.6:
            break
        if radius < 1e-15:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        if int(sel.sum()) > 1800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1800]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        csel = c[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def fingerprint_top(x, k=200):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    part = np.partition(c, -k)[-k:]
    return np.sort(part)[::-1]

def fingerprint_eig(x):
    G = x @ x.T
    return np.linalg.eigvalsh(G)[::-1]

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline = t0 + TLIM
    n, d = 68, 12
    rng = np.random.default_rng(20261031)

    # ---- build pools (de novo, no best.npy) ----
    # Sobol-on-sphere pool
    try:
        from scipy.stats import qmc
        from scipy.special import ndtri
        sampler = qmc.Sobol(d=d, scramble=True, seed=12345)
        # 4096 points
        u = sampler.random_base2(m=12)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z = ndtri(u)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        sobol_pool = z  # (4096,12)
    except Exception:
        sobol_pool = rng.standard_normal((4096, d))
        sobol_pool /= np.linalg.norm(sobol_pool, axis=1, keepdims=True)

    gauss_pool = rng.standard_normal((12000, d))
    gauss_pool /= np.linalg.norm(gauss_pool, axis=1, keepdims=True)

    # D12 lattice pool: integer vectors in {-2..2} with even sum
    raw = rng.integers(-2, 3, size=(30000, d))
    sm = raw.sum(axis=1)
    nrm = np.linalg.norm(raw.astype(float), axis=1)
    mask = (sm % 2 == 0) & (nrm > 1e-9)
    # ensure even negative modulo handling: in python -1%2==1, fine
    d12 = raw[mask].astype(float)
    if d12.shape[0] < 2000:
        extra = rng.standard_normal((5000, d))
        d12_pool = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    else:
        d12 = d12 / np.linalg.norm(d12, axis=1, keepdims=True)
        # dedup roughly
        d12_pool = d12[:12000]

    mega_pool = np.vstack([sobol_pool[:4096], gauss_pool[:6000], d12_pool[:4000]])

    # ---- launch many cheap independent starts ----
    starts = []
    # 6 random Gaussian
    for _ in range(6):
        x = rng.standard_normal((n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        starts.append(('gauss', x))
    # 4 Sobol blocks
    for b in range(4):
        s = b * 68
        # wrap around
        idx = (np.arange(68) + s * 7 + b * 131) % sobol_pool.shape[0]
        starts.append(('sobol', sobol_pool[idx].copy()))
    # 4 D12 samples
    for _ in range(4):
        ii = rng.choice(d12_pool.shape[0], size=n, replace=False)
        starts.append(('d12', d12_pool[ii].copy()))
    # 8 greedy maximin from pools
    for g in range(8):
        # vary source pool subset
        if g % 3 == 0:
            src = sobol_pool
        elif g % 3 == 1:
            src = gauss_pool
        else:
            # mixed mega subset
            ii = rng.choice(mega_pool.shape[0], size=3500, replace=False)
            src = mega_pool[ii]
            starts.append(('greedy-mega', greedy_select(src, n, rng)))
            continue
        # subsample 3500 then greedy
        ii = rng.choice(src.shape[0], size=min(3500, src.shape[0]), replace=False)
        sub = src[ii]
        starts.append(('greedy', greedy_select(sub, n, rng)))
    # 2 random + small perturbation of sobol/gauss mix
    for _ in range(2):
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        starts.append(('gauss2', a))

    # ---- cheap anneal 200-600 steps beta 30->300 ----
    annealed = []
    for si, (tag, x0) in enumerate(starts):
        steps = int(rng.integers(200, 601))
        # vary schedule slightly
        b0 = 30.0
        b1 = float(rng.uniform(200, 350))
        lr0 = float(rng.uniform(0.03, 0.07))
        lr1 = float(rng.uniform(0.002, 0.008))
        mu = float(rng.uniform(0.75, 0.90))
        x = anneal(x0, steps, b0, b1, lr0, lr1, mu)
        v = max_cos(x)
        annealed.append((v, tag, x))

    # sort by score
    annealed.sort(key=lambda t: t[0])

    # ---- fingerprint + diversity clustering ----
    tops = [fingerprint_top(x) for _, _, x in annealed]
    eigs = [fingerprint_eig(x) for _, _, x in annealed]
    kept_idx = []
    THRESH_TOP = 0.012  # L2 distance on top200
    THRESH_EIG = 0.6
    for i in range(len(annealed)):
        dup = False
        for j in kept_idx:
            dt = float(np.linalg.norm(tops[i] - tops[j]))
            de = float(np.linalg.norm(eigs[i] - eigs[j]))
            if dt < THRESH_TOP and de < THRESH_EIG:
                dup = True
                break
        if not dup:
            kept_idx.append(i)
    # kept in score order; take best few dissimilar
    # ensure at least 4
    if len(kept_idx) == 0:
        kept_idx = [0]
    # select up to 5 dissimilar finalists (best per cluster)
    KFIN = min(5, len(kept_idx))
    finalists = [(annealed[i][0], annealed[i][2].copy()) for i in kept_idx[:KFIN]]
    # if fewer than 4, pad with next best annealed not kept
    if len(finalists) < 4 and len(annealed) > len(finalists):
        for i in range(len(annealed)):
            if i in kept_idx:
                continue
            finalists.append((annealed[i][0], annealed[i][2].copy()))
            if len(finalists) >= 4:
                break

    # ---- spend remaining time LP-polishing best few dissimilar ----
    # medium polish each finalist
    polished = []
    # reserve ~25s for final winner polish
    for fi, (v0, xf) in enumerate(finalists):
        if time.time() > deadline - 35.0:
            polished.append((max_cos(xf), xf))
            continue
        # two-stage L-BFGS
        try:
            xf = run_lbfgs(xf, 120.0, 70)
        except Exception:
            pass
        try:
            xf = run_lbfgs(xf, 380.0, 70)
        except Exception:
            pass
        # short LP
        rem = deadline - time.time()
        lp_sec = min(3.0, max(0.8, rem - 28.0 / max(len(finalists), 1)))
        if lp_sec > 0.7 and time.time() < deadline - 28.0:
            try:
                xf = lp_polish(xf, lp_sec, deadline=deadline - 25.0)
            except Exception:
                pass
        polished.append((max_cos(xf), xf))

    polished.sort(key=lambda t: t[0])
    x_best = polished[0][1].copy()
    best_val = float(polished[0][0])

    # ---- final intensive polish on overall winner ----
    remain = deadline - time.time()
    if remain > 6.0:
        try:
            x_best = run_lbfgs(x_best, 700.0, 90)
        except Exception:
            pass
        best_val = max_cos(x_best)
    remain = deadline - time.time()
    if remain > 6.0:
        try:
            x_best = run_lbfgs(x_best, 1200.0, 70)
        except Exception:
            pass
        best_val = max_cos(x_best)
    remain = deadline - time.time()
    if remain > 3.5:
        try:
            x_best = lp_polish(x_best, min(remain - 1.0, 22.0), deadline=deadline - 0.8)
        except Exception:
            pass
        best_val = max_cos(x_best)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    # ---- final comparison (only now load best.npy) ----
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if xb.shape == (68, 12):
            vb = max_cos(xb)
            if vb < best_val:
                return xb
    except Exception:
        pass
    return x_best
