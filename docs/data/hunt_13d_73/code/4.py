import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, r_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        Gfull = (x @ x.T)[iu]
        m = float(Gfull.max())
        thresh = max(20.0 * radius, 1e-9)
        # limit constraints for speed
        for _ in range(6):
            sel = Gfull > m - thresh
            if int(np.sum(sel)) <= 1200 or thresh < 1e-9:
                break
            thresh *= 0.5
        I = iu[0][sel]
        J = iu[1][sel]
        csub = Gfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols_t = np.full(p, n * d)
        # interleave per row: I-block, J-block, t
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2*d+1] = 0  # placeholder to allocate, will overwrite
        # build correctly row by row via reshape
        tmp = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], axis=1).ravel()
        cols = tmp
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def anneal_repulse(x, steps, T0, T1, s0, s1):
    n, d = x.shape
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - f) + T1 * f
        step = s0 * (1 - f) + s1 * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if not np.isfinite(m):
            break
        D = (G - m) / T
        # exp, underflow to 0 is fine
        W = np.exp(D)
        # sparsify far pairs
        W[G < m - 6 * T] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        proj = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - proj * x
        mx = float(np.max(np.sum(grad * grad, axis=1)))
        mx = np.sqrt(mx)
        if mx < 1e-14 or not np.isfinite(mx):
            continue
        grad *= (step / mx)
        x = x - grad
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def best_hole_vec(x, idx, rng, n_cand=4000):
    n, d = x.shape
    cand = rng.randn(n_cand, d)
    cn = np.linalg.norm(cand, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cand = cand / cn
    others = np.delete(x, idx, axis=0)
    dots = cand @ others.T
    scores = np.max(dots, axis=1)
    return cand[int(np.argmin(scores))]

def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    # load starting points
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (73, 13):
                continue
            nr = np.linalg.norm(v, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            v = v / nr
            if not np.all(np.isfinite(v)):
                continue
            c = max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        v = rng.randn(73, 13)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        best = v
        best_val = max_cos(best)
    n, d = best.shape
    # hopping phase until ~135s
    HOP_END = 135.0
    sigmas = [0.03, 0.08, 0.015, 0.15, 0.05, 0.01, 0.06, 0.025]
    it = 0
    # precompute contact counts of global best for jam selection (updated on improvement)
    while time.time() - t0 < HOP_END:
        time_left_hop = HOP_END - (time.time() - t0)
        if time_left_hop < 3.0:
            break
        sigma = sigmas[it % len(sigmas)]
        # perturb
        pert = rng.randn(n, d)
        x = best + sigma * pert
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        # choose index to reinsert: alternate jammed vs random
        try:
            if it % 2 == 0:
                Gg = best @ best.T
                np.fill_diagonal(Gg, -np.inf)
                mg = float(np.max(Gg))
                cnt = np.sum(Gg > mg - 5e-4, axis=1)
                idx = int(np.argmax(cnt))
                # randomize among top-3 to diversify deterministically via rng
                order = np.argsort(-cnt)
                # pick among top 3 with rng
                idx = int(order[rng.randint(0, min(3, n))])
            else:
                idx = int(rng.randint(0, n))
        except Exception:
            idx = int(rng.randint(0, n))
        try:
            x[idx] = best_hole_vec(x, idx, rng, n_cand=4000)
            if sigma > 0.06 and rng.rand() < 0.5:
                idx2 = int(rng.randint(0, n))
                if idx2 != idx:
                    x[idx2] = best_hole_vec(x, idx2, rng, n_cand=3000)
        except Exception:
            pass
        # smooth anneal to change topology
        try:
            x = anneal_repulse(x, steps=650, T0=0.02, T1=0.002, s0=0.02, s1=0.002)
        except Exception:
            pass
        # short LP polish
        tl = HOP_END - (time.time() - t0)
        if tl < 1.0:
            break
        try:
            x = lp_polish(x, seconds=min(2.5, tl), r_init=3e-3)
        except Exception:
            pass
        try:
            v = max_cos(x)
        except Exception:
            it += 1
            continue
        if v < best_val and np.all(np.isfinite(x)):
            best = x
            best_val = v
        it += 1
        # safety: total hard limit
        if time.time() - t0 > 150:
            break
    # final long polish with remaining time (leave 3s margin to 180)
    remain = 177.0 - (time.time() - t0)
    if remain < 5.0:
        remain = 5.0
    # cap to 35s to avoid timeout in verifier (verifier allows 180)
    remain = min(remain, 35.0)
    try:
        best = lp_polish(best, seconds=remain, r_init=1e-3)
    except Exception:
        pass
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
