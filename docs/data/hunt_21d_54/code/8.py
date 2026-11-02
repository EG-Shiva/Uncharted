import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, r0=5e-3, max_cons=2000, factor=20.0):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(factor * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > max_cons:
            # keep largest
            idx = np.argpartition(c, -max_cons)[-max_cons:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    return x

def _repel(x, steps, T0, T1, lr0, lr1, normed=True):
    x = _norm(x.copy())
    n, d = x.shape
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if normed:
            nr = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
            gt = gt / nr
        else:
            # scale to keep step stable
            nr = np.linalg.norm(gt, axis=1, keepdims=True).mean() + 1e-12
            gt = gt / (nr * 10.0 + 1e-12)
        x = x - lr * gt
        x = _norm(x)
    return x

def _contacts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2)
    m = G.max()
    C = (G > m - tol).sum(axis=1)
    return C, m, G

def solve():
    import time as _t
    t0 = _t.time()
    DEADLINE = 175.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands0 = []
    if xb is not None:
        cands0.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands0.append(np.asarray(xr, dtype=float))
    if len(cands0) == 0:
        cands0.append(rng.randn(54, 21))
    # normalize and fix shape
    clean = []
    for c in cands0:
        c = np.asarray(c, dtype=float)
        if c.shape != (54, 21):
            continue
        c = _norm(c)
        if np.isnan(c).any():
            continue
        clean.append(c)
    if not clean:
        clean = [_norm(rng.randn(54, 21))]
    # pick best start
    vals0 = [_max_cos(c) for c in clean]
    best = clean[int(np.argmin(vals0))].copy()
    best_val = float(min(vals0))
    # elite set
    elites = [(best_val, best.copy())]
    # add slightly perturbed versions as diversity
    for s in [0.01, 0.03]:
        e = _norm(best + s * rng.randn(*best.shape))
        elites.append((_max_cos(e), e))
    elites = sorted(elites, key=lambda z: z[0])[:4]

    def push_elite(v, x):
        for i, (ev, ex) in enumerate(elites):
            if abs(v - ev) < 1e-12:
                return
        elites.append((v, x.copy()))
        elites.sort(key=lambda z: z[0])
        del elites[4:]

    sigmas = [0.05, 0.025, 0.012, 0.08, 0.035, 0.008, 0.06, 0.15, 0.02]
    trial = 0
    # main basin loop, reserve 35s for final
    while _t.time() - t0 < DEADLINE - 35:
        trial += 1
        # choose base: mostly best, sometimes 2nd elite
        if trial % 5 == 0 and len(elites) > 1:
            base = elites[1][1]
            base_val = elites[1][0]
        else:
            base = elites[0][1]
            base_val = elites[0][0]
        sigma = sigmas[(trial - 1) % len(sigmas)]
        mode = trial % 6
        cand = base.copy()
        n, d = cand.shape
        if mode == 0:
            cand = cand + sigma * rng.randn(n, d)
            cand = _norm(cand)
            cand = _repel(cand, steps=800, T0=0.02, T1=0.0025, lr0=0.04, lr1=0.004, normed=True)
        elif mode == 1:
            C, m, G = _contacts(base)
            idx = int(np.argmax(C + rng.rand(n) * 2.0))
            cand[idx] = cand[idx] + (sigma * 2.5) * rng.randn(d)
            cand = _norm(cand)
            cand = cand + (sigma * 0.25) * rng.randn(n, d)
            cand = _norm(cand)
            cand = _repel(cand, steps=650, T0=0.015, T1=0.002, lr0=0.03, lr1=0.003, normed=True)
        elif mode == 2:
            # kill worst into largest hole
            C, m, G = _contacts(base)
            idx = int(np.argmax(C))
            keep = np.delete(base, idx, axis=0)
            S = _norm(rng.randn(3500, d))
            md = (S @ keep.T).max(axis=1)
            hole = S[int(np.argmin(md))]
            cand[idx] = hole + 0.015 * rng.randn(d)
            cand = _norm(cand)
            cand = _repel(cand, steps=550, T0=0.01, T1=0.0015, lr0=0.02, lr1=0.003, normed=True)
        elif mode == 3:
            # worst-pair split
            G = base @ base.T
            np.fill_diagonal(G, -2)
            # find worst pair
            a, b = np.unravel_index(int(np.argmax(G)), G.shape)
            # tangent directions away from each other
            for (p_, q_) in [(a, b), (b, a)]:
                direction = cand[p_] - (cand[p_] @ cand[q_]) * cand[q_]
                # project to tangent at p_
                direction = direction - (direction @ cand[p_]) * cand[p_]
                nd = np.linalg.norm(direction) + 1e-12
                direction /= nd
                cand[p_] = cand[p_] + sigma * 1.5 * direction + sigma * 0.4 * rng.randn(d)
            cand = cand + (sigma * 0.2) * rng.randn(n, d)
            cand = _norm(cand)
            cand = _repel(cand, steps=700, T0=0.018, T1=0.002, lr0=0.035, lr1=0.003, normed=True)
        elif mode == 4:
            # cluster rotation of overloaded neighbourhood
            C, m, G = _contacts(base)
            idx = int(np.argmax(C))
            # neighbours most aligned with idx
            row = (base @ base[idx])
            nbr = np.argsort(-row)[:8]
            ang = sigma * 2.0
            # random 2D plane rotation
            u = rng.randn(d); u /= np.linalg.norm(u)
            v = rng.randn(d); v -= (v @ u) * u; v /= np.linalg.norm(v) + 1e-12
            # rotate nbr points slightly in (u,v) plane: apply Givens-like tangent move
            for k in nbr:
                # components
                cu = cand[k] @ u
                cv = cand[k] @ v
                cand[k] = cand[k] + ang * (rng.choice([-1.0, 1.0])) * (cu * v - cv * u) * 0.5 + sigma * 0.3 * rng.randn(d)
            cand = _norm(cand)
            cand = _repel(cand, steps=750, T0=0.022, T1=0.002, lr0=0.04, lr1=0.004, normed=True)
        else:
            # global shake + raw-magnitude repel (different basin character)
            cand = cand + sigma * rng.randn(n, d)
            cand = _norm(cand)
            cand = _repel(cand, steps=500, T0=0.03, T1=0.005, lr0=0.05, lr1=0.005, normed=False)
            cand = _repel(cand, steps=300, T0=0.01, T1=0.002, lr0=0.02, lr1=0.003, normed=True)
        remain = DEADLINE - 35 - (_t.time() - t0)
        if remain < 0.8:
            break
        budget = min(2.2, remain)
        r0 = min(0.02, max(0.003, sigma * 0.35))
        # quick pre-check: skip LP if repel made it much worse? still polish
        cand = _lp_polish(cand, seconds=budget, r0=r0, max_cons=1100, factor=20.0)
        v = _max_cos(cand)
        if v < elites[0][0] - 1e-12:
            elites.insert(0, (v, cand.copy()))
            elites = elites[:4]
        else:
            push_elite(v, cand)
        # keep best_val synced
        best_val, best = elites[0]
        if _t.time() - t0 > DEADLINE - 36:
            break
    # final squeeze: alternate fine repel + tight LP on top elites
    elites.sort(key=lambda z: z[0])
    # spend remaining time: deep polish best, then try second
    for ei in range(min(2, len(elites))):
        rem = DEADLINE - (_t.time() - t0)
        if rem < 2:
            break
        x = elites[ei][1].copy()
        # tiny smooth descent to break ties symmetrically
        x = _repel(x, steps=250, T0=0.003, T1=0.0008, lr0=0.006, lr1=0.0015, normed=True)
        # tight LP with full constraints
        b = min(14.0, max(3.0, rem - 1.5 if ei == 0 else rem - 0.5))
        x = _lp_polish(x, seconds=b, r0=2.5e-3, max_cons=5000, factor=30.0)
        # second pass ultra-fine
        rem2 = DEADLINE - (_t.time() - t0)
        if rem2 > 2:
            x = _repel(x, steps=150, T0=0.0012, T1=0.0005, lr0=0.003, lr1=0.0008, normed=True)
            x = _lp_polish(x, seconds=min(8.0, rem2 - 0.8), r0=8e-4, max_cons=5000, factor=30.0)
        v = _max_cos(x)
        # update
        if v < elites[0][0]:
            elites.insert(0, (v, x.copy()))
            elites = elites[:4]
    best_val, best = min(elites, key=lambda z: z[0])
    best = _norm(best)
    return best
