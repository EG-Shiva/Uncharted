import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac if T1 != T0 else T0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-12)
        step = gt / rs
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # clip large steps to 1
        scale = np.maximum(sn, 1.0)
        step = step / scale
        x = x - lr * step
        x = _norm(x)
    return x

def _hard(x, iters=80, lr0=0.005, lr1=0.001, topk=2):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        idx = np.argpartition(-G, topk, axis=1)[:, :topk]
        grad = np.zeros_like(x)
        for t in range(topk):
            w = 1.0 / (t + 1.0)
            nb = x[idx[:, t]]
            dprod = np.sum(nb * x, axis=1, keepdims=True)
            grad += w * (nb - dprod * x)
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        gn = np.maximum(gn, 1e-12)
        grad = grad / gn
        m = float(G.max())
        rm = G.max(axis=1)
        mask = (rm > m - 0.01).astype(float)[:, None]
        x = x - lr * grad * (0.3 + 0.7 * mask)
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=2e-3, cap=1200):
    x = _norm(x)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            # keep top cap
            kth = np.partition(v, -cap)[-cap]
            sel2 = np.where(c_all >= kth)[0]
            if sel2.size > cap + 300:
                # deterministic thinning
                sel2 = sel2[::2][:cap]
                if sel2.size < 100:
                    # fallback to top cap indices
                    o = np.argsort(-v)[-cap:]
                    sel = sel[o]
                else:
                    sel = sel2[:cap]
            else:
                # take highest cap among sel2
                vals2 = c_all[sel2]
                o = np.argsort(-vals2)[:cap]
                sel = sel2[o]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        # time check: if we exceeded, stop
        if time.time() - start >= seconds:
            if v < best:
                x = y
            break
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _find_void(X, rng, n_samples=8000, refine=60):
    n, d = X.shape
    cand = rng.normal(size=(n_samples, d))
    cand = _norm(cand)
    scores = (cand @ X.T).max(axis=1)
    bi = int(np.argmin(scores))
    v = cand[bi].copy()
    for _ in range(refine):
        dots = X @ v
        dm = float(dots.max())
        w = np.exp(60.0 * (dots - dm))
        g = w @ X
        gt = g - float(g @ v) * v
        nn = float(np.linalg.norm(gt))
        if nn < 1e-12 or not np.isfinite(nn):
            break
        gt /= nn
        v = v - 0.08 * gt
        nn2 = float(np.linalg.norm(v))
        if nn2 < 1e-12 or not np.isfinite(nn2):
            break
        v /= nn2
    return v

def solve():
    t0 = time.time()
    LIM = 168.0
    FINAL_RESERVE = 26.0
    rng = np.random.default_rng(20261031)
    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (n, d):
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                c = _norm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        # from-scratch fallback: multiple random starts screened by smooth
        best = None
        best_val = 1e9
        for _ in range(4):
            X = _norm(rng.normal(size=(n, d)))
            X = _smooth(X, 300, 0.03, 0.003, 0.07, 0.015)
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best = X.copy()
        cands.append(best)
    # pick best loaded
    incumbent = None
    inc_val = 1e9
    for c in cands:
        try:
            v = _max_cos(c)
        except Exception:
            continue
        if v < inc_val:
            inc_val = v
            incumbent = c.copy()
    cur_best = incumbent.copy()
    cur_val = inc_val
    # short initial tighten (2-3s) if time allows
    if LIM - (time.time() - t0) > FINAL_RESERVE + 10:
        try:
            tmp = _lp_polish(cur_best, seconds=3.0, r0=2e-3, cap=1200)
            tv = _max_cos(tmp)
            if tv < cur_val:
                cur_best = tmp
                cur_val = tv
        except Exception:
            pass
    hop_deadline = t0 + LIM - FINAL_RESERVE
    it = 0
    # hopping loop
    while time.time() < hop_deadline:
        it += 1
        remain = hop_deadline - time.time()
        if remain < 0.8:
            break
        r = rng.random()
        X = cur_best.copy()
        try:
            if r < 0.38:
                # rattler to void
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                # most isolated = smallest rowmax
                order = np.argsort(rowmax)
                # pick among 3 most isolated randomly
                cand_idx = order[:3]
                idx = int(cand_idx[rng.integers(0, len(cand_idx))])
                v = _find_void(X, rng, n_samples=8000, refine=50)
                X[idx] = v
                # tiny global jitter to break symmetry
                if rng.random() < 0.7:
                    sig = 0.008 + 0.012 * rng.random()
                    X = _norm(X + rng.normal(scale=sig, size=(n, d)) * 0.5)
                    # keep inserted void nearly intact: restore it partially
                    # re-project idx to void direction mixed with jitter
                    # (already jittered, ok)
            elif r < 0.72:
                # patch shake around random center or overloaded point
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                # degree with slack 0.005
                deg = (G > m - 0.005).sum(axis=1)
                if rng.random() < 0.5:
                    # overloaded center
                    center = int(np.argmax(deg + rng.normal(scale=0.5, size=n)))
                else:
                    center = int(rng.integers(0, n))
                dots = X @ X[center]
                k = int(rng.integers(5, 13))
                nn_idx = np.argsort(-dots)[0:k]  # includes center
                sig = float(0.05 + 0.20 * rng.random())
                noise = rng.normal(scale=sig, size=(k, d))
                # tangent projection
                for t_, gi in enumerate(nn_idx):
                    a = X[gi]
                    nnv = noise[t_] - float(noise[t_] @ a) * a
                    X[gi] = a + nnv
                X = _norm(X)
            else:
                # global shake
                sig = float(0.008 + 0.03 * rng.random())
                if rng.random() < 0.3:
                    sig = float(0.04 + 0.06 * rng.random())
                noise = rng.normal(scale=sig, size=(n, d))
                # make tangent
                dotn = np.sum(noise * X, axis=1, keepdims=True)
                noise = noise - dotn * X
                X = _norm(X + noise)
        except Exception:
            continue
        # annealed smooth screening with randomized schedule
        try:
            T0 = float(rng.choice([0.015, 0.02, 0.03]))
            T1 = float(rng.choice([0.0015, 0.002, 0.003]))
            lr0 = float(0.05 + 0.03 * rng.random())
            lr1 = float(0.008 + 0.008 * rng.random())
            iters = int(rng.choice([220, 260, 320]))
            X = _smooth(X, iters, T0, T1, lr0, lr1)
            if rng.random() < 0.6:
                X = _hard(X, iters=60, lr0=0.004, lr1=0.001, topk=2)
            v = _max_cos(X)
        except Exception:
            continue
        # decide to LP-polish
        do_lp = False
        lp_sec = 1.2
        if v < cur_val - 2e-7:
            do_lp = True
            lp_sec = 1.6
        elif v < cur_val + 8e-4 and rng.random() < 0.07 and remain > 6:
            do_lp = True
            lp_sec = 1.0
        if do_lp:
            if time.time() > hop_deadline - 0.5:
                # no time for LP, but keep if better without LP
                if v < cur_val:
                    cur_best = X.copy()
                    cur_val = v
                break
            try:
                Y = _lp_polish(X, seconds=lp_sec, r0=2e-3, cap=1100)
                vy = _max_cos(Y)
                if vy < cur_val - 1e-12:
                    cur_best = Y.copy()
                    cur_val = vy
                    # early exit if big win? continue to exploit new basin
            except Exception:
                continue
        # occasional keep of near-miss to diversify base? no, keep elitist
    # final long polish
    rem = t0 + LIM - time.time() - 1.0
    if rem > 1.0:
        fsec = min(FINAL_RESERVE - 1.0, rem)
        try:
            Yf = _lp_polish(cur_best, seconds=fsec, r0=2e-3, cap=1600)
            vf = _max_cos(Yf)
            if vf < cur_val:
                cur_best = Yf
                cur_val = vf
        except Exception:
            pass
    if cur_val > inc_val and incumbent is not None:
        return _norm(incumbent)
    return _norm(cur_best)
