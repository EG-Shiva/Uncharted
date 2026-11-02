import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return np.nan_to_num(x)

def _contacts(x, thresh=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - thresh).sum(axis=1)
    return C, m, G

def _adam_spread(Y, steps, lr0, T0, T1, rng_lr_decay=True):
    n, d = Y.shape
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps)
        T = T0 * (1 - frac) + T1 * frac
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        E = (G - mx) / max(T, 1e-12)
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum()) + 1e-12
        Gr = (W @ Y) / S
        dot = np.einsum('ij,ij->i', Gr, Y)
        Gr -= dot[:, None] * Y
        # Adam
        m1 = b1 * m1 + (1 - b1) * Gr
        m2 = b2 * m2 + (1 - b2) * (Gr * Gr)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mh = m1 / c1
        vh = m2 / c2
        lr = lr0 * (1.0 - 0.5 * frac) if rng_lr_decay else lr0
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _proj(Y)
    return Y

def _polish_point(p, Ycur, steps=300, lr0=0.05, T=0.004):
    # minimize LSE max dot(p, Ycur) over unit p
    for s in range(steps):
        frac = s / max(1, steps)
        d = Ycur @ p  # (m,)
        mx = float(d.max())
        E = (d - mx) / max(T, 1e-12)
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        S = float(W.sum()) + 1e-12
        W /= S
        g = W @ Ycur  # (d,)
        g = g - float(g @ p) * p
        lr = lr0 * (1.0 - 0.6 * frac)
        p = p - lr * g
        nrm = float(np.linalg.norm(p))
        if nrm == 0 or not np.isfinite(nrm):
            break
        p = p / nrm
    return p

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIMIT = 172.0
    DEADLINE = T0 + 150.0  # hard internal deadline for delete-expand-reinsert
    FINAL_LP = 14.0
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    best = b0.copy()
    bv = _max_cos(best)
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        rv = _max_cos(r0)
        if rv < bv:
            best, bv = r0.copy(), rv
    except Exception:
        pass
    n, d = 54, 21
    it = 0
    # precompute nothing; cycle deletion modes
    while time.time() < DEADLINE - 1.0:
        it += 1
        if time.time() > DEADLINE - 1.0:
            break
        mode = it % 3  # 0 high, 1 low, 2 random
        k = 3 + (it % 3)  # 3,4,5 cycle (3-5 points)
        if (it // 3) % 2 == 1:
            # vary k slightly deterministically
            k = 3 + ((it * 7) % 3)
        cnts, m, G = _contacts(best, thresh=1e-9)
        if mode == 0:
            order = np.argsort(-cnts, kind='stable')
            dels = order[:k].tolist()
        elif mode == 1:
            order = np.argsort(cnts, kind='stable')
            dels = order[:k].tolist()
        else:
            dels = rng.choice(n, size=k, replace=False).tolist()
        keep = [i for i in range(n) if i not in set(dels)]
        Y = best[keep].copy()  # (m,d) m=49..51
        # expand survivors: smooth log-sum-exp max-cosine, projected Adam
        # vary temperature/lr per restart for diversity
        T0_ = 0.012 + 0.008 * ((it % 4) / 3.0)
        T1_ = 0.0015 + 0.001 * ((it % 3) / 2.0)
        lr0_ = 0.025 + 0.015 * ((it % 5) / 4.0)
        steps_surv = 2200 if Y.shape[0] > 50 else 2600
        # time guard: skip if too close to deadline
        if time.time() + 0.6 > DEADLINE:
            break
        Y = _adam_spread(Y, steps=steps_surv, lr0=lr0_, T0=T0_, T1=T1_)
        # reinsert one at a time: best of 20000 random voids
        Ycur = Y.copy()
        # randomize insertion order
        order_ins = np.array(dels)
        # shuffle deterministically
        perm = rng.permutation(k)
        order_ins = [dels[i] for i in perm]
        ok = True
        for _ii in range(k):
            if time.time() > DEADLINE - 0.5:
                ok = False
                break
            # sample 20000 random unit vectors
            C = rng.normal(size=(20000, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            # score by max cosine against current set
            # chunk to bound memory/time
            best_p = None
            best_s = 1e9
            # process in 2 chunks of 10000
            for ch in (0, 1):
                Cb = C[ch * 10000:(ch + 1) * 10000]
                D = Cb @ Ycur.T  # (10000, mcur)
                s = D.max(axis=1)
                j = int(np.argmin(s))
                if float(s[j]) < best_s:
                    best_s = float(s[j])
                    best_p = Cb[j].copy()
            p = _polish_point(best_p, Ycur, steps=250, lr0=0.06, T=0.004)
            # local joint polish of p + neighbours with cosine>0 (few Adam steps)
            # build neighbour set
            dots = Ycur @ p
            nb = np.where(dots > 0)[0]
            # cap to closest 25 to keep it cheap
            if len(nb) > 25:
                nb = nb[np.argsort(-dots[nb])[:25]]
            if len(nb) > 0:
                small = np.vstack([Ycur[nb], p[None, :]])
                small = _adam_spread(small, steps=150, lr0=0.01, T0=0.005, T1=0.0015)
                # write back neighbours, keep polished p
                Ycur[nb] = small[:-1]
                p = small[-1] / (np.linalg.norm(small[-1]) + 1e-12)
                Ycur = _proj(Ycur)
            Ycur = np.vstack([Ycur, p[None, :]])
        if not ok:
            break
        # Ycur should be (54,21)
        if Ycur.shape != (54, 21):
            continue
        Ycur = _proj(Ycur)
        # quick full-set smooth-max cool-down (cheap)
        Ycur = _adam_spread(Ycur, steps=400, lr0=0.008, T0=0.004, T1=0.0012)
        v = _max_cos(Ycur)
        if v < bv - 1e-12:
            best, bv = Ycur, v
        # early exit if we beat record meaningfully? keep going to use budget
        if time.time() >= DEADLINE - 0.5:
            break
    # final single LP polish within remaining budget (<= ~15s)
    tleft = LIMIT - (time.time() - T0) - 1.0
    if tleft > 1.0:
        budget = min(FINAL_LP, tleft)
        try:
            polished = _lp_polish(best, budget)
            pv = _max_cos(polished)
            if pv < bv and np.all(np.isfinite(polished)):
                best = polished
        except Exception:
            pass
    best = _proj(best)
    return best
