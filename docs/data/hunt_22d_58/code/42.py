import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_quench(x, steps, b0, b1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b_1 = 0.9
    b_2 = 0.999
    eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        m1h = m1 / c1
        m2h = m2 / c2
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        scale = np.ones_like(sn)
        scale[over] = 0.05 / (sn[over] + 1e-12)
        step = step * scale
        x = x - step
        x = _norm(x)
    return x

def _refine_one(v, Y, steps=120):
    # locally optimize inserted point v with others Y fixed
    # minimize max(Y @ v) via softmax tangential descent, normalized direction
    v = v / (np.linalg.norm(v) + 1e-12)
    beta = 600.0
    for s in range(steps):
        d = Y @ v
        m = float(d.max())
        W = np.exp(beta * (d - m))
        sW = float(W.sum())
        if not np.finite(sW) or sW == 0:
            break
        W /= sW
        g = W @ Y  # (d,)
        gv = float(g @ v)
        g = g - gv * v
        gn = float(np.linalg.norm(g)) + 1e-12
        f = s / max(1, steps - 1)
        lr = 0.02 * (0.002 / 0.02) ** f
        v = v - lr * (g / gn)
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _lp_polish_capped(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HOP_END = t0 + 130.0
    FINAL_END = t0 + 165.0
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = _norm(v)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        best = _norm(z)
        best_val = _max_cos(best)
    incumbent = best.copy()
    incumbent_val = float(best_val)

    # tiny initial squeeze (1s) to ensure local optimum baseline
    try:
        if time.time() < HOP_END:
            best = _lp_polish_capped(best, 1.0, HARD_END)
            best_val = _max_cos(best)
            if best_val < incumbent_val:
                incumbent = best.copy()
                incumbent_val = float(best_val)
            else:
                best = incumbent.copy()
                best_val = float(incumbent_val)
    except Exception:
        pass

    ks = [3, 4, 5, 6, 8]
    trial = 0
    NCAND = 8192
    while time.time() < HOP_END:
        if time.time() >= HARD_END - 22.0:
            break
        trial += 1
        k = ks[(trial - 1) % len(ks)]
        mode = (trial - 1) % 3  # 0 random, 1 hubs, 2 cluster
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
        except Exception:
            continue
        # choose indices to excise
        try:
            if mode == 0:
                idx = rng.choice(n, size=k, replace=False)
            elif mode == 1:
                # highest-contact hubs
                gap = 5e-4
                cnt = (G0 > m0 - gap).sum(axis=1)
                # break ties randomly
                jitter = rng.random(n) * 1e-6
                order = np.argsort(-(cnt.astype(float) + jitter))
                idx = order[:k].copy()
            else:
                seed = int(rng.integers(0, n))
                row = G0[seed].copy()
                row[seed] = -2.0
                nn = np.argsort(-row)[:k-1]
                idx = np.array([seed] + nn.tolist(), dtype=int)
            idx = np.array(idx, dtype=int)
        except Exception:
            continue
        # sequential reinsertion
        try:
            Xc = best.copy()
            # order of reinsertion: random shuffle
            porder = idx.copy()
            rng.shuffle(porder)
            # active set: initially all except idx
            removed_set = set(int(v) for v in idx)
            # for each p, candidate pool vs currently active points
            for p in porder:
                if time.time() >= HOP_END:
                    break
                active = np.array([i for i in range(n) if i not in removed_set], dtype=int)
                Y = Xc[active]
                C = rng.standard_normal((NCAND, d))
                C = _norm(C)
                dots = C @ Y.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                v0 = C[bi].copy()
                v0 = _refine_one(v0, Y, steps=100)
                Xc[int(p)] = v0
                removed_set.discard(int(p))
            # short all-point soft-max quench
            Xc = _adam_quench(Xc, 450, 80, 700, 0.004, 0.0005)
            if time.time() >= HOP_END:
                # quick check without LP
                vv = _max_cos(Xc)
                if vv < best_val - 1e-12:
                    # do tiny LP if time permits
                    try:
                        Xc2 = _lp_polish_capped(Xc, 0.8, HARD_END)
                        vv2 = _max_cos(Xc2)
                        if vv2 < vv:
                            Xc, vv = Xc2, vv2
                    except Exception:
                        pass
                    if vv < best_val - 1e-12:
                        best = Xc.copy()
                        best_val = vv
                        if vv < incumbent_val:
                            incumbent = best.copy()
                            incumbent_val = vv
                continue
            # capped-pair LP polish
            Xc = _lp_polish_capped(Xc, 1.2, HARD_END)
            vv = _max_cos(Xc)
            if vv < best_val - 1e-12:
                best = Xc.copy()
                best_val = vv
                if vv < incumbent_val:
                    incumbent = best.copy()
                    incumbent_val = vv
        except Exception:
            continue
        # early stop if time nearly up
        if trial > 300:
            break

    # reserve final 20s for LP polishing of best
    try:
        remain = FINAL_END - time.time()
        if remain > 3.0:
            lp_sec = min(remain - 1.0, 18.0)
            polished = _lp_polish_capped(best, lp_sec, HARD_END)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
    except Exception:
        pass

    # guaranteed fallback to incumbent if no improvement
    if best_val > incumbent_val:
        best = incumbent.copy()
        best_val = incumbent_val
    best = _norm(best)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    best = _norm(best)
    return best
