import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def quench(x_init, steps, T0, T1, lr, rng=None, noise=0.0):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lrs = float(lr * (1.0 - 0.5 * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        P = E / S
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = lrs * (m / bc1) / (np.sqrt(v / bc2) + eps)
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        x = x - step
        if noise > 0 and rng is not None:
            nz = rng.normal(size=x.shape) * noise
            nz = nz - np.sum(nz * x, axis=1, keepdims=True) * x
            x = x + nz
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
    return x

def hole_reinsert(X_rem, k_need, rng, n_cand=4000):
    d = X_rem.shape[1]
    cand = rng.normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cur = X_rem.copy()
    outs = []
    for _ in range(k_need):
        sc = (cand @ cur.T).max(axis=1)
        bi = int(np.argmin(sc))
        q = cand[bi].copy()
        # cheap hole refine
        for __ in range(80):
            dots = cur @ q
            mm = float(dots.max())
            w = np.exp((dots - mm) / 0.004)
            s = float(w.sum())
            if s == 0 or not np.isfinite(s):
                break
            w /= s
            g = w @ cur
            g = g - float(g @ q) * q
            q = q - 0.05 * g
            nq = float(np.linalg.norm(q))
            if nq == 0 or not np.isfinite(nq):
                break
            q /= nq
        outs.append(q)
        cur = np.vstack([cur, q[None, :]])
    return np.array(outs)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-cfull[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            p = 2500
            csel = cfull[sel][idx]
        else:
            csel = cfull[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def fingerprint(x, k=200):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    c = np.sort(c)[::-1][:k]
    return c

def align_and_child(A, B, rng, mode):
    # match B to A
    S = A @ B.T
    r, c = linear_sum_assignment(-S)
    # r should be 0..n-1 sorted; build permuted B
    Bp = np.empty_like(B)
    Bp[r] = B[c]
    # Procrustes: Bp @ R ~= A
    M = Bp.T @ A
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Bal = Bp @ R
    n, d = A.shape
    if mode == 0:
        mask = rng.random(n) < 0.5
        C = np.where(mask[:, None], A, Bal)
        C = C + rng.normal(size=C.shape) * 0.008
    else:
        t = rng.random((n, 1))
        C = (1 - t) * A + t * Bal
        C = C + rng.normal(size=C.shape) * 0.008
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng0 = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.normal(size=(102, 12))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (102, 12):
        tmp = rng0.normal(size=(102, 12))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m0 = min(xb.shape[0], 102); m1 = min(xb.shape[1], 12)
        tmp[:m0, :m1] = xb[:m0, :m1]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    n, d = 102, 12
    best_overall = max_cos(xb)
    x_overall = xb.copy()

    # Phase 1: population of distinct optima, cheap polish, no LP
    NPOP = 10
    sigmas = [0.06, 0.12, 0.20, 0.30, 0.10, 0.18, 0.26, 0.08, 0.15, 0.35]
    dfracs = [0.0, 0.05, 0.10, 0.12, 0.07, 0.03, 0.14, 0.09, 0.05, 0.12]
    T0s = [0.012, 0.020, 0.028, 0.018, 0.025, 0.015, 0.030, 0.012, 0.022, 0.026]
    T1s = [0.001, 0.002, 0.003, 0.0015, 0.0025, 0.0012, 0.002, 0.0008, 0.0018, 0.0022]
    lrs = [0.005, 0.007, 0.006, 0.008, 0.004, 0.006, 0.007, 0.005, 0.0065, 0.0075]
    pop = []
    scores = []
    for i in range(NPOP):
        if time.time() - t0 > 75:
            break
        rng = np.random.default_rng(1000 + i)
        x = xb.copy()
        kd = int(round(dfracs[i] * n))
        if kd > 0:
            # delete random + worst mix for diversity
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mv = float(G.max())
            cnt = (G > mv - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            nworst = kd // 2
            worst = order[:nworst].tolist() if nworst > 0 else []
            remaining = [j for j in range(n) if j not in worst]
            extra = rng.choice(remaining, size=kd - len(worst), replace=False).tolist() if kd - len(worst) > 0 else []
            dels = np.array(worst + extra, dtype=int)
            keep = np.ones(n, dtype=bool); keep[dels] = False
            Xrem = x[keep]
            newpts = hole_reinsert(Xrem, kd, rng)
            x[dels] = newpts
        s = sigmas[i]
        x = x + rng.normal(size=x.shape) * s
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        steps = 650 if i % 2 == 0 else 450
        # shrink steps if running late
        if time.time() - t0 > 55:
            steps = 350
        xq = quench(x, steps, T0s[i], T1s[i], lrs[i], rng=rng, noise=0.0)
        v = max_cos(xq)
        pop.append(xq)
        scores.append(v)
        if v < best_overall:
            best_overall = v
            x_overall = xq.copy()
    if len(pop) == 0:
        pop = [xb.copy()]
        scores = [best_overall]
    scores = np.array(scores)
    fps = [fingerprint(p) for p in pop]
    # retain diverse subset D<=6
    D = min(6, len(pop))
    order_best = np.argsort(scores)
    selected = [int(order_best[0])]
    while len(selected) < D:
        best_gain = -1e100
        best_c = -1
        for c in range(len(pop)):
            if c in selected:
                continue
            if scores[c] > float(scores[order_best[0]]) + 0.004:
                continue
            dmin = min(float(np.linalg.norm(fps[c] - fps[s])) for s in selected)
            gain = dmin - 200.0 * max(0.0, float(scores[c] - scores[selected[0]]))
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c < 0:
            break
        selected.append(int(best_c))
    # Phase 2: crossover children
    rngx = np.random.default_rng(777)
    pairs = []
    # pairs of best with others + a few among rest
    s0 = selected[0]
    for s in selected[1:]:
        pairs.append((s0, s))
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            if (selected[a], selected[b]) not in pairs and (selected[b], selected[a]) not in pairs:
                pairs.append((selected[a], selected[b]))
            if len(pairs) >= 8:
                break
        if len(pairs) >= 8:
            break
    pairs = pairs[:8]
    children = []
    cscores = []
    for pi, (ia, ib) in enumerate(pairs):
        if time.time() - t0 > 120:
            break
        A = pop[ia]; B = pop[ib]
        try:
            C = align_and_child(A, B, rngx, mode=pi % 2)
        except Exception:
            continue
        steps = 400
        if time.time() - t0 > 100:
            steps = 250
        Cq = quench(C, steps, 0.012, 0.0012, 0.006, rng=rngx)
        v = max_cos(Cq)
        children.append(Cq)
        cscores.append(v)
        if v < best_overall:
            best_overall = v
            x_overall = Cq.copy()
    # fast soft-max screen already done; pick best two children for LP
    if len(children) > 0:
        o = np.argsort(np.array(cscores))
        top2 = [children[o[k]] for k in range(min(2, len(children)))]
    else:
        top2 = []
    # Phase 3: LP polish only best two children (+ overall if time)
    rem_total = BUDGET - (time.time() - t0) - 2.0
    per = 0.0
    if len(top2) > 0 and rem_total > 6:
        per = min(20.0, rem_total / (len(top2) + 1))
        new_top = []
        for c in top2:
            if time.time() - t0 > BUDGET - 8:
                break
            try:
                cp = lp_polish(c, per)
            except Exception:
                cp = c
            v = max_cos(cp)
            new_top.append((v, cp))
            if v < best_overall:
                best_overall = v
                x_overall = cp.copy()
    # final polish of overall with leftover time
    rem2 = BUDGET - (time.time() - t0) - 1.0
    if rem2 > 4:
        try:
            x_overall = lp_polish(x_overall, min(rem2, 25.0))
            best_overall = max_cos(x_overall)
        except Exception:
            pass
    x_overall = x_overall / np.linalg.norm(x_overall, axis=1, keepdims=True)
    x_overall = np.nan_to_num(x_overall, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(x_overall, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_overall = x_overall / nrm
    return x_overall
