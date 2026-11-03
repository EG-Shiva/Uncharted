import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

def _adam_lse(x_init, steps, beta0, beta1, lr0, lr1, deadline):
    x = x_init.copy()
    n, d = x.shape
    M = np.zeros_like(x)
    V = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = float(np.log(beta0)); lb1 = float(np.log(beta1))
    ll0 = float(np.log(lr0)); ll1 = float(np.log(lr1))
    cnt = 0
    best = _max_cos(x)
    bestX = x.copy()
    steps = int(steps)
    for t in range(steps):
        if (t & 255) == 0 and time.time() > deadline:
            break
        f = float(t) / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = float(np.max(G))
        if (t & 127) == 0 and gmax < best:
            best = gmax
            bestX = x.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(np.sum(G))
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gr -= dot * x
        cnt += 1
        M *= b1
        M += (1.0 - b1) * Gr
        V *= b2
        V += (1.0 - b2) * (Gr * Gr)
        b1c = 1.0 - (0.9 ** cnt) if cnt < 2000 else 1.0
        b2c = 1.0 - (0.999 ** cnt) if cnt < 20000 else 1.0
        if b1c < 1e-12:
            b1c = 1e-12
        if b2c < 1e-12:
            b2c = 1e-12
        Mh = M / b1c
        Vh = V / b2c
        step = Mh / (np.sqrt(Vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step -= sdot * x
        x -= lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1.0
        x /= nr
    v = _max_cos(x)
    if v < best:
        return x, v
    return bestX, best

def _hill(x, cur_val, n_steps, sigma0, sigma1, rng, deadline):
    n, d = x.shape
    best = float(cur_val)
    n_steps = int(n_steps)
    for s in range(n_steps):
        if (s & 255) == 0 and time.time() > deadline:
            break
        frac = float(s) / float(max(1, n_steps - 1))
        sigma = float(sigma0 + (sigma1 - sigma0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        row_max = np.max(G, axis=1)
        r = float(rng.random())
        if r < 0.2:
            p = int(rng.integers(0, n))
        else:
            thr = m - 0.004
            cand = np.where(row_max >= thr)[0]
            if cand.size == 0:
                p = int(rng.integers(0, n))
            else:
                p = int(cand[int(rng.integers(0, cand.size))])
        rv = rng.standard_normal(d)
        dp = float(np.dot(rv, x[p]))
        rv -= dp * x[p]
        rn = float(np.linalg.norm(rv))
        if rn < 1e-12:
            continue
        rv /= rn
        nv = x[p] + sigma * rv
        nn = float(np.linalg.norm(nv))
        if nn < 1e-12:
            continue
        nv /= nn
        dots = x @ nv
        dots[p] = -2.0
        new_others = float(np.max(dots))
        # max without p
        # compute quickly: mask row/col p
        # use precomputed G
        G[p, :] = -2.0
        G[:, p] = -2.0
        old_without = float(np.max(G))
        new_m = new_others if new_others > old_without else old_without
        if new_m < m - 1e-12:
            x[p] = nv
            if new_m < best:
                best = new_m
        elif new_m <= m + 1e-12 and row_max[p] < m - 1e-3:
            if float(rng.random()) < 0.4:
                x[p] = nv
    return x, best

def _hole_reinsert(X, victims, rng, n_cand=15000):
    n, d = X.shape
    Y = X.copy()
    victims = list(int(v) for v in victims)
    keep_idx = [i for i in range(n) if i not in set(victims)]
    Keep = Y[keep_idx].copy() if len(keep_idx) > 0 else np.zeros((0, d))
    # random order
    order = victims[:]
    rng.shuffle(order)
    for v in order:
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        if Keep.shape[0] > 0:
            D = C @ Keep.T
            scores = np.max(D, axis=1)
        else:
            scores = np.zeros(n_cand)
        # also avoid being close to already reinserted beyond Keep? Keep includes them after each step
        bi = int(np.argmin(scores))
        c = C[bi].copy()
        # local refine c against Keep: minimize LSE / max
        if Keep.shape[0] > 0:
            # few projected steps
            for it in range(120):
                dots = Keep @ c  # K
                m = float(np.max(dots))
                # softmax weights
                w = np.exp(60.0 * (dots - m))
                w /= (np.sum(w) + 1e-30)
                g = w @ Keep  # d
                g -= float(np.dot(g, c)) * c
                gn = float(np.linalg.norm(g))
                if gn < 1e-12:
                    break
                lr = 0.05 * (1.0 - it / 120.0) + 0.005
                c -= lr * g
                c /= (np.linalg.norm(c) + 1e-30)
        Y[v] = c
        # add to Keep for next victims
        if Keep.shape[0] > 0:
            Keep = np.vstack([Keep, c[None, :]])
        else:
            Keep = c[None, :].copy()
    return Y

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    best_x = None
    best_val = float("inf")
    for c in cands:
        try:
            a = np.asarray(c, dtype=np.float64)
            if a.shape != (59, 22):
                if a.size == 59 * 22:
                    a = a.reshape(59, 22)
                else:
                    continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.standard_normal((59, 22)))
        best_val = _max_cos(best_x)

    # Stage 0: solid polish of incumbent
    if time.time() < deadline - 20:
        try:
            x0, v0 = _adam_lse(best_x, 3000, 60.0, 900.0, 0.005, 0.0006, deadline)
            if v0 < best_val:
                best_x, best_val = x0, v0
            x0, v0 = _hill(best_x.copy(), best_val, 4000, 0.003, 0.0003, rng, deadline)
            if v0 < best_val:
                best_x, best_val = x0, v0
        except Exception:
            pass

    trial = 0
    # pre-generate a pool of random starts polished lightly to find new basin
    # do a few quick random explorations first
    try:
        for _ in range(4):
            if time.time() > deadline - 40:
                break
            Xr = _norm(rng.standard_normal((59, 22)))
            Xt, vt = _adam_lse(Xr, 2200, 50.0, 500.0, 0.006, 0.0008, deadline)
            if vt < best_val:
                best_x, best_val = Xt, vt
            elif vt < best_val + 0.015:
                # polish further once
                Xt2, vt2 = _adam_lse(Xt, 1500, 200.0, 1500.0, 0.002, 0.0004, deadline)
                if vt2 < best_val:
                    best_x, best_val = Xt2, vt2
    except Exception:
        pass

    while time.time() < deadline - 28.0:
        trial += 1
        if trial > 500:
            break
        mode = trial % 7
        try:
            if mode == 0:
                # kill 1 worst + hole reinsert
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                order = np.argsort(-rm)
                k = 1 if (trial % 14 != 0) else 2
                victims = order[:k].tolist()
                Y = _hole_reinsert(best_x, victims, rng, n_cand=15000)
                Yt, vt = _adam_lse(Y, 1600, 150.0, 1300.0, 0.0025, 0.0004, deadline)
                if vt < best_val + 0.002:
                    Yt, vt = _hill(Yt, vt, 1500, 0.0025, 0.0003, rng, deadline)
                if vt < best_val:
                    best_x, best_val = Yt, vt
            elif mode == 1:
                # kill 3 mixed worst+random
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                order = np.argsort(-rm)
                victims = order[:2].tolist()
                # add one random
                victims.append(int(rng.integers(0, 59)))
                victims = list(dict.fromkeys(victims))
                Y = _hole_reinsert(best_x, victims, rng, n_cand=12000)
                Yt, vt = _adam_lse(Y, 1600, 120.0, 1200.0, 0.0025, 0.0004, deadline)
                if vt < best_val + 0.003:
                    Yt, vt = _hill(Yt, vt, 1200, 0.002, 0.0003, rng, deadline)
                if vt < best_val:
                    best_x, best_val = Yt, vt
            elif mode == 2:
                # kill 5-7 random for big topology jump
                k = 5 + int(rng.integers(0, 3))
                victims = rng.choice(59, size=k, replace=False).tolist()
                Y = _hole_reinsert(best_x, victims, rng, n_cand=12000)
                Yt, vt = _adam_lse(Y, 1800, 80.0, 900.0, 0.003, 0.0005, deadline)
                if vt < best_val + 0.004:
                    Yt2, vt2 = _adam_lse(Yt, 1200, 300.0, 1600.0, 0.0015, 0.0003, deadline)
                    Yt, vt = (Yt2, vt2) if vt2 < vt else (Yt, vt)
                if vt < best_val:
                    best_x, best_val = Yt, vt
            elif mode == 3:
                # subset shake
                k = 4 + int(rng.integers(0, 5))
                idx = rng.choice(59, size=k, replace=False)
                sigma = float([0.06, 0.12, 0.2, 0.08][(trial // 7) % 4])
                Y = best_x.copy()
                Y[idx] += sigma * rng.standard_normal((k, 22))
                Y = _norm(Y)
                Yt, vt = _adam_lse(Y, 1600, 120.0, 1300.0, 0.0025, 0.0004, deadline)
                if vt < best_val:
                    best_x, best_val = Yt, vt
            elif mode == 4:
                # fresh random restart short polish (new basin hunt)
                Xr = _norm(rng.standard_normal((59, 22)))
                Yt, vt = _adam_lse(Xr, 2000, 50.0, 600.0, 0.006, 0.0007, deadline)
                if vt < best_val + 0.01:
                    Yt2, vt2 = _adam_lse(Yt, 1200, 250.0, 1400.0, 0.0018, 0.00035, deadline)
                    if vt2 < vt:
                        Yt, vt = Yt2, vt2
                if vt < best_val:
                    best_x, best_val = Yt, vt
            elif mode == 5:
                # global small shake
                sigma = 0.012 if (trial % 2 == 0) else 0.025
                Y = _norm(best_x + sigma * rng.standard_normal((59, 22)))
                Yt, vt = _adam_lse(Y, 1400, 150.0, 1400.0, 0.002, 0.00035, deadline)
                if vt < best_val:
                    best_x, best_val = Yt, vt
                elif vt < best_val + 0.0015:
                    Yt, vt = _hill(Yt, vt, 1000, 0.002, 0.0003, rng, deadline)
                    if vt < best_val:
                        best_x, best_val = Yt, vt
            else:
                # targeted: most-contact points
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                cnt = np.sum(G > (m - 0.004), axis=1)
                order = np.argsort(-cnt)
                victims = order[:3].tolist()
                Y = _hole_reinsert(best_x, victims, rng, n_cand=15000)
                Yt, vt = _adam_lse(Y, 1500, 130.0, 1200.0, 0.0025, 0.0004, deadline)
                if vt < best_val:
                    best_x, best_val = Yt, vt
        except Exception:
            continue

    # final long polish
    if time.time() < deadline - 12.0:
        try:
            x_f, v_f = _adam_lse(best_x, 3000, 300.0, 3500.0, 0.0012, 0.00025, deadline)
            if v_f < best_val:
                best_x, best_val = x_f, v_f
        except Exception:
            pass
    if time.time() < deadline - 6.0:
        try:
            x_f, v_f = _adam_lse(best_x, 2500, 800.0, 6000.0, 0.0007, 0.00012, deadline)
            if v_f < best_val:
                best_x, best_val = x_f, v_f
        except Exception:
            pass
        try:
            if time.time() < deadline - 3.0:
                x_h, v_h = _hill(best_x.copy(), best_val, 6000, 0.0015, 0.00015, rng, deadline)
                if v_h < best_val:
                    best_x, best_val = x_h, v_h
        except Exception:
            pass

    best_x = _norm(best_x)
    return best_x
