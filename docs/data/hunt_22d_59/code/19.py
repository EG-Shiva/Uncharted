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

def _adam_lse(x_init, beta0, beta1, lr0, lr1, steps, deadline, rng_off=0):
    x = x_init.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    cnt = 0
    # log schedules
    if steps <= 1:
        lb = 0.0
        ll = 0.0
    else:
        lb = (np.log(float(beta1)) - np.log(float(beta0))) / float(steps - 1)
        ll = (np.log(float(lr1)) - np.log(float(lr0))) / float(steps - 1)
    lb0 = np.log(float(beta0))
    ll0 = np.log(float(lr0))
    for k in range(int(steps)):
        if (k & 255) == 0 and time.time() > deadline:
            break
        beta = float(np.exp(lb0 + lb * k))
        lr = float(np.exp(ll0 + ll * k))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = float(np.max(G))
        E = np.exp(beta * (G - mval))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            continue
        Gr = (E / S) @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        cnt += 1
        m1 *= 0.9
        m1 += 0.1 * Gt
        m2 *= 0.999
        m2 += 0.001 * (Gt * Gt)
        b1 = 1.0 - 0.9 ** cnt
        b2 = 1.0 - 0.999 ** cnt
        m1h = m1 / max(b1, 1e-12)
        m2h = m2 / max(b2, 1e-12)
        step = m1h / (np.sqrt(m2h) + 1e-8)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x -= lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1.0
        x /= nr
    return x

def _sa_sweep(x, cur_best, steps, sigma, T0, T1, rng, deadline, G_cache=None):
    n, d = x.shape
    best = float(cur_best)
    # maintain Gram
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    rowmax = np.max(G, axis=1)
    for s in range(int(steps)):
        if (s & 255) == 0 and time.time() > deadline:
            break
        frac = float(s) / float(max(1, int(steps) - 1))
        T = float(T0 + (T1 - T0) * frac)
        # pick point biased to worst
        r = float(rng.random())
        if r < 0.7:
            # softmax pick among rowmax
            # cheap: pick among top candidates
            thr = m - 0.004
            cand = np.where(rowmax >= thr)[0]
            if cand.size == 0:
                p = int(rng.integers(0, n))
            else:
                p = int(cand[int(rng.integers(0, cand.size))])
        else:
            p = int(rng.integers(0, n))
        # propose
        rv = rng.standard_normal(d)
        dp = float(np.dot(rv, x[p]))
        rv -= dp * x[p]
        rn = float(np.linalg.norm(rv))
        if rn < 1e-12:
            continue
        rv /= rn
        new_v = x[p] + float(sigma) * rv
        nv = float(np.linalg.norm(new_v))
        if nv < 1e-12 or not np.isfinite(nv):
            continue
        new_v /= nv
        dots = x @ new_v
        dots[p] = -2.0
        new_others = float(np.max(dots))
        # old without p: max of G excluding row/col p
        # fast: mask
        # use precomputed? compute via max of G with row/col masked
        # to avoid alloc, compute in two steps
        # G has -2 on diagonal
        om = -2.0
        # max excluding p: iterate? use numpy max on masked view
        # small n so copy is fine
        # we can compute quickly:
        # temporarily set row/col p to -2
        old_row = G[p, :].copy()
        old_col = G[:, p].copy()
        G[p, :] = -2.0
        G[:, p] = -2.0
        old_without = float(np.max(G))
        G[p, :] = old_row
        G[:, p] = old_col
        new_m = new_others if new_others > old_without else old_without
        if new_m < m - 1e-12:
            # accept improving
            x[p] = new_v
            # update G row/col
            d2 = x @ new_v
            G[p, :] = d2
            G[:, p] = d2
            G[p, p] = -2.0
            m = new_m
            rowmax = np.max(G, axis=1)
            if new_m < best:
                best = new_m
        elif T > 0 and new_m <= m + T:
            # neutral / uphill SA: accept with prob exp((m-new_m)/T)
            if new_m <= m + 1e-12:
                # always allow small neutral drift with 0.3 prob if not worsening much
                if float(rng.random()) < 0.3:
                    x[p] = new_v
                    d2 = x @ new_v
                    G[p, :] = d2
                    G[:, p] = d2
                    G[p, p] = -2.0
                    m = new_m
                    rowmax = np.max(G, axis=1)
            else:
                prob = float(np.exp((m - new_m) / max(T, 1e-12)))
                if float(rng.random()) < prob * 0.5:
                    x[p] = new_v
                    d2 = x @ new_v
                    G[p, :] = d2
                    G[:, p] = d2
                    G[p, p] = -2.0
                    m = new_m
                    rowmax = np.max(G, axis=1)
        # else reject
    # recompute best as current m vs best
    return x, min(best, m)

def _best_hole(Z, rng, n_try=2500, refine_steps=400, beta=300.0):
    # Z: (m,d) fixed points, find y on sphere minimizing max_i dot(y,Z)
    m, d = Z.shape
    C = rng.standard_normal((int(n_try), d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    dots = C @ Z.T
    vals = np.max(dots, axis=1)
    idx = np.argsort(vals)
    k_top = min(8, int(n_try))
    best_y = C[idx[0]].copy()
    best_v = float(vals[idx[0]])
    for t in range(k_top):
        y = C[idx[t]].copy()
        # refine with smoothed max
        for k in range(int(refine_steps)):
            s = Z @ y  # (m,)
            mv = float(np.max(s))
            w = np.exp(beta * (s - mv))
            sw = float(np.sum(w))
            if sw < 1e-300 or not np.isfinite(sw):
                break
            w /= sw
            g = w @ Z  # (d,)
            # riemannian
            g = g - float(np.dot(g, y)) * y
            lr = 0.05 / (1.0 + k * 0.01)
            y = y - lr * g
            y /= (np.linalg.norm(y) + 1e-12)
        v = float(np.max(Z @ y))
        if v < best_v:
            best_v = v
            best_y = y.copy()
    return best_y, best_v

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 170.0
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
            if not np.all(np.isfinite(a)):
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

    # Stage 1: strong annealed polish of incumbent
    try:
        x = _adam_lse(best_x, 60.0, 500.0, 0.006, 0.0012, 3500, deadline)
        v = _max_cos(x)
        if v < best_val:
            best_x, best_val = x, v
        x = _adam_lse(best_x, 500.0, 2500.0, 0.0012, 0.0004, 3500, deadline)
        v = _max_cos(x)
        if v < best_val:
            best_x, best_val = x, v
    except Exception:
        pass
    try:
        x2, v2 = _sa_sweep(best_x.copy(), best_val, 6000, 0.003, 0.0004, 0.0, rng, deadline)
        vv = _max_cos(x2)
        if min(vv, v2) < best_val:
            best_val = min(vv, v2)
            best_x = _norm(x2)
    except Exception:
        pass

    trial = 0
    sigmas = [0.02, 0.008, 0.035, 0.012, 0.05, 0.006, 0.025, 0.015]
    while time.time() < deadline - 30.0:
        trial += 1
        if trial > 300:
            break
        mode = trial % 4
        try:
            if mode == 0 or mode == 2:
                # kill-and-fill 1 or 2 worst points
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                order = np.argsort(-rowmax)
                kkill = 1 if mode == 0 else 2
                kill = order[:kkill].tolist()
                # occasionally kill random instead
                if trial % 11 == 0:
                    kill = [int(rng.integers(0, 59))]
                    if kkill == 2:
                        kill.append(int(rng.integers(0, 59)))
                keep_mask = np.ones(59, dtype=bool)
                keep_mask[kill] = False
                Z = best_x[keep_mask]  # (59-k,22)
                cand_x = best_x.copy()
                for p in kill:
                    y, _ = _best_hole(Z, rng, n_try=2000, refine_steps=300, beta=250.0)
                    cand_x[p] = y
                    Z = np.vstack([Z, y[None, :]]) if Z.shape[0] < 59 else Z
                    # for second kill, Z includes first reinserted; need to exclude second point
                    # rebuild Z correctly for next p: keep all except remaining kills
                    # simple: after first insert, keep Z as is but remove next kill handling via mask update
                    # instead recompute: fixed = all rows except not-yet-filled kills
                # for k=2 sequential hole search is approximate; refine by re-optimizing both holes jointly
                if kkill == 2:
                    # joint refine of the two killed points with LSE Adam on 2 vectors
                    sub = cand_x[kill].copy()
                    fixed = cand_x.copy()
                    # quick local adam on sub vs all
                    for _ in range(300):
                        if time.time() > deadline:
                            break
                        # gradient from fixed + mutual
                        Sall = fixed @ sub[0]
                        # use soft max over all 59? approximate with beta 300
                        mv = float(np.max(Sall[np.arange(59) != kill[0]])) if True else 0.0
                        # compute weights over other points
                        vals = fixed @ sub[0]
                        vals[kill[0]] = -10.0
                        mv0 = float(np.max(vals))
                        w0 = np.exp(300.0 * (vals - mv0))
                        w0[kill[0]] = 0.0
                        sw0 = float(np.sum(w0))
                        if sw0 > 1e-300:
                            g0 = (w0 / sw0) @ fixed
                            g0 = g0 - float(np.dot(g0, sub[0])) * sub[0]
                            sub[0] = sub[0] - 0.03 * g0
                            sub[0] /= (np.linalg.norm(sub[0]) + 1e-12)
                        vals1 = fixed @ sub[1]
                        # update fixed row for second? use updated first
                        fixed[kill[0]] = sub[0]
                        vals1 = fixed @ sub[1]
                        vals1[kill[1]] = -10.0
                        mv1 = float(np.max(vals1))
                        w1 = np.exp(300.0 * (vals1 - mv1))
                        w1[kill[1]] = 0.0
                        sw1 = float(np.sum(w1))
                        if sw1 > 1e-300:
                            g1 = (w1 / sw1) @ fixed
                            g1 = g1 - float(np.dot(g1, sub[1])) * sub[1]
                            sub[1] = sub[1] - 0.03 * g1
                            sub[1] /= (np.linalg.norm(sub[1]) + 1e-12)
                        fixed[kill[1]] = sub[1]
                    cand_x[kill[0]] = sub[0]
                    cand_x[kill[1]] = sub[1]
                cand_x = _norm(cand_x)
                # short polish
                cand_x = _adam_lse(cand_x, 150.0, 1200.0, 0.0015, 0.0006, 1200, deadline)
                v = _max_cos(cand_x)
                if v < best_val + 0.0015:
                    cand_x2, v2 = _sa_sweep(cand_x, v, 2000, 0.002, 0.0002, 0.0, rng, deadline)
                    v2m = _max_cos(cand_x2)
                    if v2m < v:
                        cand_x, v = cand_x2, v2m
                if v < best_val:
                    best_x, best_val = cand_x, v
            else:
                sigma = float(sigmas[(trial - 1) % len(sigmas)])
                # subset shake: perturb all lightly + a few strongly
                pert = best_x.copy()
                if trial % 2 == 1:
                    noise = rng.standard_normal((59, 22))
                    pert = pert + sigma * noise
                else:
                    # shake 5 random points hard
                    idx = rng.choice(59, size=5, replace=False)
                    noise = np.zeros((59, 22))
                    noise[idx] = rng.standard_normal((5, 22)) * (sigma * 2.0)
                    noise += rng.standard_normal((59, 22)) * (sigma * 0.3)
                    pert = pert + noise
                pert = _norm(pert)
                pert = _adam_lse(pert, 120.0, 1500.0, 0.0012, 0.0005, 1400, deadline)
                v = _max_cos(pert)
                if v < best_val + 0.002:
                    pert2, v2 = _sa_sweep(pert, v, 2000, 0.0025, 0.0003, 0.0, rng, deadline)
                    vm = _max_cos(pert2)
                    if vm < v:
                        pert, v = pert2, vm
                if v < best_val:
                    best_x, best_val = pert, v
        except Exception:
            continue

    # Final squeeze: high-beta tiny-lr polish
    if time.time() < deadline - 8.0:
        try:
            x = _adam_lse(best_x, 800.0, 4000.0, 0.0006, 0.00015, 4000, deadline)
            v = _max_cos(x)
            if v < best_val:
                best_x, best_val = x, v
        except Exception:
            pass
    if time.time() < deadline - 4.0:
        try:
            x2, v2 = _sa_sweep(best_x.copy(), best_val, 5000, 0.0012, 0.0001, 0.0, rng, deadline)
            vm = _max_cos(x2)
            if vm < best_val:
                best_x, best_val = x2, vm
        except Exception:
            pass

    best_x = _norm(best_x)
    # tiny shrink guard not needed (normalization exact)
    return best_x
