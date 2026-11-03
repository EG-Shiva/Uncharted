import time
import numpy as np


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _adam_polish(x_init, alphas, iters_list, lr_base, deadline):
    x = x_init.copy()
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    for ai, alpha in enumerate(alphas):
        it = int(iters_list[ai])
        m1.fill(0.0)
        m2.fill(0.0)
        cnt = 0
        lr_a0 = float(lr_base * np.sqrt(200.0 / float(alpha)))
        for k in range(it):
            if (k & 127) == 0 and time.time() > deadline:
                return x
            lr = lr_a0 * (1.0 - 0.5 * float(k) / float(it))
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            m_val = float(np.max(G))
            E = np.exp(alpha * (G - m_val))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E) * 0.5)
            if S < 1e-300:
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
            if b1 < 1e-12:
                b1 = 1e-12
            if b2 < 1e-12:
                b2 = 1e-12
            m1h = m1 / b1
            m2h = m2 / b2
            step = m1h / (np.sqrt(m2h) + 1e-8)
            sdot = np.sum(step * x, axis=1, keepdims=True)
            step_t = step - sdot * x
            x -= lr * step_t
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr < 1e-12] = 1.0
            x /= nr
    return x


def _hill_climb(x, cur_val, n_steps, sigma0, sigma1, rng, deadline):
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
        rpick = float(rng.random())
        if rpick < 0.15:
            p = int(rng.integers(0, n))
        else:
            thr = m - 0.003
            cand = np.where(row_max >= thr)[0]
            if cand.size == 0:
                p = int(rng.integers(0, n))
            else:
                p = int(cand[int(rng.integers(0, cand.size))])
        rvec = rng.standard_normal(d)
        dp = float(np.dot(rvec, x[p]))
        rvec -= dp * x[p]
        rn = float(np.linalg.norm(rvec))
        if rn < 1e-12:
            continue
        rvec /= rn
        new_v = x[p] + sigma * rvec
        nv = float(np.linalg.norm(new_v))
        if nv < 1e-12:
            continue
        new_v /= nv
        dots = x @ new_v
        dots[p] = -2.0
        new_others = float(np.max(dots))
        G[p, :] = -2.0
        G[:, p] = -2.0
        old_without = float(np.max(G))
        new_m = new_others if new_others > old_without else old_without
        if new_m < m - 1e-12:
            x[p] = new_v
            if new_m < best:
                best = new_m
        elif new_m <= m + 1e-12 and row_max[p] < m - 1e-3:
            if float(rng.random()) < 0.5:
                x[p] = new_v
    return x, best


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0

    cands = []
    try:
        r = np.load("record.npy")
        cands.append(np.asarray(r, dtype=np.float64))
    except Exception:
        pass
    try:
        b = np.load("best.npy")
        cands.append(np.asarray(b, dtype=np.float64))
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
            nrm = np.linalg.norm(a, axis=1, keepdims=True)
            nrm[nrm < 1e-12] = 1.0
            a = a / nrm
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a
        except Exception:
            continue
    if best_x is None:
        best_x = rng.standard_normal((59, 22))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)

    # Stage 1: strong annealed polish of incumbent
    try:
        x_pol = _adam_polish(best_x, [40, 120, 350, 900, 2200],
                             [1200, 1500, 2000, 2500, 3500], 0.0009, deadline)
        v_pol = _max_cos(x_pol)
        if v_pol < best_val:
            best_x, best_val = x_pol, v_pol
    except Exception:
        pass

    try:
        x_h, v_h = _hill_climb(best_x.copy(), best_val, 12000,
                               0.004, 0.0003, rng, deadline)
        if v_h < best_val:
            best_x, best_val = x_h, v_h
    except Exception:
        pass

    # Basin hopping: perturb best to change contact topology
    trial = 0
    sigmas = [0.008, 0.015, 0.03, 0.005, 0.02]
    while time.time() < deadline - 25.0:
        trial += 1
        sigma = float(sigmas[(trial - 1) % len(sigmas)])
        try:
            pert = best_x + sigma * rng.standard_normal((59, 22))
            nrm = np.linalg.norm(pert, axis=1, keepdims=True)
            nrm[nrm < 1e-12] = 1.0
            pert /= nrm
            x_t = _adam_polish(pert, [150, 600, 2000],
                               [700, 1000, 1400], 0.0008, deadline)
            v_t = _max_cos(x_t)
            if v_t < best_val + 0.002:
                x_t, v_t = _hill_climb(x_t, v_t, 2500,
                                       0.003, 0.0005, rng, deadline)
            if v_t < best_val:
                best_x, best_val = x_t, v_t
        except Exception:
            continue
        if trial > 200:
            break

    # Final long polish
    if time.time() < deadline - 10.0:
        try:
            x_f = _adam_polish(best_x, [300, 1000, 3000],
                               [1500, 2000, 3000], 0.0006, deadline)
            v_f = _max_cos(x_f)
            if v_f < best_val:
                best_x, best_val = x_f, v_f
        except Exception:
            pass
        try:
            if time.time() < deadline - 5.0:
                x_f2, v_f2 = _hill_climb(best_x.copy(), best_val, 8000,
                                         0.002, 0.0002, rng, deadline)
                if v_f2 < best_val:
                    best_x, best_val = x_f2, v_f2
        except Exception:
            pass

    nrm = np.linalg.norm(best_x, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    best_x = best_x / nrm
    return best_x
