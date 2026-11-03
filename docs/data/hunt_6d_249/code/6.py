import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _pair_polish(x0, steps, s0=2e-3):
    x = x0.copy()
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    best_val = float(np.max(g))
    best = x.copy()
    s = s0
    n = x.shape[0]
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        dot = m
        di = x[j] - dot * x[i]
        dj = x[i] - dot * x[j]
        ni = float(np.linalg.norm(di))
        nj = float(np.linalg.norm(dj))
        if ni < 1e-14 or nj < 1e-14:
            s *= 0.5
            if s < 1e-13:
                break
            continue
        ui = di / ni
        uj = dj / nj
        xi = x[i] - s * ui
        xj = x[j] - s * uj
        xi /= np.linalg.norm(xi)
        xj /= np.linalg.norm(xj)
        xp = x.copy()
        xp[i] = xi
        xp[j] = xj
        Gp = xp @ xp.T
        np.fill_diagonal(Gp, -2.0)
        mp = float(np.max(Gp))
        if mp < m - 1e-14:
            x = xp
            if mp < best_val:
                best_val = mp
                best = x.copy()
            s *= 1.08
            if s > 0.01:
                s = 0.01
        else:
            s *= 0.55
            if s < 1e-13:
                break
    return best, best_val

def _smooth_opt(x0, steps, s0, s1, T0, T1):
    x = x0.copy()
    n, d = x.shape
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(np.max(G0))
    best = x.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        s = s0 * (s1 / s0) ** frac
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights focused on top
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        Wx = W @ x
        wG = np.sum(W * G, axis=1)
        F = -Wx + wG[:, None] * x
        nrm = np.sqrt(np.sum(F * F, axis=1))
        nz = nrm > 1e-15
        if np.any(nz):
            x[nz] += (s / nrm[nz])[:, None] * F[nz]
            inv = 1.0 / np.sqrt(np.sum(x * x, axis=1))
            x *= inv[:, None]
        if (it % 20 == 0) or (it == steps - 1):
            Gc = x @ x.T
            np.fill_diagonal(Gc, -2.0)
            mv = float(np.max(Gc))
            if mv < best_val:
                best_val = mv
                best = x.copy()
    return best, best_val

def _reinsert(x0, rng, n_cand=15000, n_try=4):
    x = x0.copy()
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mi = np.max(G, axis=1)
    order = np.argsort(-mi)
    Gfull = x @ x.T
    np.fill_diagonal(Gfull, -2.0)
    best_val = float(np.max(Gfull))
    best = x.copy()
    for k in order[:n_try]:
        C = rng.standard_normal(size=(n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ x.T
        D[:, k] = -2.0
        vals = np.max(D, axis=1)
        idx = int(np.argmin(vals))
        if vals[idx] < mi[k] - 1e-12:
            xn = x.copy()
            xn[k] = C[idx]
            Gn = xn @ xn.T
            np.fill_diagonal(Gn, -2.0)
            mn = float(np.max(Gn))
            if mn < best_val - 1e-14:
                x = xn
                best = xn.copy()
                best_val = mn
                mi = np.max(Gn, axis=1)
    return best, best_val

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 168.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(249, 6))
    if xb.shape != (249, 6):
        # reshape / truncate / pad deterministically
        flat = xb.reshape(-1, 6)
        if flat.shape[0] >= 249:
            xb = flat[:249]
        else:
            extra = rng.standard_normal(size=(249 - flat.shape[0], 6))
            xb = np.vstack([flat, extra])
    xb = _norm(xb)
    best_val = _max_cos(xb)
    best = xb.copy()

    # 1) kill single-pair max
    xb1, v1 = _pair_polish(best, steps=1200, s0=3e-3)
    if v1 < best_val:
        best, best_val = xb1, v1
    if time.time() - t0 > BUDGET - 20:
        return _norm(best)

    # 2) smooth rebalance
    xb2, v2 = _smooth_opt(best, steps=1200, s0=6e-3, s1=8e-05, T0=4e-3, T1=2e-4)
    if v2 < best_val:
        best, best_val = xb2, v2

    # 3) hole reinsertion on current best
    try:
        xr, vr = _reinsert(best, rng, n_cand=20000, n_try=6)
        if vr < best_val:
            best, best_val = xr, vr
            xb2b, v2b = _smooth_opt(best, steps=600, s0=2e-3, s1=5e-05, T0=2e-3, T1=2e-4)
            if v2b < best_val:
                best, best_val = xb2b, v2b
    except Exception:
        pass

    sigmas = [0.020, 0.010, 0.005, 0.015, 0.008, 0.030, 0.006, 0.012]
    trial = 0
    while True:
        trial += 1
        tnow = time.time() - t0
        if tnow > BUDGET - 9:
            break
        if trial > 400:
            break
        sig = sigmas[(trial - 1) % len(sigmas)]
        # alternate global vs targeted shake
        if trial % 2 == 0:
            # targeted: shake worst points more
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            mi = np.max(G, axis=1)
            order = np.argsort(-mi)
            noise = rng.standard_normal(size=best.shape) * (sig * 0.25)
            k = 30
            idx = order[:k]
            noise[idx] *= 4.0
            xt = best + noise
            xt = _norm(xt)
        else:
            xt = _norm(best + sig * rng.standard_normal(size=best.shape))
        # every 6th trial: reinsert-first diversification
        if trial % 6 == 0:
            try:
                xtr, vtr = _reinsert(xt, rng, n_cand=8000, n_try=3)
                xt = xtr
            except Exception:
                pass
        # local opt: pair + smooth
        # adapt steps to remaining time
        rem = BUDGET - 9 - (time.time() - t0)
        if rem < 2:
            break
        steps = 700 if rem > 30 else 350
        xp, vp = _pair_polish(xt, steps=150, s0=1e-3)
        # smooth
        xs, vs = _smooth_opt(xp, steps=steps, s0=5e-3, s1=6e-05, T0=3e-3, T1=2e-4)
        if vs < best_val - 1e-14:
            best, best_val = xs, vs
            # intensify around new best with hole search
            try:
                xr2, vr2 = _reinsert(best, rng, n_cand=12000, n_try=4)
                if vr2 < best_val:
                    best, best_val = xr2, vr2
                # small extra smooth
                xs2, vs2 = _smooth_opt(best, steps=400, s0=1.5e-3, s1=3e-05, T0=1.5e-3, T1=1.5e-4)
                if vs2 < best_val:
                    best, best_val = xs2, vs2
            except Exception:
                pass
        elif vp < best_val - 1e-14:
            best, best_val = xp, vp

    # final fine polish
    try:
        if time.time() - t0 < BUDGET - 2:
            xf, vf = _pair_polish(best, steps=400, s0=5e-4)
            if vf < best_val:
                best, best_val = xf, vf
        if time.time() - t0 < BUDGET - 2:
            xs, vs = _smooth_opt(best, steps=800, s0=1.2e-3, s1=2e-05, T0=1.2e-3, T1=1.2e-4)
            if vs < best_val:
                best, best_val = xs, vs
    except Exception:
        pass

    best = _norm(best)
    # safety: remove NaN
    best = np.where(np.isfinite(best), best, 0.0)
    # if any zero row, replace deterministically
    nr = np.linalg.norm(best, axis=1)
    for i in range(best.shape[0]):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            v = rng.standard_normal(size=6)
            v /= np.linalg.norm(v)
            best[i] = v
    best = _norm(best)
    return best
