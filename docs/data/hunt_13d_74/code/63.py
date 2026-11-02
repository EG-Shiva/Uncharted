import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    SYM_DEADLINE = t0 + 100.0
    rng = np.random.default_rng(0)
    n, d = 74, 13
    D = 13

    # load baselines
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    # ---------- symmetric builders ----------
    def build_X_C37(Yflat, freqs):
        Y = Yflat.reshape(2, D)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Gg = Y / r  # 2x13
        N = 37
        X = np.zeros((74, D))
        # precompute cos/sin 6xN
        k = np.arange(N)
        for a in range(2):
            ga = Gg[a]
            ax = float(ga[12])
            X[a * N:(a + 1) * N, 12] = ax
            for j, m in enumerate(freqs):
                th = 2.0 * np.pi * k * float(m) / float(N)
                c = np.cos(th); s = np.sin(th)
                gx = float(ga[2 * j]); gy = float(ga[2 * j + 1])
                X[a * N:(a + 1) * N, 2 * j] = gx * c - gy * s
                X[a * N:(a + 1) * N, 2 * j + 1] = gx * s + gy * c
        return X, Gg, r.ravel()

    def build_X_C36(Yflat, freqs):
        Y = Yflat.reshape(2, D)
        r = np.linalg.norm(Y, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Gg = Y / r
        N = 36
        X = np.zeros((74, D))
        k = np.arange(N)
        for a in range(2):
            ga = Gg[a]
            ax = float(ga[12])
            base = a * N
            X[base:base + N, 12] = ax
            for j, m in enumerate(freqs):
                th = 2.0 * np.pi * k * float(m) / float(N)
                c = np.cos(th); s = np.sin(th)
                gx = float(ga[2 * j]); gy = float(ga[2 * j + 1])
                X[base:base + N, 2 * j] = gx * c - gy * s
                X[base:base + N, 2 * j + 1] = gx * s + gy * c
        X[72, 12] = 1.0
        X[73, 12] = -1.0
        return X, Gg, r.ravel()

    def sym_fun_grad(Yflat, freqs, N, tau, is36):
        if is36:
            X, Gg, rn = build_X_C36(Yflat, freqs)
        else:
            X, Gg, rn = build_X_C37(Yflat, freqs)
        M = X.shape[0]
        Gmat = X @ X.T
        # triu
        iu = np.triu_indices(M, 1)
        g = Gmat[iu]
        mmax = float(g.max())
        e = np.exp((g - mmax) / tau)
        S = float(e.sum())
        if not np.isfinite(S) or S <= 0:
            return float(mmax), np.zeros_like(Yflat)
        F = mmax + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        # GradX = W @ X
        Wx = np.zeros_like(X)
        np.add.at(Wx, iu[0], w[:, None] * X[iu[1]])
        np.add.at(Wx, iu[1], w[:, None] * X[iu[0]])
        # GradG 2x13
        GradG = np.zeros((2, D))
        if is36:
            NN = 36
            k = np.arange(NN)
            for a in range(2):
                base = a * NN
                Gb = Wx[base:base + NN]  # NN x13
                # axial
                GradG[a, 12] = float(Gb[:, 12].sum())
                for j, m in enumerate(freqs):
                    th = 2.0 * np.pi * k * float(m) / 36.0
                    c = np.cos(th); s = np.sin(th)
                    dx = Gb[:, 2 * j]; dy = Gb[:, 2 * j + 1]
                    GradG[a, 2 * j] = float((c * dx + s * dy).sum())
                    GradG[a, 2 * j + 1] = float((-s * dx + c * dy).sum())
        else:
            NN = 37
            k = np.arange(NN)
            for a in range(2):
                base = a * NN
                Gb = Wx[base:base + NN]
                GradG[a, 12] = float(Gb[:, 12].sum())
                for j, m in enumerate(freqs):
                    th = 2.0 * np.pi * k * float(m) / 37.0
                    c = np.cos(th); s = np.sin(th)
                    dx = Gb[:, 2 * j]; dy = Gb[:, 2 * j + 1]
                    GradG[a, 2 * j] = float((c * dx + s * dy).sum())
                    GradG[a, 2 * j + 1] = float((-s * dx + c * dy).sum())
        Y = Yflat.reshape(2, D)
        # to Y
        GradY = np.zeros_like(Y)
        for a in range(2):
            ga = Gg[a]; ra = float(rn[a])
            gg = GradG[a]
            dot = float(gg @ ga)
            GradY[a] = (gg - dot * ga) / max(ra, 1e-12)
        return float(F), GradY.ravel()

    def sym_optimize(freqs, is36, Yinit, tau, maxiter):
        N = 36 if is36 else 37
        y0 = Yinit.ravel().copy()

        def fun(yf):
            return sym_fun_grad(yf, freqs, N, tau, is36)
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-6, "maxfun": int(maxiter * 10)})
            yopt = res.x
        except Exception:
            return Yinit.copy(), 1e9
        if is36:
            Xopt, _, _ = build_X_C36(yopt, freqs)
        else:
            Xopt, _, _ = build_X_C37(yopt, freqs)
        v = _max_cos(_normalize(Xopt))
        return yopt, v

    # ---------- symmetric scan ----------
    sym_best_val = 1e9
    sym_best_X = None
    sym_best_info = None
    sym_best_Y = None
    sym_best_freq = None
    sym_best_is36 = False

    trial = 0
    # quick screening
    while time.time() < SYM_DEADLINE - 6.0 and trial < 160:
        trial += 1
        is36 = (trial % 3 == 0)
        if is36:
            freqs = tuple(sorted(rng.choice(np.arange(1, 18), size=6, replace=False).tolist()))
            # avoid complementary pairs summing to 36 and m=18 already excluded
            # if any pair sums to 36, skip (degenerate mirror)
            bad = False
            for a in range(6):
                for b in range(a + 1, 6):
                    if freqs[a] + freqs[b] == 36:
                        bad = True
            if bad:
                continue
        else:
            freqs = tuple(sorted(rng.choice(np.arange(1, 19), size=6, replace=False).tolist()))
        Yinit = rng.standard_normal(26)
        # short anneal: tau 0.01 then 0.004
        y1, v1 = sym_optimize(freqs, is36, Yinit, 0.01, 60)
        if time.time() >= SYM_DEADLINE - 5.0:
            # evaluate v1 only
            if v1 < sym_best_val:
                if is36:
                    Xo, _, _ = build_X_C36(y1, freqs)
                else:
                    Xo, _, _ = build_X_C37(y1, freqs)
                sym_best_val = float(v1)
                sym_best_X = _normalize(Xo)
                sym_best_Y = y1.copy(); sym_best_freq = freqs; sym_best_is36 = is36
            break
        y2, v2 = sym_optimize(freqs, is36, y1, 0.004, 50)
        v = v2 if v2 < v1 else v1
        yb = y2 if v2 < v1 else y1
        if v < sym_best_val:
            if is36:
                Xo, _, _ = build_X_C36(yb, freqs)
            else:
                Xo, _, _ = build_X_C37(yb, freqs)
            sym_best_val = float(v)
            sym_best_X = _normalize(Xo)
            sym_best_Y = yb.copy(); sym_best_freq = freqs; sym_best_is36 = is36
        # early stop if time
        if time.time() >= SYM_DEADLINE - 5.0:
            break

    # refine top symmetric candidate a bit if found and time remains
    if sym_best_X is not None and time.time() < SYM_DEADLINE - 1.0:
        try:
            yR, vR = sym_optimize(sym_best_freq, sym_best_is36, sym_best_Y, 0.0015, 80)
            if vR < sym_best_val:
                if sym_best_is36:
                    Xo, _, _ = build_X_C36(yR, sym_best_freq)
                else:
                    Xo, _, _ = build_X_C37(yR, sym_best_freq)
                sym_best_val = float(vR)
                sym_best_X = _normalize(Xo)
                sym_best_Y = yR
            # extra random restarts on same freq (2 tries)
            for _ in range(2):
                if time.time() >= SYM_DEADLINE:
                    break
                Yr = rng.standard_normal(26)
                ya, va = sym_optimize(sym_best_freq, sym_best_is36, Yr, 0.008, 60)
                if time.time() >= SYM_DEADLINE:
                    break
                yb2, vb2 = sym_optimize(sym_best_freq, sym_best_is36, ya, 0.0015, 60)
                if vb2 < sym_best_val:
                    if sym_best_is36:
                        Xo, _, _ = build_X_C36(yb2, sym_best_freq)
                    else:
                        Xo, _, _ = build_X_C37(yb2, sym_best_freq)
                    sym_best_val = float(vb2)
                    sym_best_X = _normalize(Xo)
                    sym_best_Y = yb2
        except Exception:
            pass

    # ---------- full polish helpers ----------
    def full_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    # release symmetry: polish symmetric best
    if sym_best_X is not None:
        # only proceed if vaguely reasonable (<0.5) else skip to save time
        if sym_best_val < 0.6:
            Xp = sym_best_X.copy()
            for tau, it in [(0.008, 60), (0.002, 70), (5e-4, 80)]:
                if time.time() - t0 > LIM - 22:
                    break
                Xp = full_polish(Xp, tau, it)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = float(vp)
                best = Xp.copy()
            # keep Xp as walker even if not best? use for further squeeze if close
            if vp < best_val + 0.01:
                # extra fine polish on Xp if it became competitive
                if time.time() - t0 < LIM - 20:
                    Xq = full_polish(Xp, 2e-4, 90)
                    vq = _max_cos(Xq)
                    if vq < best_val:
                        best_val = float(vq); best = Xq.copy()

    # always polish baseline best to ensure we at least match lab best
    if time.time() - t0 < LIM - 12:
        Xb = best.copy()
        # light anneal
        Xb = full_polish(Xb, 5e-4, 70)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val = float(vb); best = Xb.copy()
        if time.time() - t0 < LIM - 6:
            Xc = full_polish(best, 1.5e-4, 90)
            vc = _max_cos(Xc)
            if vc < best_val:
                best_val = float(vc); best = Xc.copy()

    best = _normalize(best)
    return best
