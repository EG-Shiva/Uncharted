import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iu_c > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(iu_c[idx])[-3000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        c = iu_c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

# ---- symmetric orbit machinery ----
def _expand_and_grad_info(M, a, b):
    th = 2.0 * np.pi * np.arange(M) / M
    ca = np.cos(a * th); sa = np.sin(a * th)
    cb = np.cos(b * th); sb = np.sin(b * th)
    return ca, sa, cb, sb

def _apply_Q(R, ca, sa, cb, sb, ks):
    # R: (K,5) -> (K,len(ks),5)
    K = R.shape[0]
    L = len(ks)
    out = np.empty((K, L, 5))
    for li, k in enumerate(ks):
        c1 = ca[k]; s1 = sa[k]; c2 = cb[k]; s2 = sb[k]
        out[:, li, 0] = c1 * R[:, 0] - s1 * R[:, 1]
        out[:, li, 1] = s1 * R[:, 0] + c1 * R[:, 1]
        out[:, li, 2] = c2 * R[:, 2] - s2 * R[:, 3]
        out[:, li, 3] = s2 * R[:, 2] + c2 * R[:, 3]
        out[:, li, 4] = R[:, 4]
    return out

def _apply_QT_back(G, ca, sa, cb, sb, ks):
    # G: (K,L,5) grad wrt X -> grad wrt R (K,5) via Q^T
    K, L, _ = G.shape
    out = np.zeros((K, 5))
    for li, k in enumerate(ks):
        c1 = ca[k]; s1 = sa[k]; c2 = cb[k]; s2 = sb[k]
        g0 = G[:, li, 0]; g1 = G[:, li, 1]
        g2 = G[:, li, 2]; g3 = G[:, li, 3]
        g4 = G[:, li, 4]
        out[:, 0] += c1 * g0 + s1 * g1
        out[:, 1] += -s1 * g0 + c1 * g1
        out[:, 2] += c2 * g2 + s2 * g3
        out[:, 3] += -s2 * g2 + c2 * g3
        out[:, 4] += g4
    return out

def optimize_symmetric(M, a, b, Kf, Ka, Kb, n_poles, rng, time_left):
    # orbit sizes
    import math
    # full size M (assume coprime-ish; compute actual generic size by sim is M)
    size_full = M
    # A: r1=0 -> period M/gcd(b,M)
    size_A = M // math.gcd(max(1, b), M) if Ka > 0 else 0
    size_B = M // math.gcd(max(1, a), M) if Kb > 0 else 0
    ca, sa, cb, sb = _expand_and_grad_info(M, a, b)
    ks_full = list(range(size_full))
    ks_A = list(range(size_A)) if Ka > 0 else []
    ks_B = list(range(size_B)) if Kb > 0 else []
    # poles
    poles = []
    if n_poles >= 1:
        poles.append([0, 0, 0, 0, 1.0])
    if n_poles >= 2:
        poles.append([0, 0, 0, 0, -1.0])
    poles = np.array(poles, dtype=float) if len(poles) else np.zeros((0, 5))
    N = Kf * size_full + Ka * size_A + Kb * size_B + len(poles)
    # N must be 173 for valid cfgs; if not, return None
    if N != 173:
        return None, 1e9
    iu = np.triu_indices(N, 1)
    # variable dims
    dim = Kf * 5 + Ka * 3 + Kb * 3
    # closures
    def expand_from_Y(yflat):
        Rf = np.zeros((max(1, Kf), 5)) if Kf == 0 else None
        # parse
        pos = 0
        if Kf > 0:
            Yf = yflat[pos:pos + Kf * 5].reshape(Kf, 5); pos += Kf * 5
            nf = np.linalg.norm(Yf, axis=1, keepdims=True); nf[nf == 0] = 1
            Rf = Yf / nf
        else:
            Rf = np.zeros((0, 5))
        if Ka > 0:
            Ya = yflat[pos:pos + Ka * 3].reshape(Ka, 3); pos += Ka * 3
            na = np.linalg.norm(Ya, axis=1, keepdims=True); na[na == 0] = 1
            Ra3 = Ya / na
            Ra = np.zeros((Ka, 5)); Ra[:, 2:5] = Ra3
        else:
            Ra = np.zeros((0, 5)); Ra3 = None
        if Kb > 0:
            Yb = yflat[pos:pos + Kb * 3].reshape(Kb, 3); pos += Kb * 3
            nb = np.linalg.norm(Yb, axis=1, keepdims=True); nb[nb == 0] = 1
            Rb3 = Yb / nb
            Rb = np.zeros((Kb, 5)); Rb[:, 0:2] = Rb3[:, 0:2]; Rb[:, 4] = Rb3[:, 2]
        else:
            Rb = np.zeros((0, 5))
        # build X
        blocks = []
        if Kf > 0:
            Xf = _apply_Q(Rf, ca, sa, cb, sb, ks_full)  # Kf,M,5
            blocks.append(Xf.reshape(-1, 5))
        if Ka > 0:
            Xa = _apply_Q(Ra, ca, sa, cb, sb, ks_A)
            blocks.append(Xa.reshape(-1, 5))
        if Kb > 0:
            Xb = _apply_Q(Rb, ca, sa, cb, sb, ks_B)
            blocks.append(Xb.reshape(-1, 5))
        if len(poles):
            blocks.append(poles)
        X = np.concatenate(blocks, axis=0) if len(blocks) else np.zeros((0, 5))
        return X, Rf, Ra, Rb

    def fun_grad(yflat, tau):
        X, Rf, Ra, Rb = expand_from_Y(yflat)
        G = X @ X.T
        gv = G[iu]
        m0 = float(gv.max())
        e = np.exp((gv - m0) / tau)
        S = float(e.sum())
        f = m0 + tau * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        W = np.zeros((N, N)); W[iu] = w; W = W + W.T
        gradX = W @ X  # N,5
        # split gradX into blocks
        pos = 0
        grads = []
        if Kf > 0:
            gXf = gradX[pos:pos + Kf * size_full].reshape(Kf, size_full, 5); pos += Kf * size_full
            gRf = _apply_QT_back(gXf, ca, sa, cb, sb, ks_full)
            # project to tangent wrt Yf
            Yf = yflat[0:Kf * 5].reshape(Kf, 5)
            nf = np.linalg.norm(Yf, axis=1, keepdims=True); nf[nf == 0] = 1
            Rn = Yf / nf
            gYf = (gRf - (gRf * Rn).sum(1, keepdims=True) * Rn) / nf
            grads.append(gYf.ravel())
        if Ka > 0:
            gXa = gradX[pos:pos + Ka * size_A].reshape(Ka, size_A, 5); pos += Ka * size_A
            gRa_full = _apply_QT_back(gXa, ca, sa, cb, sb, ks_A)  # Ka,5 first two ~0
            gRa3 = gRa_full[:, 2:5]
            off = Kf * 5
            Ya = yflat[off:off + Ka * 3].reshape(Ka, 3)
            na = np.linalg.norm(Ya, axis=1, keepdims=True); na[na == 0] = 1
            Rn3 = Ya / na
            gYa = (gRa3 - (gRa3 * Rn3).sum(1, keepdims=True) * Rn3) / na
            grads.append(gYa.ravel())
        if Kb > 0:
            gXb = gradX[pos:pos + Kb * size_B].reshape(Kb, size_B, 5); pos += Kb * size_B
            gRb_full = _apply_QT_back(gXb, ca, sa, cb, sb, ks_B)
            # map: vars are (x1,x2,x5)
            gRb3 = np.stack([gRb_full[:, 0], gRb_full[:, 1], gRb_full[:, 4]], axis=1)
            off = Kf * 5 + Ka * 3
            Yb = yflat[off:off + Kb * 3].reshape(Kb, 3)
            nb = np.linalg.norm(Yb, axis=1, keepdims=True); nb[nb == 0] = 1
            Rn3 = Yb / nb
            gYb = (gRb3 - (gRb3 * Rn3).sum(1, keepdims=True) * Rn3) / nb
            grads.append(gYb.ravel())
        gflat = np.concatenate(grads) if len(grads) else np.zeros(0)
        return f, gflat, X

    best_X = None; best_v = 1e9
    n_restart = 2
    # reduce restarts if big dim / little time
    if dim > 80:
        n_restart = 1
    for rs in range(n_restart):
        y0 = rng.normal(size=(dim,))
        # stage taus
        for tau, maxiter in [(0.03, 120), (0.008, 120)]:
            def fg(y):
                f, g, _ = fun_grad(y, tau)
                return f, g
            try:
                res = minimize(lambda y: fg(y)[0], y0, jac=lambda y: fg(y)[1],
                               method='L-BFGS-B', options={'maxiter': maxiter, 'maxfun': maxiter * 5})
                y0 = res.x
            except Exception:
                break
            if time_left() < 25:
                break
        _, _, Xf = fun_grad(y0, 0.005)
        v = _max_cos(Xf / np.linalg.norm(Xf, axis=1, keepdims=True))
        if v < best_v:
            best_v = v; best_X = Xf / np.linalg.norm(Xf, axis=1, keepdims=True)
        if time_left() < 25:
            break
    return best_X, best_v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    def time_left():
        return deadline - time.time()
    rng = np.random.default_rng(12345)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (173, 5):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v; best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(173, 5))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = 173, 5
    # symmetric configs: (M,a,b,Kf,Ka,Kb,poles)
    cfgs = [
        (43, 1, 2, 4, 0, 0, 1),
        (43, 1, 3, 4, 0, 0, 1),
        (19, 1, 2, 9, 0, 0, 2),
        (19, 1, 3, 9, 0, 0, 2),
        (57, 1, 2, 3, 0, 0, 2),
        (12, 1, 3, 13, 4, 0, 1),
        (9, 1, 2, 19, 0, 0, 2),
    ]
    sym_best = None; sym_val = 1e9
    for (M, a, b, Kf, Ka, Kb, npol) in cfgs:
        if time_left() < 55:
            break
        try:
            X, v = optimize_symmetric(M, a, b, Kf, Ka, Kb, npol, rng, time_left)
        except Exception:
            continue
        if X is None:
            continue
        # quick symmetry-reduced polish: small full LP polish (keeps near symmetric)
        if time_left() > 20:
            try:
                Xp = _lp_polish(X, min(2.0, time_left() - 20))
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = X, v
        else:
            Xp, vp = X, v
        if vp < sym_val:
            sym_val = vp; sym_best = Xp.copy()
        # release symmetry: perturb slightly and polish to break symmetry
        if time_left() > 30:
            try:
                Y = Xp + 0.004 * rng.normal(size=Xp.shape)
                Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
                Yp = _lp_polish(Y, min(3.0, time_left() - 18))
                vyp = _max_cos(Yp)
                if vyp < sym_val:
                    sym_val = vyp; sym_best = Yp.copy()
                if vyp < best_val:
                    best_val = vyp; best = Yp.copy()
            except Exception:
                pass
        if vp < best_val:
            best_val = vp; best = Xp.copy()
    # basin-hopping from the symmetric basin (different topology) + from record
    start_pts = []
    if sym_best is not None:
        start_pts.append(sym_best)
    start_pts.append(best.copy())
    # hopping loop for remaining time
    trial = 0
    cur = sym_best if (sym_best is not None and sym_val < best_val + 0.02) else best
    cur_val = _max_cos(cur)
    while True:
        rem = time_left()
        if rem < 8.0:
            break
        pt = 1.2
        if rem < 22:
            pt = max(0.5, rem - 6.5)
        pt = min(pt, rem - 6.0)
        if pt < 0.4:
            break
        y = cur.copy()
        mode = trial % 5
        if mode == 0:
            idx = rng.choice(n, 10, replace=False)
            y[idx] += 0.06 * rng.normal(size=(len(idx), d))
        elif mode == 1:
            y += 0.012 * rng.normal(size=(n, d))
        elif mode == 2:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G >= m - 1e-9).sum(axis=1)
            victim = int(np.argmax(cnt))
            y[victim] = rng.normal(size=(d,))
            neigh = np.where(G[victim] > m - 0.02)[0]
            if len(neigh):
                y[neigh] += 0.05 * rng.normal(size=(len(neigh), d))
        elif mode == 3:
            idx = rng.choice(n, 25, replace=False)
            y[idx] += 0.035 * rng.normal(size=(len(idx), d))
        else:
            # big kick to jump basin from symmetric start
            idx = rng.choice(n, 6, replace=False)
            y[idx] += 0.12 * rng.normal(size=(len(idx), d))
        yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn == 0] = 1
        y = y / yn
        try:
            y = _lp_polish(y, pt)
        except Exception:
            trial += 1
            continue
        v = _max_cos(y)
        if v < cur_val:
            cur, cur_val = y, v
        if v < best_val:
            best, best_val = y, v
        trial += 1
    rem = time_left()
    if rem > 1.5:
        try:
            best = _lp_polish(best, rem - 0.7)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
