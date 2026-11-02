import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _degrees(X, thr=1e-9):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    m = float(c.max())
    sel = c > m - thr
    deg = np.zeros(n, dtype=int)
    I = iu0[sel]; J = iu1[sel]
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    return m, deg, G

def _rand_unit(rng, n):
    A = rng.randn(n, 4)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return A

def _rand_rot(rng):
    A = rng.randn(4, 4)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _anneal(X0, steps, T0, T1, lr, mu, rng, noise0=0.0, deadline=None):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = X.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lrk = lr * (1.0 - 0.55 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ X) / S
        Grad -= (Grad * X).sum(axis=1, keepdims=True) * X
        vel = mu * vel + (1.0 - mu) * Grad
        X = X - lrk * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * X).sum(axis=1, keepdims=True) * X
                X = X + nz
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = X.copy()
    return bx, bv

def _lbfgs(X_init, taus, iters, deadline=None):
    X = X_init / np.maximum(np.linalg.norm(X_init, axis=1, keepdims=True), 1e-12)
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    logNp = np.log(Np)
    y = X.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if deadline is not None and time.time() > deadline - 0.5:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            Xn = Y / rn
            g = (Xn @ Xn.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * (np.log(s) - logNp)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ Xn
            GY = (GX - (GX * Xn).sum(axis=1, keepdims=True) * Xn) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4)})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf, _max_cos(Yf)

def _greedy_max(X, steps, s0, deadline=None):
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    bv = _max_cos(X)
    bx = X.copy()
    n = X.shape[0]
    step = s0
    for k in range(steps):
        if deadline is not None and (k & 31) == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # all near-max pairs
        iu0, iu1 = np.triu_indices(n, 1)
        c = G[iu0, iu1]
        sel = c > m - 2e-4
        if int(sel.sum()) == 0:
            sel[np.argmax(c)] = True
        I = iu0[sel]; J = iu1[sel]
        # subgradient of mean of selected dots
        Grad = np.zeros_like(X)
        for a, b in zip(I, J):
            d = float(X[a] @ X[b])
            Grad[a] += X[b] - d * X[a]
            Grad[b] += X[a] - d * X[b]
        Grad /= max(1, len(I))
        # normalize tangential part
        gn = np.linalg.norm(Grad, axis=1, keepdims=True)
        # propose
        Y = X - step * Grad
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        v = _max_cos(Y)
        if v < bv:
            bv = v; bx = Y.copy(); X = Y
            step *= 1.08
        else:
            step *= 0.6
            # keep X but reduce step; occasionally accept sideways
            if step < 1e-12:
                break
    return bx, bv

def _holes(X, rng, m=2500, exclude=None):
    C = _rand_unit(rng, m)
    S = C @ X.T
    if exclude is not None and len(exclude) > 0:
        mask = np.ones(X.shape[0], dtype=bool)
        mask[np.asarray(exclude)] = False
        if int(mask.sum()) > 0:
            S = C @ X[mask].T
    best = S.max(axis=1)
    order = np.argsort(best)
    return C[order]

def solve():
    import time as _t
    import numpy as _np
    t0 = _t.time()
    deadline = t0 + 168.0
    rng = _np.random.default_rng(12345)
    n = 163
    # load best
    xb = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = _np.load(fn)
            A = _np.asarray(A, dtype=float)
            if A.shape == (163, 4):
                A = A / _np.maximum(_np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
                v = _max_cos(A)
                if v < bv:
                    bv = v; xb = A.copy()
        except Exception:
            continue
    if xb is None:
        xb = _rand_unit(rng, n)
        bv = _max_cos(xb)
    # quick initial smooth polish of loaded best (cheap)
    try:
        if _t.time() < deadline - 20:
            Xp, vp = _lbfgs(xb, taus=(0.0015, 0.0005), iters=(50, 60), deadline=deadline - 8)
            if vp < bv:
                xb, bv = Xp, vp
    except Exception:
        pass
    # basin hopping with defect injection
    it = 0
    # precompute for speed
    while _t.time() < deadline - 9:
        it += 1
        remaining = deadline - 9 - _t.time()
        if remaining <= 0:
            break
        mode = it % 7
        Xt = xb.copy()
        try:
            _, deg, _ = _degrees(xb, thr=1e-9)
        except Exception:
            deg = _np.zeros(n, dtype=int)
        order_hi = _np.argsort(-deg)
        if mode == 0:
            # global small shake
            sig = 0.03 + 0.05 * rng.random()
            nz = rng.randn(n, 4) * sig
            nz -= (nz * Xt).sum(axis=1, keepdims=True) * Xt
            Xt = Xt + nz
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            steps, T0, T1, lr, mu, nz0 = 450, 0.0035, 0.00025, 0.18, 0.86, 0.004
        elif mode == 1:
            # subset strong shake biased to hubs
            k = int(5 + rng.integers(0, 9))
            if rng.random() < 0.75:
                pool = order_hi[:max(20, 2 * k)]
                idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            else:
                idx = rng.choice(n, size=k, replace=False)
            sig = 0.35 + 0.25 * rng.random()
            nz = _np.zeros_like(Xt)
            nz[idx] = rng.randn(len(idx), 4) * sig
            nz -= (nz * Xt).sum(axis=1, keepdims=True) * Xt
            Xt = Xt + nz
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            steps, T0, T1, lr, mu, nz0 = 550, 0.004, 0.0002, 0.20, 0.86, 0.002
        elif mode == 2:
            # hub-to-hole relocation (defect)
            nh = int(1 + rng.integers(0, 3))  # 1-3
            hubs = order_hi[:max(nh * 4, 10)]
            idx = rng.choice(hubs, size=nh, replace=False)
            H = _holes(xb, rng, m=3000, exclude=idx)
            # pick deepest distinct holes
            chosen = []
            used = []
            for h in H:
                if len(chosen) >= nh:
                    break
                ok = True
                for u in used:
                    if float(h @ u) > 0.85:
                        ok = False; break
                if ok:
                    chosen.append(h); used.append(h)
            for a, h in zip(idx, chosen):
                Xt[a] = h + rng.randn(4) * 0.01
                Xt[a] /= max(_np.linalg.norm(Xt[a]), 1e-12)
            # also jitter neighbours of hubs slightly
            steps, T0, T1, lr, mu, nz0 = 650, 0.0045, 0.00018, 0.20, 0.87, 0.003
        elif mode == 3:
            # rigid cap rotation
            u = rng.randn(4); u /= max(_np.linalg.norm(u), 1e-12)
            proj = Xt @ u
            thr = _np.quantile(proj, 0.80 + 0.15 * rng.random())
            cap = _np.where(proj >= thr)[0]
            if len(cap) < 3:
                cap = rng.choice(n, size=8, replace=False)
            R = _rand_rot(rng)
            # small rotation: interpolate
            ang = 0.05 + 0.20 * rng.random()
            # build small rotation via QR of I+ang*skew
            S = rng.randn(4, 4) * ang
            S = S - S.T
            try:
                Rs = _np.linalg.qr(_np.eye(4) + S)[0]
                if _np.linalg.det(Rs) < 0:
                    Rs[:, 0] = -Rs[:, 0]
            except Exception:
                Rs = R
            Xt[cap] = Xt[cap] @ Rs.T
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            steps, T0, T1, lr, mu, nz0 = 500, 0.003, 0.0002, 0.18, 0.86, 0.002
        elif mode == 4:
            # fresh uniform restart (rare exploration)
            if it % 21 == 0:
                Xt = _rand_unit(rng, n)
                steps, T0, T1, lr, mu, nz0 = 800, 0.006, 0.00025, 0.22, 0.87, 0.005
            else:
                # low-degree rescue: move light points toward holes
                order_lo = _np.argsort(deg)
                k = int(3 + rng.integers(0, 5))
                idx = order_lo[:k]
                H = _holes(xb, rng, m=2000, exclude=[])
                for ii, a in enumerate(idx):
                    Xt[a] = H[ii % len(H)] + rng.randn(4) * 0.02
                    Xt[a] /= max(_np.linalg.norm(Xt[a]), 1e-12)
                steps, T0, T1, lr, mu, nz0 = 550, 0.004, 0.0002, 0.19, 0.86, 0.002
        elif mode == 5:
            # antipodal-ish pair split: pick closest pair, split midpoint, free one
            G = xb @ xb.T
            _np.fill_diagonal(G, -2.0)
            flat = _np.argmax(G)
            a = flat // n; b = flat % n
            mid = xb[a] + xb[b]; mid /= max(_np.linalg.norm(mid), 1e-12)
            Xt[a] = mid + rng.randn(4) * 0.02
            Xt[a] /= max(_np.linalg.norm(Xt[a]), 1e-12)
            H = _holes(xb, rng, m=2000, exclude=[b])
            Xt[b] = H[0] + rng.randn(4) * 0.01
            Xt[b] /= max(_np.linalg.norm(Xt[b]), 1e-12)
            steps, T0, T1, lr, mu, nz0 = 600, 0.004, 0.0002, 0.20, 0.86, 0.002
        else:
            # medium global jitter
            sig = 0.10 + 0.08 * rng.random()
            nz = rng.randn(n, 4) * sig
            nz -= (nz * Xt).sum(axis=1, keepdims=True) * Xt
            Xt = Xt + nz
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            steps, T0, T1, lr, mu, nz0 = 450, 0.003, 0.00022, 0.18, 0.85, 0.003
        # anneal
        try:
            Xa, va = _anneal(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=mu, rng=rng, noise0=nz0, deadline=deadline - 6)
        except Exception:
            continue
        # short lbfgs
        try:
            Xb2, vb2 = _lbfgs(Xa, taus=(0.0012, 0.00045), iters=(45, 55), deadline=deadline - 5)
        except Exception:
            Xb2, vb2 = Xa, va
        # accept if better (strict) ; also occasional sideways to diversify base?
        if vb2 < bv - 1e-12:
            xb, bv = Xb2, vb2
        elif rng.random() < 0.02 and vb2 < bv + 5e-5 and _t.time() < deadline - 60:
            # rare sideways move to escape same basin early on
            xb, bv = Xb2, vb2
        # early exit if big win? continue to squeeze
    # final intensive polish of best
    try:
        Xf, vf = _lbfgs(xb, taus=(0.0015, 0.0006, 0.00022), iters=(90, 90, 110), deadline=deadline - 2)
        if vf < bv:
            xb, bv = Xf, vf
    except Exception:
        pass
    try:
        Xg, vg = _greedy_max(xb, steps=250, s0=0.02, deadline=deadline - 1)
        if vg < bv:
            xb, bv = Xg, vg
    except Exception:
        pass
    xb = xb / _np.maximum(_np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    return xb
