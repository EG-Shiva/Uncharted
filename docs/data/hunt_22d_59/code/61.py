import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse_short(X0, steps, beta0, beta1, lr, t_end):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1
    pow2 = b2
    for t in range(1, steps + 1):
        if (t & 31) == 0 and time.time() > t_end:
            break
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1)
        vhat = v / (1.0 - pow2)
        pow1 *= b1
        pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if (t % 25) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur
                bestX = X.copy()
    return bestX, best

def _tangent_bases(X):
    n, d = X.shape
    dm1 = d - 1
    T = np.empty((n, d, dm1))
    I = np.eye(d)
    for i in range(n):
        x = X[i]
        e = np.zeros(d)
        e[0] = 1.0
        if abs(float(x[0])) > 0.9:
            e[0] = 0.0
            e[1] = 1.0
        u = e - x
        nu = float(u @ u)
        if nu < 1e-12:
            T[i] = I[:, 1:]
        else:
            Q = I - (2.0 / nu) * np.outer(u, u)
            T[i] = Q[:, 1:]
    return T

def _build_Jtan(X, T, thr, pcap=1200):
    n, d = X.shape
    dm1 = d - 1
    Ntan = n * dm1
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    I_, J_ = np.where(G > m - thr)
    mask = I_ < J_
    I_ = I_[mask]; J_ = J_[mask]
    c = G[I_, J_] if len(I_) else np.array([])
    if len(I_) > pcap:
        # keep tightest
        idx = np.argsort(-c)[:pcap]
        I_ = I_[idx]; J_ = J_[idx]; c = c[idx]
    p = len(I_)
    if p == 0:
        return None, None, m, p
    Jm = np.zeros((p, Ntan), dtype=np.float64)
    for k in range(p):
        i = int(I_[k]); j = int(J_[k])
        a = T[i].T @ X[j]
        b = T[j].T @ X[i]
        Jm[k, i*dm1:(i+1)*dm1] = a
        Jm[k, j*dm1:(j+1)*dm1] = b
    return Jm, (I_, J_, c), m, p

def _modes_from_J(Jm, T, rng, Ktail=25, Nrand=10, Ncombo=5):
    p, Ntan = Jm.shape
    n = T.shape[0]
    dm1 = T.shape[2]
    try:
        _, S, Vh = np.linalg.svd(Jm, full_matrices=False)
    except Exception:
        return []
    modes = []
    nt = min(Ktail, p)
    for k in range(nt):
        modes.append(Vh[p-1-k].copy())
    # random exact-null via projection out of row space
    for _ in range(Nrand):
        r = rng.standard_normal(Ntan)
        # project: r - Vh^T (Vh r)
        coef = Vh @ r
        rnull = r - Vh.T @ coef
        nr = float(np.linalg.norm(rnull))
        if nr < 1e-9:
            r = rng.standard_normal(Ntan)
            coef = Vh @ r
            rnull = r - Vh.T @ coef
            nr = float(np.linalg.norm(rnull))
            if nr < 1e-9:
                continue
        rnull /= nr
        modes.append(rnull)
    # random combos of tail + nulls
    if len(modes) >= 4:
        base = np.stack(modes[:min(len(modes), 30)], axis=0)
        for _ in range(Ncombo):
            w = rng.standard_normal(base.shape[0])
            cmb = w @ base
            nr = float(np.linalg.norm(cmb))
            if nr < 1e-12:
                continue
            cmb /= nr
            # if p large, re-project small row component? keep as is (near-null)
            modes.append(cmb)
    return modes

def _tan_to_ambient(v, T):
    n = T.shape[0]
    d = T.shape[1]
    dm1 = T.shape[2]
    U = np.empty((n, d))
    for i in range(n):
        coeff = v[i*dm1:(i+1)*dm1]
        U[i] = T[i] @ coeff
    return U

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end_global = t0 + BUDGET
    t_dead = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            if np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    gbestX = cands[bi].copy()
    gbest = float(vals[bi])
    fallbackX = gbestX.copy()
    fallbackV = gbest

    eps_list = [0.01, 0.03, 0.1, 0.2]
    thresholds = [1e-3, 1e-4, 1e-5]

    # quick short polish of loaded best to have fresh baseline (cheap)
    try:
        if time.time() < t_dead - 5:
            Xp, vp = _adam_lse_short(gbestX, 400, 100.0, 600.0, 0.008, min(t_dead, t_end_global-1))
            if vp < gbest:
                gbest = vp; gbestX = Xp.copy()
    except Exception:
        pass

    outer = 0
    while time.time() < t_dead - 3.0 and outer < 12:
        improved_outer = False
        for thr in thresholds:
            if time.time() > t_dead - 3.0:
                break
            # rebuild bases/J for current best
            try:
                T = _tangent_bases(gbestX)
                Jm, _, mcur, p = _build_Jtan(gbestX, T, thr)
            except Exception:
                continue
            if Jm is None or p < 5:
                continue
            try:
                modes = _modes_from_J(Jm, T, rng, Ktail=25, Nrand=10, Ncombo=5)
            except Exception:
                continue
            if not modes:
                continue
            # screen signed displacements
            scored = []
            for mi, vtan in enumerate(modes):
                try:
                    U = _tan_to_ambient(vtan, T)
                except Exception:
                    continue
                for eps in eps_list:
                    for sgn in (1.0, -1.0):
                        Xd = _normalize(gbestX + sgn * eps * U)
                        val = _max_cos(Xd)
                        scored.append((val, mi, eps, sgn))
            if not scored:
                continue
            scored.sort(key=lambda t: t[0])
            # keep top diverse: best 12, at most 3 per mode
            kept = []
            cnt = {}
            for val, mi, eps, sgn in scored:
                c = cnt.get(mi, 0)
                if c >= 2 and len(kept) >= 8:
                    continue
                kept.append((val, mi, eps, sgn))
                cnt[mi] = c + 1
                if len(kept) >= 12:
                    break
            # polish kept
            for val, mi, eps, sgn in kept:
                if time.time() > t_dead - 2.0:
                    break
                vtan = modes[mi]
                U = _tan_to_ambient(vtan, T)
                Xd = _normalize(gbestX + sgn * eps * U)
                rem = min(t_dead, t_end_global-1.0) - time.time()
                if rem < 1.0:
                    break
                steps = 900 if rem > 30 else (600 if rem > 12 else 400)
                # prune hopeless: if displaced val > gbest+0.02 skip large polish, use tiny
                if val > gbest + 0.02:
                    continue
                try:
                    Xp, vp = _adam_lse_short(Xd, steps, 100.0, 600.0, 0.01, min(t_dead, t_end_global-1.0))
                except Exception:
                    continue
                if vp < gbest:
                    gbest = vp
                    gbestX = Xp.copy()
                    improved_outer = True
                    # refresh T for remaining kept? keep old T, fine
            # also try one fresh random combo polish from best mode pair
            if time.time() > t_dead - 2.0:
                break
        outer += 1
        if not improved_outer and outer >= 3:
            # inject one isotropic shake + polish to diversify, cheap
            if time.time() < t_dead - 8.0:
                try:
                    Xs = _normalize(gbestX + 0.08 * rng.standard_normal((n, d)))
                    Xp, vp = _adam_lse_short(Xs, 700, 100.0, 600.0, 0.012, min(t_dead, t_end_global-1.0))
                    if vp < gbest:
                        gbest = vp; gbestX = Xp.copy()
                except Exception:
                    pass

    # final fine polish
    try:
        if time.time() < t_end_global - 1.0:
            Xf, vf = _adam_lse_short(gbestX, 800, 150.0, 600.0, 0.006, t_end_global-0.2)
            if vf < gbest:
                gbest = vf; gbestX = Xf.copy()
    except Exception:
        pass

    if not np.all(np.isfinite(gbestX)) or gbest >= fallbackV + 1e-12:
        # ensure we never regress vs fallback due to NaN
        try:
            fv = _max_cos(fallbackX)
            gv = _max_cos(gbestX)
            if fv < gv:
                gbestX = fallbackX.copy()
        except Exception:
            gbestX = fallbackX.copy()
    gbestX = _normalize(gbestX)
    return gbestX
