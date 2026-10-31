import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _sym_adam(Y0, steps=1500, beta0=30.0, beta1=300.0, lr0=0.01, lr1=0.001):
    Y = _norm(Y0.copy())
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = 1e9
    best_Y = Y.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 * ((lr1 / lr0) ** frac)
        X = np.vstack([Y, -Y])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        # track best by true max
        if M < best_val:
            best_val = M
            best_Y = Y.copy()
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        GradX = (W @ X) / S
        Gt = GradX[:Y.shape[0]] - GradX[Y.shape[0]:]
        # Riemannian projection
        Gt = Gt - (np.sum(Gt * Y, axis=1, keepdims=True)) * Y
        t = s + 1
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        sd = mhat / (np.sqrt(vhat) + eps)
        sd = sd - (np.sum(sd * Y, axis=1, keepdims=True)) * Y
        Y = Y - lr * sd
        Y = _norm(Y)
    Xb = np.vstack([best_Y, -best_Y])
    return best_Y, _maxcos(Xb), best_val

def _smooth(X, steps, beta0, beta1, s0, s1, mu=0.7):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        step = s0 + (s1 - s0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ X
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        nrm = np.sqrt(np.sum(gt ** 2, axis=1, keepdims=True)) + 1e-15
        direc = gt / nrm
        vel = mu * vel + (1.0 - mu) * direc
        vn = np.sqrt(np.sum(vel ** 2, axis=1, keepdims=True)) + 1e-15
        vdir = vel / vn
        X = X - step * vdir
        X = _norm(X)
    return X

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    best_x = X.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thresh = m - max(20 * radius, 1e-9)
        idx = np.where(c_all > thresh)[0]
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
        sel = np.zeros_like(c_all, dtype=bool)
        sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def _build_group():
    # 6-cycles shifts + even sign flips -> 192 transforms
    masks = []
    for k in range(64):
        bits = [(k >> i) & 1 for i in range(6)]
        nneg = sum(1 for b in bits if b == 1)
        # even number of -1 => product +1
        if nneg % 2 == 0:
            sv = np.ones(6)
            for i, b in enumerate(bits):
                if b:
                    sv[i] = -1.0
            masks.append(sv)
    trans = []
    for sh in range(6):
        for sv in masks:
            trans.append((sh, sv))
    return trans  # 192

def _apply_trans(g, sh, sv):
    return np.roll(g, sh) * sv

def _orbit_seed_Y(gen, trans, rng):
    orb = np.array([_apply_trans(gen, sh, sv) for (sh, sv) in trans])
    orb = _norm(orb)
    order = rng.permutation(len(orb))
    sel = []
    for oi in order:
        vv = orb[oi]
        ok = True
        for s in sel:
            if abs(float(vv @ s)) > 0.999:
                ok = False
                break
        if ok:
            sel.append(vv)
            if len(sel) >= 58:
                break
    if len(sel) < 58:
        extra = _norm(rng.standard_normal((58 - len(sel), 6)))
        sel.extend([e for e in extra])
    Y0 = np.array(sel[:58])
    return _norm(Y0)

def _orbit_gens_opt(gens0, trans_groups, steps=800, beta0=30.0, beta1=250.0, lr0=0.01, lr1=0.001):
    # gens0 (k,6), trans_groups: list per generator of list of (sh,sv); total 116
    k = gens0.shape[0]
    G0 = _norm(gens0.copy())
    m = np.zeros_like(G0); v = np.zeros_like(G0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = 1e9
    best_G = G0.copy()
    # precompute inverse info: applying trans then inverse is roll(-sh) * roll(sv?)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 * ((lr1 / lr0) ** frac)
        # expand
        Xs = []
        for gi in range(k):
            g = G0[gi]
            for (sh, sv) in trans_groups[gi]:
                Xs.append(_apply_trans(g, sh, sv))
        X = _norm(np.array(Xs))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        if M < best_val:
            best_val = M
            best_G = G0.copy()
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        GradX = (W @ X) / S
        # pullback to generators
        Ggrad = np.zeros_like(G0)
        ptr = 0
        for gi in range(k):
            acc = np.zeros(6)
            for (sh, sv) in trans_groups[gi]:
                gx = GradX[ptr]
                # inverse: unapply sign then unroll: gx*sv then roll(-sh)
                # since (roll(g)*sv), dL/dg = roll^{-1}(dL/dx * sv)
                tmp = gx * sv
                tmp = np.roll(tmp, -sh)
                acc += tmp
                ptr += 1
            Ggrad[gi] = acc
        Ggrad = Ggrad - (np.sum(Ggrad * G0, axis=1, keepdims=True)) * G0
        t = s + 1
        m = b1 * m + (1 - b1) * Ggrad
        v = b2 * v + (1 - b2) * (Ggrad * Ggrad)
        mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
        sd = mhat / (np.sqrt(vhat) + eps)
        sd = sd - (np.sum(sd * G0, axis=1, keepdims=True)) * G0
        G0 = _norm(G0 - lr * sd)
    # rebuild best X
    Xs = []
    for gi in range(k):
        g = best_G[gi]
        for (sh, sv) in trans_groups[gi]:
            Xs.append(_apply_trans(g, sh, sv))
    Xb = _norm(np.array(Xs))
    return best_G, Xb, _maxcos(Xb)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        return _norm(a)
    try:
        xb = clean(np.load("best.npy"))
    except Exception:
        xb = None
    try:
        xr = clean(np.load("record.npy"))
    except Exception:
        xr = None
    cands0 = []
    if xb is not None:
        cands0.append(xb)
    if xr is not None:
        cands0.append(xr)
    if not cands0:
        cands0.append(_norm(np.random.default_rng(0).standard_normal((116, 6))))
    vals0 = [_maxcos(c) for c in cands0]
    incumbent = cands0[int(np.argmin(vals0))].copy()
    inc_val = float(min(vals0))
    # ---- symmetric search, cap 120s ----
    sym_deadline = t0 + 120.0
    sym_best_val = 1e9
    sym_best_Y = None
    try:
        trans = _build_group()
        rng = np.random.default_rng(20261031)
        Yseeds = []
        # random seeds
        Yseeds.append(_norm(rng.standard_normal((58, 6))))
        Yseeds.append(_norm(np.random.default_rng(7).standard_normal((58, 6))))
        # signed-permutation / cyclic orbit seeds from generic generators
        for gi in range(3):
            gen = rng.standard_normal(6)
            gen = gen / (np.linalg.norm(gen) + 1e-15)
            Yseeds.append(_orbit_seed_Y(gen, trans, rng))
        # extra orbit seed with repeated-magnitude generator (smaller orbit structure)
        gen_rep = np.array([0.8, 0.8, 0.4, -0.3, 0.5, -0.6])
        gen_rep /= np.linalg.norm(gen_rep)
        Yseeds.append(_orbit_seed_Y(gen_rep, trans, rng))
        for Y0 in Yseeds:
            if time.time() > sym_deadline - 8.0:
                break
            # 1500 steps as assigned
            Yopt, Xv, bv = _sym_adam(Y0, steps=1500, beta0=30.0, beta1=300.0, lr0=0.01, lr1=0.001)
            if bv < sym_best_val:
                sym_best_val = float(bv)
                sym_best_Y = Yopt.copy()
            if time.time() > sym_deadline - 8.0:
                break
        # purely orbit-based variant: only generator seeds are variables
        if time.time() < sym_deadline - 10.0:
            # 4 generators x 29 transforms = 116
            rng2 = np.random.default_rng(999)
            perm = rng2.permutation(len(trans))
            groups = [ [trans[perm[i * 29 + j]] for j in range(29)] for i in range(4)]
            gens0 = _norm(rng2.standard_normal((4, 6)))
            _, Xorb, vorb = _orbit_gens_opt(gens0, groups, steps=700)
            if vorb < sym_best_val:
                # store as Y-like? convert Xorb to Y for release: take first 58? keep X directly
                sym_best_val = float(vorb)
                # stash Xorb as special
                sym_best_Y = None
                sym_best_Xorb = Xorb.copy()
            else:
                sym_best_Xorb = None
        else:
            sym_best_Xorb = None
    except Exception:
        sym_best_Y = None
        sym_best_val = 1e9
        sym_best_Xorb = None
        try:
            sym_best_Xorb
        except Exception:
            sym_best_Xorb = None
    # decide accepted symmetric candidate
    sym_X = None
    if sym_best_val < 0.605:
        if sym_best_Y is not None:
            sym_X = np.vstack([sym_best_Y, -sym_best_Y])
            sym_X = _norm(sym_X)
        elif 'sym_best_Xorb' in locals() and sym_best_Xorb is not None:
            sym_X = _norm(sym_best_Xorb)
    # ---- symmetry-breaking release + requench ----
    released_best = None
    released_val = 1e9
    try:
        if sym_X is not None:
            rng3 = np.random.default_rng(555)
            noises = [1e-3, 3e-3, 1e-2]
            for ns in noises:
                if time.time() > t0 + BUDGET - 35.0:
                    break
                Xn = sym_X.copy()
                R = rng3.standard_normal(Xn.shape)
                R = R - np.sum(R * Xn, axis=1, keepdims=True) * Xn
                rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-15
                R = R / rn * ns
                Xn = _norm(Xn + R)
                # requench unconstrained
                Xq = _smooth(Xn, steps=500, beta0=30, beta1=250, s0=0.02, s1=0.002)
                Xq = _smooth(Xq, steps=200, beta0=200, beta1=300, s0=0.004, s1=0.001)
                vv = _maxcos(Xq)
                if vv < released_val:
                    released_val = float(vv)
                    released_best = Xq.copy()
    except Exception:
        released_best = None
    # ---- final selection + LP polish ----
    # candidates to polish: incumbent always, released if promising
    final_X = incumbent.copy()
    final_val = inc_val
    if released_best is not None and released_val < inc_val + 0.005:
        # prefer released if close, LP will decide
        pass
    # polish released first with small budget, then incumbent with remainder
    try:
        remain = BUDGET - (time.time() - t0) - 1.0
        if released_best is not None and released_val < inc_val + 0.005 and remain > 8:
            # give ~1/3 to released
            rb = min(12.0, remain * 0.4)
            polished_rel = _lp_polish(released_best, rb)
            # re-evaluate
            vr = _maxcos(polished_rel)
            if vr < final_val:
                final_X = polished_rel
                final_val = vr
            remain = BUDGET - (time.time() - t0) - 1.0
        if remain > 4:
            polished_inc = _lp_polish(final_X if final_val < inc_val else incumbent, remain)
            vp = _maxcos(polished_inc)
            if vp < final_val:
                final_X = polished_inc
                final_val = vp
        else:
            final_X = _norm(final_X)
    except Exception:
        final_X = _norm(incumbent if released_best is None else (released_best if released_val < inc_val else incumbent))
    final_X = _norm(final_X)
    return final_X
