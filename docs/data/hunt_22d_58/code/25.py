import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees(X, thr):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return (G > thr).sum(axis=1)

def _pump(X, t, steps, lr, beta0, beta1, seed_off=0):
    # minimise sum_{active} softplus_beta(G-t)^2, active: G > t-0.01
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        G = X @ X.T
        # active mask (symmetric, no diag)
        Act = G > (t - 0.01)
        np.fill_diagonal(Act, False)
        D = G - t
        z = beta * D
        # sigmoid stable
        Sig = np.zeros_like(G)
        # only where active to save time (still compute all, cheap)
        # clip z
        zc = np.clip(z, -30, 30)
        Sig = 1.0 / (1.0 + np.exp(-zc))
        # softplus = logaddexp(0,z)/beta
        SP = np.logaddexp(0, z) / beta
        SP[~Act] = 0.0
        Sig[~Act] = 0.0
        Fp = 2.0 * SP * Sig  # dL/dG for i<j and i>j (will be halved by symmetry? use half)
        # Fp is symmetric; for gradient we need sum over j !=k Fp_kj * X_j
        # since L sums i<j only, Fp for i!=j should be counted once; our Fp counts both,
        # so divide by 2 for off-diag symmetry: actually SP*Sig defined for both (i,j),(j,i)
        # L = sum_{i<j} f => grad factor = f' once. Using symmetric Fp gives 2x, so *0.5
        Fp *= 0.5
        np.fill_diagonal(Fp, 0.0)
        Grad = Fp @ X
        # Riemannian projection
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Rg = Grad - X * dot
        m = b1 * m + (1 - b1) * Rg
        v = b2 * v + (1 - b2) * (Rg * Rg)
        # bias correction
        mhat = m / (1 - b1 ** (s + 2))
        vhat = v / (1 - b2 ** (s + 2))
        Step = mhat / (np.sqrt(vhat) + eps)
        # project step to tangent
        sdot = (Step * X).sum(axis=1, keepdims=True)
        Step = Step - X * sdot
        X = X - lr * Step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, stress=None, lam=1e-3, t0=None):
    # trust-region LP: min t + lam * <c_stress, dx>
    # lazy import
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    if t0 is None:
        t0 = time.time()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    radius = 1e-3
    # stress vector for secondary: stress matrix (n,n) symmetric or None
    # build linear secondary coeffs: c_dx[k*d:(k+1)*d] = lam * sum_j S_kj * X_j
    it = 0
    while time.time() - t0 < seconds and radius > 1e-12 and it < 30:
        it += 1
        Gfull = (X @ X.T)[iu]
        m = float(Gfull.max())
        sel = Gfull > m - max(20 * radius, 1e-9)
        # cap contacts to keep LP small
        idx = np.where(sel)[0]
        if len(idx) > 600:
            # keep largest 600
            order = np.argsort(-Gfull[idx])[:600]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
            idx = np.where(sel)[0]
        I, J = iu[0][sel], iu[1][sel]
        cvals = Gfull[sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        if stress is not None:
            # stress (n,n) symmetric nonneg
            S = stress + stress.T
            # coeff = lam * S @ X
            Coef = lam * (S @ X)
            obj[:n * d] = Coef.ravel()
        b_ub = -cvals
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best - 1e-12:
            X, best = y, vv
            # extract duals as new stress if available
            try:
                marg = None
                if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                    marg = np.array(res.ineqlin.marginals, dtype=float)
                elif hasattr(res, 'ineqlin') and isinstance(res.ineqlin, dict):
                    marg = np.array(res.ineqlin.get('marginals', None), dtype=float)
                if marg is not None and len(marg) == p:
                    w = np.maximum(-marg, 0.0)
                    # map back to full pair matrix for next iteration
                    Snew = np.zeros((n, n))
                    Snew[I, J] = w
                    Snew[J, I] = w
                    # keep scale sane
                    if Snew.max() > 0:
                        Snew /= Snew.max()
                    stress = Snew
            except Exception:
                pass
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - t0 >= seconds:
            break
    return X, best, stress

def solve():
    T_START = time.time()
    HARD = 165.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.where(np.isfinite(c), c, 0.0)
                nrm = np.linalg.norm(c, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                c = c / nrm
                cands.append(c)
        except Exception:
            pass
    if not cands:
        c = rng.standard_normal((58, 22))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    incumbent = best.copy()
    inc_val = best_val
    # contact baseline of incumbent
    G0 = incumbent @ incumbent.T
    np.fill_diagonal(G0, -2)
    m0 = float(G0.max())
    deg0 = (G0 > m0 - 5e-4).sum(axis=1)
    med0 = float(np.median(deg0))

    # stress init from near contacts
    n = 58
    stress = np.zeros((n, n))
    # softmax stress at incumbent
    Gw = incumbent @ incumbent.T
    np.fill_diagonal(Gw, -1.0)
    W0 = np.exp(np.clip((Gw - m0) / 0.002, -30, 0))
    np.fill_diagonal(W0, 0.0)
    stress = W0 / (W0.max() + 1e-12)

    targets = [0.0663, 0.0660, 0.0655]
    # filter targets below incumbent only (slightly below)
    targets = [t for t in targets if t < inc_val]
    if not targets:
        targets = [inc_val - 0.0002]

    for ti, t in enumerate(targets):
        if time.time() - T_START > HARD - 25:
            break
        # start from incumbent + small kick (bigger kick for later targets)
        kick = 0.004 + 0.004 * ti
        X = incumbent + kick * rng.standard_normal(incumbent.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # annealed pump: beta up, lr modest
        steps = 1200 if ti == 0 else 900
        # leave time guard
        X = _pump(X, t, steps=steps, lr=0.004, beta0=250.0, beta1=1800.0)
        if time.time() - T_START > HARD - 22:
            v = _max_cos(X)
            if v < best_val:
                best, best_val = X, v
            break
        v = _max_cos(X)
        # count above old max / above t
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cnt_t = int((G > t).sum() // 2)
        cnt_old = int((G > inc_val - 1e-12).sum() // 2)
        deg = (G > v - 5e-4).sum(axis=1)
        mindeg = int(deg.min())
        # improvement? accept
        if v < best_val - 1e-12:
            # LP polish this t-feasible-ish point briefly with stress secondary
            X2, v2, stress = _lp_polish(X, seconds=4, stress=stress, lam=2e-3, t0=time.time())
            if v2 < best_val:
                best, best_val = X2, v2
            else:
                best, best_val = X, v
            # if we beat t, lower: update incumbent and continue
            if best_val < t:
                incumbent, inc_val = best.copy(), best_val
                # recompute stress baseline
                Gw2 = incumbent @ incumbent.T
                np.fill_diagonal(Gw2, -1.0)
                mI = float(Gw2.max())
                W0 = np.exp(np.clip((Gw2 - mI) / 0.002, -30, 0))
                np.fill_diagonal(W0, 0.0)
                stress = W0 / (W0.max() + 1e-12)
                continue
        # de-jamming check: markedly fewer contacts -> transplant that point into incumbent for next round
        # condition: mindeg < 0.4*median OR at least one point with deg <= 3 while median large
        if mindeg < max(3, int(0.5 * med0)):
            # transplant lowest-degree point into incumbent (keep diversity)
            j = int(np.argmin(deg))
            incumbent_try = incumbent.copy()
            incumbent_try[j] = X[j] / (np.linalg.norm(X[j]) + 1e-16)
            # quick small pump around transplant to test
            Xt = _pump(incumbent_try, t, steps=300, lr=0.002, beta0=400.0, beta1=1200.0)
            vt = _max_cos(Xt)
            if vt < best_val - 1e-12:
                best, best_val = Xt, vt
                if vt < inc_val:
                    incumbent, inc_val = Xt.copy(), vt
            # LP step with secondary stress, dropping unfavourable contacts:
            # drop contacts with tiny stress (<0.05) implicitly by capping: keep stress thresholded
            Sthr = np.where(stress > 0.08, stress, 0.0)
            Xlp, vlp, stress = _lp_polish(incumbent, seconds=3, stress=Sthr, lam=5e-3, t0=time.time())
            if vlp < best_val - 1e-12:
                best, best_val = Xlp, vlp
                incumbent, inc_val = Xlp.copy(), vlp
            # restart next target round from incumbent (as required)
            continue
        else:
            # no de-jamming: alternate LP with secondary stress objective
            Sthr = np.where(stress > 0.05, stress, 0.0)
            Xlp, vlp, stress = _lp_polish(X, seconds=3, stress=Sthr, lam=3e-3, t0=time.time())
            if vlp < best_val - 1e-12:
                best, best_val = Xlp, vlp
                if vlp < inc_val:
                    incumbent, inc_val = Xlp.copy(), vlp
            elif v < best_val + 5e-5:
                # penalise unfavourable contacts: downweight high-stress pairs by extra pump focused on them
                # one more short pump with higher beta
                X3 = _pump(X, t - 0.0001, steps=400, lr=0.002, beta0=800.0, beta1=2500.0)
                v3 = _max_cos(X3)
                if v3 < best_val - 1e-12:
                    best, best_val = X3, v3
                    if v3 < inc_val:
                        incumbent, inc_val = X3.copy(), v3
        # time guard
        if time.time() - T_START > HARD - 20:
            break

    # final LP polish of best for remaining time (max ~12s)
    remain = HARD - (time.time() - T_START)
    if remain > 4:
        try:
            Xf, vf, _ = _lp_polish(best, seconds=min(10.0, remain - 2), stress=stress, lam=1e-3, t0=time.time())
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    # safety: if somehow worse than loaded best, return loaded best
    try:
        bl = np.load("best.npy").astype(float)
        bl /= np.linalg.norm(bl, axis=1, keepdims=True)
        if _max_cos(bl) < _max_cos(best):
            best = bl
    except Exception:
        pass
    return best
