import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 93, 13
    rng = np.random.default_rng(12345)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)

    # baseline
    base_best = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base_best = a.copy()
        except Exception:
            pass
    if base_best is None:
        base_best = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base_best)

    # ---- symmetric parametrization ----
    A_mult = np.array([1, 5, 7, 11, 13, 17], dtype=np.float64)  # mod 31
    K31 = 31
    ks = np.arange(K31)  # 0..30
    # alpha[k,j] = 2pi*a_j*k/31  shape (31,6)
    ALPHA = (2.0 * np.pi / 31.0) * (ks[:, None] * A_mult[None, :])

    def build_X(params):
        params = np.asarray(params, dtype=np.float64)
        X = np.empty((93, 13), dtype=np.float64)
        for o in range(3):
            v = params[o * 13:o * 13 + 7]
            ph = params[o * 13 + 7:o * 13 + 13]
            nv = float(np.dot(v, v) ** 0.5)
            if nv < 1e-12:
                w = np.full(7, 1.0 / np.sqrt(7))
            else:
                w = v / nv
            r = np.abs(w[:6])
            c = float(w[6])
            TH = ph[None, :] + ALPHA  # (31,6)
            CT = np.cos(TH)
            ST = np.sin(TH)
            blk = np.empty((31, 13), dtype=np.float64)
            for j in range(6):
                blk[:, 2 * j] = r[j] * CT[:, j]
                blk[:, 2 * j + 1] = r[j] * ST[:, j]
            blk[:, 12] = c
            X[o * 31:(o + 1) * 31, :] = blk
        # renormalize (should already be unit)
        return _norm(X)

    def obj_grad(params, tau):
        params = np.asarray(params, dtype=np.float64)
        # build per-orbit quantities, keep for backward
        ws = []
        rs = []
        cs = []
        THs = []
        for o in range(3):
            v = params[o * 13:o * 13 + 7]
            ph = params[o * 13 + 7:o * 13 + 13]
            nv = float(np.sqrt(np.dot(v, v) + 1e-300))
            if nv < 1e-12:
                w = np.full(7, 1.0 / np.sqrt(7))
                nv = 1e-12
            else:
                w = v / nv
            ws.append((w, v, nv))
            r = np.abs(w[:6])
            c = float(w[6])
            rs.append(r)
            cs.append(c)
            THs.append(ph[None, :] + ALPHA)
        X = np.empty((93, 13), dtype=np.float64)
        for o in range(3):
            r = rs[o]
            c = cs[o]
            TH = THs[o]
            CT = np.cos(TH)
            ST = np.sin(TH)
            blk = np.empty((31, 13), dtype=np.float64)
            for j in range(6):
                blk[:, 2 * j] = r[j] * CT[:, j]
                blk[:, 2 * j + 1] = r[j] * ST[:, j]
            blk[:, 12] = c
            X[o * 31:(o + 1) * 31, :] = blk
        # X rows already unit if w unit, but keep as is for grad (norm=1)
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        # weights
        E = np.exp((c_all - m) / tau)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            return float(m), np.zeros_like(params)
        L = m + tau * np.log(S)
        # grad wrt X
        # build symmetric weight matrix via accumulation
        Wmat = np.zeros((n, n), dtype=np.float64)
        # fill both triangles
        Wmat[iu0, iu1] = E
        Wmat[iu1, iu0] = E
        gradX = (Wmat @ X) / S  # (93,13)
        grad = np.zeros_like(params)
        for o in range(3):
            w, v, nv = ws[o]
            r = rs[o]
            TH = THs[o]
            CT = np.cos(TH)
            ST = np.sin(TH)
            Gblk = gradX[o * 31:(o + 1) * 31, :]  # (31,13)
            # dL/dr_j, dL/dphi_j, dL/dc
            dLdr = np.zeros(6)
            dLdphi = np.zeros(6)
            for j in range(6):
                gx = Gblk[:, 2 * j]
                gy = Gblk[:, 2 * j + 1]
                cj = CT[:, j]
                sj = ST[:, j]
                dLdr[j] = float(np.sum(gx * cj + gy * sj))
                dLdphi[j] = float(np.sum(r[j] * (-gx * sj + gy * cj)))
            dLdc = float(np.sum(Gblk[:, 12]))
            # chain to v
            sgn = np.sign(w[:6])
            sgn[sgn == 0] = 1.0
            h = np.empty(7, dtype=np.float64)
            h[:6] = dLdr * sgn
            h[6] = dLdc
            wdot = float(np.dot(w, h))
            gv = (h - wdot * w) / max(nv, 1e-12)
            grad[o * 13:o * 13 + 7] = gv
            grad[o * 13 + 7:o * 13 + 13] = dLdphi
        return float(L), grad

    # ---- symmetric optimization with restarts ----
    try:
        from scipy.optimize import minimize
        has_sp = True
    except Exception:
        has_sp = False

    sym_best_val = 1e9
    sym_best_params = None
    sym_best_X = None

    def random_start():
        p = np.empty(39, dtype=np.float64)
        for o in range(3):
            p[o * 13:o * 13 + 7] = rng.standard_normal(7)
            p[o * 13 + 7:o * 13 + 13] = rng.uniform(0, 2 * np.pi, size=6)
        return p

    TAUS = [0.02, 0.01, 0.005, 0.002]
    N_RESTART = 55 if has_sp else 0
    # quick evaluation of random starts to prioritize? just run
    if has_sp:
        for ri in range(N_RESTART):
            if time.time() - t0 > 62.0:
                break
            p0 = random_start()
            # evaluate true max to skip hopeless? no, run opt
            p = p0.copy()
            try:
                for ti, tau in enumerate(TAUS):
                    if time.time() - t0 > 64.0:
                        break
                    def fg(pp, _tau=tau):
                        L, g = obj_grad(pp, _tau)
                        if not np.all(np.isfinite(g)):
                            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                        return float(L), g.astype(np.float64)
                    maxit = 120 if ti < 2 else 180
                    res = minimize(fg, p, method='L-BFGS-B', jac=True,
                                   options={'maxiter': maxit, 'ftol': 1e-12, 'gtol': 1e-8})
                    p = res.x
                    if time.time() - t0 > 64.0:
                        break
            except Exception:
                continue
            try:
                Xc = build_X(p)
                vc = _maxcos(Xc)
            except Exception:
                continue
            if vc < sym_best_val and np.all(np.isfinite(Xc)):
                sym_best_val = float(vc)
                sym_best_params = p.copy()
                sym_best_X = Xc.copy()

    if sym_best_X is None:
        # fallback: random symmetric
        p = random_start()
        sym_best_X = build_X(p)
        sym_best_val = _maxcos(sym_best_X)

    # ---- polish utilities (full 93x13) ----
    def adam_lse(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for s in range(steps):
            if time.time() - t0 > BUDGET - 6:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * f)
            lr = lr0 * np.exp(llr * f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            cur = float(G.max())
            if cur < bv:
                bv = cur
                bX = X.copy()
            W = np.exp((G - cur) / tau)
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                continue
            grad = (W @ X) / S
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            tt += 1
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            d2 = np.einsum('ij,ij->i', step, X)[:, None]
            step = step - d2 * X
            X = _norm(X - step)
        return bX, bv

    def hard_topk(X0, iters, K0, K1, step0):
        X = _norm(X0.copy())
        bv = _maxcos(X)
        bX = X.copy()
        step = float(step0)
        for it in range(iters):
            if time.time() - t0 > BUDGET - 5:
                break
            f = it / max(1, iters - 1)
            K = int(round(K0 + (K1 - K0) * f))
            K = max(1, min(K, P))
            G = X @ X.T
            c = G[iu0, iu1]
            if K < P:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(P)
            I = iu0[idx]
            J = iu1[idx]
            grad = np.zeros_like(X)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-15:
                break
            direc = gt / ng
            Xt = _norm(X - step * direc * np.sqrt(n))
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv
                bX = Xt.copy()
                X = Xt
                step = min(step * 1.2, 0.02)
            else:
                step *= 0.5
                if step < 1e-12:
                    step = float(step0) * 0.3
                    rr = rng.standard_normal((n, d)) * 1e-4
                    rr -= np.einsum('ij,ij->i', rr, X)[:, None] * X
                    X = _norm(X + rr)
                    if it > iters - 20:
                        break
        return bX, bv

    def tangent_perturb(X, sigma):
        R = rng.standard_normal(size=X.shape) * sigma
        R -= np.einsum('ij,ij->i', R, X)[:, None] * X
        return _norm(X + R)

    # global best starts as baseline
    x_best = base_best.copy()
    best_val = float(base_val)

    # consider symmetric result
    if sym_best_X is not None and np.all(np.isfinite(sym_best_X)):
        if sym_best_val < best_val:
            best_val = float(sym_best_val)
            x_best = sym_best_X.copy()

    # build candidate list: symmetric + perturbed, plus baseline
    cands = []
    if sym_best_X is not None:
        cands.append(sym_best_X.copy())
        # small symmetry breaks at multiple scales
        for sig in (1e-4, 1e-3, 1e-2):
            for rep in range(2):
                if time.time() - t0 > BUDGET - 60:
                    break
                try:
                    cands.append(tangent_perturb(sym_best_X, sig))
                except Exception:
                    pass
    # also perturbed baselines to ensure competitiveness
    for sig in (0.01, 0.05):
        try:
            cands.append(tangent_perturb(base_best, sig))
        except Exception:
            pass
    cands.append(base_best.copy())

    # polish each candidate briefly, keep best
    # allocate steps adaptively based on remaining time and num cands
    for ci, Xc in enumerate(list(cands)):
        if time.time() - t0 > BUDGET - 18:
            break
        try:
            # short anneal
            bX, bv = adam_lse(Xc, 350, 0.008, 0.0006, 0.004, 0.0009)
            # hard polish if promising
            if bv < best_val + 0.003:
                bXh, bvh = hard_topk(bX, 100, 300, 30, 0.007)
                if bvh < bv:
                    bv, bX = bvh, bXh
            if bv < best_val and np.all(np.isfinite(bX)):
                best_val = float(bv)
                x_best = bX.copy()
        except Exception:
            continue

    # final intensive polish on current best
    while time.time() - t0 < BUDGET - 8:
        try:
            bX, bv = adam_lse(x_best, 300, 0.004, 0.00025, 0.0025, 0.0006)
            bXh, bvh = hard_topk(bX if bv < best_val + 1 else x_best, 150, 400, 25, 0.006)
            improved = False
            if bvh < best_val:
                best_val, x_best = float(bvh), bXh.copy()
                improved = True
            elif bv < best_val:
                best_val, x_best = float(bv), bX.copy()
                improved = True
            if not improved:
                # random kick and retry once
                Xk = tangent_perturb(x_best, 0.02)
                bX2, bv2 = adam_lse(Xk, 250, 0.006, 0.0004, 0.003, 0.0008)
                if bv2 < best_val:
                    best_val, x_best = float(bv2), bX2.copy()
                else:
                    break
        except Exception:
            break
        if time.time() - t0 > BUDGET - 9:
            break

    x_best = _norm(x_best)
    # tiny shrink guard (no-op, ensures finite)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    # replace any zero rows (should not happen)
    nr = np.linalg.norm(x_best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = _norm(x_best)
    return x_best
