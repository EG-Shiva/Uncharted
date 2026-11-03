import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contact_info(x, tight=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - tight, axis=1)
    rowmax = G.max(axis=1)
    tied = rowmax > m - 1e-9
    return m, cnt, tied, G


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_refine_all(x0, taus=(0.015, 0.005), maxiters=(80, 80)):
    n, d = x0.shape
    flat = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    for tau, mi in zip(taus, maxiters):
        def fun(f):
            X = f.reshape(n, d)
            norms = np.linalg.norm(X, axis=1)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms[:, None]
            G = Xn @ Xn.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0:
                S = 1e-300
            val = m + tau * np.log(S / 2.0 + 1e-300)
            W = E / S
            Gw = W @ Xn
            grad_Xn = 2.0 * Gw
            dot = (grad_Xn * Xn).sum(axis=1, keepdims=True)
            grad_X = (grad_Xn - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            grad_X = grad_X + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, grad_X.ravel()
        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": mi * 3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def find_holes(Xfrozen, k, rng, n_samp=50000, sep=0.65):
    d = Xfrozen.shape[1]
    C = rng.randn(n_samp, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ Xfrozen.T
    scores = S.max(axis=1)
    order = np.argsort(scores, kind="mergesort")
    picked_idx = []
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
            picked_idx.append(int(idx))
        if len(picked) >= k:
            break
    while len(picked) < k:
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        picked.append(v)
    P = np.array(picked)
    # local minimisation of max cosine to frozen code (projected gradient on LSE)
    tau = 0.01
    for a in range(P.shape[0]):
        v = P[a].copy()
        lr = 0.08
        for it in range(200):
            dots = Xfrozen @ v
            m = float(dots.max())
            E = np.exp((dots - m) / tau)
            s = float(E.sum())
            if s <= 0 or not np.isfinite(s):
                break
            W = E / s
            grad = W @ Xfrozen
            gt = grad - float(grad @ v) * v
            ng = float(np.linalg.norm(gt))
            if ng < 1e-12:
                break
            v = v - lr * gt
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                break
            v /= nv
            lr *= 0.995
        P[a] = v
    # re-separate after local moves: if two too close, jitter one
    for a in range(k):
        for b in range(a):
            if float(P[a] @ P[b]) > 0.85:
                vv = P[a] + 0.1 * rng.randn(d)
                P[a] = vv / np.linalg.norm(vv)
    return P


def joint_optimize_free(Xfree_init, Xfrozen, taus=(0.01, 0.003, 0.001), iters=150):
    kf = Xfree_init.shape[0]
    d = Xfree_init.shape[1]
    nfr = Xfrozen.shape[0]
    npair_denom = float(kf * nfr + kf * (kf - 1) / 2.0)
    flat = Xfree_init.ravel().copy()

    for tau in taus:
        def fun(f):
            X = f.reshape(kf, d)
            norms = np.linalg.norm(X, axis=1)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms[:, None]
            Gfr = Xn @ Xfrozen.T  # k x nfr
            Gff = Xn @ Xn.T
            # max over free-involved pairs
            m1 = float(Gfr.max()) if Gfr.size else -2.0
            if kf > 1:
                tmp = Gff.copy()
                np.fill_diagonal(tmp, -10.0)
                m2 = float(tmp.max())
            else:
                m2 = -2.0
            m = m1 if m1 > m2 else m2
            Efr = np.exp((Gfr - m) / tau)
            if kf > 1:
                Eff = np.exp((Gff - m) / tau)
                # zero diag
                for a in range(kf):
                    Eff[a, a] = 0.0
                # keep only upper for sum, but symmetric weights
                # sum unordered: half of full off-diag sum
                Sff = float(Eff.sum()) / 2.0
            else:
                Eff = None
                Sff = 0.0
            Sfr = float(Efr.sum())
            S = Sfr + Sff
            if S <= 0 or not np.isfinite(S):
                S = 1e-300
            val = m + tau * np.log(S / npair_denom + 1e-300)
            # weights normalized by S
            Wfr = Efr / S  # k x nfr
            grad = Wfr @ Xfrozen  # k x d
            if kf > 1:
                # symmetric unordered weight: Wsym = (Eff/S) with half weight?
                # unordered: each pair once, gradient = Wup * Xn
                # Wup[a,b] = Eff[a,b]/S for a<b (Eff symmetric, so Eff/S counts twice if full)
                # To avoid double, use Wsym = Eff/(2S) off-diag then (Wsym+Wsym.T)=Eff/S
                Wsym = Eff / (2.0 * S)
                # make symmetric contribution
                grad = grad + (Wsym + Wsym.T) @ Xn
            # convert to X gradient + penalty
            dot = (grad * Xn).sum(axis=1, keepdims=True)
            grad_X = (grad - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            grad_X = grad_X + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, grad_X.ravel()
        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": iters, "ftol": 1e-14, "gtol": 1e-7, "maxfun": iters * 3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(kf, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    rng = np.random.RandomState(20261103)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    LOOP_END = t0 + 150.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(162, 8)
    xb = np.asarray(xb, dtype=float).reshape(162, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    k_cycle = [3, 4, 6, 8, 10]
    trial = -1
    while time.time() < LOOP_END:
        trial += 1
        if time.time() > LOOP_END - 1.0:
            break
        k = k_cycle[trial % len(k_cycle)]
        mode = (trial // len(k_cycle)) % 3  # 0 top, 1 cluster, 2 random
        try:
            m, cnt, tied, G = contact_info(x_best, tight=1e-7)
        except Exception:
            m, cnt = best_val, np.zeros(162)
        rattler = int(np.argmin(cnt))
        order_desc = np.argsort(cnt, kind="mergesort")[::-1]
        # build deletion set always including rattler
        if mode == 0:
            # rattler + highest-contact
            sel = [rattler]
            for idx in order_desc:
                if int(idx) == rattler:
                    continue
                sel.append(int(idx))
                if len(sel) >= k:
                    break
            del_idx = np.array(sel[:k], dtype=int)
        elif mode == 1:
            # one spatial cluster + rattler
            # seed: pick among top-12 high-contact (deterministic rotate + random)
            top12 = [int(i) for i in order_desc[:12] if int(i) != rattler]
            if len(top12) == 0:
                top12 = [int(order_desc[0])]
            seed = top12[rng.randint(len(top12))]
            dots = x_best @ x_best[seed]
            dots[seed] = 2.0
            dots[rattler] = 2.0  # exclude rattler from nearest, add separately
            nn = np.argsort(-dots, kind="mergesort")[: max(0, k - 1)]
            sel = [rattler] + [int(i) for i in nn]
            # if k> len, fill random
            while len(sel) < k:
                cand = int(rng.randint(162))
                if cand not in sel:
                    sel.append(cand)
            del_idx = np.array(sel[:k], dtype=int)
        else:
            # random mixture + rattler (mix high and low)
            sel = [rattler]
            # half from high, half random
            nh = (k - 1) // 2
            for idx in order_desc:
                if int(idx) == rattler:
                    continue
                if len(sel) >= 1 + nh:
                    break
                sel.append(int(idx))
            while len(sel) < k:
                cand = int(rng.randint(162))
                if cand not in sel:
                    sel.append(cand)
            # shuffle (keep rattler inside)
            perm = rng.permutation(len(sel))
            sel = [sel[i] for i in perm]
            del_idx = np.array(sel[:k], dtype=int)

        mask = np.ones(162, dtype=bool)
        mask[del_idx] = False
        Xfrozen = x_best[mask].copy()
        # holes
        try:
            Xholes = find_holes(Xfrozen, k, rng, n_samp=50000, sep=0.65)
        except Exception:
            continue
        if time.time() > LOOP_END - 1.0:
            break
        # joint optimisation of free points only
        try:
            Xfree = joint_optimize_free(Xholes, Xfrozen, taus=(0.01, 0.003, 0.001), iters=150)
        except Exception:
            continue
        if time.time() > LOOP_END - 0.5:
            break
        Xnew = np.empty_like(x_best)
        Xnew[mask] = Xfrozen
        # map free back to del_idx order
        Xnew[del_idx] = Xfree
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        # short smooth refinement unfrozen
        try:
            Xnew = smooth_refine_all(Xnew, taus=(0.015, 0.005), maxiters=(60, 60))
        except Exception:
            pass
        if time.time() > LOOP_END:
            v = max_cos(Xnew)
            if v < best_val:
                x_best, best_val = Xnew, v
            break
        # trust-region LP polish 3-5 s
        remain_loop = LOOP_END - time.time()
        if remain_loop < 1.0:
            v = max_cos(Xnew)
            if v < best_val:
                x_best, best_val = Xnew, v
            break
        budget = 3.0
        if trial % 5 == 4:
            budget = 5.0
        budget = min(budget, remain_loop)
        if budget < 0.8:
            v = max_cos(Xnew)
            if v < best_val:
                x_best, best_val = Xnew, v
            continue
        try:
            Xpol = lp_polish(Xnew, budget)
        except Exception:
            continue
        vpol = max_cos(Xpol)
        vsmooth = max_cos(Xnew)
        # keep every post-LP result if best
        cand_best = vpol
        cand_X = Xpol
        if vsmooth < vpol:
            # keep smooth if better (rare)
            cand_best = vsmooth
            cand_X = Xnew
        if cand_best < best_val - 1e-12:
            x_best = cand_X.copy()
            best_val = float(cand_best)
        # also allow equal to diversify? keep exploring from best only
        if time.time() > DEADLINE - 3.0:
            break
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
