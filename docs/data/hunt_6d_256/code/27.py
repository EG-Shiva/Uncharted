import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _adam_riesz(x0, steps, s, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, 1.0)
        D2 = 2.0 - 2.0 * G
        np.maximum(D2, 1e-4, out=D2)
        # f' = s * D2^{-s/2-1}
        Fp = s * np.power(D2, -s / 2.0 - 1.0)
        Fp = np.minimum(Fp, 1e6)
        np.fill_diagonal(Fp, 0.0)
        # normalize to keep scale stable
        sc = float(np.median(Fp[Fp > 0])) if np.any(Fp > 0) else 1.0
        if np.isfinite(sc) and sc > 0:
            Fp = Fp / sc
        grad = Fp @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # clip tangent for stability
        gn = np.linalg.norm(gt, axis=1, keepdims=True)
        # Adam
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _build_sign_mats():
    import itertools
    S64 = np.array(list(itertools.product([-1.0, 1.0], repeat=6)), dtype=float)
    # subgroup flipping first 3 coords only (order 8)
    S8 = np.ones((8, 6), dtype=float)
    k = 0
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            for c in (-1.0, 1.0):
                S8[k, 0] = a
                S8[k, 1] = b
                S8[k, 2] = c
                k += 1
    return S64, S8

def _adam_sym(B0, S, steps, K0, K1, lr, fold_all, rng_seed_off=0):
    # B0 (k,6), S (g,6) -> X (k*g,6)
    k, d = B0.shape
    g = S.shape[0]
    B = B0.copy()
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    if fold_all:
        B = np.abs(B)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    def expand(Bb):
        X = (Bb[:, None, :] * S[None, :, :]).reshape(k * g, d)
        # already unit
        return X
    Xb = expand(B)
    best_val = _max_cos(Xb)
    best_B = B.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        X = expand(B)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum() * 0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        gradX = (W @ X) / Ssum  # (n,6)
        Gr = gradX.reshape(k, g, d)
        # dL/dB[j,i] = sum_s gradX[j,s,i]*S[s,i]
        gradB = np.sum(Gr * S[None, :, :], axis=1)
        dot = np.sum(gradB * B, axis=1, keepdims=True)
        gt = gradB - dot * B
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * B, axis=1, keepdims=True)
        step = step - sdot * B
        B = B - step
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        if fold_all:
            B = np.abs(B)
            B /= np.linalg.norm(B, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(expand(B))
            if val < best_val:
                best_val = val
                best_B = B.copy()
    return expand(best_B), best_val, best_B

def _void_replace_points(x, idxs, rng, n_samples=8000):
    n, d = x.shape
    for r in idxs:
        Z = rng.normal(size=(n_samples, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        S = Z @ x.T
        S[:, int(r)] = -2.0
        mm = S.max(axis=1)
        k2 = int(np.argmin(mm))
        if float(mm[k2]) < float(np.max(x[int(r)] @ x.T[np.arange(n) != int(r)])) - 1e-12:
            x[int(r)] = Z[k2]
        else:
            # accept if it improves global max when relevant
            # check global improvement opportunistically
            cur_global = _max_cos(x)
            if float(mm[k2]) < cur_global - 1e-12:
                x[int(r)] = Z[k2]
    return x

def _greedy_sweep(x, best, rng, deadline, top_k=24, batch=1600):
    n, d = x.shape
    sigmas = (0.03, 0.01, 0.0035)
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        prow = G.max(axis=1)
        order = np.argsort(-prow)[:top_k]
        moved = False
        for ii in order:
            if time.time() >= deadline:
                break
            i = int(ii)
            cur = float(prow[i])
            if cur < best - 5e-4:
                continue
            Gm = G.copy()
            Gm[i, :] = -2.0
            Gm[:, i] = -2.0
            other_max = float(Gm.max())
            if other_max >= best - 1e-12 and cur >= best - 1e-12:
                # tied: must beat best with both
                need = best
            else:
                need = min(best, other_max) if other_max >= best - 1e-12 else best
                # moving i alone can beat best only if other_max < best
                if other_max >= best - 1e-12:
                    # need new_local < best and other_max < best impossible
                    # still try to lower cur to help future (allow if cur is worst)
                    if cur < best - 1e-9:
                        continue
            xt = x[i].copy()
            best_local = cur
            best_cand = None
            for sig in sigmas:
                if time.time() >= deadline:
                    break
                R = rng.normal(size=(batch, d))
                R -= (R @ xt)[:, None] * xt[None, :]
                nr = np.linalg.norm(R, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                R /= nr
                C = xt[None, :] + sig * R
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                D = C @ x.T
                D[:, i] = -2.0
                lm = D.max(axis=1)
                kk = int(np.argmin(lm))
                if float(lm[kk]) < best_local - 1e-13:
                    best_local = float(lm[kk])
                    best_cand = C[kk].copy()
            # also try pure void samples
            if time.time() < deadline:
                Z = rng.normal(size=(batch, d))
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                D = Z @ x.T
                D[:, i] = -2.0
                lm = D.max(axis=1)
                kk = int(np.argmin(lm))
                if float(lm[kk]) < best_local - 1e-13:
                    best_local = float(lm[kk])
                    best_cand = Z[kk].copy()
            if best_cand is not None:
                new_global = float(max(best_local, other_max))
                if new_global < best - 1e-12:
                    x[i] = best_cand
                    best = new_global
                    G = x @ x.T
                    np.fill_diagonal(G, -2.0)
                    prow = G.max(axis=1)
                    moved = True
                    break
                elif best_local < cur - 2e-6 and new_global < best + 2e-5:
                    # sideways equalization: accept small worsening guard? only if not worsening best
                    if new_global < best + 1e-12:
                        x[i] = best_cand
                        G = x @ x.T
                        np.fill_diagonal(G, -2.0)
                        prow = G.max(axis=1)
                        moved = True
                        # continue to next point without updating best
        if not moved:
            break
    return x, best

def _lp_polish(x, deadline, r0=2.0e-3, cap=3500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(25.0 * radius, 8e-5)
        sel = c > (m - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.015)
        else:
            radius *= 0.5
    return x, best

def _rand_orth_close(rng, d, t):
    M = np.eye(d) + t * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    SYM_END = t0 + 32.0
    FRESH_END = t0 + 48.0
    ANNEAL_END = t0 + 132.0
    GREEDY_END = t0 + 157.0
    rng = np.random.default_rng(0)
    n, d = 256, 6
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    S64, S8 = _build_sign_mats()
    # --- symmetric full-sign trials (k=4 -> 256) ---
    try:
        for trial in range(3):
            if time.time() > SYM_END:
                break
            B0 = np.abs(rng.normal(size=(4, d)))
            B0 /= np.linalg.norm(B0, axis=1, keepdims=True)
            Xs, vs, Bs = _adam_sym(B0, S64, steps=900, K0=60.0, K1=600.0, lr=0.02, fold_all=True)
            # break symmetry slightly and relax with full optimizer
            Xn = Xs + 1.5e-3 * rng.normal(size=Xs.shape)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xr, vr = _adam_riesz(Xn, steps=400, s=8.0, lr=0.008)
            Xr2, vr2 = _adam_sm(Xr, steps=400, K0=150.0, K1=1200.0, lr=0.007)
            if vr2 < best - 1e-12:
                best = vr2
                x_best = Xr2.copy()
    except Exception:
        pass
    # --- partial symmetry trials (k=32 x8 ->256) ---
    try:
        for trial in range(2):
            if time.time() > SYM_END + 6.0:
                break
            B0 = rng.normal(size=(32, d))
            B0 /= np.linalg.norm(B0, axis=1, keepdims=True)
            Xs, vs, Bs = _adam_sym(B0, S8, steps=600, K0=80.0, K1=700.0, lr=0.015, fold_all=False)
            Xn = Xs + 1.5e-3 * rng.normal(size=Xs.shape)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xr, vr = _adam_riesz(Xn, steps=300, s=8.0, lr=0.008)
            Xr2, vr2 = _adam_sm(Xr, steps=300, K0=150.0, K1=1000.0, lr=0.007)
            if vr2 < best - 1e-12:
                best = vr2
                x_best = Xr2.copy()
    except Exception:
        pass
    # --- structured hypercube-rotated starts ---
    try:
        if time.time() < FRESH_END:
            H = S64 / np.sqrt(6.0)  # 64 x6, max 0.666
            for trial in range(2):
                if time.time() > FRESH_END:
                    break
                blocks = []
                for _ in range(4):
                    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]
                    blocks.append(H @ Q.T)
                X0 = np.vstack(blocks)
                X0 += 1e-3 * rng.normal(size=X0.shape)
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
                Xr, vr = _adam_riesz(X0, steps=500, s=7.0, lr=0.009)
                Xr2, vr2 = _adam_sm(Xr, steps=500, K0=120.0, K1=1200.0, lr=0.007)
                if vr2 < best - 1e-12:
                    best = vr2
                    x_best = Xr2.copy()
    except Exception:
        pass
    # --- fresh random Riesz starts ---
    try:
        for trial in range(3):
            if time.time() > FRESH_END:
                break
            X0 = rng.normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xr, vr = _adam_riesz(X0, steps=700, s=7.0, lr=0.01)
            Xr2, vr2 = _adam_sm(Xr, steps=500, K0=120.0, K1=1000.0, lr=0.008)
            if vr2 < best - 1e-12:
                best = vr2
                x_best = Xr2.copy()
    except Exception:
        pass
    # --- elitist basin hopping ---
    cycle = 0
    while time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - time.time()
        if rem < 4.0:
            break
        x = x_best.copy()
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        flat = int(np.argmax(G))
        ai = flat // n
        aj = flat % n
        prow = G.max(axis=1)
        cnt = np.sum(G > gmax - 3e-4, axis=1)
        order_cnt = np.argsort(-cnt)
        order_row = np.argsort(-prow)
        mode = cycle % 5
        if mode == 0:
            # worst-pair split + void reinsertion of worst 4
            eps = 0.04 + 0.03 * rng.random()
            x[ai] = x[ai] - eps * x[aj]
            x[aj] = x[aj] - eps * x[ai]
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            worst4 = order_row[:4]
            x = _void_replace_points(x, worst4, rng, n_samples=12000)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            # coherent cap rotation around worst midpoint
            c = x[ai] + x[aj]
            nn = np.linalg.norm(c)
            if nn == 0:
                c = x[ai]
            else:
                c = c / nn
            dots = x @ c
            thr = 0.55 + 0.2 * rng.random()
            idx = np.where(dots > thr)[0]
            if idx.size < 2 or idx.size > 200:
                idx = np.argsort(-dots)[:30]
            t_ang = 0.05 + 0.10 * rng.random()
            Q = _rand_orth_close(rng, d, t_ang)
            x[idx] = (x[idx] @ Q.T)
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            rnd = rng.choice(n, size=4, replace=False)
            x[rnd] += 0.02 * rng.normal(size=(4, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            # block replacement of 8 worst by voids (topology change)
            worst8 = order_row[:8]
            # sample large void pool once
            Z = rng.normal(size=(20000, 6))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            for r in worst8:
                S = Z @ x.T
                S[:, int(r)] = -2.0
                mm = S.max(axis=1)
                k2 = int(np.argmin(mm))
                # accept if better than current row max minus small tol
                if float(mm[k2]) < float(prow[int(r)]) - 1e-5:
                    x[int(r)] = Z[k2]
                    # update prow approx for next r
                    prow[int(r)] = float(mm[k2])
            nb = np.argsort(-(x @ x[ai].T))[:6] if False else order_row[:6]
            x[nb] += 0.015 * rng.normal(size=(len(nb), 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 3:
            # patch shatter around worst + overloaded
            nb = np.argsort(-G[ai])[:12]
            x[nb] += 0.10 * rng.normal(size=(12, 6))
            over = order_cnt[:6]
            x[over] += 0.04 * rng.normal(size=(6, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # global moderate kick
            over = order_cnt[:10]
            x[over] += 0.05 * rng.normal(size=(10, 6))
            rnd = rng.choice(n, size=10, replace=False)
            x[rnd] += 0.04 * rng.normal(size=(10, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # relax: balanced Riesz then soft-max
        try:
            xr, _ = _adam_riesz(x, steps=280, s=9.0, lr=0.007)
            xn, vn = _adam_sm(xr, steps=420, K0=150.0, K1=1300.0, lr=0.007)
        except Exception:
            continue
        if vn < best - 2e-9:
            best = vn
            x_best = xn.copy()
    # --- final intensification ---
    try:
        if time.time() < GREEDY_END:
            xb, vb = _adam_sm(x_best, steps=900, K0=200.0, K1=2500.0, lr=0.006)
            if vb < best:
                best = vb
                x_best = xb.copy()
    except Exception:
        pass
    try:
        if time.time() < GREEDY_END:
            x_best, best = _greedy_sweep(x_best, best, rng, min(GREEDY_END, T_END - 14.0))
    except Exception:
        pass
    try:
        if time.time() < T_END - 4.0:
            x_best, best = _lp_polish(x_best, min(T_END - 2.0, time.time() + 11.0))
    except Exception:
        pass
    try:
        if time.time() < T_END - 1.0:
            # tiny final soft-max to re-equalize after LP
            xb, vb = _adam_sm(x_best, steps=250, K0=300.0, K1=3000.0, lr=0.004)
            if vb < best:
                best = vb
                x_best = xb.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
