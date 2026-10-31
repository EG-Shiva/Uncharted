import time
import numpy as np
from scipy.optimize import linprog, nnls
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4000 - 1)[:4000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def tangent_bases(x):
    n, d = x.shape
    B = np.zeros((n, d, 3))
    for i in range(n):
        xi = x[i]
        e = int(np.argmax(np.abs(xi)))
        cols = []
        for k in range(d):
            if k == e:
                continue
            ek = np.zeros(d)
            ek[k] = 1.0
            v = ek - float(xi[k]) * xi
            cols.append(v)
        M = np.stack(cols, axis=1)  # 4x3
        Q, Rq = np.linalg.qr(M)
        # Q is 4x3 orthonormal (M full rank); ensure tangent
        # re-orthogonalize against xi numerically
        for c in range(3):
            q = Q[:, c] - float(Q[:, c] @ xi) * xi
            nn = np.linalg.norm(q)
            if nn > 1e-12:
                Q[:, c] = q / nn
        B[i] = Q
    return B


def rigidity_analysis(x, tol=1e-7, K=20):
    n, d = x.shape
    assert d == 4
    B = tangent_bases(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c_all = G[iu]
    sel = c_all >= m - tol
    I = iu[0][sel]
    J = iu[1][sel]
    p = len(I)
    N = n * 3
    if p == 0:
        return None
    R = np.zeros((p, N))
    # rows: block i: B_i^T x_j, block j: B_j^T x_i
    for t in range(p):
        i = int(I[t]); j = int(J[t])
        R[t, i * 3:(i + 1) * 3] = B[i].T @ x[j]
        R[t, j * 3:(j + 1) * 3] = B[j].T @ x[i]
    # SVD for floppy modes
    try:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
    except Exception:
        return None
    kmin = S.shape[0]
    K = min(K, kmin)
    Vflop = Vt[-K:].T.copy()  # N x K, floppiest last
    Sflop = S[-K:].copy()
    # equilibrium stress via augmented NNLS: min ||R^T w|| s.t. w>=0, mean~1
    w = np.ones(p) / p * p  # fallback ~1
    try:
        gamma = 0.5
        A_aug = np.vstack([R.T, gamma * np.ones((1, p))])
        b_aug = np.zeros(A_aug.shape[0])
        b_aug[-1] = gamma * p
        # scale for stability
        # nnls can be slow; cap p if huge
        if p <= 3000:
            ww, _ = nnls(A_aug, b_aug)
            if np.sum(ww) > 1e-9 and np.all(np.isfinite(ww)):
                w = ww
            else:
                w = np.ones(p)
        else:
            w = np.ones(p)
    except Exception:
        w = np.ones(p)
    order = np.argsort(-w)
    return {"B": B, "R": R, "I": I, "J": J, "m": m, "p": p,
            "Vflop": Vflop, "Sflop": Sflop, "w": w, "order": order}


def apply_tangent_step(x, B, u, step_scale=1.0):
    # u: (N,) tangent coords; scale then retract
    n = x.shape[0]
    U = (u.reshape(n, 3)) * step_scale
    dX = np.einsum('nij,nj->ni', B, U)
    y = x + dX
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.standard_normal((272, 4))
    bx = np.asarray(bx, dtype=np.float64)
    if bx.shape != (272, 4):
        bx = np.resize(bx, (272, 4))
    bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    best_x = bx.copy()
    best_v = max_cos(best_x)
    try:
        rx = np.load("record.npy")
        rx = np.asarray(rx, dtype=np.float64)
        rx /= np.linalg.norm(rx, axis=1, keepdims=True)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    except Exception:
        pass

    # initial quick analysis
    ana = None
    try:
        ana = rigidity_analysis(best_x, tol=1e-7, K=20)
    except Exception:
        ana = None

    # build unload direction if possible
    unload_vec = None
    if ana is not None:
        try:
            R = ana["R"]; Vflop = ana["Vflop"]; w = ana["w"]; order = ana["order"]
            K = Vflop.shape[1]
            Q = min(80, ana["p"])
            top = order[:Q]
            Rtop = R[top]  # Q x N
            M = Rtop @ Vflop  # Q x K
            rhs = -np.ones(Q)
            a, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
            if np.linalg.norm(a) < 1e-12 or not np.all(np.isfinite(a)):
                a = rng.standard_normal(K)
            a = a / (np.linalg.norm(a) + 1e-15)
            unload_vec = Vflop @ a  # N,
            # normalize to unit global norm
            unload_vec = unload_vec / (np.linalg.norm(unload_vec) + 1e-15)
        except Exception:
            unload_vec = None

    # candidate step sizes (RMS per point)
    stepsizes = [0.005, 0.010, 0.020, 0.035, 0.055, 0.080]

    trial = 0
    # reserve final polish
    while time.time() - t0 < BUDGET - 13.0:
        trial += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 4:
            break
        # refresh analysis occasionally or after improvement
        if trial % 6 == 1 and trial > 1:
            try:
                ana2 = rigidity_analysis(best_x, tol=1e-7, K=20)
                if ana2 is not None:
                    ana = ana2
                    try:
                        R = ana["R"]; Vflop = ana["Vflop"]; w = ana["w"]; order = ana["order"]
                        K = Vflop.shape[1]
                        Q = min(80, ana["p"])
                        top = order[:Q]
                        M = R[top] @ Vflop
                        a, _, _, _ = np.linalg.lstsq(M, -np.ones(Q), rcond=None)
                        if np.linalg.norm(a) < 1e-12 or not np.all(np.isfinite(a)):
                            a = rng.standard_normal(K)
                        a = a / (np.linalg.norm(a) + 1e-15)
                        unload_vec = Vflop @ a
                        unload_vec = unload_vec / (np.linalg.norm(unload_vec) + 1e-15)
                    except Exception:
                        pass
            except Exception:
                pass

        # choose direction type
        mode_choice = trial % 5
        u = None
        if ana is not None and mode_choice in (0, 1, 2, 3):
            try:
                Vflop = ana["Vflop"]; B = ana["B"]
                K = Vflop.shape[1]
                if mode_choice == 0 and unload_vec is not None:
                    # stress-unloading direction (+/-)
                    s = 1.0 if (rng.random() < 0.8) else -1.0
                    u = s * unload_vec
                elif mode_choice == 1:
                    # single floppy mode +- (cycle through floppiest)
                    idx = (trial // 5) % K
                    s = 1.0 if (rng.random() < 0.5) else -1.0
                    u = s * Vflop[:, idx] / (np.linalg.norm(Vflop[:, idx]) + 1e-15)
                elif mode_choice == 2:
                    # random combo of 5-20 floppiest
                    kk = int(rng.integers(5, K + 1)) if K > 5 else K
                    # bias to floppiest: weight by 1/(S+eps)
                    Sf = ana["Sflop"][-kk:] if kk <= len(ana["Sflop"]) else ana["Sflop"]
                    wgt = 1.0 / (Sf + 0.05)
                    wgt = wgt / wgt.max()
                    coeff = rng.standard_normal(kk) * wgt
                    # map kk last modes
                    Vsub = Vflop[:, -kk:]
                    u = Vsub @ coeff
                    u = u / (np.linalg.norm(u) + 1e-15)
                else:
                    # combo targeting top-stress subset sign: random then flip to unload on average
                    kk = min(K, 12)
                    Vsub = Vflop[:, -kk:]
                    coeff = rng.standard_normal(kk)
                    u0 = Vsub @ coeff
                    # check mean change on top-stress contacts; flip to make negative
                    try:
                        Q2 = min(50, ana["p"])
                        top2 = ana["order"][:Q2]
                        ch = float(np.mean(ana["R"][top2] @ (u0 / (np.linalg.norm(u0) + 1e-15))))
                        if ch > 0:
                            u0 = -u0
                    except Exception:
                        pass
                    u = u0 / (np.linalg.norm(u0) + 1e-15)
            except Exception:
                u = None
        # fallback isotropic tangent gaussian
        if u is None:
            try:
                B = ana["B"] if ana is not None else tangent_bases(best_x)
            except Exception:
                B = tangent_bases(best_x)
            # random tangent coords
            ur = rng.standard_normal((272, 3))
            u = ur.ravel() / (np.linalg.norm(ur.ravel()) + 1e-15)
        else:
            try:
                B = ana["B"]
            except Exception:
                B = tangent_bases(best_x)

        step_rms = stepsizes[(trial - 1) % len(stepsizes)]
        # occasionally jitter step
        step_rms = step_rms * (0.7 + 0.6 * rng.random())
        global_scale = step_rms * np.sqrt(272)  # u is unit global norm
        xp = apply_tangent_step(best_x, B, u, step_scale=global_scale)

        # brief annealed softmax phase beta 50 -> 800
        # adapt steps to remaining time
        if remain < 25:
            gd_steps = 250
        elif remain < 60:
            gd_steps = 350
        else:
            gd_steps = 450 + int(rng.integers(0, 200))
        # jitter softmax params slightly but keep 50->800 endpoints per spec
        b0 = 50.0 * (0.9 + 0.2 * rng.random())
        b1 = 800.0 * (0.8 + 0.4 * rng.random())
        s0 = 0.006 + 0.004 * rng.random()
        s1 = 0.0006 + 0.0006 * rng.random()
        xp = smooth_gd(xp, gd_steps, b0, b1, s0, s1, mu=0.8)
        v = max_cos(xp)
        # decide LP polish: always polish if within 0.003 of best or better; else only sometimes
        do_lp = False
        lp_time = 2.5
        if v < best_v + 0.003:
            do_lp = True
            # more time if promising
            if v < best_v + 0.0005:
                lp_time = min(4.0, max(2.5, remain / 8.0))
            else:
                lp_time = 2.5
        else:
            # occasional long-shot polish
            if rng.random() < 0.05 and remain > 30:
                do_lp = True
                lp_time = 2.0
        if do_lp:
            if remain - 1.0 < lp_time:
                lp_time = max(1.0, remain - 1.0)
            yp = lp_polish(xp, lp_time)
            vp = max_cos(yp)
            if vp < best_v - 1e-12:
                best_x, best_v = yp, vp
                # refresh analysis immediately after improvement
                try:
                    ana = rigidity_analysis(best_x, tol=1e-7, K=20)
                    if ana is not None:
                        try:
                            R = ana["R"]; Vflop = ana["Vflop"]; order = ana["order"]
                            Kk = Vflop.shape[1]
                            Q = min(80, ana["p"])
                            top = order[:Q]
                            M = R[top] @ Vflop
                            aa, _, _, _ = np.linalg.lstsq(M, -np.ones(Q), rcond=None)
                            if np.linalg.norm(aa) < 1e-12 or not np.all(np.isfinite(aa)):
                                aa = rng.standard_normal(Kk)
                            aa = aa / (np.linalg.norm(aa) + 1e-15)
                            unload_vec = Vflop @ aa
                            unload_vec = unload_vec / (np.linalg.norm(unload_vec) + 1e-15)
                        except Exception:
                            pass
                except Exception:
                    pass
            elif v < best_v - 5e-5:
                # GD alone improved a lot but LP didn't beat; still adopt GD point as new base
                best_x, best_v = xp, v
        else:
            if v < best_v - 5e-5:
                best_x, best_v = xp, v

        # opportunistic isotropic kick interleave every 7th trial to avoid stagnation
        if trial % 7 == 0 and time.time() - t0 < BUDGET - 20:
            try:
                sigma = 0.015 if rng.random() < 0.5 else 0.040
                noise = rng.standard_normal(best_x.shape) * sigma
                xq = best_x + noise
                xq /= np.linalg.norm(xq, axis=1, keepdims=True)
                xq = smooth_gd(xq, 300, 40.0, 500.0, 0.008, 0.0008, mu=0.8)
                vq = max_cos(xq)
                if vq < best_v + 0.001:
                    yq = lp_polish(xq, 2.0)
                    vqq = max_cos(yq)
                    if vqq < best_v - 1e-12:
                        best_x, best_v = yq, vqq
            except Exception:
                pass

    remain = BUDGET - (time.time() - t0)
    if remain > 3:
        try:
            best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
