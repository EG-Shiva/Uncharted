import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

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
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
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

def lbfgs_smooth(x, alpha=250.0, maxiter=80, ref=None):
    n, d = x.shape
    if ref is None:
        ref = max_cos(x)
    z0 = x.ravel().copy()
    # scale: use ref for shift stability
    def fun_and_jac(z):
        Z = z.reshape(n, d)
        rn = np.linalg.norm(Z, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Z / rn
        G = X @ X.T
        # shift by ref (fixed)
        E = alpha * (G - ref)
        # clip to avoid overflow: exp(50) huge
        E = np.clip(E, -50, 30)
        np.fill_diagonal(E, -50)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5  # sum i<j
        if S <= 0:
            S = 1e-300
        L = np.log(S) / alpha + ref
        P = W / (2.0 * S)  # symmetric, sum_{j} P_kj X_j gives grad? check: dL = sum_{i<j} W_ij/S dG_ij
        # For k: sum_{j!=k} W_kj/S * X_j  (since each pair once, symmetric W)
        # W/(2S) *2 = W/S, but using symmetric multiply: (P+P.T)@X? W symmetric so P symmetric, P@X *2?
        # Let's compute: grad_Xk = sum_{j} (W_kj/S) X_j
        grad_X = (W / S) @ X
        # chain to Z
        dot = np.sum(grad_X * X, axis=1, keepdims=True)
        grad_Z = (grad_X - dot * X) / rn
        return float(L), grad_Z.ravel()
    try:
        res = minimize(fun_and_jac, z0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfun': maxiter*2})
        Z = res.x.reshape(n, d)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        return Z
    except Exception:
        return x

def refine_single(Y, c0, steps=120, s=0.03):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        g = Y[sel].sum(axis=0) if np.count_nonzero(sel) else Y[np.argmax(dots)]
        gt = g - np.dot(g, c) * c
        nn = np.linalg.norm(gt)
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.5 * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c /= np.linalg.norm(c)
            s *= 0.995
    return c / np.linalg.norm(c)

def hole_surgery(x, rng, n_samples=15000, n_move=1):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rowm = G.max(axis=1)
    # pick worst points
    order = np.argsort(-rowm)
    y = x.copy()
    for t in range(n_move):
        # choose among top ~10 with randomness
        cand_pool = order[:12]
        move = cand_pool[rng.integers(len(cand_pool))] if t > 0 else order[0]
        if rng.random() < 0.15:
            move = rng.integers(n)
        Y = np.delete(y, move, axis=0)
        C = rng.standard_normal((n_samples, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        best_c = None; best_s = 2.0
        bs = 3000
        for a in range(0, n_samples, bs):
            Cb = C[a:a+bs]
            S = Cb @ Y.T
            sm = S.max(axis=1)
            k2 = int(np.argmin(sm))
            if sm[k2] < best_s:
                best_s = sm[k2]
                best_c = Cb[k2].copy()
        best_c = refine_single(Y, best_c)
        y[move] = best_c
    return y

def random_tangent_kick(x, rng, sigma, rattler_boost=None, rowm=None, mcur=None):
    n, d = x.shape
    noise = rng.standard_normal((n, d))
    if rattler_boost is not None and rowm is not None:
        is_r = rowm < (mcur - 5e-4)
        boost = np.ones(n)
        boost[is_r] = rattler_boost
        noise *= boost[:, None]
    noise *= sigma
    # project to tangent
    dot = np.sum(noise * x, axis=1, keepdims=True)
    noise -= dot * x
    y = x + noise
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(12345)
    n, d = 272, 4
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands0 = []
    if bx is not None:
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands0.append(bx / np.linalg.norm(bx, axis=1, keepdims=True))
    if rx is not None:
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands0.append(rx / np.linalg.norm(rx, axis=1, keepdims=True))
    if not cands0:
        r = rng.standard_normal((n, d)); r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands0.append(r)
    best_x = None; best_v = 2.0
    for c in cands0:
        v = max_cos(c)
        if v < best_v:
            best_v = v; best_x = c.copy()
    cur_x = best_x.copy(); cur_v = best_v
    # quick GD+LBFGS evaluation to get true local value of starting point
    # (already polished, skip heavy)

    it = 0
    sigmas = [0.003, 0.007, 0.015, 0.030, 0.060, 0.010, 0.025, 0.050]
    while time.time() - t0 < BUDGET - 16.0:
        it += 1
        remain = BUDGET - 16.0 - (time.time() - t0)
        if remain <= 1.0:
            break
        # pick operator
        u = rng.random()
        rowm, mcur = row_max(cur_x)
        if u < 0.45:
            sigma = sigmas[(it-1) % len(sigmas)] * (0.7 + 0.6*rng.random())
            boost = 3.0 + 4.0*rng.random() if rng.random() < 0.6 else None
            y = random_tangent_kick(cur_x, rng, sigma, rattler_boost=boost, rowm=rowm, mcur=mcur)
        elif u < 0.65:
            # hole surgery 1-2 points
            nm = 1 if rng.random() < 0.7 else 2
            y = hole_surgery(cur_x, rng, n_samples=12000, n_move=nm)
            # small jitter others
            y = random_tangent_kick(y, rng, 0.002, rattler_boost=None, rowm=None, mcur=None)
        elif u < 0.80:
            # subset rotation: pick k points, apply small random rotation
            k = int(rng.integers(8, 40))
            idx = rng.choice(n, size=k, replace=False)
            # random skew matrix
            A = rng.standard_normal((d, d)); A = (A - A.T) * (0.02 + 0.08*rng.random())
            # exp approx: I + A
            R = np.eye(d) + A
            # orthonormalize via QR
            Q, _ = np.linalg.qr(R)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            y = cur_x.copy()
            y[idx] = (y[idx] @ Q.T)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # large global shake + hole
            y = random_tangent_kick(cur_x, rng, 0.08 + 0.05*rng.random())
        # fast annealed GD
        a0 = 60.0; a1 = 400.0 + 400.0*rng.random()
        steps = 250 if time.time()-t0 < BUDGET-40 else 150
        s0 = 0.02; s1 = 0.002
        y = smooth_gd(y, steps=steps, a0=a0, a1=a1, s0=s0, s1=s1, mu=0.8)
        v_gd = max_cos(y)
        # filter: only promising go to LBFGS/LP
        # adaptive threshold: allow worse for walker diversity
        if v_gd > cur_v + 0.03 and v_gd > best_v + 0.01:
            # too bad, skip expensive polish, maybe still move walker rarely
            continue
        # L-BFGS refinement (cheap, ~0.2-0.5s)
        alpha_l = 200.0 + 150.0*rng.random()
        y2 = lbfgs_smooth(y, alpha=alpha_l, maxiter=60, ref=min(v_gd, best_v))
        v2 = max_cos(y2)
        if v2 < v_gd:
            y, v_gd = y2, v2
        else:
            # sometimes LBFGS worsens max (softmax vs max mismatch); keep gd version
            pass
        # decide LP budget
        if v_gd < best_v + 0.002:
            lp_sec = min(3.5, max(1.0, remain*0.15))
            y3 = lp_polish(y, lp_sec)
            v3 = max_cos(y3)
        else:
            y3, v3 = y, v_gd
        # update best
        if v3 < best_v:
            best_x, best_v = y3.copy(), float(v3)
            cur_x, cur_v = y3.copy(), float(v3)
        else:
            # walker acceptance (basin hopping)
            T = 0.0012
            if v3 < cur_v or rng.random() < np.exp(-(v3 - cur_v) / T):
                cur_x, cur_v = y3.copy(), float(v3)
            # occasional restart to best to avoid drift
            if rng.random() < 0.08:
                cur_x, cur_v = best_x.copy(), float(best_v)
        # early exit if clearly record-breaking? keep improving
        if best_v < 0.8975 and time.time()-t0 > BUDGET-30:
            break
    # final polish best
    fstart = time.time()
    # alternate LBFGS + LP in remaining time
    try:
        best_x = lbfgs_smooth(best_x, alpha=350.0, maxiter=120, ref=best_v)
        best_v = max_cos(best_x)
    except Exception:
        pass
    left = BUDGET - (time.time() - t0)
    left = max(2.0, min(13.0, left))
    best_x = lp_polish(best_x, left)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
