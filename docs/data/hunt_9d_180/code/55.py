import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def load_best():
    try:
        b = np.asarray(np.load("best.npy"), dtype=float)
        if b.shape == (180, 9):
            n = np.linalg.norm(b, axis=1, keepdims=True)
            n[~np.isfinite(n)] = 1.0
            n[n < 1e-12] = 1.0
            b = b / n
            b[~np.isfinite(b)] = 0.0
            # renormalize again
            n2 = np.linalg.norm(b, axis=1, keepdims=True)
            n2[n2 < 1e-12] = 1.0
            b = b / n2
            return b
    except Exception:
        pass
    return None

def antipodal_adam(Y0, betas, lrs, stage_budgets, t_end_hard, track_best=True):
    Y = Y0.copy()
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step_cnt = 0
    # best tracking
    def eval_X(Y_):
        X = np.vstack([Y_, -Y_])
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return Xn, max_cos(Xn)
    bestY = Y.copy()
    Xb, bv = eval_X(Y)
    t0 = time.time()
    for si, (beta, lr, budget) in enumerate(zip(betas, lrs, stage_budgets)):
        T = 1.0 / float(beta)
        stage_end = time.time() + budget
        # cap by hard end
        if stage_end > t_end_hard:
            stage_end = t_end_hard
        while time.time() < stage_end:
            step_cnt += 1
            G = Y @ Y.T
            # max of |off-diag|
            # compute max efficiently
            # set diag to -inf for max
            np.fill_diagonal(G, -np.inf)
            gmax_pos = float(np.max(G))
            gmax_neg = float(np.max(-G[np.isfinite(G)])) if np.any(np.isfinite(G)) else -np.inf
            # Actually G has -inf diag, -G has +inf diag -> need handle
            # recompute correctly: off-diag max abs
            # simpler: temp copy
            # We already filled diag -inf, so -G diag = +inf -> max_neg would be inf. Fix:
            # compute off-diag max of -G by masking
            # redo:
            # Use raw Ye Gram without fill for safety
            # Let's recompute cleanly every 1 step is cheap (90x90)
            # Do clean:
            Gr = Y @ Y.T
            # off diag mask
            off = ~np.eye(n, dtype=bool)
            gmax = float(np.max(np.abs(Gr[off])))
            Eplus = np.exp((Gr - gmax) / T)
            Eminus = np.exp((-Gr - gmax) / T)
            np.fill_diagonal(Eplus, 0.0)
            np.fill_diagonal(Eminus, 0.0)
            Z = float(np.sum(Eplus) + np.sum(Eminus)) + 1e-30
            E = Eplus - Eminus
            grad = (E @ Y) / Z
            # tangent projection
            dot = np.sum(grad * Y, axis=1, keepdims=True)
            grad = grad - dot * Y
            # Adam
            m = b1 * m + (1.0 - b1) * grad
            v = b2 * v + (1.0 - b2) * (grad * grad)
            bc1 = 1.0 - b1 ** step_cnt
            bc2 = 1.0 - b2 ** step_cnt
            mhat = m / bc1
            vhat = v / bc2
            Y = Y - lr * mhat / (np.sqrt(vhat) + eps)
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            if (step_cnt & 127) == 0:
                _, vv = eval_X(Y)
                if vv < bv:
                    bv = vv
                    bestY = Y.copy()
                if time.time() >= stage_end:
                    break
            # quick time check every 32 steps to avoid overhead? we check loop cond each iter
        # end stage: reset momentum slightly? keep
        _, vv = eval_X(Y)
        if vv < bv:
            bv = vv
            bestY = Y.copy()
        if time.time() >= t_end_hard:
            break
    return bestY, bv

def full_adam_relax(X0, seconds, T0=0.02, T1=0.0015, lr0=0.004, seed_step_offset=0):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = X.copy()
    bv = max_cos(X)
    t_end = time.time() + seconds
    step_cnt = 0
    # precompute iu not needed
    while time.time() < t_end:
        frac = 1.0  # will compute from elapsed
        # use elapsed fraction for annealing
        # approximate: we don't know total steps, use time fraction
        elapsed = seconds - (t_end - time.time())
        frac = min(1.0, max(0.0, elapsed / max(1e-9, seconds)))
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        step_cnt += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        Z = float(np.sum(E)) + 1e-30
        grad = (E @ X) / Z
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        m = b1 * m + (1.0 - b1) * grad
        vv = b2 * v + (1.0 - b2) * (grad * grad)
        v = vv
        bc1 = 1.0 - b1 ** step_cnt
        bc2 = 1.0 - b2 ** step_cnt
        mhat = m / bc1
        vhat = v / bc2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        if (step_cnt & 63) == 0:
            cur = max_cos(X)
            if cur < bv:
                bv = cur
                best = X.copy()
            if time.time() >= t_end:
                break
    cur = max_cos(X)
    if cur < bv:
        bv = cur
        best = X.copy()
    return best, bv

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    t_start = time.time()
    if deadline is None:
        deadline = t_start + seconds
    else:
        deadline = min(deadline, t_start + seconds)
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() - t_start >= seconds:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        mcur = float(np.max(c))
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 178.0
    rng = np.random.RandomState(12345)
    # fallback best
    xb = load_best()
    if xb is not None:
        bbest = max_cos(xb)
    else:
        xb = rng.randn(180, 9)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        bbest = max_cos(xb)
    # Stage A: antipodal ansatz, <=60s but leave room; use at most 55s
    anti_end = min(t0 + 55.0, HARD - 100.0)
    # if very little time, skip
    Y_best = None
    bv_anti = np.inf
    if time.time() < anti_end - 2.0:
        Y0 = rng.randn(90, 9)
        Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
        betas = [30, 100, 300, 800]
        lrs = [0.005, 0.003, 0.0015, 0.0007]
        remaining = max(1.0, anti_end - time.time())
        # split proportionally
        w = np.array([12.0, 13.0, 14.0, 16.0])
        w = w / w.sum() * remaining
        try:
            Y_best, bv_anti = antipodal_adam(Y0, betas, lrs, list(w), anti_end)
        except Exception:
            Y_best = Y0
            bv_anti = np.inf
    if Y_best is None:
        # fallback: derive Y from xb by taking 90 points (first 90) orthogonalized? just random
        Y_best = rng.randn(90, 9)
        Y_best /= np.linalg.norm(Y_best, axis=1, keepdims=True)
        X0 = np.vstack([Y_best, -Y_best])
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv_anti = max_cos(X0)
    else:
        X0 = np.vstack([Y_best, -Y_best])
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv_anti = max_cos(X0)
    # branches with symmetry breaking
    sigmas = [0.0, 1e-4, 1e-3, 3e-3]
    rng2 = np.random.RandomState(777)
    global_best = xb.copy()
    global_val = bbest
    # consider antipodal point itself
    if bv_anti < global_val:
        global_best = X0.copy()
        global_val = bv_anti
    # time budgeting for branches: divide remaining time
    # reserve ~5s final margin
    for bi, sigma in enumerate(sigmas):
        if time.time() > HARD - 5.0:
            break
        # build branch init
        if sigma == 0.0:
            Xb = X0.copy()
        else:
            noise = rng2.randn(180, 9)
            # tangent projection of noise
            dotn = np.sum(noise * X0, axis=1, keepdims=True)
            noise = noise - dotn * X0
            Xb = X0 + sigma * noise
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
        # annealed relaxation budget: aim 14s but adapt to remaining
        n_rem_branches = len(sigmas) - bi
        time_left = HARD - time.time() - 5.0
        if time_left <= 1.0:
            break
        # reserve LP time ~7s per remaining branch
        relax_budget = min(14.0, max(2.0, (time_left - n_rem_branches * 7.0) / n_rem_branches))
        # if last branch, give a bit more
        relax_budget = max(1.0, relax_budget)
        try:
            Xr, vr = full_adam_relax(Xb, seconds=relax_budget, T0=0.02, T1=0.0012, lr0=0.004)
        except Exception:
            Xr, vr = Xb, max_cos(Xb)
        if vr < global_val:
            global_val = vr
            global_best = Xr.copy()
        # LP polish budget up to 8s
        time_left2 = HARD - time.time() - 2.0
        if time_left2 <= 0.5:
            # no time for LP, keep relax result
            continue
        lp_budget = min(8.0, max(1.0, time_left2 / (n_rem_branches)))
        try:
            Xp = lp_polish(Xr, seconds=lp_budget, deadline=HARD - 1.0)
            vp = max_cos(Xp)
        except Exception:
            Xp, vp = Xr, vr
        if vp < global_val:
            global_val = vp
            global_best = Xp.copy()
        # also try LP on unrelaxed if relax hurt? already covered by global
    # final fallback: if global worse than xb, return xb
    # ensure finite
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # final safety: pick best among global_best and xb
    try:
        if max_cos(xb) < max_cos(global_best):
            return xb
    except Exception:
        pass
    return global_best
