import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def adam_smooth(X0, steps, s0, s1, lr, max_step=0.025, noise_amp=0.0, rng=None):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    if steps <= 0:
        return X
    log_ratio = np.log(s1 / s0) / max(1, steps)
    for t in range(1, steps + 1):
        s = s0 * np.exp(log_ratio * t)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        M = np.max(G)
        E = np.exp(s * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E) * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        E /= S
        Grad = E @ X
        dot = np.einsum('ij,ij->i', Grad, X)[:, None]
        Gt = Grad - dot * X
        if noise_amp > 0 and rng is not None:
            # decaying Langevin noise on tangent space
            decay = np.exp(-3.0 * t / steps)
            R = rng.normal(size=X.shape)
            rd = np.einsum('ij,ij->i', R, X)[:, None]
            Rt = R - rd * X
            Gt = Gt + (noise_amp * decay) * Rt
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        step *= lr
        nrm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        over = nrm[:, 0] > max_step
        if np.any(over):
            sc = np.ones_like(nrm)
            sc[over, 0] = max_step / nrm[over, 0]
            step *= sc
        X -= step
        nrmX = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-30
        X /= nrmX
        if not np.all(np.isfinite(X)):
            break
    return X


def refine_single(vec, others, rng, steps=150, lr=0.02, s=120.0):
    # maximise minimal angle of vec to others (minimise max dot)
    v = vec.copy()
    v /= (np.linalg.norm(v) + 1e-30)
    m = np.zeros_like(v)
    vv = np.zeros_like(v)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        dots = others @ v  # (m,)
        Mx = np.max(dots)
        w = np.exp(s * (dots - Mx))
        sw = np.sum(w)
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        grad = w @ others  # (d,)
        gt = grad - np.dot(grad, v) * v
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        st = lr * mhat / (np.sqrt(vhat) + eps)
        # clip
        nn = np.linalg.norm(st) + 1e-12
        if nn > 0.05:
            st *= 0.05 / nn
        v = v - st
        v /= (np.linalg.norm(v) + 1e-30)
    return v


def greedy_reinsert_single(others, rng, n_try=2000, refine=True):
    d = others.shape[1]
    C = rng.normal(size=(n_try, d))
    C /= (np.sqrt(np.sum(C * C, axis=1, keepdims=True)) + 1e-30)
    D = C @ others.T
    Mx = np.max(D, axis=1)
    b = int(np.argmin(Mx))
    vec = C[b]
    if refine:
        vec = refine_single(vec, others, rng, steps=120, lr=0.02, s=150.0)
    return vec


def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 175.0
    PHASE1 = t0 + 156.0
    rng = np.random.default_rng(0)
    n, d = 94, 13
    bestX = None
    best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    vv = max_cos(a)
                    if vv < best:
                        best = vv
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.normal(size=(n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = max_cos(bestX)
    cands = [(best, bestX.copy())]

    def add_cand(val, mat):
        cands.append((val, mat.copy()))
        cands.sort(key=lambda z: z[0])
        del cands[5:]

    trial = 0
    sigmas = np.array([0.008, 0.02, 0.05, 0.12, 0.25, 0.4])
    s1_choices = np.array([200.0, 350.0, 550.0, 800.0])
    lr_choices = np.array([0.003, 0.006, 0.010, 0.014])
    while time.time() < PHASE1 and (time.time() - t0) < BUDGET - 20:
        trial += 1
        r = rng.random()
        use_noise = rng.random() < 0.30
        noise_amp = float(rng.choice(np.array([0.002, 0.005, 0.012]))) if use_noise else 0.0
        if r < 0.12:
            Y = rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps = int(rng.choice(np.array([1200, 1800, 2400])))
            s1 = float(rng.choice(s1_choices))
            lr = float(rng.choice(lr_choices))
            Yopt = adam_smooth(Y, steps, 25.0, s1, lr, max_step=0.03, noise_amp=noise_amp, rng=rng)
        elif r < 0.50:
            if rng.random() < 0.65:
                base = bestX
            else:
                base = cands[int(rng.integers(len(cands)))][1]
            sigma = float(rng.choice(sigmas))
            # occasionally shake only a subset (block shake)
            Y = base + sigma * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps = int(rng.choice(np.array([1000, 1500, 2200])))
            s1 = float(rng.choice(s1_choices))
            lr = float(rng.choice(lr_choices))
            Yopt = adam_smooth(Y, steps, 30.0, s1, lr, max_step=0.025, noise_amp=noise_amp, rng=rng)
        else:
            if rng.random() < 0.60:
                base = bestX.copy()
            else:
                base = cands[int(rng.integers(len(cands)))][1].copy()
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                M = float(np.max(G))
                deg_strict = np.sum(G > M - 1e-9, axis=1)
                deg_loose = np.sum(G > M - 5e-4, axis=1)
                order_strict = np.argsort(-deg_strict)
                order_loose = np.argsort(deg_loose)
            except Exception:
                order_strict = np.arange(n)
                order_loose = np.arange(n)
                M = 0.24
            k = int(rng.integers(2, 11))
            kill_idx = []
            mode = rng.random()
            if mode < 0.45:
                # kill most jammed points
                perm = rng.permutation(len(order_strict))
                for ii in range(min(k, n)):
                    # mix top-strict with randomness
                    if rng.random() < 0.75:
                        kill_idx.append(int(order_strict[ii % n]))
                    else:
                        kill_idx.append(int(rng.integers(n)))
            elif mode < 0.75:
                # cluster kill around worst pair
                try:
                    iu0 = np.triu_indices(n, 1)
                    cc = (base @ base.T)[iu0]
                    wpos = int(np.argmax(cc))
                    a0 = int(iu0[0][wpos]); b0 = int(iu0[1][wpos])
                    # neighbours of a0,b0
                    sima = base @ base[a0]
                    simb = base @ base[b0]
                    score = np.maximum(sima, simb)
                    order = np.argsort(-score)
                    for ii in range(k):
                        kill_idx.append(int(order[ii]))
                except Exception:
                    kill_idx = [int(x) for x in rng.choice(n, size=k, replace=False)]
            else:
                # kill low-degree + random (rattler-like / loose)
                for ii in range(k):
                    if rng.random() < 0.5:
                        kill_idx.append(int(order_loose[int(rng.integers(min(12, n)))]))
                    else:
                        kill_idx.append(int(rng.integers(n)))
            # dedup
            uniq = []
            for q in kill_idx:
                if q not in uniq:
                    uniq.append(q)
            # fill up if dedup shortened
            while len(uniq) < k:
                q = int(rng.integers(n))
                if q not in uniq:
                    uniq.append(q)
            kill_idx = uniq[:k]
            Y = base.copy()
            nt_base = 2500 if k <= 4 else 1200
            for idx in kill_idx:
                others = np.delete(Y, idx, axis=0)
                Y[idx] = greedy_reinsert_single(others, rng, n_try=nt_base, refine=True)
            steps = int(rng.choice(np.array([1000, 1600, 2200])))
            s1 = float(rng.choice(np.array([250.0, 450.0, 700.0])))
            lr = float(rng.choice(np.array([0.004, 0.008, 0.012])))
            Yopt = adam_smooth(Y, steps, 30.0, s1, lr, max_step=0.025, noise_amp=noise_amp, rng=rng)
        try:
            val = max_cos(Yopt)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        if val < cands[-1][0] + 0.008 or val < best:
            add_cand(val, Yopt)
        if val < best - 1e-12:
            # quick squeeze with short LP to confirm true basin value
            try:
                if time.time() - t0 < BUDGET - 25:
                    Ys = lp_polish(Yopt, seconds=1.5, deadline=t0 + BUDGET - 3.0)
                    vals = max_cos(Ys)
                    if np.isfinite(vals) and vals < val:
                        Yopt = Ys
                        val = vals
            except Exception:
                pass
            best = val
            bestX = Yopt.copy()
        if time.time() >= PHASE1:
            break

    # final polish of best (and runner-up if time)
    cands.sort(key=lambda z: z[0])
    # ensure bestX is first
    plist = [bestX.copy()]
    for vv, mm in cands:
        if len(plist) >= 2:
            break
        try:
            if np.linalg.norm(mm - plist[0]) > 1e-7:
                plist.append(mm.copy())
        except Exception:
            pass
    remaining = (t0 + BUDGET - 2.0) - time.time()
    if remaining < 4:
        remaining = 4.0
    # give most time to best
    try:
        per_best = min(remaining * 0.75, 14.0)
        mp = lp_polish(plist[0], seconds=per_best, deadline=t0 + BUDGET - 2.0)
        v = max_cos(mp)
        if v < best and np.all(np.isfinite(mp)):
            best = v
            bestX = mp.copy()
    except Exception:
        pass
    if len(plist) > 1 and (time.time() < t0 + BUDGET - 4.0):
        try:
            left = (t0 + BUDGET - 2.0) - time.time()
            if left > 2:
                mp2 = lp_polish(plist[1], seconds=min(left, 6.0), deadline=t0 + BUDGET - 2.0)
                v2 = max_cos(mp2)
                if v2 < best and np.all(np.isfinite(mp2)):
                    best = v2
                    bestX = mp2.copy()
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    return bestX
