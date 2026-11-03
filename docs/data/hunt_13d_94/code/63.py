import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _tangent_perturb(X, sigma, rng):
    N = rng.standard_normal(size=X.shape)
    dot = (N * X).sum(axis=1, keepdims=True)
    Nt = N - dot * X
    nrm = np.linalg.norm(Nt, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    # scale to have rms tangent norm ~ sigma
    Y = X + sigma * Nt
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _adam_quantile(x, steps, M, lr0, lr1, iu0, iu1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        vals = G[iu0, iu1]
        if M >= vals.size:
            t = float(vals.min())
        else:
            t = float(np.partition(vals, -M)[-M])
        # hinge weights above t
        H = G - t
        H[H < 0] = 0.0
        np.fill_diagonal(H, 0.0)
        cnt = float((H > 0).sum() * 0.5)
        if cnt < 1.0:
            cnt = 1.0
        Grad = (H @ x) / cnt
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _adam_lse(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eye = np.eye(n)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - eye * 2.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _greedy_best_vec(Xkeep, rng, n_samp=15000):
    n, d = Xkeep.shape
    best_val = 1e9
    best_vec = None
    chunk = 5000
    done = 0
    while done < n_samp:
        m = min(chunk, n_samp - done)
        C = rng.standard_normal((m, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = (C @ Xkeep.T).max(axis=1)
        j = int(np.argmin(S))
        if float(S[j]) < best_val:
            best_val = float(S[j])
            best_vec = C[j].copy()
        done += m
    return best_vec

def solve():
    T0 = time.time()
    BUDGET = 172.0
    END = T0 + BUDGET
    HOP_END = T0 + 138.0
    rng = np.random.default_rng(0)
    # load incumbent
    incumb = None
    incumb_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < incumb_val:
                    incumb_val = v
                    incumb = a.copy()
        except Exception:
            pass
    if incumb is None:
        incumb = rng.standard_normal((94, 13))
        incumb /= np.linalg.norm(incumb, axis=1, keepdims=True)
        incumb_val = _max_cos(incumb)
    n, d = 94, 13
    iu0, iu1 = np.triu_indices(n, 1)
    best_x = incumb.copy()
    best_v = incumb_val
    # pool of top solutions for diversity
    pool = [(best_v, best_x.copy())]

    sigmas = [0.03, 0.07, 0.13, 0.22, 0.35, 0.55, 0.80]
    Ms = [800, 600, 400, 250, 600, 800, 350]
    it = 0
    while time.time() < HOP_END and it < 500:
        it += 1
        sig = sigmas[(it - 1) % len(sigmas)]
        M = Ms[(it - 1) % len(Ms)]
        # choose base: mostly best, sometimes 2nd best / incumb for diversity
        if len(pool) > 1 and (it % 7 == 0):
            # perturb second best
            base = pool[1][1]
        else:
            base = best_x
        mode = it % 5
        try:
            if mode in (0, 1, 2, 3):
                # full jitter
                x0 = _tangent_perturb(base, sig, rng)
                # stage 1: crush plateau
                x1 = _adam_quantile(x0, steps=450, M=M, lr0=0.012, lr1=0.0012, iu0=iu0, iu1=iu1)
                # stage 2: sharpen max
                # vary sharpness
                if it % 2 == 0:
                    x2 = _adam_lse(x1, steps=350, k0=80.0, k1=650.0, lr0=0.006, lr1=0.0009)
                else:
                    x2 = _adam_lse(x1, steps=350, k0=200.0, k1=1500.0, lr0=0.004, lr1=0.0007)
            else:
                # worst-point excision + greedy reinsert (topology change)
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                # remove 1 or 2 worst points
                krm = 1 if (it % 10 != 4) else 2
                idx = np.argsort(rm)[-krm:]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[idx] = False
                Xkeep = base[keep_mask]
                for _ in range(krm):
                    v = _greedy_best_vec(Xkeep, rng, n_samp=15000)
                    Xkeep = np.vstack([Xkeep, v.reshape(1, -1)])
                x0 = Xkeep
                x1 = _adam_quantile(x0, steps=400, M=M, lr0=0.012, lr1=0.0012, iu0=iu0, iu1=iu1)
                x2 = _adam_lse(x1, steps=350, k0=100.0, k1=800.0, lr0=0.005, lr1=0.0008)
            v2 = _max_cos(x2)
            # keep pool of 3 distinct good
            if v2 < pool[-1][0] + 5e-06 or v2 < best_v:
                # insert sorted
                pool.append((v2, x2.copy()))
                pool = sorted(pool, key=lambda t: t[0])[:3]
            if v2 < best_v - 1e-12:
                best_x = x2.copy()
                best_v = v2
            # opportunistic: if time low break
            if time.time() > HOP_END:
                break
        except Exception:
            continue

    # final polish: LP polish top pool members
    # order pool best first
    pool = sorted(pool, key=lambda t: t[0])
    # ensure incumb in pool
    pool.append((incumb_val, incumb.copy()))
    # dedup roughly: keep best 2 with different gram spectra
    cands = [pool[0][1]]
    cvals = [pool[0][0]]
    for vv, xx in pool[1:]:
        try:
            # distance check via max cos diff or frobenius of gram sorted
            if abs(vv - cvals[0]) > 1e-9:
                cands.append(xx); cvals.append(vv)
                break
        except Exception:
            pass
    if len(cands) == 1 and len(pool) > 1:
        cands.append(pool[1][1]); cvals.append(pool[1][0])

    t_rem = max(2.0, END - time.time() - 1.0)
    per = t_rem / max(1, len(cands))
    polished = []
    polished_vals = []
    for xx in cands:
        dl = min(END - 0.5, time.time() + per)
        try:
            # pre-sharpen briefly
            if time.time() < dl - 2.0:
                xx = _adam_quantile(xx, steps=200, M=500, lr0=0.004, lr1=0.001, iu0=iu0, iu1=iu1)
            if time.time() < dl - 1.0:
                xx = _adam_lse(xx, steps=250, k0=200.0, k1=1200.0, lr0=0.003, lr1=0.0006)
            x2, v2 = _lp_polish_deadline(xx, dl)
            # post-sharpen + re-polish if time
            if time.time() < dl - 1.0:
                x2 = _adam_lse(x2, steps=150, k0=300.0, k1=1500.0, lr0=0.002, lr1=0.0005)
                x2, v2 = _lp_polish_deadline(x2, dl)
            polished.append(x2); polished_vals.append(v2)
        except Exception:
            polished.append(xx); polished_vals.append(_max_cos(xx))
    polished.append(best_x); polished_vals.append(best_v)
    polished.append(incumb); polished_vals.append(incumb_val)
    bi = int(np.argmin(np.array(polished_vals)))
    out = polished[bi] / np.linalg.norm(polished[bi], axis=1, keepdims=True)
    # safety: return best found
    try:
        if not np.all(np.isfinite(out)):
            return incumb
        return out
    except Exception:
        return incumb
