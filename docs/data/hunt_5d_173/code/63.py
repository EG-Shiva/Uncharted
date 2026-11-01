import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _soft_refine(X, beta, steps, lr):
    X = X.copy()
    for _ in range(int(steps)):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            break
        WX = W @ X
        grad = WX / S
        dot = (grad * X).sum(axis=1, keepdims=True)
        Gt = grad - dot * X
        gn = float(np.linalg.norm(Gt, axis=1).max())
        if not np.isfinite(gn) or gn <= 1e-15:
            break
        X = X - (lr / gn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _lp_polish(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            part = np.argpartition(c_all[idx], len(idx) - 3000)[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[part]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_void(y0, Y, steps=250, s0=0.06):
    y = y0 / (np.linalg.norm(y0) + 1e-300)
    step = s0
    for _ in range(int(steps)):
        dots = Y @ y
        j = int(np.argmax(dots))
        d = float(dots[j])
        g = Y[j] - d * y
        gn = float(np.linalg.norm(g))
        if gn > 1e-12:
            y = y - step * (g / gn)
            yn = float(np.linalg.norm(y))
            if yn == 0 or not np.isfinite(yn):
                break
            y = y / yn
        step *= 0.995
    return y

def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    rng = np.random.default_rng(1)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (_N, _D):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        b = rng.normal(size=(_N, _D))
        best = b / np.linalg.norm(b, axis=1, keepdims=True)
        best_val = _max_cos(best)

    global_best = best.copy()
    global_val = float(best_val)
    cur = best.copy()
    cur_val = float(best_val)

    # quick soft spread to widen basin info
    try:
        cur = _soft_refine(cur, 250.0, 150, 0.03)
        cur = _soft_refine(cur, 800.0, 120, 0.015)
        v = _max_cos(cur)
        if v < cur_val:
            cur_val = v
        if v < global_val:
            global_val = v
            global_best = cur.copy()
        else:
            # keep original if soft worsened
            if v > cur_val + 1e-6:
                cur = global_best.copy()
                cur_val = global_val
    except Exception:
        pass

    # short LP to settle
    try:
        if time.time() < deadline - 30:
            cur = _lp_polish(cur, min(3.0, max(0.5, deadline - time.time() - 25)), deadline)
            cur_val = _max_cos(cur)
            if cur_val < global_val:
                global_val = cur_val
                global_best = cur.copy()
    except Exception:
        pass

    it = 0
    # basin hopping by void re-insertion
    while time.time() < deadline - 16.0:
        it += 1
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            # loose contacts for overload ranking
            m = float(per.max())
            cnt_loose = (G >= m - 1e-3).sum(axis=1)
            score = per + 1e-4 * cnt_loose
            order = np.argsort(-score)
            K = 15
            cand_top = order[:K]
            # round-robin + random among top
            if it % 3 == 0:
                i_out = int(cand_top[rng.integers(0, K)])
            else:
                i_out = int(cand_top[(it - 1) % K])
            old_x = cur[i_out].copy()
            Y = np.delete(cur, i_out, axis=0)  # (172,5)

            # void sampling
            ns = 12000
            C = rng.normal(size=(ns, _D))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            dots = C @ Y.T
            vals = dots.max(axis=1)
            nb = 8
            idx = np.argpartition(vals, nb)[:nb]
            # refine
            refined = []
            for k in idx:
                y0 = C[k]
                yr = _refine_void(y0, Y, steps=220, s0=0.06)
                # evaluate void quality
                q = float((Y @ yr).max())
                refined.append((q, yr))
            refined.sort(key=lambda t: t[0])
            # choose most diverse among good: prefer far from old_x
            chosen = refined[0][1]
            for q, yr in refined[:4]:
                if float(yr @ old_x) < 0.92:
                    chosen = yr
                    break
            # if still very close to old, perturb
            if float(chosen @ old_x) > 0.98:
                pert = chosen + 0.05 * rng.normal(size=_D)
                pert /= np.linalg.norm(pert)
                chosen = pert

            trial = Y.copy()
            # re-insert at same index to keep shapes
            trial_full = np.empty_like(cur)
            # insert: place chosen at i_out, shift? simpler: copy cur and replace row
            trial_full = cur.copy()
            trial_full[i_out] = chosen
            raw = _max_cos(trial_full)
            # quick soft polish
            t1 = _soft_refine(trial_full, 350.0, 180, 0.03)
            t1 = _soft_refine(t1, 900.0, 100, 0.012)
            v1 = _max_cos(t1)
            # gate LP: only if promising
            # accept tolerance decays
            progress = min(1.0, (time.time() - t0) / 150.0)
            gate = cur_val + 0.004 * (1.0 - progress) + 0.0002
            if v1 < gate:
                rem = deadline - time.time()
                lp_sec = 0.9 if (v1 < cur_val + 1e-4) else 0.5
                lp_sec = min(lp_sec, max(0.2, rem - 14.0))
                if lp_sec > 0.15:
                    t1 = _lp_polish(t1, lp_sec, deadline)
                    v1 = _max_cos(t1)
            # update global
            if v1 < global_val - 1e-12:
                global_val = v1
                global_best = t1.copy()
                cur = t1.copy()
                cur_val = v1
            else:
                # sideways acceptance to change basin
                accept_tol = 0.0012 * (1.0 - progress) + 1e-5
                if v1 < cur_val + accept_tol:
                    # avoid cycling back to same basin: require some movement
                    cur = t1.copy()
                    cur_val = v1
                else:
                    # occasional random restart of cur to global with kick
                    if rng.random() < 0.06:
                        sig = 0.02
                        xk = global_best + sig * rng.normal(size=global_best.shape)
                        xk /= np.linalg.norm(xk, axis=1, keepdims=True)
                        xk = _soft_refine(xk, 300.0, 120, 0.03)
                        vk = _max_cos(xk)
                        if vk < cur_val + 0.002:
                            cur = xk
                            cur_val = vk
            # safety: if cur drifts too bad, reset to global
            if cur_val > global_val + 0.004:
                cur = global_best.copy()
                cur_val = global_val
        except Exception:
            continue

    # final polish of global best
    try:
        rem = deadline - time.time()
        if rem > 3.0:
            # soft + long LP
            global_best = _soft_refine(global_best, 500.0, 150, 0.02)
            global_best = _soft_refine(global_best, 1200.0, 100, 0.01)
            global_best = _lp_polish(global_best, max(1.0, rem - 1.0), deadline)
    except Exception:
        pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
