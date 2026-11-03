import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normed(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds, t_deadline=None):
    """Trust-region LP from parent, with deadline awareness."""
    x = _normed(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_gd(x_init, iters=2500, b0=30.0, b1=350.0, lr0=0.015, lr1=0.0015, mu=0.8):
    """Annealed tangential soft-max repulsion. Keeps best max-cos seen."""
    x = _normed(x_init.copy())
    n, d = x.shape
    vel = np.zeros_like(x)
    best_x = x.copy()
    best_v = _max_cos(x)
    lb = np.log(b0); ub = np.log(b1)
    for it in range(iters):
        f = it / max(1, iters - 1)
        beta = float(np.exp(lb + (ub - lb) * f))
        lr = float(lr0 + (lr1 - lr0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # mask far pairs for stability
        thr = m - 14.0 / beta
        # weights
        W = np.exp(beta * (G - m))
        W[G < thr] = 0.0
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        nrm = np.sqrt(np.sum(Ft * Ft, axis=1, keepdims=True)) + 1e-12
        Ft_hat = Ft / nrm
        vel = mu * vel + Ft_hat
        # damp velocity to avoid explosion
        x = x - lr * vel
        x = _normed(x)
        if (it & 63) == 0 or it == iters - 1:
            v = _max_cos(x)
            if v < best_v:
                best_v = v
                best_x = x.copy()
            # early abort if exploded
            if v > best_v + 0.05 and it > 300:
                # reset to best with smaller vel
                x = best_x.copy()
                vel *= 0.3
    return best_x

def refine_single(x, idx, iters=400, beta=250.0, lr=0.02, rng=None):
    """Move single point away from others, others fixed."""
    x = _normed(x.copy())
    for it in range(iters):
        xi = x[idx:idx+1]  # 1x8
        dots = (x @ xi.ravel())  # n
        dots[idx] = -2.0
        m = dots.max()
        w = np.exp(beta * (dots - m))
        w[idx] = 0.0
        # force = sum w_j * x_j
        F = (w[:, None] * x).sum(axis=0, keepdims=True)  # 1x8
        # tangent at xi
        d = float(F @ xi.ravel())
        Ft = F - d * xi
        nrm = float(np.sqrt((Ft * Ft).sum())) + 1e-12
        step = lr * (1.0 - 0.7 * it / iters)
        xn = xi - step * Ft / nrm
        xn = xn / np.linalg.norm(xn)
        x[idx] = xn.ravel()
    return x

def solve():
    import time as _time
    t0 = _time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(0)

    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (184, 8):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.normal(size=(184, 8))
        a = _normed(a)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    # quick initial smooth attempt (cheap, may already help)
    try:
        if DEADLINE - _time.time() > 20:
            tmp = smooth_gd(x_best, iters=1500, b0=40.0, b1=400.0, lr0=0.010, lr1=0.001, mu=0.75)
            tv = _max_cos(tmp)
            if tv < best_val:
                x_best, best_val = tmp, tv
    except Exception:
        pass

    trial = 0
    # basin-hopping phase: leave ~35s for final LP
    while _time.time() < DEADLINE - 35.0 and trial < 200:
        trial += 1
        cur = x_best.copy()
        n, d = cur.shape
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        per_max = G.max(axis=1)
        order_worst = np.argsort(-per_max)
        order_best = np.argsort(per_max)

        # schedule shake size
        if trial < 10:
            sigma = 0.05
            K = 14
        elif trial < 30:
            sigma = 0.03
            K = 10
        else:
            sigma = 0.015 + 0.02 * ((trial % 5) / 4.0)
            K = 8 + (trial % 3) * 4

        # shake worst points
        idx_shake = order_worst[:K]
        # vary subset deterministically
        if trial % 2 == 1:
            idx_shake = idx_shake[::2]
        noise = rng.normal(size=(len(idx_shake), d))
        cur[idx_shake] += sigma * noise
        cur = _normed(cur)

        # rattler / void re-insertion every other trial
        if trial % 2 == 0:
            try:
                # pick 1-2 loosest points
                n_re = 1 if (trial % 4 != 0) else 2
                for k in range(n_re):
                    # recompute loosest after previous replacement
                    Gt = cur @ cur.T
                    np.fill_diagonal(Gt, -2.0)
                    pm = Gt.max(axis=1)
                    r_idx = int(np.argmin(pm))
                    # sample void: best of random candidates
                    Nc = 4000
                    cand = rng.normal(size=(Nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    dots = cand @ cur.T
                    dots[:, r_idx] = -2.0
                    worst = dots.max(axis=1)
                    bc = cand[int(np.argmin(worst))]
                    cur[r_idx] = bc
                    cur = refine_single(cur, r_idx, iters=300, beta=200.0, lr=0.03)
            except Exception:
                pass

        # annealed repulsion
        try:
            # vary beta schedule slightly per trial
            b1 = 300.0 + 150.0 * ((trial % 3) / 2.0)
            cur_s = smooth_gd(cur, iters=2200, b0=30.0, b1=b1, lr0=0.012, lr1=0.0012, mu=0.8)
        except Exception:
            continue
        v = _max_cos(cur_s)
        if v < best_val:
            # squeeze with short LP
            remain = DEADLINE - _time.time()
            if remain > 12:
                try:
                    cur_p = lp_polish(cur_s, seconds=min(5.0, remain - 7.0), t_deadline=DEADLINE - 6.0)
                    vp = _max_cos(cur_p)
                except Exception:
                    cur_p, vp = cur_s, v
                if vp < best_val:
                    x_best, best_val = cur_p, vp
                elif v < best_val:
                    x_best, best_val = cur_s, v
            else:
                x_best, best_val = cur_s, v
        # opportunistic: if very close, occasionally LP-polish to check true basin value
        elif v < best_val + 0.0008 and (trial % 12 == 0) and (DEADLINE - _time.time() > 40):
            try:
                cur_p = lp_polish(cur_s, seconds=3.0, t_deadline=DEADLINE - 6.0)
                vp = _max_cos(cur_p)
                if vp < best_val:
                    x_best, best_val = cur_p, vp
            except Exception:
                pass
        if best_val < 0.4990:
            break

    # final polish with remaining time
    try:
        remain = DEADLINE - _time.time() - 2.0
        if remain > 3:
            x_best = lp_polish(x_best, seconds=remain, t_deadline=DEADLINE - 0.5)
            best_val = _max_cos(x_best)
    except Exception:
        pass
    x_best = _normed(x_best)
    # safety: no NaN / zero rows
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    x_best = x_best / nr
    return x_best
