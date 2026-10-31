import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)


def _smooth_quench(x0, steps, tau0, tau1, lr0, lr1, beta=0.9):
    x = x0.copy()
    n = x.shape[0]
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gm = float(G.max())
        if Gm < bv:
            bv = Gm
            bx = x.copy()
        E = np.exp((G - Gm) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        np.maximum(rs, 1e-30, out=rs)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        x /= nrm
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def _lp_polish_to_deadline(x, deadline, init_radius=1e-3, max_pairs=3500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_full = G[iu0]
        m = float(c_full.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c_full > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > max_pairs:
            # keep largest
            part = np.argpartition(c_full[idx], -max_pairs)[-max_pairs:]
            idx = idx[part]
        I = iu0[0][idx]
        J = iu0[1][idx]
        csel = c_full[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return best_x, best


def _largest_hole(x, rng, samples=8000):
    d = x.shape[1]
    s = rng.standard_normal((samples, d))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    # max dot to x
    md = s @ x.T
    m = md.max(axis=1)
    i = int(np.argmin(m))
    return s[i]


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d = 153, 4

    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal((n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)

    cur_x = best_x.copy()
    cur_v = best_v

    T0, T1 = 0.0009, 0.00002
    it = 0

    STEPS = 750
    TAU0, TAU1 = 0.022, 0.0012
    LR0, LR1 = 0.055, 0.004

    # precompute iu for worst pair search reuse
    while time.time() < EXPLORE_END and time.time() < HARD_END - 15:
        it += 1
        elapsed = time.time() - t0
        frac = min(1.0, max(0.0, elapsed / max(1e-9, (EXPLORE_END - t0))))
        Temp = T0 * ((T1 / T0) ** frac)

        # base: mostly walk from cur, sometimes restart from best
        if it % 7 == 0:
            base = best_x
        else:
            base = cur_x
        y = base.copy()
        mode = int(rng.integers(0, 6))

        if mode == 0:
            # global log-uniform sigma
            ls = rng.uniform(np.log(0.006), np.log(0.40))
            sigma = float(np.exp(ls))
            noise = rng.standard_normal((n, d)) * sigma
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise -= dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            # subset re-randomize
            k = int(rng.integers(4, 26))
            idx = rng.choice(n, size=k, replace=False)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
            # small jitter to rest
            noise = rng.standard_normal((n, d)) * 0.008
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise -= dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            # subset big kick
            k = int(rng.integers(8, 32))
            idx = rng.choice(n, size=k, replace=False)
            sigma = float(rng.uniform(0.25, 0.7))
            noise = rng.standard_normal((k, d)) * sigma
            dot = np.sum(noise * y[idx], axis=1, keepdims=True)
            noise -= dot * y[idx]
            y[idx] = y[idx] + noise
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            # tiny global jitter
            g = rng.standard_normal((n, d)) * 0.005
            dot2 = np.sum(g * y, axis=1, keepdims=True)
            g -= dot2 * y
            y = y + g
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 3:
            # hole insertion: move worst / rattler to largest hole
            try:
                rm = _row_max(y)
            except Exception:
                rm = None
            if rm is not None:
                # pick among worst 8 or best (rattler) 3 at random
                order = np.argsort(rm)
                if rng.random() < 0.6:
                    victim = int(rng.choice(order[-8:]))
                else:
                    victim = int(rng.choice(order[:4]))
            else:
                victim = int(rng.integers(0, n))
            hole = _largest_hole(y, rng, samples=6000)
            # jitter hole a bit
            hole = hole + rng.standard_normal(d) * 0.02
            hole /= np.linalg.norm(hole)
            y[victim] = hole
            sigma = float(rng.uniform(0.01, 0.05))
            noise = rng.standard_normal((n, d)) * sigma
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise -= dot * y
            y = y + noise
            # keep victim close to hole: reduce its noise
            y[victim] = hole + rng.standard_normal(d) * 0.005
            y[victim] /= np.linalg.norm(y[victim])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 4:
            # worst-pair split + global noise
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = divmod(flat, n)
            # push apart along tangent
            s = float(rng.uniform(0.05, 0.22))
            vi = y[ii] - G[ii, jj] * y[jj]
            vj = y[jj] - G[ii, jj] * y[ii]
            ni = np.linalg.norm(vi) + 1e-12
            nj = np.linalg.norm(vj) + 1e-12
            vi /= ni
            vj /= nj
            y[ii] = y[ii] + s * vi
            y[jj] = y[jj] + s * vj
            y[ii] /= np.linalg.norm(y[ii])
            y[jj] /= np.linalg.norm(y[jj])
            sigma = float(rng.uniform(0.01, 0.08))
            noise = rng.standard_normal((n, d)) * sigma
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise -= dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # coherent non-rigid shear field
            M = rng.standard_normal((d, d)) * 0.35
            # tangent vector field v_k = M x_k - (x_k^T M x_k) x_k
            Mx = y @ M.T
            coeff = np.sum(Mx * y, axis=1, keepdims=True)
            V = Mx - coeff * y
            eps = float(rng.uniform(0.15, 0.5))
            y = y + eps * V
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            sigma = float(rng.uniform(0.005, 0.05))
            noise = rng.standard_normal((n, d)) * sigma
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise -= dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # smooth quench
        bx, bv = _smooth_quench(y, STEPS, TAU0, TAU1, LR0, LR1, beta=0.88)

        # decide whether worth LP-evaluating: skip if far worse
        # allow larger tolerance early
        tol_skip = 0.006 if frac < 0.5 else 0.0035
        ref = min(cur_v, best_v)
        if bv > ref + tol_skip:
            # still occasionally evaluate to allow big jumps? rarely
            if rng.random() > 0.05:
                # Metropolis on unpolished value to update cur? no, skip
                continue

        # short LP evaluation to get true basin value
        lp_time = 0.45
        # shrink if near end
        remain_explore = EXPLORE_END - time.time()
        if remain_explore < 3:
            break
        lp_deadline = time.time() + min(lp_time, remain_explore - 0.2)
        try:
            px, pv = _lp_polish_to_deadline(bx, lp_deadline, init_radius=1.2e-3)
        except Exception:
            continue
        # use polished value
        bx, bv = px, pv

        # update best
        if bv < best_v:
            best_v = float(bv)
            best_x = bx.copy()
            cur_x, cur_v = bx.copy(), float(bv)
        else:
            # Metropolis acceptance for cur walk (allows uphill)
            if bv <= cur_v:
                cur_x, cur_v = bx.copy(), float(bv)
            else:
                gap = float(bv - cur_v)
                # cap drift: if cur already bad, force reset to best
                if cur_v > best_v + 0.004 and bv > cur_v:
                    # reset to best with some prob
                    if rng.random() < 0.7:
                        cur_x, cur_v = best_x.copy(), best_v
                    continue
                prob = float(np.exp(-gap / max(1e-12, Temp)))
                if rng.random() < prob:
                    cur_x, cur_v = bx.copy(), float(bv)
        # adapt steps if time low (speed up)
        if time.time() > EXPLORE_END - 25 and STEPS > 500:
            STEPS = 500

    # final long polish of best
    remain = HARD_END - time.time() - 1.5
    if remain > 2:
        use = min(remain, 42.0)
        deadline = time.time() + use
        try:
            best_x, best_v2 = _lp_polish_to_deadline(best_x, deadline, init_radius=1e-3)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
