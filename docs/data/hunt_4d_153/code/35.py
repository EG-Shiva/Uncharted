import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 3e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _feas_adam(x0, t, steps, lr0, lr1, check_every=100):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # track lowest max along trajectory
    best_x = x.copy()
    # compute initial max
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    if best_m <= t:
        return x, best_m, best_x, True
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    feasible = False
    for s in range(steps):
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        # violation, ignore diagonal (self dot =1)
        D = G - t
        # zero diagonal in place
        np.fill_diagonal(D, -1.0)
        # relu
        # use in-place maximum for speed
        np.maximum(D, 0.0, out=D)
        # coeff = 2*D + 4*D^3  (grad of sum relu^2+relu^4)
        # D3 = D^3 where D>0
        # to avoid extra alloc: C = D*(2+4*D*D)
        C = D * (2.0 + 4.0 * D * D)
        grad = C @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1.0 - b1) * gt
        v1 = b2 * v1 + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        # re-project step to tangent
        sdot = (step * x).sum(axis=1, keepdims=True)
        step = step - sdot * x
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
        if (s + 1) % check_every == 0 or (s + 1) == steps:
            Gc = x @ x.T
            np.fill_diagonal(Gc, -2.0)
            cm = float(Gc.max())
            if cm < best_m:
                best_m = cm
                best_x = x.copy()
            if cm <= t:
                feasible = True
                # keep going a bit for margin? break to let outer lower t
                # do small extra margin push: break
                break
            # early abort if completely stuck with huge max? continue
    # final eval
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    fm = float(Gf.max())
    if fm < best_m:
        best_m = fm
        best_x = x.copy()
    if fm <= t or best_m <= t:
        feasible = True
    return x, fm, best_x, feasible

def solve():
    t0 = time.time()
    HARD_END = 176.0
    rng = np.random.default_rng(777)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    global_val = _max_cos(xb)
    global_best = xb.copy()
    n, d = 153, 4

    # current chain point (best feasible for higher target)
    cur_x = global_best.copy()
    cur_val = global_val
    # start just below record
    cur_t = min(0.8490, global_val - 0.00005)
    # if global already below 0.8490, start lower
    if global_val < 0.8490:
        cur_t = global_val - 0.00015

    # ladder step
    STEP = 0.0002

    it = 0
    # diverse sigma schedule
    sigmas = [0.008, 0.02, 0.05, 0.10, 0.20, 0.03, 0.07, 0.15]
    while time.time() - t0 < 150.0:
        if time.time() - t0 > HARD_END - 28:
            break
        it += 1
        # choose base and sigma
        cyc = (it - 1) % 10
        fresh = (cyc == 9) or (cyc == 5 and (it % 3 == 0))
        if fresh:
            x0 = rng.normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = 1500
            lr0, lr1 = 0.02, 0.002
        else:
            sigma = sigmas[(it - 1) % len(sigmas)]
            # alternate base between cur_x and global_best
            base = cur_x if (it % 2 == 1) else global_best
            x0 = base + rng.normal(0, sigma, size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # larger sigma -> more steps
            if sigma >= 0.1:
                steps = 1300
            else:
                steps = 1000
            lr0, lr1 = 0.015, 0.0015
            # vary lr slightly
            if it % 4 == 0:
                lr0 = 0.03

        # pick target: mostly cur_t, sometimes lower jump
        t_try = cur_t
        if it % 11 == 0:
            t_try = cur_t - 0.0003  # probe lower
        # clamp to sensible range
        if t_try < 0.8455:
            t_try = 0.8475

        x_end, fmax, x_traj_best, feas = _feas_adam(x0, t_try, steps, lr0, lr1)

        # track trajectory best (lowest max seen)
        Gt = x_traj_best @ x_traj_best.T
        np.fill_diagonal(Gt, -2.0)
        tbest = float(Gt.max())
        # candidates to consider
        for cand, cv in ((x_end, fmax), (x_traj_best, tbest)):
            if cv < global_val:
                global_val = cv
                global_best = cand.copy() / np.linalg.norm(cand, axis=1, keepdims=True)

        # if feasible for t_try -> advance chain
        # feasibility means max <= t_try (allow tiny tol 2e-9)
        best_for_try = min(fmax, tbest)
        best_x_for_try = x_end if fmax <= tbest else x_traj_best
        if best_for_try <= t_try + 2e-9:
            # short LP polish to gain margin
            rem_lp = 1.0
            if time.time() - t0 < HARD_END - 30:
                try:
                    yp, vp = _lp_polish(best_x_for_try, rem_lp)
                except Exception:
                    yp, vp = best_x_for_try, best_for_try
                if vp < global_val:
                    global_val = vp
                    global_best = yp.copy()
                    best_x_for_try, best_for_try = yp, vp
            cur_x = best_x_for_try.copy()
            cur_val = best_for_try
            # lower target
            cur_t = min(cur_t - STEP, best_for_try - 0.00012)
            # don't go absurdly low too fast
            if cur_t < 0.8455:
                cur_t = 0.8455
            continue

        # not feasible: maybe LP can close small gap
        gap = best_for_try - t_try
        if gap < 0.001 and best_for_try < global_val + 0.002:
            if time.time() - t0 < HARD_END - 30:
                try:
                    yp, vp = _lp_polish(best_x_for_try, 0.8)
                except Exception:
                    yp, vp = best_x_for_try, best_for_try
                if vp < global_val:
                    global_val = vp
                    global_best = yp.copy()
                if vp <= t_try + 2e-9:
                    cur_x = yp.copy()
                    cur_val = vp
                    cur_t = min(cur_t - STEP, vp - 0.00012)
                    continue

        # targeted kick resume if small violating set
        # estimate violations at x_end
        if gap < 0.002 and gap > 0:
            Ge = x_end @ x_end.T
            np.fill_diagonal(Ge, -2.0)
            viol_mask = Ge > t_try
            nviol = int(viol_mask.sum() // 2)
            if 1 <= nviol <= 120:
                # find endpoints: per-point worst excess
                per = (Ge * viol_mask).max(axis=1)
                # nodes involved
                involved = np.where(per > 0)[0]
                # kick only involved nodes
                y0 = x_end.copy()
                kick_sigma = 0.04 if nviol < 30 else 0.08
                y0[involved] += rng.normal(0, kick_sigma, size=(len(involved), d))
                y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
                if time.time() - t0 < HARD_END - 28:
                    y_end, yf, y_best, yfeas = _feas_adam(y0, t_try, 600, 0.015, 0.002)
                    Gy = y_best @ y_best.T
                    np.fill_diagonal(Gy, -2.0)
                    yb = float(Gy.max())
                    if yf < global_val:
                        global_val = yf
                        global_best = y_end.copy() / np.linalg.norm(y_end, axis=1, keepdims=True)
                    if yb < global_val:
                        global_val = yb
                        global_best = y_best.copy() / np.linalg.norm(y_best, axis=1, keepdims=True)
                    if min(yf, yb) <= t_try + 2e-9:
                        bx = y_end if yf <= yb else y_best
                        bv = min(yf, yb)
                        # polish
                        try:
                            yp2, vp2 = _lp_polish(bx, 0.8)
                        except Exception:
                            yp2, vp2 = bx, bv
                        if vp2 < global_val:
                            global_val = vp2
                            global_best = yp2.copy()
                            bx, bv = yp2, vp2
                        cur_x = bx.copy()
                        cur_val = bv
                        cur_t = min(cur_t - STEP, bv - 0.00012)
                        continue
        # if we improved global but not feasible, optionally move cur_x toward better region?
        # keep cur_x as is (feasible for higher t). Occasionally update cur_x to global_best if global improved a lot
        # to avoid stagnation, if global_val < cur_val - 0.0003, set cur_x = global_best and cur_t = global_val - 0.0001
        if global_val < cur_val - 1e-4 and global_val < cur_t + 0.0005:
            # adopt global as new chain if it is close to cur_t
            if global_val <= cur_t + 0.0004:
                cur_x = global_best.copy()
                cur_val = global_val
                # keep cur_t (don't lower aggressively)
                # nudge cur_t down a bit if global close
                if global_val <= cur_t + 1e-4:
                    cur_t = global_val - 0.00012

    rem = HARD_END - (time.time() - t0)
    if rem > 1.0:
        try:
            global_best, global_val = _lp_polish(global_best, min(rem, 12.0))
        except Exception:
            pass
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
