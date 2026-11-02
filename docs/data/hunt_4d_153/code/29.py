import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x, iu0=None, iu1=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, budget, iu):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu0, iu1 = iu
    try:
        best = _max_cos(x)
    except Exception:
        best = 1.0
    radius = 1e-3
    start = time.time()
    while time.time() - start < budget and radius > 1e-14:
        if time.time() - start >= budget:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
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
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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

def _feas_adam(x0, t, steps, lr0, lr1, iu, deadline, p=4):
    iu0, iu1 = iu
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    # initial E
    G = x @ x.T
    gv0 = G[iu0, iu1] - t
    pos0 = gv0[gv0 > 0]
    best_E = float((pos0 ** p).sum()) if pos0.size else 0.0
    best_x = x.copy()
    lrat = np.log(lr1 / lr0) / max(steps - 1, 1)
    for s in range(steps):
        if (s & 255) == 0 and time.time() > deadline:
            break
        lr = lr0 * np.exp(lrat * s)
        G = x @ x.T
        gv = G[iu0, iu1] - t
        mask = gv > 0
        if not np.any(mask):
            best_x = x.copy()
            best_E = 0.0
            break
        vv = gv[mask]
        Ecur = float((vv ** p).sum())
        if Ecur < best_E:
            best_E = Ecur
            best_x = x.copy()
        w = p * (vv ** (p - 1))
        I = iu0[mask]; J = iu1[mask]
        grad = np.zeros_like(x)
        # accumulate: grad[I]+=w*x[J], grad[J]+=w*x[I]
        np.add.at(grad, I, w[:, None] * x[J])
        np.add.at(grad, J, w[:, None] * x[I])
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return best_x, best_E

def _kick_viol(x, t, sigma, rng, iu, nrep_extra=False):
    iu0, iu1 = iu
    y = x.copy()
    n = x.shape[0]
    G = x @ x.T
    gv = G[iu0, iu1] - t
    mask = gv > 0
    if np.any(mask):
        I = iu0[mask]; J = iu1[mask]
        vv = gv[mask]
        cnt = np.zeros(n)
        np.add.at(cnt, I, vv)
        np.add.at(cnt, J, vv)
        order = np.argsort(-cnt)
        # kick top violators harder
        k = min(n, max(4, int(mask.sum() // 2) + 2))
        worst = order[:k]
        y[worst] += rng.normal(0, sigma, size=(len(worst), 4))
        # also random replacement of single worst occasionally
        if nrep_extra and rng.random() < 0.5:
            w0 = order[0]
            y[w0] = rng.normal(size=4)
    else:
        # feasible: small global jitter
        y += rng.normal(0, sigma * 0.3, size=y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    HARD = 172.0
    deadline_main = t0 + HARD
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 153, 4
    iu0, iu1 = np.triu_indices(n, 1)
    iu = (iu0, iu1)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    # continuation schedule strictly decreasing, step 2e-4 from above record
    sched = [0.8492, 0.8490, 0.8488, 0.8486, 0.8484, 0.8482, 0.8480, 0.8478, 0.8475, 0.8470]
    # filter to those below current best+slack and above hard floor
    x_cur = x_best.copy()
    # quick feasibility check: start at first t that is infeasible
    # keep x_cur feasible for previous t
    for ti, t in enumerate(sched):
        if time.time() > deadline_main - 30:
            break
        # skip targets already satisfied with margin? still need to try
        # warm-start attempt
        steps = 2500 if t >= 0.8482 else 3200
        dl = min(deadline_main - 26, time.time() + 12)
        y, E = _feas_adam(x_cur, t, steps, 0.008, 0.0004, iu, dl, p=4)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best, best_val = y.copy(), v
        if E == 0.0 or v <= t + 1e-12:
            x_cur = y.copy()
            # short LP polish to gain slack for next stage
            if time.time() < deadline_main - 28:
                yp, vp = _lp_polish(x_cur, 1.2, iu)
                if vp < best_val:
                    x_best, best_val = yp.copy(), vp
                # use polished for continuation if still feasible-ish
                if vp <= t + 5e-4:
                    x_cur = yp.copy()
            continue
        # stalled: targeted kicks on violating endpoints, resume
        improved = False
        cur_y, cur_E, cur_v = y, E, v
        for k in range(3):
            if time.time() > deadline_main - 28:
                break
            sigma = 0.02 + 0.02 * k
            yk0 = _kick_viol(cur_y, t, sigma, rng, iu, nrep_extra=(k >= 1))
            dl2 = min(deadline_main - 26, time.time() + 8)
            yk, Ek = _feas_adam(yk0, t, 2000, 0.006, 0.0003, iu, dl2, p=4)
            vk = _max_cos(yk)
            if vk < best_val - 1e-12:
                x_best, best_val = yk.copy(), vk
            if Ek < cur_E:
                cur_y, cur_E, cur_v = yk, Ek, vk
            if Ek == 0.0 or vk <= t + 1e-12:
                x_cur = yk.copy()
                improved = True
                if time.time() < deadline_main - 28:
                    yp, vp = _lp_polish(x_cur, 1.0, iu)
                    if vp < best_val:
                        x_best, best_val = yp.copy(), vp
                    if vp <= t + 5e-4:
                        x_cur = yp.copy()
                break
        if improved:
            continue
        # fresh random code attempt for this t
        if time.time() < deadline_main - 30:
            xr = rng.normal(size=(n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            dl3 = min(deadline_main - 26, time.time() + 8)
            yr, Er = _feas_adam(xr, t, 2500, 0.010, 0.0005, iu, dl3, p=4)
            vr = _max_cos(yr)
            if vr < best_val - 1e-12:
                x_best, best_val = yr.copy(), vr
            if Er == 0.0 or vr <= t + 1e-12:
                x_cur = yr.copy()
                continue
        # cannot solve this t: keep best-E state as cur for further tries but do not go lower
        # do a couple more random sweeps at this same t if time allows, else break to final sweep
        # update cur to best violating state to allow barrier crossing
        x_cur = cur_y.copy()
        # if this t is far below best, stop schedule; go to broad sweep
        break

    # broad sweep of lowest targets: spend remaining budget on feasibility just below best
    while time.time() < deadline_main - 24:
        # target just below current best
        t_try = best_val - 0.00015
        # alternate warm perturbed best vs fresh random
        r = rng.random()
        if r < 0.6:
            # perturbed best / cur with mixed sigma
            base = x_best if rng.random() < 0.7 else x_cur
            sigma = float(rng.choice([0.01, 0.03, 0.06, 0.12]))
            y0 = base + rng.normal(0, sigma, size=base.shape)
            # occasionally reinsert 1-2 points
            if rng.random() < 0.4:
                rep = rng.choice(n, size=int(rng.integers(1, 3)), replace=False)
                y0[rep] = rng.normal(size=(len(rep), 4))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            steps = 1800
            lr0, lr1 = 0.007, 0.0004
        else:
            y0 = rng.normal(size=(n, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            steps = 2200
            lr0, lr1 = 0.010, 0.0005
        dl = min(deadline_main - 22, time.time() + 6)
        y, E = _feas_adam(y0, t_try, steps, lr0, lr1, iu, dl, p=4)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            x_best, best_val = y.copy(), v
            x_cur = y.copy()
            # quick LP snap on new record
            if time.time() < deadline_main - 26:
                yp, vp = _lp_polish(x_best, 1.0, iu)
                if vp < best_val:
                    x_best, best_val = yp.copy(), vp
        elif E < 1e-10:
            x_cur = y.copy()
        # targeted resume if close (few violations)
        G = y @ y.T
        cnt_viol = int(((G[iu0, iu1] - t_try) > 0).sum())
        if 0 < cnt_viol <= 30 and time.time() < deadline_main - 26:
            yk0 = _kick_viol(y, t_try, 0.025, rng, iu, nrep_extra=True)
            dl2 = min(deadline_main - 22, time.time() + 4)
            yk, Ek = _feas_adam(yk0, t_try, 1200, 0.005, 0.0003, iu, dl2, p=4)
            vk = _max_cos(yk)
            if vk < best_val - 1e-12:
                x_best, best_val = yk.copy(), vk
                x_cur = yk.copy()

    # final polish of incumbent with remaining time
    remain = deadline_main - (time.time()) - 1.5
    if remain > 2.0:
        remain = min(remain, 20.0)
        try:
            yp, vp = _lp_polish(x_best, remain, iu)
            if vp < best_val:
                x_best, best_val = yp, vp
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
