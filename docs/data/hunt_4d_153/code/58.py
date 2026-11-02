import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _rewire_smooth(x0, Iu, Ju, broken_idx, target, steps, beta0, beta1, lr0, lr1, mu, rng):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    M = Iu.shape[0]
    has_broken = broken_idx.size > 0
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        # softmax weights
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        # penalty weights for broken edges with hinge (g-target)+
        if has_broken:
            gb = g[broken_idx]
            r = gb - target
            pos = r > 0.0
            if np.any(pos):
                wp_vals = 2.0 * mu * r[pos]
                # build combined weight matrix
                W = np.zeros((n, n), dtype=float)
                W[Iu, Ju] = w
                W = W + W.T
                # add penalty contributions
                bi = broken_idx[pos]
                ii = Iu[bi]
                jj = Ju[bi]
                # accumulate (multiple edges distinct, no collision)
                np.add.at(W, (ii, jj), wp_vals)
                np.add.at(W, (jj, ii), wp_vals)
            else:
                W = np.zeros((n, n), dtype=float)
                W[Iu, Ju] = w
                W = W + W.T
        else:
            W = np.zeros((n, n), dtype=float)
            W[Iu, Ju] = w
            W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1)
        c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x


def _lp_polish_exclude(x, budget, exclude_mask, IuJu=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1) if IuJu is None else IuJu
    Iu0, Ju0 = iu
    has_ex = exclude_mask is not None
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[Iu0, Ju0]
        m = c_all.max()
        thr = max(20 * radius, 1e-9)
        sel = c_all > m - thr
        if has_ex:
            # broken edges excluded unless they rise above max-1e-6
            ex = exclude_mask
            # only for selected excluded ones, check re-add
            # sel[ex] currently True if within thr; force False unless > m-1e-6
            readd = c_all > m - 1e-6
            sel = sel & (~ex | readd)
        I = Iu0[sel]
        J = Ju0[sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        if time.time() - start >= budget:
            break
    return x, best


def solve():
    t0 = time.time()
    LIMIT = 175.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 153, 4
    Iu, Ju = np.triu_indices(n, 1)
    M = Iu.shape[0]
    best = _max_cos(xb)
    # also check record
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float).reshape(153, 4)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best:
            best = vr
            xb = xr.copy()
    except Exception:
        pass
    x_best = xb.copy()
    x_cur = xb.copy()
    m_cur = best
    taboo = np.full(M, -100, dtype=int)
    deltas = [3e-4, 1e-3, 3e-3]
    it = 0
    # main rewiring loop, reserve ~12s for final polish
    while time.time() - t0 < LIMIT - 14.0:
        it += 1
        remain = LIMIT - 14.0 - (time.time() - t0)
        if remain < 3.0:
            break
        Gc = x_cur @ x_cur.T
        gall = Gc[Iu, Ju]
        mc = float(gall.max())
        contact_mask = gall > mc - 1e-9
        contact_pos = np.where(contact_mask)[0]
        if contact_pos.size < 15:
            # take top 60 closest if degenerate
            contact_pos = np.argsort(-gall)[:60]
            contact_mask = np.zeros(M, dtype=bool)
            contact_mask[contact_pos] = True
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, Iu[contact_mask], 1)
        np.add.at(cnt, Ju[contact_mask], 1)
        num = int(rng.integers(15, 41))
        num = min(num, int(contact_pos.size))
        scores = cnt[Iu[contact_pos]] + cnt[Ju[contact_pos]]
        w = scores.astype(float) ** 2 + 0.5
        # taboo: do not re-pick edges broken in last 2 iterations
        cool = (it - taboo[contact_pos]) <= 2
        w[cool] = 0.0
        if w.sum() <= 0:
            w = np.ones_like(w, dtype=float)
        p = w / w.sum()
        try:
            chosen = rng.choice(contact_pos, size=num, replace=False, p=p)
        except Exception:
            chosen = rng.choice(contact_pos, size=num, replace=False)
        taboo[chosen] = it
        broken_mask = np.zeros(M, dtype=bool)
        broken_mask[chosen] = True
        delta = deltas[(it - 1) % 3]
        # vary mu slightly with delta to keep force comparable
        if delta < 5e-4:
            mu = 500.0
        elif delta < 2e-3:
            mu = 300.0
        else:
            mu = 200.0
        target = mc - delta
        # small jitter before breaking to avoid exact saddle
        x_start = x_cur + rng.normal(0, 1e-4, size=x_cur.shape)
        x_start /= np.linalg.norm(x_start, axis=1, keepdims=True)
        # steps adapt to remaining time
        steps = 500
        if remain < 20:
            steps = 300
        x_broken = _rewire_smooth(x_start, Iu, Ju, np.asarray(chosen, dtype=int),
                                 target, steps, 100.0, 500.0, 0.02, 0.004, mu, rng)
        # LP polish with broken edges excluded
        budget = min(3.0, max(1.0, remain / 8.0))
        x_pol, v_pol = _lp_polish_exclude(x_broken, budget, broken_mask, (Iu, Ju))
        if not np.all(np.isfinite(x_pol)):
            continue
        if v_pol < best - 1e-13:
            best = float(v_pol)
            x_best = x_pol.copy()
        # accept within 0.002 of best (uses updated best, so improvements always accepted)
        if v_pol < best + 0.002 + 1e-12:
            x_cur = x_pol.copy()
            m_cur = float(v_pol)
        # else reject: keep x_cur; also occasionally force move if stuck
        # to avoid re-forming same graph, taboo already forces disjoint next set
        if it % 10 == 0:
            # opportunistic squeeze of global best without exclusion (short)
            if time.time() - t0 < LIMIT - 16.0:
                x_best, v2 = _lp_polish_exclude(x_best, 1.0, np.zeros(M, dtype=bool), (Iu, Ju))
                if v2 < best:
                    best = float(v2)
    # final squeeze of best with standard active set
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 1.0:
        rem = min(rem, 10.0)
        x_best, v = _lp_polish_exclude(x_best, rem, np.zeros(M, dtype=bool), (Iu, Ju))
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
