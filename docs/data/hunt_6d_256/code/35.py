import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _top_stats(x):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    # top 1000 needed; partition is enough but sort fully for clarity (32k)
    s = np.sort(c)[::-1]
    t1 = float(s[0])
    t100 = float(s[99]) if len(s) >= 100 else float(s[-1])
    t1000 = float(s[999]) if len(s) >= 1000 else float(s[-1])
    return t1, t100, t1000

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(K * (G - gmax), -60, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, 1e-4)
        sel = c > (m - margin)
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _refine_y(y0, X, steps=80, K=1800.0, lr=0.02):
    d = X.shape[1]
    y = y0.copy() / (np.linalg.norm(y0) + 1e-16)
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    def _rowmax(yy):
        dd = X @ yy
        return float(dd.max())
    best_y = y.copy()
    best_r = _rowmax(y)
    for t in range(1, steps + 1):
        dots = X @ y
        dm = float(dots.max())
        w = np.exp(np.clip(K * (dots - dm), -60, 0.0))
        S = float(w.sum())
        if not np.isfinite(S) or S <= 0:
            break
        grad = (w @ X) / S
        gt = grad - float(grad @ y) * y
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        st = lr * mh / (np.sqrt(vh) + eps)
        st = st - float(st @ y) * y
        y = y - st
        nrm = np.linalg.norm(y) + 1e-16
        y = y / nrm
        if t % 20 == 0 or t == steps:
            r = _rowmax(y)
            if r < best_r:
                best_r = r
                best_y = y.copy()
    return best_y, best_r

def _bulk_surgery(x0, k, rng, cand_n=200000, refine_steps=80):
    n, d = x0.shape
    G = x0 @ x0.T
    np.fill_diagonal(G, -2.0)
    iuI, iuJ = np.triu_indices(n, 1)
    c = G[iuI, iuJ]
    Ktop = min(1500, len(c))
    part = np.argpartition(-c, Ktop - 1)[:Ktop]
    I = iuI[part]; J = iuJ[part]
    partic = np.zeros(n, dtype=int)
    np.add.at(partic, I, 1)
    np.add.at(partic, J, 1)
    prow = G.max(axis=1)
    order_p = np.argsort(-partic, kind="stable")
    order_r = np.argsort(-prow, kind="stable")
    nhub = k // 2
    nrest = k - nhub
    del_set = set()
    for idx in order_p[:nhub]:
        del_set.add(int(idx))
    # remaining high-row-max pool (top 80 excluding hubs)
    pool = [int(i) for i in order_r if int(i) not in del_set][:80]
    if len(pool) < nrest:
        pool = [int(i) for i in order_r if int(i) not in del_set]
    # sample half deterministically via rng from pool
    if nrest > 0:
        # bias to highest row-max but with randomness
        perm = rng.permutation(len(pool))
        # take first nrest from a weighted mix: first 40 are highest, sample among them
        # to keep determinism use permutation of top-40 + rest
        cand_pool = pool[:40] + [pool[i] for i in perm[:min(len(pool), 40)]]
        # dedup preserve order
        seen = set(); uniq = []
        for v in cand_pool:
            if v not in del_set and v not in seen:
                seen.add(v); uniq.append(v)
            if len(uniq) >= nrest:
                break
        # fill if short
        for v in pool:
            if len(uniq) >= nrest:
                break
            if v not in del_set and v not in seen:
                seen.add(v); uniq.append(v)
        for v in uniq[:nrest]:
            del_set.add(int(v))
    # if still short (ties), fill by participation order
    for v in order_p:
        if len(del_set) >= k:
            break
        del_set.add(int(v))
    del_list = sorted(list(del_set))[:k]
    keep = np.array([i for i in range(n) if i not in set(del_list)], dtype=int)
    F = x0[keep].copy()  # (n-k, 6)
    # generate candidates
    Z = rng.normal(size=(cand_n, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # base rowmax to frozen set, batched
    base = np.empty(cand_n, dtype=float)
    chunk = 20000
    for s in range(0, cand_n, chunk):
        e = min(s + chunk, cand_n)
        S = Z[s:e] @ F.T
        base[s:e] = S.max(axis=1)
    cur = base.copy()
    used = np.zeros(cand_n, dtype=bool)
    Y = []
    for s_idx in range(len(del_list)):
        # mask used
        cur_masked = cur.copy()
        cur_masked[used] = 2.0
        bi = int(np.argmin(cur_masked))
        y0 = Z[bi].copy()
        Xf = F if len(Y) == 0 else np.vstack([F] + [y[None, :] for y in Y])
        yref, _ = _refine_y(y0, Xf, steps=refine_steps, K=1800.0, lr=0.02)
        Y.append(yref)
        used[bi] = True
        # update cur with new point
        dots = Z @ yref
        # cur = max(cur, dots)
        np.maximum(cur, dots, out=cur)
        cur[bi] = 2.0
    Y = np.array(Y)
    x_new = np.vstack([F, Y])
    # restore to n rows (order irrelevant)
    return x_new, del_list

def solve():
    import time as _time
    t0 = _time.time()
    T_END = t0 + 172.0
    SURG_END = t0 + 138.0
    rng = np.random.default_rng(20261103)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    n, d = 256, 6
    # quick initial LP equalization
    try:
        if _time.time() < SURG_END:
            x_best = _lp_polish(x_best, min(_time.time() + 4.0, SURG_END), r0=2e-3)
            best = _max_cos(x_best)
    except Exception:
        pass
    bt1, bt100, bt1000 = _top_stats(x_best)
    bsum = bt1 + bt100 + bt1000
    # bulk surgery trials
    for k in (8, 16, 32, 48):
        if _time.time() >= SURG_END:
            break
        try:
            x_raw, _ = _bulk_surgery(x_best, k, rng, cand_n=200000, refine_steps=80)
        except Exception:
            continue
        # unfreeze: LogSumExp Adam
        try:
            rem = SURG_END - _time.time()
            if rem < 4:
                break
            steps = 450 if rem > 20 else 300
            x_pol, v_pol = _adam_sm(x_raw, steps, 80.0, 2200.0, 0.008)
        except Exception:
            continue
        try:
            x_pol2 = _lp_polish(x_pol, min(_time.time() + 3.0, SURG_END + 6.0), r0=1.5e-3)
            v_pol2 = _max_cos(x_pol2)
            if v_pol2 < v_pol:
                x_pol, v_pol = x_pol2, v_pol2
        except Exception:
            pass
        try:
            nt1, nt100, nt1000 = _top_stats(x_pol)
        except Exception:
            continue
        nsum = nt1 + nt100 + nt1000
        # retention: top-1, top-100, top-1000 must improve as a group without max regression
        # require max strictly better and combined triple better, with no triple element much worse
        cond_max = (nt1 < bt1 - 1e-12)
        cond_sum = (nsum < bsum - 1e-12)
        cond_tol = (nt100 <= bt100 + 2e-4) and (nt1000 <= bt1000 + 2e-4)
        if cond_max and cond_sum and cond_tol:
            x_best = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
            best = v_pol
            bt1, bt100, bt1000 = nt1, nt100, nt1000
            bsum = nsum
        elif v_pol < best - 2e-6 and cond_tol and (nsum < bsum):
            # strong max win with spread not regressed: accept as well
            x_best = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
            best = v_pol
            bt1, bt100, bt1000 = nt1, nt100, nt1000
            bsum = nsum
        # else discard
    # extra second-round surgery at promising scale if time remains
    if _time.time() < SURG_END - 10:
        try:
            x_raw, _ = _bulk_surgery(x_best, 32, rng, cand_n=200000, refine_steps=60)
            x_pol, v_pol = _adam_sm(x_raw, 400, 60.0, 1800.0, 0.009)
            x_pol2 = _lp_polish(x_pol, min(_time.time() + 3.0, T_END - 12.0), r0=1.5e-3)
            v_pol2 = _max_cos(x_pol2)
            if v_pol2 < v_pol:
                x_pol, v_pol = x_pol2, v_pol2
            nt1, nt100, nt1000 = _top_stats(x_pol)
            nsum = nt1 + nt100 + nt1000
            if (nt1 < bt1 - 1e-12) and (nsum < bsum - 1e-12):
                x_best = x_pol / np.linalg.norm(x_pol, axis=1, keepdims=True)
                best = v_pol
                bt1, bt100, bt1000 = nt1, nt100, nt1000
        except Exception:
            pass
    # final polish
    try:
        if _time.time() < T_END - 3:
            xb, vb = _adam_sm(x_best, 600, 400.0, 3500.0, 0.004)
            if vb < best and np.all(np.isfinite(xb)):
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 2:
            x2 = _lp_polish(x_best, T_END - 1.0, r0=1.2e-3)
            v2 = _max_cos(x2)
            if v2 < best:
                x_best, best = x2, v2
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
