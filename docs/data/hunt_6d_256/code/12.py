import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal_best(x0, steps, a0, a1, lr0, lr1):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(x)
    best_x = x.copy()
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # softmax weights, G-gmax <= 0
        W = np.exp((G - gmax) * alpha)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * gt
        v *= b2
        v += (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x /= nrm
        if (t % 150 == 0) or (t == steps):
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            val = float(G2.max())
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish_budget(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    end = time.time() + budget
    iu0 = np.triu_indices(n, 1)
    iu0_0 = iu0[0]
    iu0_1 = iu0[1]
    while time.time() < end and radius > 1e-14:
        C = (x @ x.T)[iu0]
        m = float(C.max())
        win = max(25.0 * radius, 2e-9)
        sel = C > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            # keep tightest 3500
            part = np.argpartition(C[idx], -3500)[-3500:]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
            cnt = 3500
        I = iu0_0[sel]
        J = iu0_1[sel]
        csub = C[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ar = np.arange(d)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
        if time.time() >= end:
            break
    return x

def _voids(x_fixed, k, rng, n_cand=12000):
    cand = rng.standard_normal((n_cand, 6))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.empty(n_cand)
    # chunked scoring for speed
    for a in range(0, n_cand, 2000):
        b = min(a + 2000, n_cand)
        scores[a:b] = (cand[a:b] @ x_fixed.T).max(axis=1)
    order = np.argsort(scores)
    chosen = []
    for h in order:
        c = cand[h]
        ok = True
        for q in chosen:
            if abs(float(c @ cand[q])) > 0.68:
                ok = False
                break
        if ok:
            chosen.append(int(h))
            if len(chosen) >= k:
                break
    return cand[chosen] if len(chosen) > 0 else cand[order[:k]]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    LOOP_END = t0 + 147.0
    rng = np.random.default_rng(0)
    x_best = None
    best_v = np.inf
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best_v:
                    best_v = vv
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal((256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_v = _max_cos(x_best)
    cur_best = float(best_v)
    n, d = 256, 6
    it = 0
    sig_cycle = [0.02, 0.05, 0.12, 0.25]
    while time.time() < LOOP_END:
        it += 1
        remain_loop = LOOP_END - time.time()
        if remain_loop < 2.5:
            break
        # row maxima of current best
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        rmax = Gb.max(axis=1)
        gmax = float(Gb.max())
        order = np.argsort(rmax)  # ascending: rattlers first
        u = rng.random()
        x_try = x_best.copy()
        if u < 0.34:
            sigma = sig_cycle[it % len(sig_cycle)]
            # bias to jammed points
            prox = np.clip((rmax - (gmax - 0.05)) / 0.05, 0, 1)
            scale = (0.5 + 1.6 * prox)[:, None]
            x_try = x_try + sigma * scale * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 650 if sigma < 0.15 else 850
            a0, a1 = 70.0, 520.0
            lr0, lr1 = 0.030, 0.003
        elif u < 0.62:
            ks = [8, 16, 32]
            k = ks[rng.integers(0, len(ks))]
            worst_pool = order[-64:]
            nw = min(k, (k * 3) // 4 + 1)
            idx_w = rng.choice(worst_pool, size=nw, replace=False)
            sel = set(int(i) for i in idx_w)
            while len(sel) < k:
                sel.add(int(rng.integers(0, n)))
            if rng.random() < 0.5:
                sel.add(int(order[rng.integers(0, 8)]))
            idx = np.array(sorted(sel), dtype=int)
            rnd = rng.standard_normal((len(idx), d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = x_try + 0.008 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 950
            a0, a1 = 55.0, 450.0
            lr0, lr1 = 0.032, 0.0025
        elif u < 0.84:
            k = int(rng.choice(np.array([8, 12, 18])) )
            rep_idx = order[:k].copy()
            # voids wrt all points except those to be replaced? use x_best
            try:
                holes = _voids(x_best, k, rng, n_cand=12000)
            except Exception:
                holes = None
            if holes is not None and holes.shape[0] >= 1:
                nh = min(k, holes.shape[0])
                for ii in range(nh):
                    x_try[int(rep_idx[ii])] = holes[ii]
                # if fewer holes than k, randomize rest
                if nh < k:
                    for ii in range(nh, k):
                        rr = rng.standard_normal(6)
                        rr /= np.linalg.norm(rr)
                        x_try[int(rep_idx[ii])] = rr
            else:
                for ii in rep_idx:
                    rr = rng.standard_normal(6)
                    rr /= np.linalg.norm(rr)
                    x_try[int(ii)] = rr
            x_try = x_try + 0.015 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 800
            a0, a1 = 80.0, 550.0
            lr0, lr1 = 0.025, 0.0025
        else:
            # jammed cluster expansion: hub + nearest neighbours
            hub = int(np.argmax(rmax))
            dots_hub = Gb[hub].copy()
            dots_hub[hub] = -2.0
            nn = np.argsort(-dots_hub)[:12]
            cluster = np.concatenate([[hub], nn])
            cent = x_try[cluster].mean(axis=0)
            cent /= (np.linalg.norm(cent) + 1e-12)
            for ci in cluster:
                push = x_try[ci] - cent
                # tangentize push
                push = push - np.dot(push, x_try[ci]) * x_try[ci]
                pn = np.linalg.norm(push)
                if pn > 1e-12:
                    push /= pn
                else:
                    push = rng.standard_normal(6)
                    push /= np.linalg.norm(push)
                x_try[ci] = x_try[ci] + 0.18 * push + 0.05 * rng.standard_normal(6)
            # global tiny jitter
            x_try = x_try + 0.012 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps = 800
            a0, a1 = 60.0, 500.0
            lr0, lr1 = 0.030, 0.003
        # adapt steps to remaining time
        if remain_loop < 8:
            steps = min(steps, 400)
        try:
            x_opt, v_opt = _anneal_best(x_try, steps=steps, a0=a0, a1=a1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        # decide LP
        thresh = cur_best + (0.004 if u < 0.84 else 0.007)
        if v_opt < thresh:
            lp_budget = 0.9 if v_opt < cur_best + 0.0006 else 0.55
            # leave time for loop
            lp_budget = min(lp_budget, max(0.35, LOOP_END - time.time() - 0.8))
            if lp_budget > 0.3:
                try:
                    x_opt2 = _lp_polish_budget(x_opt, lp_budget)
                except Exception:
                    x_opt2 = x_opt
                v2 = _max_cos(x_opt2)
            else:
                x_opt2, v2 = x_opt, v_opt
            if v2 < cur_best:
                cur_best = float(v2)
                x_best = x_opt2.copy()
    # final long polish
    rem = LIMIT - (time.time() - t0) - 0.8
    if rem > 2.0:
        try:
            x_best = _lp_polish_budget(x_best, min(rem, 24.0))
            cur_best = _max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
