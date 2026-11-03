import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, mc, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G > (mc - tol)
    cnt = C.sum(axis=1)
    return G, cnt, C

def _adam_single(x0, Others, beta=200.0, steps=300, lr=0.02):
    d = x0.shape[0]
    x = x0 / (np.linalg.norm(x0) + 1e-30)
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        dots = Others @ x
        mx = float(dots.max())
        # softmax weights
        e = np.exp(beta * (dots - mx))
        s = e.sum()
        if not np.isfinite(s) or s == 0:
            break
        w = e / s
        grad = w @ Others
        gd = float(grad @ x)
        gt = grad - gd * x
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        x = x - step
        n = np.linalg.norm(x)
        if n == 0 or not np.isfinite(n):
            x = x0 / (np.linalg.norm(x0) + 1e-30)
            m[:] = 0; v[:] = 0
            continue
        x /= n
    return x

def _annealed_relax(X, steps=450, mu=0.85, lr0=0.08, lr1=0.008, b0=20.0, b1=250.0):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        # avoid overflow: (G-mx)<=0
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.15:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    EXPL_END = t0 + 148.0
    rng = np.random.default_rng(12345)
    n, d = 95, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if xr is not None and xr.shape == (95, 13):
        xn = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if np.all(np.isfinite(xn)):
            cands.append(xn)
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_inc = cands[bi].copy()
    inc_val = float(vals[bi])
    # best accepted trial (for final polish); init to incumbent
    x_best_acc = x_inc.copy()
    best_acc_val = inc_val
    found_better_unpolished = False
    # exploration loop
    trial = 0
    TOL = 1e-7
    while time.time() < EXPL_END and time.time() < HARD_END - 25:
        trial += 1
        # recompute contacts from incumbent-best (adapts as we improve)
        ref = x_best_acc if found_better_unpolished else x_inc
        mc = _max_cos(ref)
        G, cnt, C = _contacts(ref, mc, tol=TOL)
        rattlers = np.where(cnt == 0)[0]
        if len(rattlers) == 0:
            rattlers = np.argsort(cnt)[:2]
        r_idx = int(rattlers[rng.integers(0, len(rattlers))])
        # hub ordering
        order = np.argsort(-cnt)
        # choose k hubs 4..8
        k = int(rng.integers(4, 9))
        pool = order[:22].tolist()
        # random start among top 6 to diversify
        start_candidates = order[:6].tolist()
        s0 = int(start_candidates[rng.integers(0, len(start_candidates))])
        if s0 == r_idx and len(start_candidates) > 1:
            s0 = int(start_candidates[(rng.integers(0, len(start_candidates)))])
        hub_set = [s0]
        # greedy clique expansion: maximize contacts to current set
        avail = [p for p in pool if p != s0 and p != r_idx]
        rng.shuffle(avail)
        while len(hub_set) < k and avail:
            best_score = -1
            best_cands = []
            for a in avail:
                sc = 0
                for h in hub_set:
                    i, j = (a, h) if a < h else (h, a)
                    # C is full matrix
                    if C[a, h]:
                        sc += 1
                if sc > best_score:
                    best_score = sc
                    best_cands = [a]
                elif sc == best_score:
                    best_cands.append(a)
            # prefer mutually-in-contact (score==len(hub_set)); if none, accept best
            pick = int(rng.choice(np.array(best_cands)))
            hub_set.append(pick)
            avail.remove(pick)
            # if best_score < len(hub_set)-1 and rng.random()<0.3: break ties by random top hub outside pool
            if best_score == 0 and rng.random() < 0.2:
                # inject diversity: random hub outside pool
                extra = int(rng.integers(0, n))
                if extra not in hub_set and extra != r_idx and len(hub_set) < k:
                    hub_set.append(extra)
                    if extra in avail:
                        avail.remove(extra)
        removed = [r_idx] + hub_set
        # unique
        removed = list(dict.fromkeys(removed))
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.array(removed, dtype=int)] = False
        X_keep = ref[keep_mask].copy()
        # reinsert one at a time in random order
        order_ins = np.array(removed)
        # shuffle order
        perm = rng.permutation(len(order_ins))
        placed = []
        cur_frozen = X_keep.copy()
        ok = True
        for pi in perm:
            if time.time() > EXPL_END or time.time() > HARD_END - 25:
                ok = False
                break
            # 30k candidates
            cand = rng.standard_normal(size=(30000, d))
            cn = np.linalg.norm(cand, axis=1, keepdims=True)
            cn[cn == 0] = 1.0
            cand /= cn
            dots = cand @ cur_frozen.T
            md = dots.max(axis=1)
            j = int(np.argmin(md))
            x_start = cand[j]
            # 300 steps tangent Adam vs frozen
            x_new = _adam_single(x_start, cur_frozen, beta=200.0, steps=300, lr=0.02)
            placed.append(x_new)
            cur_frozen = np.vstack([cur_frozen, x_new.reshape(1, -1)])
        if not ok:
            break
        # assemble trial in original indexing: keep + placed (order lost, fine)
        X_trial = np.vstack([X_keep, np.array(placed)])
        # annealed softmax relaxation on all points (at most ~5s, we use ~450 steps ~0.2s)
        # adapt steps to remaining time
        remain = min(EXPL_END - time.time(), HARD_END - 25 - time.time())
        if remain < 0.05:
            break
        # use fewer steps if little time
        if remain < 0.5:
            steps_ann = 150
        else:
            steps_ann = 450
        t_ann0 = time.time()
        # cap at 5 seconds per spec
        X_trial = _annealed_relax(X_trial, steps=steps_ann)
        # safety: if anneal took >5s (shouldn't), skip
        if time.time() - t_ann0 > 5.5:
            pass
        X_trial /= np.linalg.norm(X_trial, axis=1, keepdims=True)
        v = _max_cos(X_trial)
        if v < 0.26:
            if (not found_better_unpolished) or (v < best_acc_val):
                # store best accepted (even if worse than incumbent, keep best among accepted for final polish,
                # but also keep incumbent fallback)
                if v < best_acc_val:
                    x_best_acc = X_trial.copy()
                    best_acc_val = float(v)
                    found_better_unpolished = True
                # else still could keep diversity: with small prob replace to diversify reference?
                # keep reference as best accepted to drive contacts
            # also if trial beats incumbent directly, update incumbent too
            if v < inc_val:
                x_inc = X_trial.copy()
                inc_val = float(v)
                x_best_acc = X_trial.copy()
                best_acc_val = float(v)
                found_better_unpolished = True
        # occasional time check
        if trial % 10 == 0:
            if time.time() > EXPL_END:
                break
    # final 20-second LP polish of best accepted (or incumbent)
    target = x_best_acc.copy() if found_better_unpolished else x_inc.copy()
    # ensure finite
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    remain = HARD_END - time.time() - 1.0
    if remain > 2.0:
        lp_sec = float(min(20.0, remain))
        deadline = time.time() + lp_sec
        try:
            target = _lp_polish(target, seconds=lp_sec, t_deadline=min(deadline, HARD_END - 0.5))
        except Exception:
            pass
    # choose between polished target and incumbent by true score
    try:
        vt = _max_cos(target)
    except Exception:
        return x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)
    if vt < inc_val or found_better_unpolished:
        out = target
    else:
        out = x_inc
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    # strict finite check
    if not np.all(np.isfinite(out)):
        out = x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)
    return out
