import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    # upper triangle pairs
    iu0, iu1 = np.triu_indices(x.shape[0], 1)
    c = G[iu0, iu1]
    sel = c > m - tol
    pairs = list(zip(iu0[sel].tolist(), iu1[sel].tolist()))
    deg = (mask.sum(axis=1)).astype(int)
    rowmax = G.max(axis=1)
    return m, G, deg, rowmax, pairs, c

def _lp_polish(x, deadline, r0=0.005, cap=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            sel[np.argmax(c)] = True
            cnt = 1
        if cnt > cap:
            idx = np.argsort(-c)[:cap]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx] = True
            sel = sel2
        I = iu0[sel]; J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2 * d] = J[:, None] * d + ar
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.empty((p, 2 * d + 1), dtype=float)
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        vals = vals.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(x0, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = active.copy()
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        if not np.all(act):
            vel[~act] = 0.0
        x[act] = x[act] - lr_k * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                na = int(act.sum())
                if na > 0:
                    nz = rng.randn(na, d) * amp
                    xa = x[act]
                    nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                    xa = xa + nz
                    xa /= np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
                    x[act] = xa
        nn = np.linalg.norm(x[act], axis=1, keepdims=True)
        nn = np.maximum(nn, 1e-12)
        x[act] /= nn
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _refine_holes(H, others, T0=0.008, T1=0.0008, iters=70, lr=0.15):
    # H: (nh,4), others: (m,4)
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    lratio = np.log(T1 / T0)
    for k in range(iters):
        frac = k / max(1, iters - 1)
        T = T0 * np.exp(lratio * frac)
        D = H @ others.T
        mr = D.max(axis=1, keepdims=True)
        E = np.exp((D - mr) / T)
        S = E.sum(axis=1, keepdims=True)
        S = np.maximum(S, 1e-300)
        W = E / S
        Grad = W @ others
        Grad -= (Grad * H).sum(axis=1, keepdims=True) * H
        step = lr * (1.0 - 0.5 * frac)
        H = H - step * Grad
        H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    # final values
    D = H @ others.T
    vals = D.max(axis=1)
    return H, vals

def _edge_set(x, tol=5e-5):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    sel = G[iu0, iu1] > m - tol
    s = set((int(a) * 1000 + int(b)) for a, b in zip(iu0[sel], iu1[sel]))
    # note: n<1000 so encoding unique; but indices change after deletion,
    # for full codes n=163 this is comparable across trials (same ordering roughly).
    # Better: use sorted dot-rank signature? Use set of encoded pairs directly.
    # For Jaccard across different full codes ordering differs, but still indicative.
    # Use canonical: encode sorted pair.
    return s, m

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    POLISH_END = HARD_END - 12.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(np.float64)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    base_edge, _ = _edge_set(best)
    elite = [(best_val, best.copy())]
    n_full = 163
    # quick tighten of incumbent slightly (cheap)
    if time.time() < POLISH_END - 60:
        bt = _lp_polish(best, min(time.time() + 2.0, POLISH_END), r0=0.002)
        bv = _max_cos(bt)
        if bv < best_val:
            best, best_val = bt, bv
            base_edge, _ = _edge_set(best)
            elite = [(best_val, best.copy())]
    # surgery trials
    k_list = [6, 10, 14, 18, 10, 6, 14, 18]
    trial = 0
    # precompute stress order from incumbent (degrees + rowmax)
    while time.time() < POLISH_END - 8:
        if trial >= len(k_list):
            # reshuffle with new seed ordering: randomize victim choice a bit
            k_list.append(int(rng.choice([6, 10, 14, 18])))
            if len(k_list) > 14:
                break
        k = int(k_list[trial])
        trial += 1
        if time.time() > POLISH_END - 25 and trial > 4:
            break
        # identify most stressed points from current best (recompute each trial to diversify)
        src = elite[0][1] if rng.rand() < 0.7 else best
        # add small noise to src for diversity on later trials
        m0, G0, deg, rowmax, pairs, cvec = _contacts(src, tol=1e-6)
        # dual-multiplier proxy: stress per point = sum of softmax weights over contacts
        # compute weights
        T_s = 0.001
        iu0, iu1 = np.triu_indices(src.shape[0], 1)
        cc = G0[iu0, iu1]
        w = np.exp((cc - m0) / T_s)
        wsum = np.zeros(src.shape[0])
        np.add.at(wsum, iu0, w)
        np.add.at(wsum, iu1, w)
        # score = degree primary, wsum secondary, rowmax tertiary + random jitter
        jitter = rng.rand(src.shape[0]) * 0.15
        score = deg.astype(float) * 10.0 + np.log1p(wsum) * 2.0 + (rowmax - m0 + 1e-6) * 100.0 + jitter
        # force prefer 8-9 contact points: boost deg>=8
        score += (deg >= 8).astype(float) * 2.0
        victims = np.argsort(-score)[:k]
        victims_set = set(int(v) for v in victims)
        keep = np.array([i for i in range(src.shape[0]) if i not in victims_set], dtype=int)
        rem = src[keep].copy()
        # re-polish remaining with annealed softmax until well below target
        # anneal
        t_rem = min(POLISH_END - 6, time.time() + 3.5)
        # two-stage anneal on remainder
        rem, _ = _anneal(rem, 220, 0.010, 0.0006, 0.045, 0.85, rng, active=None, noise0=0.002, deadline=t_rem)
        if time.time() > t_rem:
            pass
        rem = _lp_polish(rem, min(t_rem + 1.0, POLISH_END - 5), r0=0.006)
        m_rem = _max_cos(rem)
        # if remainder not well below full target, continue annealing a bit (want margin)
        # target margin: remainder max should be < best_val - 0.002 ideally (fewer points)
        # if not, shake more
        if m_rem > best_val - 0.0005 and time.time() < POLISH_END - 10:
            rem2, _ = _anneal(rem, 150, 0.006, 0.0004, 0.03, 0.8, rng, deadline=POLISH_END - 8)
            rem2 = _lp_polish(rem2, min(time.time() + 0.8, POLISH_END - 5), r0=0.004)
            if _max_cos(rem2) < m_rem:
                rem = rem2
                m_rem = _max_cos(rem)
        # sequential reinsertion
        # epsilon-greedy rank choice for diversity
        cur_set = rem.copy()
        order_seed = rng.randint(0, 1000000)
        r2 = np.random.RandomState(order_seed)
        stall = 0
        for ins in range(k):
            if time.time() > POLISH_END - 6:
                break
            m_fixed = cur_set.shape[0]
            # sample 20k uniform candidates
            nc = 20000
            cand = r2.randn(nc, 4)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # compute max dot vs cur_set in chunks
            dots_max = np.empty(nc)
            chunk = 5000
            for a in range(0, nc, chunk):
                b = min(nc, a + chunk)
                dots_max[a:b] = (cand[a:b] @ cur_set.T).max(axis=1)
            # best 20 holes
            nh = 20
            part = np.argpartition(dots_max, nh - 1)[:nh]
            # sort them
            o = np.argsort(dots_max[part])
            part = part[o]
            H0 = cand[part].copy()
            Href, hvals = _refine_holes(H0, cur_set, T0=0.008, T1=0.0007, iters=80, lr=0.2)
            # choose hole: usually best, sometimes 2nd/3rd for diversity (restart orders)
            if r2.rand() < 0.65 or ins == 0:
                bi = int(np.argmin(hvals))
            else:
                # pick among top 5 with prob weighted
                top5 = np.argsort(hvals)[:5]
                bi = int(r2.choice(top5))
            best_h = Href[bi].copy()
            best_hv = float(hvals[bi])
            # insert
            cur_set = np.vstack([cur_set, best_h.reshape(1, 4)])
            # locally relax inserted + neighbours cos>0.75
            last = cur_set.shape[0] - 1
            dots_new = cur_set[:-1] @ cur_set[-1]
            neigh = np.where(dots_new > 0.75)[0]
            act = np.zeros(cur_set.shape[0], dtype=bool)
            act[last] = True
            act[neigh] = True
            # ensure at least a few active; if too few, add closest 4
            if int(act.sum()) < 4:
                closest = np.argsort(-dots_new)[:4]
                act[closest] = True
            cur_set, _ = _anneal(cur_set, 140, 0.006, 0.0005, 0.03, 0.82, r2, active=act, noise0=0.0, deadline=POLISH_END - 5)
            # occasional stall detection: if new point's rowmax far above best_val, restart order?
            Grow = cur_set @ cur_set.T
            np.fill_diagonal(Grow, -2.0)
            cur_max = float(Grow.max())
            if cur_max > best_val + 0.004:
                stall += 1
            if stall >= 2 and r2.rand() < 0.5:
                # restart greedy insertion with different order: drop last 2 and redo with different picks
                # remove last inserted (keep one) and retry once
                if cur_set.shape[0] > rem.shape[0] + 1:
                    cur_set = cur_set[:-1].copy()
                    stall = 0
                    # redo this ins index by continuing (will sample fresh)
                    # to avoid infinite loop, count extra
                    if ins > 0:
                        pass
        if cur_set.shape[0] != n_full:
            continue
        v_full = _max_cos(cur_set)
        edge, _ = _edge_set(cur_set)
        jd = _jaccard(edge, base_edge)
        # keep elite archive of distinct reinsertions
        # store if diverse or good
        is_diverse = jd > 0.10
        # fully LP-polish only if contact graph differs measurably OR value already promising
        do_full = is_diverse or (v_full < best_val + 0.0004)
        if do_full and time.time() < POLISH_END - 4:
            # light anneal first then LP
            if v_full > best_val - 0.0002:
                cur_set, _ = _anneal(cur_set, 120, 0.004, 0.0005, 0.025, 0.82, r2, deadline=POLISH_END - 4)
                v_full = _max_cos(cur_set)
            cur_set = _lp_polish(cur_set, min(time.time() + 2.2, POLISH_END - 2), r0=0.004)
            v_full = _max_cos(cur_set)
            edge2, _ = _edge_set(cur_set)
            jd2 = _jaccard(edge2, base_edge)
        else:
            jd2 = jd
        # archive: keep up to 6 distinct
        elite.append((v_full, cur_set.copy()))
        # prune elite: sort by value, keep diverse
        elite = sorted(elite, key=lambda t: t[0])[:7]
        if v_full < best_val:
            best, best_val = cur_set.copy(), v_full
            base_edge, _ = _edge_set(best)
        elif jd2 > 0.10 and v_full < best_val + 0.001:
            # keep as alternate basin for next surgery source (already in elite)
            pass
        # early exit if big win? continue to use remaining time for more trials
    # final polish of best elite entries
    elite = sorted(elite, key=lambda t: t[0])[:4]
    final_dead = HARD_END - 1.5
    for idx, (vv, xx) in enumerate(list(elite)):
        if time.time() > final_dead - 2:
            break
        xp = _lp_polish(xx, min(final_dead, time.time() + 3.0), r0=0.003 if idx > 0 else 0.004)
        vp = _max_cos(xp)
        if vp < best_val:
            best, best_val = xp, vp
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
