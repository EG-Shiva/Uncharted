import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x / n

def _degrees(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    mask = g > m - tol
    deg = mask.sum(axis=1).astype(int)
    iu0, iu1 = np.triu_indices(x.shape[0], 1)
    c = (x @ x.T)[iu0, iu1]
    return deg, mask, c

def _gd(x0, steps, T0, T1, lr0, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        if not np.all(act):
            grad[~act] = 0.0
            v[~act] = 0.0
        v = mu * v + (1.0 - mu) * grad
        x[act] = x[act] - lr * v[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                xa = x[act]
                nz = rng.randn(int(act.sum()), d) * amp
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                x[act] = xa + nz
        xn = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        x = x / xn
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lp_polish(x, deadline, r0=2e-3, cap=1500):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _refine_hole(v0, Y, steps=100, T0=0.005, T1=0.0005, lr=0.25):
    v = v0 / max(np.linalg.norm(v0), 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        d = Y @ v
        m = float(d.max())
        w = np.exp((d - m) / T)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lr * (1.0 - 0.5 * frac) * g
        v /= max(np.linalg.norm(v), 1e-12)
    return v / max(np.linalg.norm(v), 1e-12)

def _jaccard_keep(inc_keep_G_mask_triu_set_a, cur_G_mask_triu_set_b):
    # inputs are sets of (a,b) with 0<=a<b<m
    if len(inc_keep_G_mask_triu_set_a) == 0 and len(cur_G_mask_triu_set_b) == 0:
        return 0.0
    inter = len(inc_keep_G_mask_triu_set_a & cur_G_mask_triu_set_b)
    union = len(inc_keep_G_mask_triu_set_a | cur_G_mask_triu_set_b)
    if union == 0:
        return 0.0
    return 1.0 - inter / union

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(917163)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.randn(n, 4))
        best_val = _max_cos(best_x)
    incumbent = best_x.copy()
    inc_val = best_val
    # contact analysis of incumbent
    Ginc_full = incumbent @ incumbent.T
    np.fill_diagonal(Ginc_full, -2.0)
    m_inc = float(Ginc_full.max())
    deg_inc, _, _ = _degrees(incumbent, m_inc, tol=1e-9)
    # stress score: degree + 0.2*neighbor-degree sum (proxy for dual multiplier)
    # build neighbor sum quickly
    mask_inc = Ginc_full > m_inc - 1e-9
    neigh_sum = mask_inc @ deg_inc.astype(float)
    stress_pt = deg_inc.astype(float) + 0.2 * neigh_sum
    # edge stress for reference
    # elite archive: list of (val, X)
    elite = [(best_val, best_x.copy())]

    def contact_set_triu(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu0, iu1 = np.triu_indices(X.shape[0], 1)
        c = G[iu0, iu1]
        sel = np.where(c > m - tol)[0]
        s = set()
        for k in sel:
            s.add((int(iu0[k]), int(iu1[k])))
        return s, m

    inc_set_full, _ = contact_set_triu(incumbent, tol=1e-9)

    ks = [6, 10, 14, 18]
    # outer surgery
    for kk in ks:
        for variant in range(2):
            if time.time() > DEADLINE - 30:
                break
            # select deletion set: most stressed points
            order = np.argsort(-stress_pt, kind="stable")
            if variant == 0:
                del_idx = order[:kk]
            else:
                # randomized among high-contact points (deg>=6) weighted by stress
                cand = np.where(deg_inc >= 6)[0]
                if len(cand) < kk:
                    cand = order[:max(kk * 2, kk)]
                w = stress_pt[cand].astype(float)
                w = w - w.min() + 0.5
                w = w / w.sum()
                # deterministic sampling via rng
                del_idx = rng.choice(cand, size=kk, replace=False, p=w)
                # ensure at least half overlap with top to keep stress focus? keep as sampled
            del_idx = np.sort(np.asarray(del_idx, dtype=int))
            keep_idx = np.array([i for i in range(n) if i not in set(del_idx.tolist())], dtype=int)
            m_keep = len(keep_idx)
            Y = incumbent[keep_idx].copy()
            # small tangent unlocking noise
            nz = rng.randn(m_keep, 4) * 0.004
            nz -= (nz * Y).sum(axis=1, keepdims=True) * Y
            Y = _norm(Y + nz)
            # annealed softmax on reduced set
            dl1 = min(time.time() + 3.5, DEADLINE - 28)
            Y, _ = _gd(Y, steps=700, T0=0.006, T1=0.0004, lr0=0.35, mu=0.85, rng=rng, noise0=0.0015, deadline=dl1)
            dl2 = min(time.time() + 2.5, DEADLINE - 26)
            Y, v_red = _lp_polish(Y, deadline=dl2, r0=1.5e-3, cap=1200)
            # if reduced max still too high, still continue (should be well below)
            # greedy reinsertion with restarts
            n_restarts = 2 if kk <= 10 else 1
            for restart in range(n_restarts):
                if time.time() > DEADLINE - 26:
                    break
                cur = Y.copy()
                # different sampling seed per restart via rng sequence (deterministic)
                ok = True
                for s in range(kk):
                    if time.time() > DEADLINE - 24:
                        ok = False
                        break
                    cur_n = cur.shape[0]
                    # sample 20k uniform candidates
                    C = _norm(rng.randn(20000, 4))
                    D = C @ cur.T
                    mx = D.max(axis=1)
                    # best 20 holes (lowest max dot)
                    part = np.argpartition(mx, 20)[:20]
                    # optionally if stalled (min mx too high), resample once
                    min_mx = float(mx[part].min())
                    if min_mx > 0.87 and s > 0:
                        # stall: extra sampling with different candidates
                        C2 = _norm(rng.randn(20000, 4))
                        D2 = C2 @ cur.T
                        mx2 = D2.max(axis=1)
                        part2 = np.argpartition(mx2, 20)[:20]
                        # merge
                        C = np.vstack([C[part], C2[part2]])
                        D = C @ cur.T
                        mx = D.max(axis=1)
                        part = np.arange(C.shape[0])
                        # refine all 40? take best 20 again
                        part = np.argpartition(mx, 20)[:20]
                    best_hole = None
                    best_hv = 1e9
                    for ii in part:
                        v0 = C[ii]
                        # refine by tangent descent against fixed set
                        vr = _refine_hole(v0, cur, steps=90, T0=0.004, T1=0.0004, lr=0.3)
                        hv = float(cur @ vr.reshape(4, 1).ravel().max() if False else (cur @ vr).max())
                        if hv < best_hv:
                            best_hv = hv
                            best_hole = vr
                    cur = np.vstack([cur, best_hole.reshape(1, 4)])
                    # local relax: inserted + neighbours cos>0.75
                    dots = (cur[:-1] @ cur[-1])
                    neigh = np.where(dots > 0.75)[0]
                    act = np.zeros(cur.shape[0], dtype=bool)
                    act[-1] = True
                    if len(neigh) > 0:
                        act[neigh] = True
                    # relax active subset
                    cur, _ = _gd(cur, steps=300, T0=0.004, T1=0.0004, lr0=0.3, mu=0.85, rng=rng, active=act, noise0=0.0, deadline=DEADLINE - 22)
                if not ok:
                    continue
                # full code obtained: compute max and Jaccard of retained subgraph
                vfull = _max_cos(cur)
                # contact sets for Jaccard: retained part only (first m_keep rows correspond to keep order)
                # incumbent keep part
                inc_keep = incumbent[keep_idx]
                set_inc, _ = contact_set_triu(inc_keep, tol=1e-9)
                set_cur, _ = contact_set_triu(cur[:m_keep], tol=1e-9)
                jd = _jaccard_keep(set_inc, set_cur)
                # also full-set comparison vs incumbent full for archive diversity (approx via degree hist)
                # keep distinct reinsertions in elite archive
                if jd > 0.05:
                    # store pre-polish distinct candidate
                    if len(elite) < 8:
                        elite.append((vfull, cur.copy()))
                    elif vfull < max(e[0] for e in elite):
                        # replace worst
                        elite.sort(key=lambda e: e[0])
                        elite[-1] = (vfull, cur.copy())
                # fully LP-polish only if measurably different
                if jd > 0.10:
                    dlp = min(time.time() + 6.0, DEADLINE - 12)
                    # quick anneal before LP to escape exact tie
                    cur_a, _ = _gd(cur, steps=250, T0=0.002, T1=0.0003, lr0=0.25, mu=0.85, rng=rng, noise0=0.0, deadline=dlp - 3.0)
                    # keep better of cur/cur_a by max
                    if _max_cos(cur_a) < vfull:
                        cur = cur_a
                    cur_p, vp = _lp_polish(cur, deadline=dlp, r0=2e-3, cap=1500)
                    vfull = vp
                    cur = cur_p
                    if vp < best_val - 1e-12 and np.all(np.isfinite(cur)):
                        best_val = vp
                        best_x = _norm(cur)
                        elite.sort(key=lambda e: e[0])
                else:
                    # light polish only if promising
                    if vfull < best_val + 0.0005:
                        dlp2 = min(time.time() + 1.5, DEADLINE - 12)
                        cur_p, vp = _lp_polish(cur, deadline=dlp2, r0=8e-4, cap=1000)
                        if vp < best_val - 1e-12:
                            # check diversity: require improvement to accept even near-tie basin
                            best_val = vp
                            best_x = _norm(cur_p)
                # different order restart: perturb order by shuffling keep correspondence? just continue with new rng
            # end restarts
        # end variants
    # final squeeze: polish best and top elites
    elite.sort(key=lambda e: e[0])
    # ensure best_x included
    cands = []
    cands.append(best_x.copy())
    for _, xe in elite[:3]:
        cands.append(xe.copy())
    # dedup roughly by max value
    for xc in cands:
        if time.time() > DEADLINE - 4.0:
            break
        xc = _norm(xc)
        # short anneal + LP
        xa, _ = _gd(xc, steps=300, T0=0.0015, T1=0.00025, lr0=0.22, mu=0.86, rng=rng, noise0=0.0, deadline=DEADLINE - 3.0)
        if _max_cos(xa) < _max_cos(xc):
            xc = xa
        xp, vp = _lp_polish(xc, deadline=DEADLINE - 0.5, r0=1.2e-3, cap=1500)
        if vp < best_val - 1e-15 and np.all(np.isfinite(xp)):
            best_val = vp
            best_x = _norm(xp)
    best_x = _norm(best_x)
    return best_x
