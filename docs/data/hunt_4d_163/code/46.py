import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _contacts_deg(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    deg = mask.sum(axis=1).astype(int)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = c > m - tol
    edges = set(zip(iu0[sel].tolist(), iu1[sel].tolist()))
    return m, deg, edges, G

def _gd_softmax(x0, steps, T0, T1, lr0, mu, rng, active=None, noise0=0.0, deadline=None):
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
                na = int(act.sum())
                if na > 0:
                    nz = rng.randn(na, d) * amp
                    xa = x[act]
                    nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                    x[act] = xa + nz
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        x /= np.maximum(xn, 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lp_polish(x, deadline, r0=0.004, cap=1200):
    x = _norm(x.copy())
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

def _refine_hole(c0, Y, T=0.002, iters=150, lr=0.06):
    c = c0 / max(np.linalg.norm(c0), 1e-12)
    for _ in range(iters):
        dots = Y @ c
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ Y
        g = g - float(g @ c) * c
        c = c - lr * g
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-12:
            break
        c = c / nrm
    return c

def _find_hole(Y, rng, n_sample=20000, n_refine=20):
    C = rng.randn(n_sample, 4)
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    D = Y @ C.T
    mx = D.max(axis=0)
    idx = np.argsort(mx)[:n_refine]
    best_c = None
    best_v = 1e9
    for j in idx:
        c = _refine_hole(C[j], Y)
        v = float((Y @ c).max())
        if v < best_v:
            best_v = v
            best_c = c
    return best_c, best_v

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(20261102)
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
        best_x = rng.randn(n, 4)
        best_x = _norm(best_x)
        best_val = _max_cos(best_x)
    incumbent = best_x.copy()
    inc_val = best_val
    # quick LP touch on incumbent if time allows (small)
    # elite archive: list of (val, X)
    elite = [(inc_val, incumbent.copy())]
    try:
        m0, deg0, edges0, _ = _contacts_deg(incumbent, tol=1e-9)
    except Exception:
        deg0 = np.zeros(n, dtype=int)
        edges0 = set()
    deg_hist0 = np.sort(deg0).astype(float)

    ks = [10, 6, 14, 18, 10, 14, 6, 18]
    trial = 0
    # time budget: leave 12s for final polish
    while time.time() < DEADLINE - 12.0 and trial < 8:
        k = ks[trial % len(ks)]
        trial += 1
        if time.time() > DEADLINE - 12.0:
            break
        try:
            _, deg, _, _ = _contacts_deg(incumbent, tol=1e-9)
        except Exception:
            deg = np.zeros(n, dtype=int)
        jitter = rng.randn(n) * 0.35
        order = np.argsort(-deg.astype(float) - jitter * 0.0 + jitter * 0.15 - deg.astype(float) * 0.0)
        # top-k by degree with jitter: sort by deg + jitter
        score_del = deg.astype(float) + jitter
        del_idx = np.argsort(-score_del)[:k]
        del_idx = np.sort(del_idx)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Y = incumbent[keep_mask].copy()
        # annealed softmax polish of remainder (fewer points -> go well below target)
        sub_dead = min(time.time() + 6.0, DEADLINE - 12.0)
        # two-stage anneal
        Y, _ = _gd_softmax(Y, steps=900, T0=0.008, T1=0.0004, lr0=0.10, mu=0.85, rng=rng, noise0=0.004, deadline=sub_dead)
        if time.time() < sub_dead - 0.5:
            Y, _ = _gd_softmax(Y, steps=700, T0=0.002, T1=0.00015, lr0=0.06, mu=0.88, rng=rng, noise0=0.0, deadline=sub_dead)
        if time.time() < sub_dead - 0.8:
            Y, vy = _lp_polish(Y, min(sub_dead - 0.2, DEADLINE - 12.0), r0=0.005)
        else:
            vy = _max_cos(Y)
        # greedy reinsertion with different orders: shuffle tie-breaking via fresh rng draws
        cur = Y.copy()
        stall = 0
        prev_hole_v = 1e9
        for t in range(k):
            if time.time() > DEADLINE - 12.0:
                break
            hc, hv = _find_hole(cur, rng, n_sample=20000, n_refine=20)
            if hc is None:
                hc = rng.randn(4); hc /= np.linalg.norm(hc)
                hv = float((cur @ hc).max())
            # restart greedy with different order when stalled: if hole not improving, sample extra batch
            if hv > prev_hole_v - 1e-6:
                stall += 1
            else:
                stall = 0
            prev_hole_v = hv
            if stall >= 2:
                # extra-large sample restart
                hc2, hv2 = _find_hole(cur, rng, n_sample=30000, n_refine=25)
                if hv2 < hv:
                    hc, hv = hc2, hv2
                stall = 0
            cur = np.vstack([cur, hc.reshape(1, 4)])
            cur = _norm(cur)
            # locally relax inserted point + neighbours with cosine > 0.75
            try:
                dots = float((cur[:-1] @ cur[-1]).max()) if cur.shape[0] > 1 else -2.0
                dvec = cur[:-1] @ cur[-1]
                nbr = np.where(dvec > 0.75)[0]
            except Exception:
                nbr = np.array([], dtype=int)
            act = np.zeros(cur.shape[0], dtype=bool)
            act[-1] = True
            if len(nbr) > 0:
                act[nbr] = True
            # if many neighbours, include them; else relax all lightly every 4th insertion
            if int(act.sum()) <= 2 or (t % 4 == 3):
                act[:] = True
            Ldead = min(time.time() + 1.2, DEADLINE - 11.0)
            cur, _ = _gd_softmax(cur, steps=350, T0=0.003, T1=0.0003, lr0=0.07, mu=0.85, rng=rng, active=act, noise0=0.0, deadline=Ldead)
        if cur.shape[0] != n:
            continue
        cur = _norm(cur)
        try:
            vcur = _max_cos(cur)
        except Exception:
            continue
        # Jaccard-style topology check: degree-histogram distance from incumbent
        try:
            _, degc, edgesc, _ = _contacts_deg(cur, tol=1e-7)
            histc = np.sort(degc).astype(float)
            # pad to same length (same n)
            denom = max(float(np.abs(deg_hist0).mean()), 1.0)
            topo_dist = float(np.abs(histc - deg_hist0).mean()) / denom
            # edge-set size difference as secondary
            jproxy = topo_dist  # >~0.05 means measurably different
        except Exception:
            jproxy = 1.0
        # fully LP-polish only if topologically different (Jaccard proxy > 0.02) or clearly promising
        do_full = (jproxy > 0.02) or (vcur < inc_val - 1e-7)
        if vcur > inc_val + 0.0015 and not do_full:
            # keep in archive anyway if distinct
            if len(elite) < 5 and jproxy > 0.05:
                elite.append((vcur, cur.copy()))
            continue
        if do_full:
            if time.time() < DEADLINE - 10.0:
                # light anneal before LP to settle
                cur, _ = _gd_softmax(cur, steps=400, T0=0.0015, T1=0.0002, lr0=0.05, mu=0.88, rng=rng, noise0=0.0, deadline=min(time.time() + 2.0, DEADLINE - 9.0))
                cur, vcur = _lp_polish(cur, min(DEADLINE - 6.0, time.time() + 7.0), r0=0.003)
            else:
                continue
        else:
            # near-tie back into same basin: perturb instead of accepting
            if len(elite) < 6:
                elite.append((vcur, cur.copy()))
            continue
        # update elite archive with distinct reinsertions
        elite.append((vcur, cur.copy()))
        # keep only best 5 distinct (by val separation 2e-7)
        elite = sorted(elite, key=lambda z: z[0])
        new_elite = []
        for vv, xx in elite:
            keep = True
            for vv2, _ in new_elite:
                if abs(vv - vv2) < 2e-7:
                    keep = False
                    break
            if keep:
                new_elite.append((vv, xx))
            if len(new_elite) >= 5:
                break
        elite = new_elite
        if vcur < inc_val - 1e-12:
            incumbent = cur.copy()
            inc_val = vcur
            try:
                _, degn, _, _ = _contacts_deg(incumbent, tol=1e-9)
                deg_hist0 = np.sort(degn).astype(float)
            except Exception:
                pass
        # if stalled repeatedly, jump to an elite member that is not incumbent to force basin change
        if trial % 3 == 0 and len(elite) > 1 and time.time() < DEADLINE - 20.0:
            # pick second-best distinct as next start occasionally
            try:
                alt = elite[1][1].copy()
                # small random tangent kick to rattler-like low-degree point
                _, degA, _, _ = _contacts_deg(alt, tol=1e-9)
                q = int(np.argmin(degA))
                nz = rng.randn(4) * 0.01
                nz -= float(nz @ alt[q]) * alt[q]
                alt[q] = alt[q] + nz
                alt = _norm(alt)
                incumbent_alt_val = _max_cos(alt)
                # do not replace incumbent, but use as base next trial by swapping
                if incumbent_alt_val < inc_val + 0.0008:
                    incumbent = alt.copy() if (trial % 6 == 0) else incumbent
                    if trial % 6 == 0:
                        inc_val = incumbent_alt_val
            except Exception:
                pass
    # final selection: best of elite + incumbent, final LP polish on the very best
    cand_list = [(inc_val, incumbent)] + elite
    cand_list = sorted(cand_list, key=lambda z: z[0])
    final_x = cand_list[0][1].copy()
    final_v = cand_list[0][0]
    if time.time() < DEADLINE - 5.0:
        try:
            f2, v2 = _lp_polish(_norm(final_x), DEADLINE - 1.5, r0=0.002)
            if v2 < final_v:
                final_x, final_v = f2, v2
        except Exception:
            pass
    final_x = _norm(final_x)
    # strict feasibility: tiny shrink not needed (unit rows already); ensure finite
    final_x = np.where(np.isfinite(final_x), final_x, 0.0)
    rn = np.linalg.norm(final_x, axis=1, keepdims=True)
    rn = np.maximum(rn, 1e-12)
    final_x = final_x / rn
    return final_x
