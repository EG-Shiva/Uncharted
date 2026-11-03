import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def smooth_polish(X, steps, lr0, lr1, tau0, tau1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        tau = tau0 * (1 - f) + tau1 * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        E = np.exp((G - m) / max(tau, 1e-9))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if S <= 0 or not np.isfinite(S):
            break
        P = E / S
        Gr = P @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Ft = Gr - dot * X
        norms = np.sqrt(np.sum(Ft * Ft, axis=1))
        mx = np.max(norms)
        if mx < 1e-12 or not np.isfinite(mx):
            break
        X = X - (lr / mx) * Ft
        X = _norm(X)
    return X

def lp_polish(x, seconds, t_end=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints for speed
        if p > 5000:
            idx = np.argsort(-c)[:5000]
            I, J, c = I[idx], J[idx], c[idx]
            p = 5000
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_pool(rng):
    # all nonzero {-1,0,1}^9 : 19682
    N = 3 ** 9
    vals = np.empty((N, 9), dtype=float)
    tmp = np.arange(N)
    for k in range(9):
        vals[:, k] = (tmp % 3) - 1
        tmp //= 3
    mask = np.any(vals != 0, axis=1)
    vals = vals[mask]
    nr = np.linalg.norm(vals, axis=1, keepdims=True)
    P1 = vals / nr
    # subsampled {-2,-1,0,1,2}^9, deterministic
    M = 12000
    Rv = rng.integers(-2, 3, size=(M, 9)).astype(float)
    z = np.all(Rv == 0, axis=1)
    Rv[z, 0] = 1.0
    # add D9-type and coordinate explicitly (already in P1) but keep
    Rn = np.linalg.norm(Rv, axis=1, keepdims=True)
    Rn = np.maximum(Rn, 1e-12)
    P2 = Rv / Rn
    P = np.vstack([P1, P2])
    # deduplicate directions roughly: keep all, duplicates harmless
    return P

def greedy_select(P, rng):
    N = P.shape[0]
    n = 131
    start = int(rng.integers(0, N))
    sel = [start]
    selset = {start}
    cur = P @ P[start]  # max dot to selected so far
    cur[start] = np.inf
    for _ in range(1, n):
        # pick minimal cur; randomize among top-8 ties for diversity
        # argpartition for speed
        k = 8
        part = np.argpartition(cur, k)[:k]
        # choose among them with small random preference weighted by cur
        # to keep deterministic but diverse: pick best with prob .7 else random of top-k
        if rng.random() < 0.7:
            bi = int(np.argmin(cur))
        else:
            # random among top k not selected
            cand = [c for c in part if c not in selset]
            if len(cand) == 0:
                bi = int(np.argmin(cur))
            else:
                bi = int(rng.choice(np.array(cand)))
        sel.append(bi)
        selset.add(bi)
        cur[bi] = np.inf
        d = P @ P[bi]
        # update running max (ignore selected)
        np.maximum(cur, d, out=cur)
        cur[bi] = np.inf
        # prevent re-selecting same direction: also forbid exact duplicates?
        # mark near-duplicate (dot>0.9999) as used to avoid same ray
        # cheap: only when d very high
        # (skip for speed)
    S = P[np.array(sel)]
    return S, sel

def swap_improve(P, S, sel, rng, sweeps=2):
    n = S.shape[0]
    N = P.shape[0]
    sel = list(sel)
    S = S.copy()
    in_sel = np.zeros(N, dtype=bool)
    in_sel[np.array(sel)] = True
    for _ in range(sweeps):
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        glob = float(np.max(G))
        # max excluding each i
        # for each i, max over pairs not involving i
        # compute via: for each i, mask
        # n=131 small, do loop
        order = rng.permutation(n)
        # precompute pool-selected dots
        PS = P @ S.T  # N x n
        # count near-max pairs
        thr = glob - 1e-4
        # number of near-max pairs (for tie-break)
        cur_cnt = int(np.sum(G[np.triu_indices(n, 1)] >= thr))
        for i in order:
            # best achievable max excluding i
            Gwo = G.copy()
            Gwo[i, :] = -2.0
            Gwo[:, i] = -2.0
            base = float(np.max(Gwo))
            # scores for each pool vector: max dot to others
            # max over j != i
            # compute fast: rowmax excluding i
            # PS max excluding column i
            sc = np.max(np.delete(PS, i, axis=1), axis=1)
            # forbid already selected (except i itself direction allowed to stay)
            sc[in_sel] = np.inf
            # allow staying: score of current
            cur_sc = float(np.max(np.delete(G[i], i)))
            best_new = base if base > cur_sc else cur_sc  # current overall = glob
            bi = int(np.argmin(sc))
            bv = float(sc[bi])
            new_max = bv if bv > base else base
            if new_max < glob - 1e-12:
                # replace
                in_sel[sel[i]] = False
                in_sel[bi] = True
                sel[i] = bi
                S[i] = P[bi]
                # update G row/col i and PS col i
                G[i] = S[i] @ S.T
                G[:, i] = G[i]
                G[i, i] = -2.0
                PS[:, i] = P @ S[i]
                glob = float(np.max(G))
                thr = glob - 1e-4
                cur_cnt = int(np.sum(G[np.triu_indices(n, 1)] >= thr))
            elif abs(new_max - glob) <= 1e-12:
                # reduce count criterion: estimate new count
                # count pairs at glob excluding i + new contacts of candidate
                # new contacts = number of j!=i with P_bi.S_j >= glob-1e-4
                nc = int(np.sum(PS[bi] >= 0) * 0)  # placeholder
                # compute new near-max count if swapped
                dots = P[bi] @ S.T
                dots[i] = -2.0
                add = int(np.sum(dots >= glob - 1e-4))
                # old contacts of i
                old = int(np.sum(G[i] >= glob - 1e-4))  # G[i] already has -2 at i? no, G[i,i]=-2
                # pairs not involving i at threshold
                tot_wo = int(np.sum(Gwo[np.triu_indices(n, 1)] >= glob - 1e-4))
                new_cnt = tot_wo + add
                if new_cnt < cur_cnt - 1:
                    in_sel[sel[i]] = False
                    in_sel[bi] = True
                    sel[i] = bi
                    S[i] = P[bi]
                    G[i] = S[i] @ S.T
                    G[:, i] = G[i]
                    G[i, i] = -2.0
                    PS[:, i] = P @ S[i]
                    cur_cnt = new_cnt
        # end order
    return S, sel

def contact_info(X, tol=0.005):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G >= (m - tol)
    deg = C.sum(axis=1)
    iu = np.triu_indices(X.shape[0], 1)
    emask = G[iu] >= (m - 1e-4)
    edges = set(zip(iu[0][emask].tolist(), iu[1][emask].tolist()))
    return m, deg, edges

def edge_overlap(e1, e2):
    if len(e1) == 0 or len(e2) == 0:
        return 0.0
    inter = len(e1 & e2)
    union = len(e1 | e2)
    return inter / max(1, union)

def solve():
    t0 = time.time()
    TL = 178.0
    t_end = t0 + TL - 1.5
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    _, base_deg, base_edges = contact_info(best)
    # build pool
    P = build_pool(np.random.default_rng(1))
    # generate discrete seeds
    seeds = []
    t_phase1 = t0 + 35.0
    n_seeds = 6
    for s in range(n_seeds):
        if time.time() > min(t_phase1, t_end - 60):
            break
        r2 = np.random.default_rng(1000 + s)
        S0, sel = greedy_select(P, r2)
        S1, sel1 = swap_improve(P, S0, sel, r2, sweeps=2)
        # quick smooth to judge
        Q = S1.copy()
        Q = smooth_polish(Q, steps=220, lr0=0.02, lr1=0.002, tau0=0.03, tau1=0.004)
        v = _max_cos(Q)
        m, deg, edges = contact_info(Q)
        seeds.append((v, Q, deg, edges, S1))
    # polish discrete seeds + random orthogonal versions, 40-60s total
    # keep diverse: no zero-contact and overlap<0.7 retained even if worse
    diverse = []
    for (v, Q, deg, edges, Sraw) in seeds:
        ov = edge_overlap(edges, base_edges)
        has_zero = np.any(deg == 0)
        # retain if no zeros and different graph
        if (not has_zero) and ov < 0.70:
            diverse.append((v, Q))
        elif v < best_val + 0.02:
            diverse.append((v, Q))
    # add random orthogonal transforms of discrete seeds
    rots = []
    for (v, Q) in list(diverse)[:4]:
        R = rng.standard_normal((d, d))
        Qr, _ = np.linalg.qr(R)
        if np.linalg.det(Qr) < 0:
            Qr[:, 0] = -Qr[:, 0]
        Qrot = Q @ Qr.T
        # small shake to leave exact rotation basin
        Qrot = Qrot + 0.005 * rng.standard_normal(Qrot.shape)
        Qrot = _norm(Qrot)
        Qrot = smooth_polish(Qrot, steps=150, lr0=0.015, lr1=0.002, tau0=0.025, tau1=0.004)
        vr = _max_cos(Qrot)
        rots.append((vr, Qrot))
    cands = [(best_val, best.copy())] + diverse + rots
    # smooth repulsion phase for all cands within budget
    smooth_deadline = t0 + 95.0
    if smooth_deadline > t_end - 40:
        smooth_deadline = t_end - 40
    newc = []
    for (v, X) in cands:
        if time.time() > smooth_deadline:
            newc.append((v, X))
            continue
        Y = smooth_polish(X.copy(), steps=300, lr0=0.025, lr1=0.002, tau0=0.035, tau1=0.003)
        vy = _max_cos(Y)
        newc.append((vy, Y))
        if vy < best_val:
            best_val = vy
            best = Y.copy()
            _, base_deg, base_edges = contact_info(best)
    newc.sort(key=lambda z: z[0])
    # final LP squeeze on top elites (include best + up to 3 diverse)
    # prefer diverse with no rattlers
    elites = []
    elites.append((best_val, best.copy()))
    for (v, X) in newc:
        if len(elites) >= 4:
            break
        # avoid duplicate of best
        if abs(v - elites[0][0]) < 1e-12 and len(elites) == 1:
            continue
        _, deg, edges = contact_info(X)
        if np.any(deg == 0):
            # allow one rattler candidate only if very good
            if v > best_val + 0.001:
                continue
        ov = edge_overlap(edges, base_edges)
        # keep if diverse or very good
        if ov < 0.70 or v < best_val + 0.002:
            # check not duplicate
            dup = False
            for (_, E) in elites:
                _, _, e2 = contact_info(E)
                if edge_overlap(edges, e2) > 0.95 and abs(v - _max_cos(E)) < 1e-9:
                    dup = True
                    break
            if not dup:
                elites.append((v, X.copy()))
    # LP budget split
    elites.sort(key=lambda z: z[0])
    for i in range(len(elites)):
        if time.time() >= t_end - 2.0:
            break
        val, X = elites[i]
        # sharp smooth before LP
        X = smooth_polish(X.copy(), steps=100, lr0=0.008, lr1=0.001, tau0=0.012, tau1=0.002)
        remain = t_end - time.time()
        per = remain / (len(elites) - i)
        per = min(max(per, 3.0), 14.0)
        X = lp_polish(X, seconds=per, t_end=t_end - 0.5)
        vv = _max_cos(X)
        elites.append((vv, X))
        if vv < best_val:
            best_val = vv
            best = X.copy()
    elites.sort(key=lambda z: z[0])
    out = elites[0][1].copy()
    out = _norm(out)
    return out
