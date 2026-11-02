import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights only for near-max pairs
        W = np.zeros_like(G)
        sel = G > (m - w)
        W[sel] = np.exp(k * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        deg = np.sum(W, axis=1, keepdims=True) + 1.0
        Ft = Ft / deg
        x = x - lr * Ft
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _lp_polish(x, seconds, t0, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _build_pool(rng):
    parts = []
    # 1) cyclic binary Golay (23,12,7) -> +/-1 / sqrt23
    g = np.zeros(23, dtype=int)
    for e in [0, 2, 4, 5, 6, 10, 11]:
        g[e] = 1
    Gmat = np.stack([np.roll(g, i) for i in range(12)])  # 12x23
    masks = np.arange(4096)[:, None]
    bits = ((masks >> np.arange(12)) & 1)  # 4096x12
    code = (bits @ Gmat) & 1
    vec = (1 - 2 * code).astype(float) / np.sqrt(23.0)
    parts.append(vec)
    # 2) Hadamard-24 Paley rows, delete one coordinate
    residues = set((i * i) % 23 for i in range(1, 23))
    C = np.full((23, 23), -1.0)
    for i in range(23):
        for j in range(23):
            if i == j:
                C[i, j] = 0.0
            else:
                C[i, j] = 1.0 if ((j - i) % 23 in residues) else -1.0
    B = C.copy()
    np.fill_diagonal(B, 1.0)
    H = np.ones((24, 24))
    H[1:, 0] = -1.0
    H[0, 1:] = 1.0
    H[1:, 1:] = B
    had = []
    for r in range(24):
        hr = H[r]
        for c_ in range(24):
            v = np.delete(hr, c_).astype(float)
            v /= np.linalg.norm(v)
            had.append(v)
            had.append(-v)
    parts.append(np.array(had))
    # 3) coordinate vectors +-e_i
    ei = []
    for i in range(23):
        for s in (-1.0, 1.0):
            v = np.zeros(23)
            v[i] = s
            ei.append(v)
    parts.append(np.array(ei))
    # 4) normalized +-e_i +- e_j
    eij = []
    for i in range(23):
        for j in range(i + 1, 23):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(23)
                    v[i] = s1
                    v[j] = s2
                    v /= np.sqrt(2.0)
                    eij.append(v)
    parts.append(np.array(eij))
    # 5) sparse supports 4-8
    sp = []
    for k in (4, 5, 6, 7, 8):
        for _ in range(2200):
            supp = rng.choice(23, size=k, replace=False)
            signs = rng.choice(np.array([-1.0, 1.0]), size=k)
            v = np.zeros(23)
            v[supp] = signs / np.sqrt(float(k))
            sp.append(v)
    parts.append(np.array(sp))
    # 6) random dense to help greedy coverage
    R = rng.randn(3500, 23)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    parts.append(R)
    pool = np.concatenate(parts, axis=0)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    # deduplicate
    pr = np.round(pool, 8)
    # use unique
    try:
        _, idx = np.unique(pr, axis=0, return_index=True)
        idx = np.sort(idx)
        pool = pool[idx]
    except Exception:
        pass
    return pool

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(20261102)
    n, d = 54, 23
    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best_inc = _max_cos(xb)
    x_best = xb.copy()

    # ---- discrete pool ----
    try:
        pool = _build_pool(rng)
    except Exception:
        pool = None
    x_disc_pol = None
    v_disc_pol = 1e9
    if pool is not None:
        try:
            P = pool.shape[0]
            # greedy best-hole insertion (incremental)
            sel = [0]
            selmask = np.zeros(P, dtype=bool)
            selmask[0] = True
            curmax = pool @ pool[0]
            curmax[0] = np.inf
            for _ in range(1, n):
                if time.time() - t0 > 40.0:
                    break
                idx = int(np.argmin(curmax))
                sel.append(idx)
                selmask[idx] = True
                dd = pool @ pool[idx]
                # incremental max
                np.maximum(curmax, dd, out=curmax)
                curmax[selmask] = np.inf
            # if early break, fill randomly
            while len(sel) < n:
                cand = int(rng.randint(0, P))
                if not selmask[cand]:
                    sel.append(cand)
                    selmask[cand] = True
            sel = np.array(sel, dtype=int)
            S = pool[sel].copy()
            # exhaustive one-for-one swaps
            swap_deadline = time.time() + 12.0
            for sweep in range(30):
                if time.time() > swap_deadline or (time.time() - t0) > 55.0:
                    break
                G = S @ S.T
                np.fill_diagonal(G, -2.0)
                M = float(G.max())
                # rest max without p
                rest = np.zeros(n)
                for p in range(n):
                    Gm = G.copy()
                    Gm[p, :] = -2.0
                    Gm[:, p] = -2.0
                    rest[p] = float(Gm.max())
                D = pool @ S.T  # P x n
                # per-p best
                best_overall = M
                best_p = -1
                best_q = -1
                # vectorized per p
                for p in range(n):
                    # max over i!=p
                    if n == 1:
                        rowex = np.full(P, -2.0)
                    else:
                        cols = [c for c in range(n) if c != p]
                        rowex = D[:, cols].max(axis=1)
                    newmax = np.maximum(rest[p], rowex)
                    newmax[selmask] = np.inf
                    q = int(np.argmin(newmax))
                    qv = float(newmax[q])
                    if qv < best_overall - 1e-14:
                        best_overall = qv
                        best_p = p
                        best_q = q
                if best_p >= 0:
                    # apply
                    old = int(sel[best_p])
                    selmask[old] = False
                    sel[best_p] = best_q
                    selmask[best_q] = True
                    S[best_p] = pool[best_q]
                    continue
                # no primary improvement: try secondary (contact count)
                Ccnt = int(np.sum(G > M - 1e-9) // 2)
                improved = False
                # only check swaps that tie M
                for p in range(n):
                    cols = [c for c in range(n) if c != p]
                    rowex = D[:, cols].max(axis=1)
                    # candidates tying
                    cand_q = np.where((np.maximum(rest[p], rowex) <= M + 1e-12) & (~selmask))[0]
                    if len(cand_q) == 0:
                        continue
                    # score by number of near-max edges + rowex value
                    # pick lowest rowex then fewest close edges
                    order = np.argsort(rowex[cand_q])[:50]
                    for qq in cand_q[order]:
                        # estimate new contact count
                        new_edges = int(np.sum(D[qq, cols] > M - 1e-9))
                        rest_edges = 0
                        # count rest pairs above M
                        # quick: total minus edges incident to p
                        inc_p = int(np.sum(G[p, :] > M - 1e-9))
                        rest_c = Ccnt - inc_p
                        new_c = rest_c + new_edges
                        # also pairs among rest unchanged; if rest[p] > M then this swap invalid (already handled)
                        if rest[p] > M + 1e-12:
                            continue
                        if new_c < Ccnt:
                            old = int(sel[p])
                            selmask[old] = False
                            sel[p] = int(qq)
                            selmask[int(qq)] = True
                            S[p] = pool[int(qq)]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                # occasional two-for-two via worst-2 reinsertion
                ppm = G.max(axis=1)
                worst = np.argsort(-ppm)[:2]
                keep = [i for i in range(n) if i not in set(worst.tolist())]
                Sk = S[keep]
                # best holes for remaining set
                Dk = pool @ Sk.T
                mk = Dk.max(axis=1)
                mk[selmask] = np.inf
                q1 = int(np.argmin(mk))
                # tentative add q1 then second
                tmp_sel = Sk.tolist()
                tmp_sel.append(pool[q1].tolist())
                Tarr = np.array(tmp_sel)
                D2 = pool @ Tarr.T
                m2 = D2.max(axis=1)
                m2[selmask] = np.inf
                m2[q1] = np.inf
                q2 = int(np.argmin(m2))
                # evaluate new set
                newS = Sk.copy()
                newS = np.vstack([newS, pool[q1], pool[q2]])
                Gn = newS @ newS.T
                np.fill_diagonal(Gn, -2.0)
                Mn = float(Gn.max())
                if Mn < M - 1e-12:
                    # replace worst with q1,q2
                    for k_, qq in zip(worst, (q1, q2)):
                        old = int(sel[k_])
                        selmask[old] = False
                        sel[k_] = int(qq)
                        selmask[int(qq)] = True
                        S[k_] = pool[int(qq)]
                    continue
                break
            # continuous polish of discrete set
            try:
                Sd = S.copy()
                # two-stage anneal
                Sd = _anneal(Sd, steps=700, k0=30.0, k1=600.0, lr0=0.30, lr1=0.03, w0=0.10, w1=0.008)
                if (time.time() - t0) < 70.0:
                    Sd = _anneal(Sd, steps=500, k0=80.0, k1=2500.0, lr0=0.10, lr1=0.008, w0=0.03, w1=0.004)
                x_disc_pol = Sd / np.linalg.norm(Sd, axis=1, keepdims=True)
                v_disc_pol = _max_cos(x_disc_pol)
            except Exception:
                pass
        except Exception:
            pass

    # choose champion so far
    champ = x_best.copy()
    champ_val = best_inc
    if x_disc_pol is not None and v_disc_pol < champ_val:
        champ = x_disc_pol.copy()
        champ_val = v_disc_pol

    # ---- basin hopping around incumbent to try to beat record ----
    # short anneal hops without LP per hop (save time)
    try:
        hops = 0
        while time.time() - t0 < BUDGET - 25.0 and hops < 12:
            hops += 1
            remaining = BUDGET - 25.0 - (time.time() - t0)
            if remaining < 3.0:
                break
            x = champ.copy()
            mode = hops % 4
            if mode == 0:
                x += rng.randn(n, d) * 0.006
            elif mode == 1:
                x += rng.randn(n, d) * 0.015
            elif mode == 2:
                # reinsert worst point at best random void
                Gc = x @ x.T
                np.fill_diagonal(Gc, -2.0)
                ppm = Gc.max(axis=1)
                j = int(np.argmax(ppm))
                cand = rng.randn(3000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ x.T
                dots[:, j] = -10.0
                md = dots.max(axis=1)
                x[j] = cand[int(np.argmin(md))]
                x += rng.randn(n, d) * 0.003
            else:
                x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 400 if remaining > 30 else 250
            xh = _anneal(x, steps=steps, k0=40.0, k1=1200.0, lr0=0.20, lr1=0.01, w0=0.06, w1=0.005)
            vh = _max_cos(xh)
            if vh < champ_val - 1e-12:
                champ = xh.copy()
                champ_val = vh
    except Exception:
        pass

    # final LP polish of champion
    try:
        left = BUDGET - (time.time() - t0)
        if left > 3.0:
            champ = _lp_polish(champ, min(left - 1.0, 20.0), t0, BUDGET)
            champ_val = _max_cos(champ)
    except Exception:
        pass
    # safety: if champ worse than loaded best, return loaded best
    try:
        if champ_val > best_inc + 1e-12:
            champ = x_best.copy()
    except Exception:
        pass
    champ /= np.linalg.norm(champ, axis=1, keepdims=True)
    return champ
