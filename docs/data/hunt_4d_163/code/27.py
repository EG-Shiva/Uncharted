import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, m=None, tol=1e-7):
    g = x @ x.T
    if m is None:
        np.fill_diagonal(g, -2.0)
        m = float(g.max())
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    c = g[iu0, iu1]
    sel = c > m - tol
    s = set((int(a)*1000+int(b)) for a, b in zip(iu0[sel], iu1[sel]))
    return s, m

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
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
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _gd_full(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
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
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _gd_cluster(x0, cidx, steps, T0, T1, lr0, mu, rng, taboo=None, taboo_w=6.0, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    k = len(cidx)
    mask = np.zeros(n, dtype=bool)
    mask[cidx] = True
    v = np.zeros_like(x)
    # prebuild weight matrix
    if taboo is not None:
        W = np.ones((n, n), dtype=float)
        # taboo: list of (i,j)
        for (a, b) in taboo:
            W[a, b] = taboo_w
            W[b, a] = taboo_w
    else:
        W = None
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # best over relevant pairs
    rel = G[np.ix_(mask, np.ones(n, dtype=bool))].max() if k > 0 else float(G.max())
    # also cluster-frozen and cluster-cluster; use full max involving cluster
    # compute initial best global
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for kk in range(steps):
        if deadline is not None and (kk & 127) == 0 and time.time() > deadline:
            break
        frac = kk / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        if W is not None:
            E = E * W
        np.fill_diagonal(E, 0.0)
        # zero frozen-frozen interactions (no grad needed there, but keep for S? zero them)
        # build mask for relevant: rows or cols in cluster
        # zero out frozen-frozen block
        # E_ff zero:
        # Use outer logic: relevant = mask[:,None]|mask[None,:]
        # apply:
        # (faster: set rows/cols)
        # we want grad only on cluster, frozen grad ignored, so frozen-frozen entries irrelevant
        # zero them to not affect S
        E[np.ix_(~mask, ~mask)] = 0.0
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad[~mask] = 0.0
        grad[cidx] -= (grad[cidx] * x[cidx]).sum(axis=1, keepdims=True) * x[cidx]
        v = mu * v + (1.0 - mu) * grad
        # only update cluster
        x[cidx] = x[cidx] - lr * v[cidx]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(k, d) * amp
                nz -= (nz * x[cidx]).sum(axis=1, keepdims=True) * x[cidx]
                x[cidx] = x[cidx] + nz
        x[cidx] /= np.maximum(np.linalg.norm(x[cidx], axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _refine_hole(h0, Xf, steps=80, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(np.linalg.norm(h0), 1e-12)
    if Xf.shape[0] == 0:
        return h
    lratio = np.log(T1 / T0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / max(T, 1e-12))
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = np.linalg.norm(h)
        if nh < 1e-12:
            break
        h /= nh
    return h

def _find_hole(Xf, rng, n_samples=6000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    if Xf.shape[0] == 0:
        return Y[0], -1.0
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:4]
    best_h = Y[idx[0]]
    best_v = float(vals[idx[0]])
    for ii in idx[:2]:
        h = _refine_hole(Y[ii], Xf)
        v = float((Xf @ h).max())
        if v < best_v:
            best_v = v
            best_h = h
    return best_h, best_v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 130.0
    rng = np.random.RandomState(12345)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands = []
    if bx is not None:
        bx = np.asarray(bx, dtype=float)
        if bx.shape == (163, 4):
            cands.append(bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12))
    if rec is not None:
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (163, 4):
            cands.append(rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12))
    if len(cands) == 0:
        c = rng.randn(n, 4)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    best_contacts, _ = _contacts(best_x, best_val, tol=1e-7)

    trial = 0
    # precompute loop
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 38:
            break
        # recompute degrees of incumbent
        G0 = best_x @ best_x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        deg = (G0 > m0 - 1e-7).sum(axis=1)
        rowm = G0.max(axis=1)
        order_high = np.argsort(deg + (rowm - m0) * 50.0)  # high at end
        order_low = np.argsort(-deg + rng.randn(n) * 0.01)
        mode = trial % 3  # 0 hub, 1 low-contact, 2 spatial
        if mode == 0:
            # high-contact hub cycling
            hub = int(order_high[-(trial // 3) % n - 1])
            # occasionally pick 2nd hub
            target = 8 + int(rng.randint(0, 9))  # 8..16
            want_low_boundary = (trial % 2 == 0)
        elif mode == 1:
            # low-contact centred
            hub = int(np.argsort(rowm + rng.randn(n) * 1e-5)[-(trial // 3) % 20 - 1]) if trial % 4 == 0 else int(order_low[(trial // 3) % n])
            # low-contact centre: still expand via contacts+spatial
            target = 10 + int(rng.randint(0, 9))  # 10..18
            want_low_boundary = True
        else:
            # spatially localised: random direction cap
            u = rng.randn(4); u /= np.linalg.norm(u)
            dots = best_x @ u
            hub = int(np.argmax(dots + rng.randn(n) * 0.02))
            target = 8 + int(rng.randint(0, 13))  # 8..20
            want_low_boundary = (rng.rand() < 0.5)
        target = int(min(max(target, 8), 20))
        # build connected cluster: hub + contact neighbours
        neigh = np.where(G0[hub] > m0 - 1e-7)[0].tolist()
        # also near neighbours by cosine if not enough
        if len(neigh) < target - 1:
            # sort by dot descending
            srt = np.argsort(-G0[hub])
            for cand in srt:
                if cand == hub:
                    continue
                if cand not in neigh:
                    neigh.append(int(cand))
                if len(neigh) >= target - 1:
                    break
        cluster = [hub] + neigh[:target - 1]
        # optionally add one low-contact boundary point nearest to hub
        if want_low_boundary and len(cluster) < 20:
            # low-contact candidates: smallest deg
            low_cands = np.argsort(deg)
            # nearest among low 30 to hub
            best_d = -2
            best_c = -1
            for lc in low_cands[:40]:
                if lc in cluster:
                    continue
                d = float(G0[hub, lc])
                # prefer spatially adjacent (high dot) but low contact
                if d > best_d:
                    best_d = d
                    best_c = int(lc)
            if best_c >= 0:
                # replace farthest or append
                if len(cluster) >= target:
                    # replace farthest from hub
                    far = min(cluster, key=lambda c: float(G0[hub, c]) if c != hub else 2.0)
                    if far != hub:
                        cluster.remove(far)
                cluster.append(best_c)
        # second-ring expansion to ensure connectivity size exactly target-ish: trim/pad
        cluster = list(dict.fromkeys(cluster))[:20]
        while len(cluster) < 8:
            srt = np.argsort(-G0[hub])
            for cand in srt:
                if int(cand) not in cluster:
                    cluster.append(int(cand))
                    break
        cidx = np.array(sorted(cluster), dtype=int)
        k = len(cidx)
        # taboo pairs: old contacts involving cluster above m0-1e-4
        iu_r, iu_c = np.where(G0 > m0 - 1e-4)
        taboo = []
        cset = set(cidx.tolist())
        for a, b in zip(iu_r.tolist(), iu_c.tolist()):
            if a >= b:
                continue
            if (a in cset) or (b in cset):
                taboo.append((int(a), int(b)))
        # cap taboo length
        if len(taboo) > 400:
            # keep tightest
            taboo_sorted = sorted(taboo, key=lambda p: -float(G0[p[0], p[1]]))[:400]
            taboo = taboo_sorted
        # --- reinsertion ---
        cur = best_x.copy()
        # greedy maximin hole placement for cluster points
        frozen_idx = np.array([i for i in range(n) if i not in cset], dtype=int)
        # sequential holes
        placed = cur[frozen_idx].copy() if len(frozen_idx) > 0 else np.zeros((0, 4))
        # random order
        order = cidx.copy()
        rng.shuffle(order)
        for w in order:
            h, _ = _find_hole(placed, rng, n_samples=6000)
            # small jitter to break symmetry
            h = h + rng.randn(4) * 0.005
            h /= max(np.linalg.norm(h), 1e-12)
            cur[w] = h
            placed = np.vstack([placed, h[None, :]])
        # cluster-only optimisation with soft-max + taboo penalty
        dl1 = min(time.time() + 2.5, SEARCH_END)
        taboo_w = 7.0 if (trial % 3 != 2) else 4.0
        cur_b, vb = _gd_cluster(cur, cidx, steps=2200, T0=0.018, T1=0.0007, lr0=0.05, mu=0.86,
                                rng=rng, taboo=taboo, taboo_w=taboo_w, noise0=0.004, deadline=dl1)
        # short joint annealing (all points, no taboo)
        dl2 = min(time.time() + 1.8, SEARCH_END)
        cur_b, vb2 = _gd_full(cur_b, steps=1800, T0=0.006, T1=0.00045, lr0=0.032, mu=0.87,
                              rng=rng, noise0=0.0012, deadline=dl2)
        vb = vb2 if vb2 < vb else vb
        # quick gate: only polish if promising
        if vb < best_val + 0.002:
            # LP polish short
            dl3 = min(time.time() + 2.2, SEARCH_END)
            pol, vp = _lp_polish(cur_b, deadline=dl3, r0=8e-4)
            # edit distance
            try:
                new_contacts, m_new = _contacts(pol, vp, tol=1e-7)
                inter = len(new_contacts & best_contacts)
                union = len(new_contacts | best_contacts)
                edit = union - inter
            except Exception:
                edit = 999
                vp = _max_cos(pol)
            # retain only when substantial edit distance, unless big gain
            need_edit = 25
            if vp < best_val - 1e-12 and (edit >= need_edit or vp < best_val - 4e-7):
                best_x, best_val = pol, vp
                best_contacts = new_contacts
            elif vp < best_val - 1e-12 and edit >= 60:
                # allow slightly worse? no, keep best only
                pass
            else:
                # even if not retained as best, if vp close and topology novel, use as basis for next trial occasionally
                if vp < best_val + 0.0005 and edit >= 80 and rng.rand() < 0.15:
                    # jump to novel basin without improving (diversify)
                    best_x, best_val = pol, vp
                    best_contacts = new_contacts
        trial += 1
        # occasional light full restart to feed diversity (rare)
        if trial % 25 == 24 and time.time() < SEARCH_END - 10:
            rc = rng.randn(n, 4)
            rc /= np.linalg.norm(rc, axis=1, keepdims=True)
            dl = min(time.time() + 2.0, SEARCH_END)
            rc_b, rv = _gd_full(rc, steps=3500, T0=0.025, T1=0.0009, lr0=0.038, mu=0.88, rng=rng, noise0=0.002, deadline=dl)
            if rv < best_val:
                dl3 = min(time.time() + 1.5, SEARCH_END)
                pol, vp = _lp_polish(rc_b, deadline=dl3)
                if vp < best_val:
                    try:
                        nc, _ = _contacts(pol, vp, tol=1e-7)
                        ed = len(nc | best_contacts) - len(nc & best_contacts)
                    except Exception:
                        ed = 999
                    if ed >= 25 or vp < best_val - 4e-7:
                        best_x, best_val = pol, vp
                        best_contacts = nc
    # final polish with remaining time
    try:
        fin_dead = t0 + BUDGET - 2.0
        best_x, best_val = _lp_polish(best_x, deadline=fin_dead, r0=6e-4)
    except Exception:
        pass
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
