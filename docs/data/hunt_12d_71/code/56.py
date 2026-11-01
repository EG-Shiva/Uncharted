import time
import numpy as np

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_stats(X, gmax=None):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if gmax is None:
        gmax = float(G.max())
    row_max = G.max(axis=1)
    cnt = (G > (gmax - 1e-3)).sum(axis=1)
    return row_max, cnt, gmax

def _refine_single(s0, R, stages=((0.004, 150, 0.12, 0.80), (0.0005, 150, 0.05, 0.80))):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    v = np.zeros_like(s)
    m = R.shape[0]
    for (T, steps, lr, mu) in stages:
        v[:] = 0.0
        invT = 1.0 / T
        for _ in range(steps):
            g = R @ s
            mg = float(g.max())
            # exp((g-mg)/T)
            e = np.exp((g - mg) * invT)
            den = float(e.sum()) + 1e-30
            grad = (R.T @ e) / den
            c = float(grad @ s)
            Gt = grad - c * s
            v = mu * v + Gt
            s = s - lr * v
            nrm = float(np.linalg.norm(s))
            if not np.isfinite(nrm) or nrm < 1e-12:
                s = s0 / (np.linalg.norm(s0) + 1e-30)
                v[:] = 0.0
                nrm = 1.0
            else:
                s = s / nrm
    return s

def _best_hole(R, rng, n_uniform=14000, n_pair=2500, n_z_top=4):
    m, d = R.shape
    # uniform
    Z = rng.standard_normal((n_uniform, d))
    zn = np.linalg.norm(Z, axis=1, keepdims=True)
    zn[zn == 0] = 1.0
    Z /= zn
    S = Z @ R.T
    smax = S.max(axis=1)
    # pair negative sums
    ia = rng.integers(0, m, size=n_pair)
    ib = rng.integers(0, m, size=n_pair)
    V = -(R[ia] + R[ib])
    vn = np.linalg.norm(V, axis=1, keepdims=True)
    # fix degenerate
    bad = (~np.isfinite(vn[:, 0])) | (vn[:, 0] < 1e-12)
    if np.any(bad):
        nb = int(bad.sum())
        Rb = rng.standard_normal((nb, d))
        Rn = np.linalg.norm(Rb, axis=1, keepdims=True)
        Rn[Rn == 0] = 1.0
        V[bad] = Rb / Rn
        vn[bad] = 1.0
    V /= vn
    S2 = V @ R.T
    smax2 = S2.max(axis=1)
    # antipodes
    A = -R
    SA = A @ R.T
    smaxA = SA.max(axis=1)
    # collect starts
    # top uniform
    if n_z_top >= n_uniform:
        z_idx = np.argsort(smax)
    else:
        z_idx = np.argpartition(smax, n_z_top)[:n_z_top]
    # top pair
    n_v_top = 2
    if n_v_top >= n_pair:
        v_idx = np.argsort(smax2)
    else:
        v_idx = np.argpartition(smax2, n_v_top)[:n_v_top]
    a_idx = int(np.argmin(smaxA))
    starts = []
    for k in z_idx[:n_z_top]:
        starts.append(Z[k])
    for k in v_idx[:n_v_top]:
        starts.append(V[k])
    starts.append(A[a_idx])
    best_y = None
    best_v = 1e100
    # quick pre-check true values to order refinement (refine best first not needed)
    for s0 in starts:
        # refine
        try:
            s1 = _refine_single(s0, R)
        except Exception:
            continue
        vv = float((R @ s1).max())
        if vv < best_v:
            best_v = vv
            best_y = s1
    # fallback
    if best_y is None:
        k0 = int(np.argmin(smax))
        best_y = Z[k0] / (np.linalg.norm(Z[k0]) + 1e-30)
        best_v = float((R @ best_y).max())
    return best_y / (np.linalg.norm(best_y) + 1e-30), float(best_v)

def _quench(X_init, steps, T0, T1, lr, mu=0.88):
    X = _norm_rows(np.array(X_init, dtype=float, copy=True))
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        mm = float(G.max())
        Q = (G - mm) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 120 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((n, d)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    best_X = _norm_rows(best_X)

    def time_left(sec=6.0):
        return time.time() < (HARD - sec)

    # ---- Phase 1: leave-1-out global reinsertion, tightest first ----
    # do up to 2 full sweeps
    for sweep in range(3):
        if not time_left(45.0):
            break
        row_max, cnt, gmax = _row_stats(best_X, best_score)
        # rank: (a) largest cosine desc, (b) count desc -> tightest first
        # also ensure loosest 20 are included (they are at tail)
        order = np.lexsort((np.arange(n), -cnt, -row_max))
        # lexsort gives sorted by -row_max primary? last key primary.
        # we want primary -row_max, secondary -cnt: so keys = (-cnt, -row_max) with last primary
        # above: keys (arange, -cnt, -row_max) -> primary -row_max correct
        improved = False
        for idx in order:
            if time.time() > HARD - 40.0:
                break
            idx = int(idx)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            R = best_X[mask]
            try:
                m_rest = _max_cos(R)
            except Exception:
                continue
            if m_rest >= best_score:
                continue
            try:
                y, v = _best_hole(R, rng, n_uniform=14000, n_pair=2500, n_z_top=4)
            except Exception:
                continue
            m_new = m_rest if m_rest > v else v
            if m_new < best_score - 1e-12:
                Xnew = best_X.copy()
                Xnew[idx] = y
                Xnew = _norm_rows(Xnew)
                # short smooth quench (not LP)
                try:
                    Xq, vq = _quench(Xnew, steps=500, T0=0.005, T1=0.0004, lr=0.16, mu=0.88)
                except Exception:
                    Xq, vq = Xnew, m_new
                if vq < best_score:
                    best_score, best_X = vq, Xq.copy()
                    improved = True
                    row_max, cnt, gmax = _row_stats(best_X, best_score)
                elif m_new < best_score:
                    best_score, best_X = m_new, Xnew.copy()
                    improved = True
                # restart sweep after acceptance to re-rank
                if improved:
                    break
        if not improved:
            break

    # ---- Phase 2: k=2 simultaneous deletions, greedy sequential reinsertion ----
    if time_left(30.0):
        # build pair list
        try:
            row_max, cnt, gmax = _row_stats(best_X, best_score)
        except Exception:
            row_max = np.zeros(n)
            cnt = np.zeros(n)
        tight = list(np.argsort(-row_max)[:12])
        loose = list(np.argsort(row_max)[:20])
        pairs = []
        # all pairs among tight
        for ii in range(len(tight)):
            for jj in range(ii + 1, len(tight)):
                pairs.append((int(tight[ii]), int(tight[jj])))
        # tight x random/loose
        for a in tight:
            # 6 random partners
            for _ in range(6):
                b = int(rng.integers(0, n))
                if b == a:
                    continue
                p = (a, b) if a < b else (b, a)
                pairs.append(p)
            # 2 loose partners
            for _ in range(2):
                b = loose[int(rng.integers(0, len(loose)))]
                if b == a:
                    continue
                p = (a, b) if a < b else (b, a)
                pairs.append(p)
        # random pairs
        for _ in range(60):
            a = int(rng.integers(0, n)); b = int(rng.integers(0, n))
            if a == b:
                continue
            p = (a, b) if a < b else (b, a)
            pairs.append(p)
        # dedupe preserve order
        seen = set()
        upairs = []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                upairs.append(p)
        # shuffle deterministically via rng
        perm = rng.permutation(len(upairs))
        upairs = [upairs[i] for i in perm]
        for (a, b) in upairs:
            if time.time() > HARD - 18.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[a] = False; mask[b] = False
            R = best_X[mask]
            try:
                m_rest = _max_cos(R)
            except Exception:
                continue
            if m_rest >= best_score:
                continue
            try:
                y1, v1 = _best_hole(R, rng, n_uniform=12000, n_pair=2000, n_z_top=3)
            except Exception:
                continue
            # early prune: if v1 already >= best, still need second point but R1 max >= best -> fail
            # R1 max = max(m_rest, v1); if v1>=best then m after 2 inserts >=best, skip
            if v1 >= best_score:
                continue
            R1 = np.vstack([R, y1])
            try:
                y2, v2 = _best_hole(R1, rng, n_uniform=12000, n_pair=2000, n_z_top=3)
            except Exception:
                continue
            m_new = m_rest
            if v1 > m_new:
                m_new = v1
            # max of R1 internal already m_new; v2 is max(y2 vs R1)
            if v2 > m_new:
                m_new = v2
            if m_new < best_score - 1e-12:
                Xnew = np.vstack([R, y1, y2])
                Xnew = _norm_rows(Xnew)
                try:
                    Xq, vq = _quench(Xnew, steps=500, T0=0.005, T1=0.0004, lr=0.16, mu=0.88)
                except Exception:
                    Xq, vq = Xnew, m_new
                if vq < best_score:
                    best_score, best_X = vq, Xq.copy()
                elif m_new < best_score:
                    best_score, best_X = m_new, Xnew.copy()

    # ---- Phase 3: k=3 triples (max pair + one other), greedy sequential ----
    if time_left(12.0):
        try:
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii, jj = divmod(flat, n)
        except Exception:
            ii, jj = 0, 1
        triples = []
        for k in range(n):
            if k == ii or k == jj:
                continue
            triples.append((ii, jj, k))
        # subsample if too many / time low: take every other + random
        # prioritize loosest k
        try:
            row_max2, _, _ = _row_stats(best_X, best_score)
            orderk = np.argsort(row_max2)
            # reorder triples by looseness of k
            triples = sorted(triples, key=lambda t: float(row_max2[t[2]]))
        except Exception:
            pass
        # limit to ~35 attempts
        if len(triples) > 35:
            # take 35 spread
            idxs = np.linspace(0, len(triples) - 1, 35).astype(int)
            triples = [triples[i] for i in idxs]
        for (a, b, c) in triples:
            if time.time() > HARD - 6.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[a] = False; mask[b] = False; mask[c] = False
            R = best_X[mask]
            try:
                m_rest = _max_cos(R)
            except Exception:
                continue
            if m_rest >= best_score:
                continue
            try:
                y1, v1 = _best_hole(R, rng, n_uniform=10000, n_pair=1500, n_z_top=3)
            except Exception:
                continue
            if v1 >= best_score:
                continue
            R1 = np.vstack([R, y1])
            try:
                y2, v2 = _best_hole(R1, rng, n_uniform=10000, n_pair=1500, n_z_top=3)
            except Exception:
                continue
            if v2 >= best_score:
                continue
            # also need max(R1)=max(m_rest,v1) < best already ensured
            R2 = np.vstack([R1, y2])
            try:
                y3, v3 = _best_hole(R2, rng, n_uniform=10000, n_pair=1500, n_z_top=3)
            except Exception:
                continue
            m_new = max(m_rest, v1, v2, v3)
            if m_new < best_score - 1e-12:
                Xnew = np.vstack([R, y1, y2, y3])
                Xnew = _norm_rows(Xnew)
                try:
                    Xq, vq = _quench(Xnew, steps=400, T0=0.004, T1=0.0004, lr=0.15, mu=0.88)
                except Exception:
                    Xq, vq = Xnew, m_new
                if vq < best_score:
                    best_score, best_X = vq, Xq.copy()
                elif m_new < best_score:
                    best_score, best_X = m_new, Xnew.copy()

    # final tiny quench polish if time allows and best still improvable
    if time.time() < HARD - 5.0:
        try:
            Xq, vq = _quench(best_X, steps=400, T0=0.003, T1=0.0003, lr=0.12, mu=0.88)
            if vq < best_score:
                best_score, best_X = vq, Xq.copy()
        except Exception:
            pass
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    return best_X
