import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _anneal(X, steps, k0, k1, lr0, lr1, w0, w1):
    n, d = X.shape
    x = X.copy()
    iu = np.triu_indices(n, 1)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        m = float(G[iu].max())
        thr = m - w
        W = np.zeros_like(G)
        mask = G > thr
        np.fill_diagonal(mask, False)
        if np.any(mask):
            W[mask] = np.exp(k * (G[mask] - m))
            Gr = W @ x
            dot = np.sum(Gr * x, axis=1, keepdims=True)
            Gt = Gr - dot * x
            deg = np.sum(W, axis=1, keepdims=True) + 1.0
            x = x - lr * Gt / deg
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
    return x

def _lp_polish(X, seconds, t0=None, budget=1e9):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 12:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
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
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            x = bx.copy()
            if radius < 5e-4:
                break
    return bx

def _paley_h12():
    q = 11
    chi = np.zeros(q)
    for a in range(1, q):
        l = pow(a, (q - 1) // 2, q)
        chi[a] = 1.0 if l == 1 else -1.0
    Q = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            Q[i, j] = chi[(j - i) % q]
    H = np.ones((12, 12))
    H[0, 1:] = 1.0
    H[1:, 0] = -1.0
    H[1:, 1:] = Q + np.eye(q)
    return H

def _build_pool(rng, d=23):
    parts = []
    # 1) cyclic Golay-mapped dense +-1 / sqrt23 : random signs (covers Golay codewords subset)
    M1 = 11000
    A = rng.randint(0, 2, size=(M1, d)).astype(float) * 2.0 - 1.0
    A /= np.sqrt(d)
    parts.append(A)
    # 2) quadratic-residue based weight-11/12 patterns with random signs
    # QR mod 23: residues
    residues = set(pow(i, 2, 23) for i in range(1, 23))
    residues = sorted(residues)
    base = np.zeros(d)
    for r in residues:
        base[r % d] = 1.0
    # cyclic shifts of QR support with random signs
    Mqr = 2000
    Q = np.zeros((Mqr, d))
    for k in range(Mqr):
        sh = rng.randint(0, d)
        sup = np.roll(base, sh)
        sgn = rng.randint(0, 2, size=d).astype(float) * 2 - 1
        v = sup * sgn
        # fill zeros with random +-1 with prob 0.5 to get dense variants half time
        if k % 2 == 0:
            z = (v == 0)
            v[z] = rng.randint(0, 2, size=np.count_nonzero(z)).astype(float) * 2 - 1
        nrm = np.linalg.norm(v)
        if nrm == 0:
            v = rng.randn(d)
            nrm = np.linalg.norm(v)
        Q[k] = v / nrm
    parts.append(Q)
    # 3) sparse supports sizes 4-8 with random signs
    for ksup in (4, 5, 6, 7, 8):
        Mk = 2200
        S = np.zeros((Mk, d))
        for i in range(Mk):
            supp = rng.choice(d, size=ksup, replace=False)
            sgn = rng.randint(0, 2, size=ksup).astype(float) * 2 - 1
            v = np.zeros(d)
            v[supp] = sgn
            v /= np.linalg.norm(v)
            S[i] = v
        parts.append(S)
    # weight 11,12 sparse
    for ksup in (11, 12):
        Mk = 1500
        S = np.zeros((Mk, d))
        for i in range(Mk):
            supp = rng.choice(d, size=ksup, replace=False)
            sgn = rng.randint(0, 2, size=ksup).astype(float) * 2 - 1
            v = np.zeros(d)
            v[supp] = sgn
            v /= np.linalg.norm(v)
            S[i] = v
        parts.append(S)
    # 4) Hadamard-24 rows with one coordinate deleted
    try:
        H12 = _paley_h12()
        H2 = np.array([[1.0, 1.0], [1.0, -1.0]])
        H24 = np.kron(H12, H2)
        # H24 is 24x24 +-1
        H24 = np.sign(H24)
        H24[H24 == 0] = 1.0
        delvecs = []
        for r in range(24):
            row = H24[r]
            for p in range(24):
                v = np.delete(row, p)  # length 23
                v = v / np.linalg.norm(v)
                delvecs.append(v)
            # negation (antipode)
            for p in range(0, 24, 4):
                v = -np.delete(row, p)
                v = v / np.linalg.norm(v)
                delvecs.append(v)
        parts.append(np.array(delvecs))
        # signed/permuted variants of hadamard rows
        Mp = 2500
        HP = np.zeros((Mp, d))
        for i in range(Mp):
            r = rng.randint(0, 24)
            p = rng.randint(0, 24)
            v = np.delete(H24[r], p).copy()
            # random column sign flips (still Hadamard-type)
            flips = rng.randint(0, 2, size=d).astype(float) * 2 - 1
            # apply flips to half to keep diversity
            if i % 2 == 0:
                v = v * flips
            perm = rng.permutation(d)
            if i % 3 == 0:
                v = v[perm]
            v = v / np.linalg.norm(v)
            HP[i] = v
        parts.append(HP)
    except Exception:
        pass
    # 5) coordinate vectors +-e_i
    E = np.zeros((2 * d, d))
    for i in range(d):
        E[2 * i, i] = 1.0
        E[2 * i + 1, i] = -1.0
    parts.append(E)
    # 6) normalized +-e_i +- e_j
    F = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = s1
                    v[j] = s2
                    v /= np.sqrt(2.0)
                    F.append(v)
    parts.append(np.array(F))
    # 7) random gaussian normalized
    Mg = 6000
    G = rng.randn(Mg, d)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    parts.append(G)
    C = np.vstack(parts)
    # normalize
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # deduplicate (round to 8 decimals)
    Cr = np.round(C, 8)
    # use unique on bytes view for speed
    _, uidx = np.unique(Cr, axis=0, return_index=True)
    uidx = np.sort(uidx)
    C = C[uidx]
    # cap size to keep greedy fast
    if len(C) > 42000:
        sel = rng.choice(len(C), size=42000, replace=False)
        C = C[np.sort(sel)]
    return C

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(20261102)
    n, d = 54, 23
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
    best = _max_cos(xb)
    x_best = xb.copy()

    # ---- discrete pool ----
    C = _build_pool(rng, d=d)
    M = len(C)
    # greedy best-hole insertion
    selected = []
    selset = np.zeros(M, dtype=bool)
    best_arr = np.full(M, -2.0)
    # start from e0-like candidate: find closest to e0
    e0 = np.zeros(d); e0[0] = 1.0
    dots0 = C @ e0
    first = int(np.argmin(dots0))  # most orthogonal/negative to e0 for diversity
    # actually start from a dense sign vector: pick index of max variance? use first
    first = 0
    # ensure first is dense +-1 type: argmax abs sum? just use 0
    cur = first
    selected.append(cur)
    selset[cur] = True
    best_arr = np.maximum(best_arr, C @ C[cur])
    best_arr[cur] = np.inf
    for k in range(1, n):
        if time.time() - t0 > 40.0:
            break
        # pick minimizer of best_arr among unselected
        # mask selected to inf already (update each iter)
        cur = int(np.argmin(best_arr))
        selected.append(cur)
        selset[cur] = True
        dc = C @ C[cur]
        best_arr = np.maximum(best_arr, dc)
        best_arr[selset] = np.inf
    selected = np.array(selected, dtype=int)
    if len(selected) < n:
        # fill randomly
        extra = rng.choice(M, size=n - len(selected), replace=False)
        selected = np.concatenate([selected, extra])
    Xd = C[selected].copy()
    cur_max = _max_cos(Xd)

    # ---- exhaustive one-for-one swaps ----
    # precompute D
    def current_D(Xs):
        return C @ Xs.T  # M x n
    D = current_D(Xd)
    Gsel = Xd @ Xd.T
    sweeps = 0
    while time.time() - t0 < 75.0 and sweeps < 4:
        sweeps += 1
        improved = False
        # internal excl max per position
        for i in range(n):
            if time.time() - t0 > 75.0:
                break
            Gs = Gsel.copy()
            Gs[i, :] = -2.0
            Gs[:, i] = -2.0
            internal_excl = float(np.max(Gs))
            # max to others per candidate
            # fast: compute row max excl i via top-two trick using D
            # compute per-i max with vectorized op
            mask = np.ones(n, dtype=bool); mask[i] = False
            mto = np.max(D[:, mask], axis=1)
            mto[selset] = np.inf
            # also exclude i's own slot: candidate equal to current selected[i] would give 1.0; it's marked selected anyway
            vals = np.maximum(mto, internal_excl)
            cstar = int(np.argmin(vals))
            vstar = float(vals[cstar])
            if vstar < cur_max - 1e-12:
                # swap
                old = int(selected[i])
                selset[old] = False
                # best_arr not needed further
                selected[i] = cstar
                selset[cstar] = True
                Xd[i] = C[cstar]
                # update D column i and Gsel
                D[:, i] = C @ Xd[i]
                Gsel = Xd @ Xd.T
                cur_max = _max_cos(Xd)
                improved = True
        if not improved:
            break
    # ---- sampled two-for-two swaps among worst points ----
    if time.time() - t0 < 85.0:
        ppm = _per_max(Xd)
        worst = np.argsort(-ppm)[:12]
        # top candidate pool by current max-to-selected
        rowmax = np.max(D, axis=1)
        rowmax[selset] = np.inf
        order = np.argsort(rowmax)
        pool = order[:2500]
        # candidate-candidate dots needed on fly
        attempts = 2500
        for a in range(attempts):
            if time.time() - t0 > 85.0:
                break
            i, j = worst[rng.randint(0, len(worst))], worst[rng.randint(0, len(worst))]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            c1 = pool[rng.randint(0, len(pool))]
            c2 = pool[rng.randint(0, len(pool))]
            if c1 == c2 or selset[c1] or selset[c2]:
                continue
            # evaluate new max: dots of c1,c2 to rest + to each other + internal of rest
            rest = np.ones(n, dtype=bool); rest[i] = False; rest[j] = False
            Grest = Gsel[np.ix_(rest, rest)]
            # careful: Grest max
            int_rest = float(np.max(Grest)) if Grest.size else -2.0
            d1r = D[c1, rest] if np.any(rest) else np.array([-2.0])
            d2r = D[c2, rest] if np.any(rest) else np.array([-2.0])
            d12 = float(C[c1] @ C[c2])
            newmax = max(int_rest, float(np.max(d1r)) if d1r.size else -2, float(np.max(d2r)) if d2r.size else -2, d12)
            if newmax < cur_max - 1e-12:
                oldi, oldj = int(selected[i]), int(selected[j])
                selset[oldi] = False; selset[oldj] = False
                selected[i] = c1; selected[j] = c2
                selset[c1] = True; selset[c2] = True
                Xd[i] = C[c1]; Xd[j] = C[c2]
                D[:, i] = C @ Xd[i]
                D[:, j] = C @ Xd[j]
                Gsel = Xd @ Xd.T
                cur_max = newmax
                ppm = _per_max(Xd)
                worst = np.argsort(-ppm)[:12]

    # ---- continuous polish of discrete set ----
    remain = BUDGET - (time.time() - t0)
    disc_polished = Xd.copy()
    if remain > 15.0:
        # anneal: steps scaled to remain
        steps = 900 if remain > 50 else 500
        disc_polished = _anneal(Xd, steps=steps, k0=120, k1=2500, lr0=0.02, lr1=0.0015, w0=0.06, w1=0.006)
        # short LP
        if time.time() - t0 < BUDGET - 30.0:
            disc_polished = _lp_polish(disc_polished, seconds=3.0, t0=t0, budget=BUDGET - 12.0)
        vdisc = _max_cos(disc_polished)
        if vdisc < best:
            best = vdisc
            x_best = disc_polished.copy()
    # ---- extra basin hopping on current best with fast anneal ----
    it = 0
    while time.time() - t0 < BUDGET - 14.0:
        it += 1
        if time.time() - t0 > BUDGET - 14.0:
            break
        Gb = x_best @ x_best.T
        np.fill_diagonal(Gb, -2.0)
        ppmb = np.max(Gb, axis=1)
        order_w = np.argsort(-ppmb)
        x = x_best.copy()
        mode = it % 4
        if mode == 0:
            x += rng.randn(n, d) * 0.010
        elif mode == 1:
            # reinsert worst point at best void
            j = int(order_w[0])
            cand = rng.randn(1500, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            kept = np.delete(x, j, axis=0)
            dots = cand @ kept.T
            mx = dots.max(axis=1)
            x[j] = cand[np.argmin(mx)] + rng.randn(d) * 0.01
            x += rng.randn(n, d) * 0.003
        elif mode == 2:
            x += rng.randn(n, d) * 0.025
        else:
            idx = order_w[:4]
            x[idx] += rng.randn(len(idx), d) * 0.08
            x += rng.randn(n, d) * 0.004
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = _anneal(x, steps=350, k0=80, k1=2200, lr0=0.012, lr1=0.0009, w0=0.05, w1=0.005)
        v = _max_cos(x)
        if v < best - 1e-12:
            best = v
            x_best = x.copy()
        if it > 30:
            break
    # final LP polish
    ft = BUDGET - (time.time() - t0) - 1.0
    ft = min(max(ft, 1.0), 10.0)
    if ft > 0.5:
        x_best = _lp_polish(x_best, seconds=ft, t0=t0, budget=BUDGET - 0.5)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
