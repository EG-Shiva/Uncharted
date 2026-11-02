import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_stats(x, tol=1e-7):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    gall = G[iu]
    mx = gall.max()
    tight = gall > mx - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    E = int(tight.sum())
    return mx, E, np.sort(cnt), cnt

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _quat_mul(g, s):
    # g:(G,4), s:(S,4) -> (G,S,4) left mult g*s, order w,x,y,z
    w1 = g[:, 0, None]; x1 = g[:, 1, None]; y1 = g[:, 2, None]; z1 = g[:, 3, None]
    w2 = s[:, 0][None, :]; x2 = s[:, 1][None, :]; y2 = s[:, 2][None, :]; z2 = s[:, 3][None, :]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def _bt24():
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for signs in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(signs))
    return np.array(pts)

def _bi120():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    a = phi / 2.0; b = 0.5; c = 1.0 / (2.0 * phi)
    base = np.array([a, b, c, 0.0])
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for signs in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(signs))
    # 96 even perms of (a,b,c,0) with signs
    for perm in itertools.permutations([0, 1, 2, 3]):
        # parity
        inv = 0
        for i in range(4):
            for j in range(i+1, 4):
                if perm[i] > perm[j]:
                    inv += 1
        if inv % 2 == 1:
            continue
        # position of zero = index i where perm[i]==3
        zero_pos = perm.index(3)
        for s0 in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    sm = {0: s0, 1: s1, 2: s2}
                    v = np.zeros(4)
                    for i in range(4):
                        bi = perm[i]
                        if bi == 3:
                            v[i] = 0.0
                        else:
                            v[i] = sm[bi] * base[bi] * 2.0 / 2.0  # base already signed magnitude
                            # base[bi] is positive magnitude; apply sign
                            # need magnitude: base values are positive
                            v[i] = sm[bi] * abs(base[bi])
                    pts.append(v)
    pts = np.array(pts)
    # deduplicate
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    # round and unique
    r = np.round(pts, 10)
    _, idx = np.unique(r, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    return pts

def _rand_so4(rng):
    A = rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d)[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _clifford_layer(nt, npp, a, ot, op):
    th = 2*np.pi*np.arange(nt)/nt + ot
    ph = 2*np.pi*np.arange(npp)/npp + op
    T, P = np.meshgrid(th, ph, indexing='ij')
    T = T.ravel(); P = P.ravel()
    ca = np.cos(a); sa = np.sin(a)
    x = np.stack([ca*np.cos(T), ca*np.sin(T), sa*np.cos(P), sa*np.sin(P)], axis=1)
    return x

def _farthest_pad(X, need, rng, pool=2000):
    # greedily add points maximizing min distance (min cosine)
    cur = X.copy()
    for _ in range(need):
        cand = rng.normal(size=(pool, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # max cosine to cur
        G = cand @ cur.T
        mx = G.max(axis=1)
        best_i = np.argmin(mx)
        cur = np.concatenate([cur, cand[best_i:best_i+1]], axis=0)
    return cur

def _greedy_trim(X, target, rng):
    cur = X.copy()
    while cur.shape[0] > target:
        G = cur @ cur.T
        np.fill_diagonal(G, -2)
        i, j = np.unravel_index(np.argmax(G), G.shape)
        # remove one of the pair with larger second-max (or random)
        # remove j
        cur = np.delete(cur, j, axis=0)
    return cur

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    # incumbent fingerprint
    _, E0, sdeg0, _ = _contact_stats(x_best)

    BT = _bt24()
    try:
        BI = _bi120()
    except Exception:
        BI = BT
    # ensure BT is 24
    if BT.shape[0] != 24:
        BT = BT[:24]

    cands = []  # list of (score, X_raw, name)
    # --- Family A1: single Clifford layer 17x9, search a ---
    best_a_score = 1e9; best_a_X = None; best_a = None
    for a in np.linspace(0.35, 1.22, 31):
        X = _clifford_layer(17, 9, float(a), 0.0, 0.0)
        v = _max_cos(X)
        if v < best_a_score:
            best_a_score = v; best_a_X = X; best_a = float(a)
    cands.append((best_a_score, best_a_X.copy(), "cliff17x9"))

    # --- Family A2: 3-layer Clifford 9x6+9x6+9x5=153, random search ---
    best2 = 1e9; best2X = None
    for _ in range(120):
        if time.time() - t0 > 12:
            break
        a1 = rng.uniform(0.25, 1.3); a2 = rng.uniform(0.25, 1.3); a3 = rng.uniform(0.25, 1.3)
        ot2 = rng.uniform(0, 2*np.pi); op2 = rng.uniform(0, 2*np.pi)
        ot3 = rng.uniform(0, 2*np.pi); op3 = rng.uniform(0, 2*np.pi)
        X1 = _clifford_layer(9, 6, a1, 0.0, 0.0)
        X2 = _clifford_layer(9, 6, a2, ot2, op2)
        X3 = _clifford_layer(9, 5, a3, ot3, op3)
        X = np.concatenate([X1, X2, X3], axis=0)
        v = _max_cos(X)
        if v < best2:
            best2 = v; best2X = X.copy()
    if best2X is not None:
        cands.append((best2, best2X, "cliff3lay"))

    # --- Family B1: BT orbits 6 seeds ->144 + pad 9 ---
    bestB = 1e9; bestBX = None
    for _ in range(60):
        if time.time() - t0 > 20:
            break
        seeds = rng.normal(size=(6, 4)); seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
        orb = _quat_mul(BT, seeds)  # (24,6,4)
        X = orb.transpose(1, 0, 2).reshape(-1, 4)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # dedup/trim not needed: 144
        X = _farthest_pad(X, 9, rng, pool=400)
        v = _max_cos(X)
        if v < bestB:
            bestB = v; bestBX = X.copy()
    if bestBX is not None:
        cands.append((bestB, bestBX, "bt6"))

    # --- Family B2: BI orbit 1 seed -> ~120 + pad ---
    # determine BI size
    bi_n = BI.shape[0]
    bestC = 1e9; bestCX = None
    for _ in range(40):
        if time.time() - t0 > 26:
            break
        seed = rng.normal(size=(1, 4)); seed /= np.linalg.norm(seed, axis=1, keepdims=True)
        orb = _quat_mul(BI, seed)  # (bi_n,1,4)
        X = orb.transpose(1, 0, 2).reshape(-1, 4)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if X.shape[0] > 153:
            X = _greedy_trim(X, 153, rng)
        elif X.shape[0] < 153:
            X = _farthest_pad(X, 153 - X.shape[0], rng, pool=400)
        v = _max_cos(X)
        if v < bestC:
            bestC = v; bestCX = X.copy()
    if bestCX is not None:
        cands.append((bestC, bestCX, "bi1"))

    # --- Family C: 600-cell(120) + rotated 24-cell(24) + 9 ring ---
    # use BI[:120] if available else first 120
    cell120 = BI[:120] if BI.shape[0] >= 120 else BI
    if cell120.shape[0] == 120:
        bestD = 1e9; bestDX = None
        # 24-cell base
        base24 = _bt24()  # same as 24-cell
        for _ in range(40):
            if time.time() - t0 > 30:
                break
            R = _rand_so4(rng)
            Y = base24 @ R.T
            # 9-ring in random 2-plane: use last rotation rows?
            R2 = _rand_so4(rng)
            ang = 2*np.pi*np.arange(9)/9 + rng.uniform(0, 2*np.pi)
            ring = np.zeros((9, 4))
            # circle in first two coords of rotated frame + small offset in others to be on S3
            # put ring at latitude: mix
            lat = rng.uniform(0.3, 1.2)
            ring[:, 0] = np.cos(lat)*np.cos(ang)
            ring[:, 1] = np.cos(lat)*np.sin(ang)
            ring[:, 2] = np.sin(lat)/np.sqrt(2)*np.ones(9)
            ring[:, 3] = np.sin(lat)/np.sqrt(2)*np.ones(9)
            # randomize phase of last two? add variation
            ring = ring @ R2.T
            ring /= np.linalg.norm(ring, axis=1, keepdims=True)
            # small random rotation of cell120 to break exact overlap
            R3 = _rand_so4(rng)
            # only rotate slightly? use full random for diversity on some draws
            X = np.concatenate([cell120 @ R3.T, Y, ring], axis=0)
            if X.shape[0] != 153:
                if X.shape[0] > 153:
                    X = _greedy_trim(X, 153, rng)
                else:
                    X = _farthest_pad(X, 153 - X.shape[0], rng, pool=400)
            v = _max_cos(X)
            if v < bestD:
                bestD = v; bestDX = X.copy()
        if bestDX is not None:
            cands.append((bestD, bestDX, "cell120+24+9"))

    # sort families by raw score, keep best 4
    cands.sort(key=lambda t: t[0])
    cands = cands[:4]

    polished = []  # (score, X, name, dissim)
    # release symmetry: softmax anneal + LP polish each family best
    for sc, X0, nm in cands:
        if time.time() - t0 > HARD - 60:
            break
        # anneal with moderate schedule
        try:
            Xa = _smooth_opt(X0, 700, 30.0, 300.0, 0.006, 0.0005, iu)
        except Exception:
            Xa = X0
        va = _max_cos(Xa)
        # quick LP
        remain = HARD - (time.time() - t0) - 40.0
        b = 1.2
        if remain < 5:
            b = 0.8
        try:
            Xp, vp = _lp_polish(Xa, b)
        except Exception:
            Xp, vp = Xa, va
        _, E, sdeg, _ = _contact_stats(Xp)
        dissim = float(np.linalg.norm(sdeg.astype(float) - sdeg0.astype(float))) + abs(E - E0) / 10.0)
        polished.append([vp, Xp, nm, dissim])
        if vp < best_val - 1e-12:
            best_val = vp; x_best = Xp.copy()

    # iterative cooperative improvement of polished pool
    it = 0
    while time.time() - t0 < HARD - 34:
        if not polished:
            break
        # round robin, bias to better + dissimilar
        polished.sort(key=lambda t: t[0])
        idx = it % len(polished)
        # occasionally pick most dissimilar
        if it % 5 == 4:
            # pick max dissim among within 0.01
            thr = min(p[0] for p in polished) + 0.01
            elig = [i for i, p in enumerate(polished) if p[0] < thr]
            if elig:
                idx = max(elig, key=lambda i: polished[i][3])
        base = polished[idx][1]
        sig = [0.02, 0.035, 0.06][it % 3]
        y = base + rng.normal(0, sig, size=base.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        beta1 = [280.0, 350.0, 220.0][it % 3]
        try:
            y2 = _smooth_opt(y, 550, 35.0, beta1, 0.006, 0.0004, iu)
        except Exception:
            it += 1
            continue
        v2 = _max_cos(y2)
        # polish if promising
        cur_best = min(best_val, min(p[0] for p in polished))
        if v2 < cur_best + 0.004:
            try:
                yp, vp = _lp_polish(y2, 1.0)
            except Exception:
                it += 1
                continue
            _, E, sdeg, _ = _contact_stats(yp)
            dissim = float(np.linalg.norm(sdeg.astype(float) - sdeg0.astype(float))) + abs(E - E0) / 10.0
            polished.append([vp, yp, f"hop{it}", dissim])
            # keep pool bounded
            polished.sort(key=lambda t: t[0])
            polished = polished[:8]
            if vp < best_val - 1e-12:
                best_val = vp; x_best = yp.copy()
        it += 1
        if time.time() - t0 > HARD - 34:
            break

    # final selection: most dissimilar within 0.003 of incumbent global best
    if polished:
        gmin = min(best_val, min(p[0] for p in polished))
        elig = [p for p in polished if p[0] < gmin + 0.003 + 1e-12]
        # also consider x_best
        if elig:
            elig.sort(key=lambda p: p[3], reverse=True)
            target = elig[0]
        else:
            polished.sort(key=lambda p: p[0])
            target = polished[0]
        remain = HARD - (time.time() - t0) - 2.0
        if remain > 2.0:
            remain = min(remain, 28.0)
            try:
                Xt, vt = _lp_polish(target[1], remain)
            except Exception:
                Xt, vt = target[1], target[0]
            if vt < best_val - 1e-12:
                best_val = vt; x_best = Xt.copy()
            else:
                # even if not better, if target is dissimilar keep best-score logic:
                # ensure x_best is global best among polished+Xt
                allc = list(polished) + [[vt, Xt, "final", target[3]]]
                allc.sort(key=lambda p: p[0])
                if allc[0][0] < best_val - 1e-12:
                    best_val = allc[0][0]; x_best = allc[0][1].copy()
    else:
        remain = HARD - (time.time() - t0) - 2.0
        if remain > 2.0:
            remain = min(remain, 28.0)
            try:
                x_best, best_val = _lp_polish(x_best, remain)
            except Exception:
                pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
