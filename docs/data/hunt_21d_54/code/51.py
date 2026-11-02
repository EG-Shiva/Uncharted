import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        csv = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-csv, 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    T = np.empty((n, d, d - 1))
    e1 = np.zeros(d); e1[0] = 1.0
    for i in range(n):
        xi = x[i]
        # Householder mapping xi to +/- e1
        s = 1.0 if xi[0] >= 0 else -1.0
        v = xi + s * e1
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            # xi ~ -s*e1, use e2
            e2 = np.zeros(d); e2[1] = 1.0
            v = xi + e2
            nv = np.linalg.norm(v)
        v = v / nv
        H = np.eye(d) - 2.0 * np.outer(v, v)
        T[i] = H[:, 1:]
    return T

def _null_modes(x, T, k=20, thr=1e-3, pcap=2000):
    n, d = x.shape
    dm = d - 1
    N = n * dm
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > m - thr
    I, J = iu[0][sel], iu[1][sel]
    order = np.argsort(-c[sel])
    I, J = I[order], J[order]
    if len(I) > pcap:
        I, J = I[:pcap], J[:pcap]
    p = len(I)
    if p < 10:
        return None, m
    A = np.empty((p, N))
    for r in range(p):
        i, j = int(I[r]), int(J[r])
        vi = T[i].T @ x[j]
        vj = T[j].T @ x[i]
        row = np.zeros(N)
        row[i*dm:(i+1)*dm] = vi
        row[j*dm:(j+1)*dm] = vj
        A[r] = row
    Gram = A.T @ A
    # symmetrize
    Gram = 0.5 * (Gram + Gram.T)
    try:
        w, V = np.linalg.eigh(Gram)
    except Exception:
        return None, m
    # smallest k
    U = V[:, :k]
    return U, m

def solve():
    T0 = time.time()
    HARD_END = T0 + 165.0
    EXPL_END = T0 + 130.0
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy").astype(float)
    except Exception:
        b0 = np.load("record.npy").astype(float)
    b0 = _proj(np.nan_to_num(b0))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.load("record.npy").astype(float))
    except Exception:
        r0 = b0.copy()
    bv = _max_cos(b0)
    rv = _max_cos(r0)
    best = b0.copy() if bv <= rv else r0.copy()
    bv = min(bv, rv)
    n, d = 54, 21
    dm = d - 1
    iu = np.triu_indices(n, 1)
    cur = best.copy()
    cur_v = bv
    thetas = [0.05, 0.1, 0.2, 0.35, 0.5]
    tan_theta = [np.tan(t) for t in thetas]

    # quick ensure at local opt with tiny polish if time allows
    # skip heavy initial polish to save budget; do 2s
    if time.time() < EXPL_END:
        _rem = min(2.0, max(0.0, EXPL_END - time.time() - 5))
        if _rem > 0.5:
            cur = _lp_polish(cur, _rem)
            cur_v = _max_cos(cur)
            if cur_v < bv:
                best, bv = cur.copy(), cur_v

    outer = 0
    while time.time() < EXPL_END:
        outer += 1
        # build tangent + null modes of cur
        try:
            Tall = _tangent_bases(cur)
        except Exception:
            break
        U, mloc = _null_modes(cur, Tall, k=20, thr=1e-3, pcap=2000)
        if U is None:
            # random tangent kick fallback
            U = None
            K = 20
        else:
            K = U.shape[1]
        # incumbent contact mask for hamming (absolute threshold)
        G0 = cur @ cur.T
        c0 = G0[iu]
        m0 = float(c0.max())
        mask0 = c0 > m0 - 1e-3
        # generate combos
        ncombo = 6
        trials = []
        for ci in range(ncombo):
            if time.time() > EXPL_END - 4:
                break
            nmode = int(rng.integers(2, 6))  # 2..5
            idx = rng.choice(K if U is not None else dm*n, size=nmode, replace=False) if U is not None else None
            coef = rng.normal(size=nmode)
            coef /= (np.linalg.norm(coef) + 1e-12)
            if U is not None:
                u = U[:, idx] @ coef  # N
            else:
                u = rng.normal(size=n*dm)
                u /= (np.linalg.norm(u)+1e-12)
            u2d = u.reshape(n, dm)
            # tangent displacements
            dx = np.einsum('ijk,ik->ij', Tall, u2d)
            norms = np.linalg.norm(dx, axis=1)
            mx = norms.max()
            if mx < 1e-12:
                continue
            for ti, tt in enumerate(tan_theta):
                scale = tt / mx
                y = cur + dx * scale
                y = _proj(y)
                Gy = y @ y.T
                cy = Gy[iu]
                mv = float(cy.max())
                if mv > 0.12:
                    continue
                maskt = cy > m0 - 1e-3
                ham = int(np.sum(maskt != mask0))
                trials.append((mv, -ham, ham, y))
        if not trials:
            # small random tangent jitter to change basin
            if time.time() > EXPL_END - 2:
                break
            step = 0.05 + 0.05 * (outer % 4)
            noise = rng.normal(size=(n, d))
            noise -= (noise * cur).sum(axis=1, keepdims=True) * cur
            nn = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12
            noise /= nn
            y = np.cos(step) * cur + np.sin(step) * noise
            y = _proj(y)
            if _max_cos(y) < 0.12:
                cur = y
                cur_v = _max_cos(cur)
            continue
        # sort by max-cos then hamming (larger ham better => smaller -ham? we stored -ham)
        trials.sort(key=lambda t: (t[0], t[1]))
        # keep materially different topology: ham > 30
        cand = [t for t in trials if t[2] > 30]
        if not cand:
            # relax to >10
            cand = [t for t in trials if t[2] > 10]
        if not cand:
            cand = trials[:1]
        # take up to 3 diverse best
        picked = cand[:3]
        polished = []
        for (_, _, ham, y) in picked:
            if time.time() > EXPL_END:
                break
            left = EXPL_END - time.time()
            if left < 1.0:
                break
            sec = min(3.0, left - 0.2)
            if sec < 0.8:
                # evaluate without polish
                polished.append((_max_cos(y), ham, y))
                continue
            try:
                yp = _lp_polish(y, sec)
            except Exception:
                continue
            vp = _max_cos(yp)
            polished.append((vp, ham, yp))
            if vp < bv - 1e-12:
                best, bv = yp.copy(), vp
                cur, cur_v = yp.copy(), vp
        if not polished:
            continue
        polished.sort(key=lambda t: t[0])
        bpv, _, byp = polished[0]
        # basin move: accept if within tolerance to keep exploring new topology
        if bpv < cur_v + 0.005 and bpv < 0.12:
            cur, cur_v = byp.copy(), bpv
            if bpv < bv:
                best, bv = byp.copy(), bpv
        # early exit if record beaten well? continue to improve
        if time.time() > EXPL_END:
            break
    # final polish of best with remaining budget (reserve ~20s, hard cap)
    try:
        left = HARD_END - time.time()
        left = min(max(left, 1.0), 22.0)
        if left > 0.5:
            best = _lp_polish(best, left)
    except Exception:
        pass
    best = _proj(best)
    return best
