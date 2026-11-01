import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

# ---------- basic metrics ----------
def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

# ---------- L-BFGS log-sum-exp ----------
def make_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7,
                              'maxfun': maxiter * 2, 'maxcor': 12})
        Y = r.x.reshape(n, d)
        Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
        return Y
    except Exception:
        return x0

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds - 0.3:
            break
        if deadline is not None and time.time() > deadline - 0.6:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        if p > 2200:
            od = np.argsort(-c[sel])[:2200]
            I, J = I[od], J[od]
            cs = c[sel][od]
        else:
            cs = c[sel]
            p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        improved = False
        for sh in (1.0, 0.3, 0.1):
            y = x + step * sh
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < best - 1e-15:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return x

# ---------- GF(4) + hexacode + K12 pool ----------
def build_pools():
    # GF4: 0=00,1=01,w=10,w2=11, add=XOR
    mul = np.array([[0, 0, 0, 0],
                    [0, 1, 2, 3],
                    [0, 2, 3, 1],
                    [0, 3, 1, 2]], dtype=int)
    def add3(a, b, c):
        return a ^ b ^ c
    # candidate generator [I | A]
    G0 = np.array([[1, 0, 0, 1, 1, 1],
                   [0, 1, 0, 1, 2, 3],
                   [0, 0, 1, 1, 3, 2]], dtype=int)
    candidates = [G0]
    # add a few random alternatives in case G0 is wrong
    rng_t = np.random.default_rng(7)
    for _ in range(8):
        A = rng_t.integers(0, 4, size=(3, 3))
        G = np.concatenate([np.eye(3, dtype=int), A], axis=1)
        candidates.append(G)
    coeffs = np.zeros((64, 3), dtype=int)
    for i in range(64):
        coeffs[i, 0] = (i >> 0) & 3 if False else 0
    # base-4 enumeration
    coeffs = np.zeros((64, 3), dtype=int)
    for idx in range(64):
        v = idx
        for k in range(3):
            coeffs[idx, k] = v % 4
            v //= 4
    best_code = None
    best_w4 = None
    for G in candidates:
        cw = np.zeros((64, 6), dtype=int)
        for r in range(64):
            c0, c1, c2 = coeffs[r]
            for j in range(6):
                cw[r, j] = mul[c0, G[0, j]] ^ mul[c1, G[1, j]] ^ mul[c2, G[2, j]]
        w = (cw != 0).sum(axis=1)
        # check linear size (distinct)
        uniq = len(set(map(tuple, cw.tolist())))
        if uniq != 64:
            continue
        mn = w[1:].min() if len(w) > 1 else 0
        if mn >= 4:
            # prefer the one with 45 weight-4 (hexacode)
            n4 = int(((w == 4).sum()))
            if n4 == 45:
                best_code = cw
                break
            if best_code is None:
                best_code = cw
    if best_code is None:
        return None
    cw = best_code
    w = (cw != 0).sum(axis=1)
    w4_idx = np.where(w == 4)[0]
    # Eisenstein reps for nonzero F4
    rep = {1: (1, 0), 2: (0, 1), 3: (1, 1)}
    sq3 = np.sqrt(3.0) / 2.0
    vecs = []
    for r in w4_idx:
        c = cw[r]
        supp = [j for j in range(6) if c[j] != 0]
        # supp len 4
        for mask in range(16):
            A = np.zeros(6, dtype=int)
            B = np.zeros(6, dtype=int)
            for k, j in enumerate(supp):
                a0, b0 = rep[int(c[j])]
                s = -1 if ((mask >> k) & 1) else 1
                A[j] = s * a0
                B[j] = s * b0
            v12 = np.zeros(12)
            for j in range(6):
                a = A[j]; b = B[j]
                v12[2 * j] = float(a) - 0.5 * float(b)
                v12[2 * j + 1] = sq3 * float(b)
            vecs.append(v12)
    # type D: 2*unit*e_i
    units = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    for pos in range(6):
        for (a0, b0) in units:
            A = np.zeros(6, dtype=int); B = np.zeros(6, dtype=int)
            A[pos] = 2 * a0; B[pos] = 2 * b0
            v12 = np.zeros(12)
            for j in range(6):
                a = A[j]; b = B[j]
                v12[2 * j] = float(a) - 0.5 * float(b)
                v12[2 * j + 1] = sq3 * float(b)
            vecs.append(v12)
    K = np.array(vecs, dtype=float)
    # norms should be 2
    nrm = np.linalg.norm(K, axis=1, keepdims=True) + 1e-300
    K = K / nrm
    # D12 roots
    D = []
    s2 = 1.0 / np.sqrt(2.0)
    for i in range(12):
        for j in range(i + 1, 12):
            for s1 in (1.0, -1.0):
                for s2v in (1.0, -1.0):
                    v = np.zeros(12)
                    v[i] = s1 * s2; v[j] = s2v * s2
                    D.append(v)
    D = np.array(D, dtype=float)
    P = np.vstack([K, D])
    # dedup (dot > 1-1e-9)
    # greedy keep
    keep = []
    kept = []
    Gp = P @ P.T
    used = np.zeros(len(P), dtype=bool)
    # simple: iterate, skip if close to already kept
    # use block to be faster
    keep_idx = []
    for i in range(len(P)):
        if used[i]:
            continue
        keep_idx.append(i)
        # mark duplicates
        dup = np.where(Gp[i] > 1 - 1e-9)[0]
        for k in dup:
            if k != i:
                used[k] = True
        used[i] = True
    P = P[np.array(keep_idx)]
    return P

def greedy_build(Gp, M, rng):
    S = []
    Sset = set()
    order = rng.permutation(M)
    # start from random point
    S.append(int(order[0])); Sset.add(S[0])
    for _ in range(1, 68):
        Nc = min(400, M)
        cand = rng.choice(M, size=Nc, replace=False)
        best_k = -1; best_v = 1e9
        for k in cand:
            k = int(k)
            if k in Sset:
                continue
            v = float(Gp[k, S].max()) if len(S) > 0 else -2.0
            if v < best_v:
                best_v = v; best_k = k
        if best_k < 0:
            # fill random
            while True:
                k = int(rng.integers(0, M))
                if k not in Sset:
                    best_k = k; break
        S.append(best_k); Sset.add(best_k)
    return S

def improve_subset(Gp, M, Sinit, rng, iters, T0=0.004, T1=0.0002):
    S = list(map(int, Sinit))
    def submax(Slist):
        return float(Gp[np.ix_(Slist, Slist)].max()) if len(Slist) > 1 else -2.0
    # ensure 68 unique
    cur = submax(S)
    best = cur; bestS = list(S)
    tabu = {}
    Sset = set(S)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        Tt = T0 * (T1 / max(T0, 1e-12)) ** frac
        sub = Gp[np.ix_(S, S)]
        # argmax
        am = int(np.argmax(sub))
        p, q = divmod(am, 68)
        cur = float(sub[p, q])
        # victim choice
        if rng.random() < 0.7:
            # pick one of the two at random weighted by row max
            rp = float(sub[p].max()); rq = float(sub[q].max())
            if rp > rq:
                victim = p
            elif rq > rp:
                victim = q
            else:
                victim = p if rng.random() < 0.5 else q
        else:
            # random overloaded position among top few
            rm = sub.max(axis=1)
            od = np.argsort(-rm)[:6]
            victim = int(od[int(rng.integers(0, len(od)))])
        S_rest = [S[k] for k in range(68) if k != victim]
        try:
            rest_max = float(Gp[np.ix_(S_rest, S_rest)].max())
        except Exception:
            rest_max = -2.0
        Ncand = 140
        cands = []
        att = 0
        while len(cands) < Ncand and att < Ncand * 12:
            att += 1
            k = int(rng.integers(0, M))
            if k in Sset:
                continue
            if tabu.get(k, -1) > it:
                continue
            cands.append(k)
        if not cands:
            continue
        cands = np.array(cands, dtype=int)
        dots = Gp[cands][:, S_rest].max(axis=1)
        newv = np.maximum(dots, rest_max)
        bi = int(np.argmin(newv))
        bn = float(newv[bi]); bk = int(cands[bi])
        accept = False
        if bn < cur - 1e-12:
            accept = True
        else:
            if bn < cur + 0.02 and rng.random() < np.exp((cur - bn) / max(Tt, 1e-12)):
                accept = True
        if accept:
            old = S[victim]
            S[victim] = bk
            Sset.remove(old); Sset.add(bk)
            tabu[old] = it + 12
            cur = bn
            if cur < best - 1e-12:
                best = cur; bestS = list(S)
    return bestS, best

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline = t0 + LIMIT
    rng = np.random.default_rng(20261031)
    # load incumbents
    cands0 = []
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands0.append(xb)
        bv0 = max_cos(xb)
    except Exception:
        bv0 = 1e9
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands0.append(xr)
    except Exception:
        pass
    if len(cands0) == 0:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands0.append(x0)
        bv0 = max_cos(x0)
    # sort cands0
    cands0 = sorted(cands0, key=lambda z: max_cos(z))
    x_best = cands0[0].copy()
    best_val = float(max_cos(x_best))
    # ---- discrete pool ----
    try:
        P = build_pools()
    except Exception:
        P = None
    disc_best_X = None
    disc_best_val = 1e9
    disc_candidates = []
    if P is not None and len(P) >= 68:
        try:
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            M = len(P)
            Gp = P @ P.T
            np.fill_diagonal(Gp, -2.0)
            # thresholds decreasing: just minimize
            n_restart = 10
            # time budget for discrete: ~45s
            disc_deadline = t0 + 55.0
            for rs in range(n_restart):
                if time.time() > disc_deadline:
                    break
                if time.time() > deadline - 90:
                    break
                rng_s = np.random.default_rng(1000 + rs * 77)
                S0 = greedy_build(Gp, M, rng_s)
                iters = 2500 if rs < 6 else 1500
                S1, v1 = improve_subset(Gp, M, S0, rng_s, iters=iters,
                                        T0=0.005, T1=0.0003)
                # extra greedy repair toward threshold 0.25->0.21 style: second pass low temp
                if time.time() < disc_deadline - 2:
                    S1, v1 = improve_subset(Gp, M, S1, rng_s, iters=800,
                                            T0=0.0008, T1=0.00005)
                if v1 < disc_best_val:
                    disc_best_val = v1
                    disc_best_X = P[np.array(S1)].copy()
                    disc_candidates.append((v1, disc_best_X.copy()))
                    disc_candidates = sorted(disc_candidates, key=lambda e: e[0])[:3]
        except Exception:
            disc_best_X = None
    # ---- continuous release ----
    # collect seeds to polish: discrete best(s) + incumbents
    seeds = []
    if disc_best_X is not None:
        for _, Xd in disc_candidates:
            seeds.append(Xd.copy())
    for c in cands0[:2]:
        seeds.append(c.copy())
    # dedup seeds limit 4
    seeds = seeds[:4]
    # polish each seed briefly, keep global best
    for si, s in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            # ladder with moderate iters
            x = s.copy()
            # small random kick for discrete seeds to break exact symmetry
            if si < len(disc_candidates):
                x = x + 1e-3 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = run_lbfgs(x, 80.0, maxiter=120)
            if time.time() > deadline - 32:
                # skip further
                v = max_cos(x)
                if v < best_val:
                    best_val = v; x_best = x.copy()
                continue
            x = run_lbfgs(x, 250.0, maxiter=120)
            x = run_lbfgs(x, 600.0, maxiter=80)
            v = max_cos(x)
            if v < best_val:
                best_val = v; x_best = x.copy()
            # if discrete-derived and competitive, give extra
            if v < bv0 + 0.003 and time.time() < deadline - 45:
                x2 = run_lbfgs(x, 900.0, maxiter=80)
                v2 = max_cos(x2)
                if v2 < best_val:
                    best_val = v2; x_best = x2.copy()
        except Exception:
            continue
    # final intensive polish of global best
    try:
        if deadline - time.time() > 8:
            x_best = run_lbfgs(x_best, 500.0, maxiter=200)
            v = max_cos(x_best)
            if v < best_val:
                best_val = v
            x_best = run_lbfgs(x_best, 900.0, maxiter=150)
            best_val = max_cos(x_best)
    except Exception:
        pass
    rem = deadline - time.time() - 0.8
    if rem > 3:
        try:
            x_lp = lp_polish(x_best, seconds=rem, deadline=deadline)
            vlp = max_cos(x_lp)
            if vlp < best_val:
                best_val = vlp
                x_best = x_lp.copy()
        except Exception:
            pass
    x_best = np.asarray(x_best, dtype=float)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
