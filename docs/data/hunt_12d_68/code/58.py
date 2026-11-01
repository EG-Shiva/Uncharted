import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def soft_descent(x, steps, beta0, beta1, lr0, lr1, mu=0.85, noise=0.0, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        beta = beta0 * (beta1 / max(beta0, 1e-12)) ** t
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg = Tg / rms
        v = mu * v + (1.0 - mu) * Tg if s > 0 else Tg
        x = x - lr * v
        if noise > 0 and rng is not None:
            amp = noise * (1.0 - t)
            x = x + amp * rng.standard_normal(x.shape)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x

def lp_polish(x, deadline, r0=8e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + step * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

# ---------- pool builders ----------
def build_K12():
    # F4 as pairs (a,b) encoded 0..3 = a+2b
    def add(a, b):
        return a ^ b
    def mul(a, b):
        a1 = a & 1; b1 = (a >> 1) & 1
        a2 = b & 1; b2 = (b >> 1) & 1
        ar = (a1 & a2) ^ (b1 & b2)
        br = (a1 & b2) ^ (b1 & a2) ^ (b1 & b2)
        return ar | (br << 1)
    w = 2
    w2 = 3
    r1 = [1, 0, 0, 1, 1, 1]
    r2 = [0, 1, 0, 1, w, w2]
    r3 = [0, 0, 1, 1, w2, w]
    codes = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                cw = []
                for k in range(6):
                    v = add(add(mul(a, r1[k]), mul(b, r2[k])), mul(c, r3[k]))
                    cw.append(v)
                codes.append(cw)
    codes = np.array(codes, dtype=int)
    # weight
    wt = np.sum(codes != 0, axis=1)
    # units: (a,b) integer pairs
    units = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (1, 1)]
    # map F4 symbol -> two units
    mp = {1: [(1, 0), (-1, 0)], 2: [(0, 1), (0, -1)], 3: [(-1, -1), (1, 1)]}
    sq3 = np.sqrt(3.0) / 2.0
    vecs = []
    # type A from zero codeword
    for pos in range(6):
        for u in units:
            arr = np.zeros((6, 2), dtype=int)
            arr[pos, 0] = 2 * u[0]
            arr[pos, 1] = 2 * u[1]
            v12 = np.zeros(12)
            for i in range(6):
                a_, b_ = int(arr[i, 0]), int(arr[i, 1])
                v12[2 * i] = a_ - 0.5 * b_
                v12[2 * i + 1] = b_ * sq3
            v12 = v12 / 2.0
            vecs.append(v12)
    # type B
    for idx in range(len(codes)):
        cw = codes[idx]
        if wt[idx] != 4:
            continue
        supp = [k for k in range(6) if cw[k] != 0]
        syms = [int(cw[k]) for k in supp]
        for mask in range(16):
            arr = np.zeros((6, 2), dtype=int)
            for j, k in enumerate(supp):
                s = syms[j]
                choice = (mask >> j) & 1
                u = mp[s][choice]
                arr[k, 0] = u[0]
                arr[k, 1] = u[1]
            v12 = np.zeros(12)
            for i in range(6):
                a_, b_ = int(arr[i, 0]), int(arr[i, 1])
                v12[2 * i] = a_ - 0.5 * b_
                v12[2 * i + 1] = b_ * sq3
            v12 = v12 / 2.0
            vecs.append(v12)
    P = np.array(vecs)
    # dedup roughly, normalize
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    # remove exact duplicates (keep first)
    # round to 8 decimals for dedup
    _, uidx = np.unique(np.round(P, 8), axis=0, return_index=True)
    P = P[np.sort(uidx)]
    return P

def build_D12():
    d = 12
    vecs = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = s1; v[j] = s2
                    vecs.append(v)
    P = np.array(vecs) / np.sqrt(2.0)
    # add coordinate axes
    for i in range(d):
        for s in (-1.0, 1.0):
            v = np.zeros(d); v[i] = s
            vecs.append(v)
    P2 = np.array(vecs)
    P2 = P2 / np.linalg.norm(P2, axis=1, keepdims=True)
    return P2

def build_cube():
    M = 4096; d = 12
    idx = np.arange(M)[:, None]
    bits = ((idx >> np.arange(d)) & 1).astype(float) * 2.0 - 1.0
    P = bits / np.sqrt(12.0)
    return P

def subset_max(G, S):
    sub = G[np.ix_(S, S)].copy()
    np.fill_diagonal(sub, -2.0)
    return float(np.max(sub))

def search_pool(pool, thresholds, rng, deadline, max_steps_each=2500):
    M = pool.shape[0]
    # Gram
    if M > 1500:
        Pf = pool.astype(np.float32)
        G = (Pf @ Pf.T).astype(np.float32)
    else:
        G = (pool @ pool.T)
    n_target = 68
    S0 = rng.choice(M, size=n_target, replace=False)
    best_val = subset_max(G, S0)
    best_S = S0.copy()
    # try thresholds descending
    for thresh in thresholds:
        if time.time() > deadline:
            break
        S_cur = best_S.copy()
        for step in range(max_steps_each):
            if time.time() > deadline:
                break
            sub = G[np.ix_(S_cur, S_cur)].copy()
            # fill diag
            for k in range(n_target):
                sub[k, k] = -2.0
            mx = float(np.max(sub))
            if mx < best_val - 1e-12:
                best_val = mx
                best_S = S_cur.copy()
            if mx <= thresh + 1e-12:
                break
            flat = int(np.argmax(sub))
            pi = flat // n_target; qi = flat % n_target
            ev_pos = pi if rng.random() < 0.5 else qi
            S_rem = np.delete(S_cur, ev_pos)
            dots = np.max(G[:, S_rem], axis=1).astype(float)
            dots[S_rem] = np.inf
            k = 5
            # argpartition
            part = np.argpartition(dots, k)[:k]
            if rng.random() < 0.7:
                bc = part[int(np.argmin(dots[part]))]
            else:
                bc = rng.choice(part)
            S_cur[ev_pos] = int(bc)
        # after thresh loop, if feasible keep it as start (best_S already updated)
        # also small random kick to diversify for next thresh if not feasible
        if time.time() > deadline:
            break
    return best_S, float(best_val), G

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            if a.shape == (68, 12):
                v = max_cos(a)
                if v < best_score:
                    best_score = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = max_cos(bestX)
    incumbent = bestX.copy()

    # build pools
    try:
        poolK = build_K12()
    except Exception:
        poolK = None
    try:
        poolD = build_D12()
    except Exception:
        poolD = None
    try:
        poolC = build_cube()
    except Exception:
        poolC = None

    disc_cands = []
    disc_deadline = t0 + 45.0
    # K12 search: quantized thresholds
    if poolK is not None and time.time() < disc_deadline:
        try:
            th = [0.51, 0.30, 0.26, 0.20, 0.10, 0.0]
            S, v, _ = search_pool(poolK, th, rng, min(disc_deadline, T_END - 100), max_steps_each=2500)
            Xd = poolK[np.array(S)].copy()
            Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
            disc_cands.append((v, Xd))
        except Exception:
            pass
    if poolD is not None and time.time() < disc_deadline:
        try:
            th = [0.55, 0.51, 0.50, 0.30, 0.0]
            S, v, _ = search_pool(poolD, th, rng, min(disc_deadline, T_END - 100), max_steps_each=2000)
            Xd = poolD[np.array(S)].copy()
            Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
            disc_cands.append((v, Xd))
        except Exception:
            pass
    if poolC is not None and time.time() < disc_deadline:
        try:
            th = [0.34, 0.26, 0.18, 0.1667, 0.16]
            S, v, _ = search_pool(poolC, th, rng, min(disc_deadline, T_END - 90), max_steps_each=2500)
            Xd = poolC[np.array(S)].copy()
            Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
            disc_cands.append((v, Xd))
        except Exception:
            pass
    # sort discrete by value
    disc_cands.sort(key=lambda z: z[0])

    # continuous release of discrete candidates
    # keep global best
    x_best = incumbent.copy()
    v_best = best_score
    for dv, Xd in disc_cands[:3]:
        if time.time() > T_END - 55:
            break
        x = Xd.copy()
        # small jitter to break exact symmetry before release
        x = x + 1e-4 * rng.standard_normal(x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        # log-sum-exp style soft descent in two stages
        x = soft_descent(x, steps=500, beta0=30.0, beta1=450.0, lr0=0.05, lr1=0.004, mu=0.85, noise=0.0, rng=rng)
        x = soft_descent(x, steps=300, beta0=400.0, beta1=700.0, lr0=0.012, lr1=0.002, mu=0.8, noise=0.0, rng=rng)
        v = max_cos(x)
        if v < v_best:
            v_best = v
            x_best = x.copy()
        # short LP squeeze
        x = lp_polish(x, deadline=min(time.time() + 6.0, T_END - 30), r0=8e-4)
        v = max_cos(x)
        if v < v_best:
            v_best = v
            x_best = x.copy()

    # also polish incumbent with a few basin hops mixing discrete info
    # quick hops: shake + soft + LP
    explore_end = T_END - 28.0
    trial = 0
    while time.time() < explore_end:
        trial += 1
        x = x_best.copy()
        u = rng.random()
        if u < 0.3 and len(disc_cands) > 0:
            # inject a discrete block: replace worst 8 points with random discrete pool points far from rest
            dv, Xdref = disc_cands[0]
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(6 + rng.integers(0, 5))
            idx = order[:k]
            # pick k random rows from a random discrete candidate pool
            src = disc_cands[int(rng.integers(0, len(disc_cands)))][1]
            perm = rng.permutation(src.shape[0])[:k]
            x[idx] = src[perm] + 0.02 * rng.standard_normal((k, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.6:
            sigma = float(rng.choice([0.01, 0.03, 0.07]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # worst-cluster kick
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            wi = ij // n
            prox = G[wi]
            order = np.argsort(-prox)
            k = int(rng.integers(3, 8))
            idx = order[:k]
            x[idx] += 0.5 * rng.standard_normal((len(idx), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = soft_descent(x, steps=int(rng.integers(400, 700)), beta0=30.0,
                         beta1=float(rng.uniform(350, 600)),
                         lr0=0.04, lr1=0.003, mu=0.85, noise=0.0, rng=rng)
        v = max_cos(x)
        if v < v_best - 2e-7:
            v_best = v
            x_best = x.copy()
        elif v < v_best:
            v_best = v
            x_best = x.copy()
        if trial % 3 == 0 and time.time() < T_END - 35:
            x_best = lp_polish(x_best, deadline=min(time.time() + 3.0, T_END - 28), r0=6e-4)
            v_best = max_cos(x_best)
        if time.time() > explore_end:
            break

    # final squeeze
    x_best = lp_polish(x_best, deadline=T_END - 1.0, r0=6e-4)
    v_best = max_cos(x_best)
    # strict feasibility shrink
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
