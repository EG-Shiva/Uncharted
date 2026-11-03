import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SCREEN_END = t0 + 90.0
    n, d = 121, 8
    rng = np.random.default_rng(0)
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    P = len(Iall)

    # load references
    def _load(fn):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    return a
        except Exception:
            pass
        return None
    Rbest = _load("best.npy")
    Rrec = _load("record.npy")
    Rref = Rrec if Rrec is not None else Rbest
    if Rbest is None and Rrec is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        Rbest = X0.copy()
    Vbest_ref = _max_cos(Rbest) if Rbest is not None else 1.0

    # record contact mask for Hamming distance
    if Rref is not None:
        Gr = Rref @ Rref.T
        mr = float(Gr[iu].max())
        rec_mask = (Gr[iu] >= mr - 1e-7)
    else:
        rec_mask = np.zeros(P, dtype=bool)
        mr = 0.0

    # ---------- build structured pools ----------
    def normed(A):
        A = np.asarray(A, dtype=float)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-18
        return A

    # E8 roots: D8 (112) + half-integer (128) = 240
    E8 = []
    s2 = 1.0 / np.sqrt(2.0)
    for i in range(d):
        for j in range(i+1, d):
            for s1 in (-1.0, 1.0):
                for s2v in (-1.0, 1.0):
                    v = np.zeros(d); v[i] = s1*s2; v[j] = s2v*s2
                    E8.append(v)
    # 2^8 half vectors with even parity
    for mask in range(256):
        signs = np.ones(d)
        nm = 0
        for k in range(d):
            if (mask >> k) & 1:
                signs[k] = -1.0; nm += 1
        if nm % 2 == 0:
            E8.append(0.5*signs)
    E8 = np.array(E8)  # 240x8
    E8 = normed(E8)

    # D8-type pool: E8 + basis + hypercube + random
    basis = np.zeros((16, d))
    for i in range(d):
        basis[2*i, i] = 1.0; basis[2*i+1, i] = -1.0
    # hypercube subset (256)
    rng1 = np.random.default_rng(1)
    cube = rng1.choice([-1.0, 1.0], size=(256, d)) / np.sqrt(float(d))
    randA = rng1.standard_normal((1200, d)); randA = normed(randA)
    pool_D8 = normed(np.vstack([E8, basis, cube, randA]))

    # S3 codebooks: 24-cell + 16-cell + random
    # 24-cell in R4: 8 +-e_i + 16 (+-0.5^4)
    cell24 = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s; cell24.append(v)
    for mask in range(16):
        v = np.ones(4)*0.5
        for k in range(4):
            if (mask >> k) & 1:
                v[k] = -0.5
        cell24.append(v)
    cell24 = np.array(cell24)  # 24x4
    rng2 = np.random.default_rng(2)
    randS3a = normed(rng2.standard_normal((200, 4)))
    randS3b = normed(rng2.standard_normal((200, 4)))
    Uset = normed(np.vstack([cell24, randS3a]))  # 224x4
    Vset = normed(np.vstack([cell24, randS3b]))

    def product_pool(a, K, seed):
        r = np.random.default_rng(seed)
        b = np.sqrt(max(1e-12, 1-a*a))
        iu_u = r.integers(0, len(Uset), size=K)
        iv_v = r.integers(0, len(Vset), size=K)
        # mix in fully random u,v for diversity half the time
        Ur = Uset[iu_u].copy(); Vr = Vset[iv_v].copy()
        # replace half with continuous random S3
        h = K//2
        idx = r.choice(K, h, replace=False)
        Ur[idx] = normed(r.standard_normal((h, 4)))
        idx2 = r.choice(K, h, replace=False)
        Vr[idx2] = normed(r.standard_normal((h, 4)))
        M = np.hstack([a*Ur, b*Vr])
        return normed(M)

    pool_P05 = product_pool(0.5, 3000, 10)
    pool_P707 = product_pool(1.0/np.sqrt(2.0), 3000, 11)
    pool_P08 = product_pool(0.8, 3000, 12)

    pools = [E8, pool_D8, pool_P05, pool_P707, pool_P08]

    # ---------- greedy max-min selection ----------
    def greedy_select(pool, start_idx, nsel=121):
        M = pool.shape[0]
        sel = np.zeros(nsel, dtype=int)
        sel[0] = int(start_idx % M)
        # cur max cosine to selected set
        dots0 = pool @ pool[sel[0]]
        cur = dots0.copy()
        cur[sel[0]] = 1e9  # exclude selected
        in_sel = np.zeros(M, dtype=bool); in_sel[sel[0]] = True
        for t in range(1, nsel):
            # pick min cur among not selected
            # argmin over masked
            best = -1
            # fast argmin with masking
            tmp = cur.copy(); tmp[in_sel] = 1e9
            best = int(np.argmin(tmp))
            sel[t] = best
            in_sel[best] = True
            cur[best] = 1e9
            # update
            nd = pool @ pool[best]
            # cur = max(cur, nd) for not selected
            np.maximum(cur, nd, out=cur)
            cur[best] = 1e9
            if time.time() > SCREEN_END - 1.0:
                # fill rest randomly if timeout (should not happen)
                pass
        return pool[sel].copy()

    def local_swaps(pool, Xsel, tries=30, seed=0):
        r = np.random.default_rng(seed)
        X = Xsel.copy()
        m = _max_cos(X)
        Mp = pool.shape[0]
        for _ in range(tries):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # worst pair
            ij = int(np.argmax(G.reshape(-1)))
            ii, jj = ij // n, ij % n
            # pick one endpoint to replace
            tgt = ii if r.random() < 0.5 else jj
            # try few random pool candidates
            for _ in range(25):
                cand = pool[int(r.integers(0, Mp))]
                # quick score: max dot to others
                rest = np.delete(X, tgt, axis=0)
                v = float((rest @ cand).max())
                if v < m - 1e-4:
                    X[tgt] = cand
                    m = _max_cos(X)
                    break
        return X

    # generate greedy codes with 200 random restarts spread over pools
    # distribute: E8 small pool -> fewer restarts, product pools -> more
    per_pool = [30, 40, 45, 45, 40]  # total 200
    greedy_codes = []
    rr = np.random.default_rng(7)
    for pi, pool in enumerate(pools):
        cnt = per_pool[pi]
        starts = rr.integers(0, pool.shape[0], size=cnt)
        for si, s in enumerate(starts):
            if time.time() > SCREEN_END - 25.0:
                break
            Xg = greedy_select(pool, int(s))
            Xg = local_swaps(pool, Xg, tries=12, seed=1000+pi*100+si)
            greedy_codes.append(Xg)
        if time.time() > SCREEN_END - 25.0:
            break
    if len(greedy_codes) == 0:
        greedy_codes = [Rbest.copy()]

    # keep best 48 by maxcos for screening (need at least 32)
    vals_g = [_max_cos(x) for x in greedy_codes]
    order = np.argsort(vals_g)
    B = min(40, len(order))
    if B < 32:
        B = len(order)
    # ensure at least 32: duplicate with noise if needed
    sel_idx = order[:B].tolist()
    Xb = [greedy_codes[i] for i in sel_idx]
    while len(Xb) < 32:
        Xb.append(Xb[len(Xb) % len(Xb)] + 0.001*rng.standard_normal((n, d)))
        Xb[-1] /= np.linalg.norm(Xb[-1], axis=1, keepdims=True)
    B = len(Xb)
    Xbatch = np.stack(Xb, axis=0)  # BxNxD

    # ---------- batched vectorised Adam LogSumExp (beta 50->1000, <=1000 steps) ----------
    T = 650
    if time.time() > SCREEN_END - 15.0:
        T = 150
    beta0, beta1 = 50.0, 1000.0
    lr0, lr1 = 0.025, 0.004
    b1, b2, eps = 0.9, 0.999, 1e-8
    mA = np.zeros_like(Xbatch); vA = np.zeros_like(Xbatch)
    # ensure normalised
    Xbatch = Xbatch / np.linalg.norm(Xbatch, axis=2, keepdims=True)
    for t in range(1, T+1):
        if time.time() > SCREEN_END:
            break
        frac = t / float(T)
        beta = beta0 * (beta1/beta0) ** frac
        lr = lr0 * (lr1/lr0) ** frac
        # Gram batched
        G = np.matmul(Xbatch, np.transpose(Xbatch, (0, 2, 1)))  # BxNxN
        # max per batch
        # mask diag by setting -2
        # efficient: set diag
        # use copy? modify in place then restore? compute mmax excluding diag
        # set diag to -2
        for b in range(B):
            np.fill_diagonal(G[b], -2.0)
        mmax = G.max(axis=(1, 2))  # B
        E = np.exp(np.clip(beta*(G - mmax[:, None, None]), -700, 0))
        for b in range(B):
            np.fill_diagonal(E[b], 0.0)
        S = E.sum(axis=(1, 2)) + 1e-300  # B
        W = E / S[:, None, None]
        GY = np.matmul(W, Xbatch)  # BxNxD
        dot = (GY * Xbatch).sum(axis=2, keepdims=True)
        Gr = GY - dot * Xbatch
        # Adam
        mA = b1*mA + (1-b1)*Gr
        vA = b2*vA + (1-b2)*(Gr*Gr)
        mhat = mA / (1 - b1**t + 1e-18)
        vhat = vA / (1 - b2**t + 1e-18)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        Xbatch = Xbatch - step
        Xbatch = Xbatch / (np.linalg.norm(Xbatch, axis=2, keepdims=True) + 1e-18)
        # NaN guard
        if not np.all(np.isfinite(Xbatch)):
            Xbatch = np.nan_to_num(Xbatch, nan=0.0, posinf=0.0, neginf=0.0)
            Xbatch = Xbatch / (np.linalg.norm(Xbatch, axis=2, keepdims=True) + 1e-18)

    scores = []
    for b in range(B):
        scores.append(_max_cos(Xbatch[b]))
    scores = np.array(scores)

    # ---------- promote: <0.475 and Hamming >100 ----------
    prom = []
    for b in range(B):
        if scores[b] < 0.475:
            Gb = Xbatch[b] @ Xbatch[b].T
            mb = float(Gb[iu].max())
            maskb = (Gb[iu] >= mb - 1e-7)
            ham = int(np.logical_xor(maskb, rec_mask).sum())
            if ham > 100:
                prom.append((scores[b], b, ham))
    if len(prom) == 0:
        # relax to <0.485
        for b in range(B):
            if scores[b] < 0.485:
                Gb = Xbatch[b] @ Xbatch[b].T
                mb = float(Gb[iu].max())
                maskb = (Gb[iu] >= mb - 1e-7)
                ham = int(np.logical_xor(maskb, rec_mask).sum())
                if ham > 50:
                    prom.append((scores[b], b, ham))
    if len(prom) == 0:
        # take best few regardless
        ob = np.argsort(scores)
        for b in ob[:4]:
            prom.append((scores[b], int(b), 999))
    prom.sort(key=lambda x: x[0])
    best_b = int(prom[0][1])
    Xprom = Xbatch[best_b].copy()
    Vprom = float(scores[best_b])

    # ---------- short L-BFGS anneal on single best promoted ----------
    def make_fun(beta):
        bb = float(beta)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            G_ = Y @ Y.T
            c = G_[iu]
            m_ = float(c.max())
            e = np.exp(np.clip(bb*(c-m_), -700, 0))
            s = float(e.sum()) + 1e-300
            f = m_ + np.log(s)/bb
            w = e/s
            W_ = np.zeros((n, n)); W_[Iall, Jall] = w; W_[Jall, Iall] = w
            GY_ = W_ @ Y
            dot_ = (GY_*Y).sum(axis=1, keepdims=True)
            GX = (GY_-dot_*Y)/nr
            return float(f), GX.ravel()
        return fun
    # only if time allows and before deadline
    for beta, mi in [(300.0, 120), (1500.0, 120)]:
        if time.time() > DEADLINE - 30.0:
            break
        fun = make_fun(beta)
        try:
            res = minimize(fun, Xprom.ravel().copy(), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*6)})
            Xn = np.asarray(res.x, dtype=float).reshape(n, d)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-18
            Xprom = Xn
        except Exception:
            break
    Vprom = _max_cos(Xprom)

    # ---------- LP polish only single best promoted seed for <=20s ----------
    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 2.0:
            it += 1
            if it > 80 or radius < 1e-14:
                break
            G_ = x @ x.T
            cf = G_[iu]
            m_ = float(cf.max())
            thr = max(20.0*radius, 2e-9)
            sel = cf > m_ - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5; continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel); mk[idxp] = True
                    sel = mk; cnt = 3000
            c_ = cf[sel]
            I_ = iu[0][sel]; J_ = iu[1][sel]
            p_ = cnt; nv = n*d+1
            rows = np.repeat(np.arange(p_), 2*d+1)
            cols = np.concatenate([I_[:, None]*d+np.arange(d), J_[:, None]*d+np.arange(d), np.full((p_,1), n*d)],1).ravel()
            vA_ = np.concatenate([x[J_], x[I_], -np.ones((p_,1))],1).ravel()
            try:
                A_ub = coo_matrix((vA_, (rows, cols)), shape=(p_, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5; continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c_, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5; continue
            if res.status != 0:
                radius *= 0.5; continue
            y = x + res.x[:-1].reshape(n, d)
            y = y / (np.linalg.norm(y, axis=1, keepdims=True)+1e-18)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    Xpol, Vpol = lp_polish(Xprom, seconds=20.0, radius_init=4e-4)

    # choose between polished promoted and reference best (fallback guarantee)
    Xfinal, Vfinal = Xpol, Vpol
    if Rbest is not None and Vbest_ref < Vfinal:
        # light polish of reference with remaining time (few seconds) to stay competitive
        rem = DEADLINE - time.time() - 1.0
        if rem > 3.0:
            Xr, Vr = lp_polish(Rbest, seconds=min(8.0, rem), radius_init=1e-4)
            if Vr < Vfinal:
                Xfinal, Vfinal = Xr, Vr
            else:
                # keep reference if still better
                if Vbest_ref < Vfinal:
                    Xfinal, Vfinal = Rbest.copy(), Vbest_ref
        else:
            Xfinal, Vfinal = Rbest.copy(), Vbest_ref

    Xfinal = Xfinal / np.linalg.norm(Xfinal, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    Xfinal = np.nan_to_num(Xfinal, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xfinal, axis=1, keepdims=True); nn[nn==0]=1.0
    Xfinal = Xfinal/nn
    return Xfinal
