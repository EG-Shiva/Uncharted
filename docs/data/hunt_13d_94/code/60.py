import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cmat = G[iu0, iu1]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        if p == 0:
            break
        sel_vals = cmat[sel]
        if p > 2500:
            order = np.argpartition(-sel_vals, 2500)[:2500]
            I, J = I[order], J[order]
            sel_vals = sel_vals[order]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-sel_vals, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _hub_ranking(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    # top pair
    fi, fj = np.unravel_index(int(np.argmax(G)), G.shape)
    # per point stats
    cnt7 = np.zeros(n, dtype=int)
    sexp = np.zeros(n)
    for i in range(n):
        row = G[i]
        # exclude self ( -2 )
        cnt7[i] = int(np.sum(row > m - 1e-7))
        # sum exp, clip to avoid overflow (slack>=0 so exp<=1)
        v = (row - m) / 1e-7
        # self entry gives huge negative, exp->0 fine
        v = np.clip(v, -50, 0)
        sexp[i] = float(np.sum(np.exp(v)))
    is_top = np.zeros(n, dtype=int)
    is_top[int(fi)] = 1; is_top[int(fj)] = 1
    order = sorted(range(n), key=lambda i: (is_top[i], int(cnt7[i]), float(sexp[i])), reverse=True)
    return order, int(fi), int(fj), float(m), G

def _build_pool(x_old, Fixed, rng, total=4096):
    d = Fixed.shape[1]
    F = Fixed.shape[0]
    pools = []
    # 1) random sphere
    n_rand = 2048
    R = rng.standard_normal((n_rand, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    pools.append(R)
    # 2) perturbations of old position
    n_pert = 1024
    sigs = [0.02, 0.07, 0.20, 0.50]
    P = np.empty((n_pert, d))
    for k in range(n_pert):
        s = sigs[k % 4]
        N = rng.standard_normal(d)
        Nt = N - float(N @ x_old) * x_old
        nn = np.linalg.norm(Nt)
        if nn < 1e-12:
            Nt = rng.standard_normal(d); Nt -= float(Nt @ x_old) * x_old; nn = np.linalg.norm(Nt)
            if nn < 1e-12:
                Nt = np.zeros(d); nn = 1.0
        Nt /= nn
        # random magnitude around s
        mag = s * (0.5 + rng.random())
        y = x_old + mag * Nt
        y /= np.linalg.norm(y)
        P[k] = y
    pools.append(P)
    # neighbour ordering by dot to x_old
    dots_old = Fixed @ x_old  # (F,)
    nidx = np.argsort(-dots_old)
    topK = min(25, F)
    top_idx = nidx[:topK]
    # 3) antipodes of all fixed
    pools.append(-Fixed)
    # 4) normalised differences (x_old - y) for top neighbours
    D = []
    for j in top_idx:
        y = Fixed[j]
        v = x_old - y
        nrm = np.linalg.norm(v)
        if nrm > 1e-12:
            D.append((v / nrm))
    if len(D) > 0:
        pools.append(np.stack(D))
    # 5) pairwise neighbour differences to fill
    cur = sum(p.shape[0] for p in pools)
    need = total - cur
    if need > 0:
        Q = np.empty((need, d))
        for k in range(need):
            a = int(rng.integers(0, F)); b = int(rng.integers(0, F))
            if a == b:
                v = rng.standard_normal(d); v /= np.linalg.norm(v)
            else:
                v = Fixed[a] - Fixed[b]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    v = rng.standard_normal(d); nrm = np.linalg.norm(v)
                v = v / nrm
            # random sign
            if rng.random() < 0.5:
                v = -v
            Q[k] = v
        pools.append(Q)
    C = np.vstack(pools)
    # trim / pad to exactly total
    if C.shape[0] > total:
        C = C[:total]
    elif C.shape[0] < total:
        extra = total - C.shape[0]
        R2 = rng.standard_normal((extra, d)); R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
        C = np.vstack([C, R2])
    # normalise
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return C

def _refine_joint(Fixed, Xb, steps=200, k0=100.0, k1=800.0, lr0=0.01, lr1=0.002):
    # Xb: (B,K,d)
    B, K, d = Xb.shape
    F = Fixed.shape[0]
    X = Xb.copy()
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eyeK = np.eye(K)
    Ft = Fixed.T
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Df = np.matmul(X, Ft)  # (B,K,F)
        Dv = np.matmul(X, np.transpose(X, (0, 2, 1)))  # (B,K,K)
        Dv = Dv - eyeK[None, :, :] * 5.0
        mDf = Df.max(axis=(1, 2))
        mDv = Dv.max(axis=(1, 2))
        m = np.maximum(mDf, mDv)
        Wf = np.exp(np.clip(kk * (Df - m[:, None, None]), -50, 0))
        Wv = np.exp(np.clip(kk * (Dv - m[:, None, None]), -50, 0))
        Wv = Wv * (1.0 - eyeK[None, :, :])
        sumF = Wf.sum(axis=(1, 2))
        sumV = Wv.sum(axis=(1, 2)) * 0.5
        S = sumF + sumV
        S = np.maximum(S, 1e-300)
        Gf = np.matmul(Wf, Fixed)  # (B,K,d)
        Gv = np.matmul(Wv, X)  # (B,K,d)
        Grad = (Gf + Gv) / S[:, None, None]
        dot = (Grad * X).sum(axis=2, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=2, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def solve():
    T0 = time.time()
    END = T0 + 165.0
    rng = np.random.default_rng(0)
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = float(best_val)
    best_overall = best.copy(); best_overall_val = float(best_val)
    it = 0
    NBEST = 16
    while time.time() < END - 5.6:
        it_rem = END - time.time()
        if it_rem < 5.6:
            break
        # ranking on cur
        try:
            order, fi, fj, mcur, Gcur = _hub_ranking(cur)
        except Exception:
            break
        double = (it % 8 == 7)
        if not double:
            h = int(order[it % len(order)])
            rm = [h]
        else:
            h = int(order[it % len(order)])
            # nearest neighbour of h (excluding self)
            row = cur[h] @ cur.T
            row[h] = -2.0
            nn = int(np.argmax(row))
            # ensure the pair is near-maximal; if not, use second hub
            rm = sorted([h, nn])
        K = len(rm)
        Fmask = np.ones(94, dtype=bool); Fmask[rm] = False
        Fixed = cur[Fmask]  # (94-K, d)
        x_olds = [cur[i].copy() for i in rm]
        # build pools
        if K == 1:
            pool = _build_pool(x_olds[0], Fixed, rng, total=4096)
            # score: max dot to Fixed
            scores = (pool @ Fixed.T).max(axis=1)
            bidx = np.argpartition(scores, NBEST)[:NBEST]
            # order best first
            bidx = bidx[np.argsort(scores[bidx])]
            Xb = pool[bidx][:, None, :]  # (16,1,13)
        else:
            pool1 = _build_pool(x_olds[0], Fixed, rng, total=4096)
            pool2 = _build_pool(x_olds[1], Fixed, rng, total=4096)
            s1 = (pool1 @ Fixed.T).max(axis=1)
            s2 = (pool2 @ Fixed.T).max(axis=1)
            inter = (pool1 * pool2).sum(axis=1)
            joint = np.maximum(np.maximum(s1, s2), inter)
            bidx = np.argpartition(joint, NBEST)[:NBEST]
            bidx = bidx[np.argsort(joint[bidx])]
            Xb = np.stack([pool1[bidx], pool2[bidx]], axis=1)  # (16,2,13)
        # refine 200 steps
        try:
            Xr = _refine_joint(Fixed, Xb, steps=200, k0=100.0, k1=800.0, lr0=0.01, lr1=0.002)
        except Exception:
            it += 1
            continue
        # evaluate refined true scores to fixed set
        B = Xr.shape[0]
        true_scores = np.empty(B)
        for b in range(B):
            mf = float((Xr[b] @ Fixed.T).max()) if Fixed.size else -2.0
            if K > 1:
                mi = float((Xr[b, 0] @ Xr[b, 1]))
                mf = max(mf, mi)
            true_scores[b] = mf
        bbest_order = np.argsort(true_scores)
        # try best refined first, LP polish only the most promising? spec: take best, run 5s LP.
        # To save time, run LP on the single best refined; if fails try next? We run LP once.
        bi = int(bbest_order[0])
        trial = cur.copy()
        for k_i, pos in enumerate(rm):
            trial[pos] = Xr[bi, k_i] / np.linalg.norm(Xr[bi, k_i])
        # quick check: if trial already much worse, still LP? do LP anyway (topology change)
        lp_dead = min(time.time() + 5.0, END - 0.5)
        try:
            Xpol, vpol = _lp_polish_deadline(trial, lp_dead)
        except Exception:
            it += 1
            continue
        # accept strictly improving vs cur
        if float(vpol) < cur_val - 1e-12:
            cur = Xpol.copy(); cur_val = float(vpol)
            if cur_val < best_overall_val:
                best_overall_val = cur_val; best_overall = cur.copy()
        else:
            # also track best even without acceptance? keep best_overall as min
            if float(vpol) < best_overall_val:
                best_overall_val = float(vpol); best_overall = Xpol.copy()
                # move cur to best? spec says restore incumbent on non-improve, so keep cur
                pass
        it += 1
    best_overall = best_overall / np.linalg.norm(best_overall, axis=1, keepdims=True)
    return best_overall
