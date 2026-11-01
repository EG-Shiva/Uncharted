import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _topk(X, k=50):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu]
    s = np.sort(c)[::-1]
    return s[:k].copy(), float(s[0])

def _lex_better(a, b, tol=2e-12):
    for x, y in zip(a, b):
        if x < y - tol:
            return True
        if x > y + tol:
            return False
    return False

def _quench(X_init, steps, T0, T1, lr, mu=0.88, seed_noise=0.0, rng=None):
    X = np.array(X_init, dtype=float, copy=True)
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
        m = float(G.max())
        Q = (G - m) / T
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
        if rng is not None and seed_noise > 0:
            N = rng.standard_normal((n, d)) * seed_noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            Gt = Gt + N
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 100 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _single_refine(s, R, steps=160, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def _find_hole(R, rng, n_sample=4000):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, 3)[:3]
    bv = 1e100
    bs = None
    for k in idx:
        s1 = _single_refine(Z[k], R, steps=150, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=150, T=0.0012, lr=0.07)
        v = float((R @ s1).max())
        if v < bv:
            bv = v
            bs = s1
    bs = bs / (np.linalg.norm(bs) + 1e-12)
    return bs, bv

def _rand_orth(d, rng):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    dd = np.sign(np.diag(R))
    dd[dd == 0] = 1.0
    Q = Q * dd[None, :]
    return Q

def _lp_polish(X0, seconds, deadline, r0=2.5e-3, cap_p=1500):
    X = _norm_rows(np.array(X0, dtype=float, copy=True))
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X, 1e100
    bestX = X.copy()
    radius = float(r0)
    start = time.time()
    iu = np.triu_indices(n, 1)
    nv = n * d + 1
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        cuv = G[iu]
        m = float(cuv.max())
        band = max(25.0 * radius, 2e-4)
        sel = cuv > (m - band)
        I = iu[0][sel]; J = iu[1][sel]
        cs = cuv[sel]
        p = len(I)
        if p == 0:
            break
        if p > cap_p:
            kk = np.argpartition(-cs, cap_p)[:cap_p]
            I = I[kk]; J = J[kk]; cs = cs[kk]
            p = cap_p
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        bv = 1e100; ba = None
        for alpha in (1.0, 0.5, 0.25):
            Y = _norm_rows(X + alpha * dx)
            v = _max_cos(Y)
            if v < bv:
                bv = v; ba = alpha
        if ba is None:
            radius *= 0.5
            continue
        Y = _norm_rows(X + ba * dx)
        if bv < m - 1e-14:
            X = Y
            if bv < best:
                best = bv; bestX = Y.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
            if bv < best - 1e-14:
                best = bv; bestX = Y.copy(); X = Y
            if radius < 2e-6:
                break
    return bestX, best

def _fingerprint_top(X, k=20):
    t, m = _topk(X, k=k)
    return tuple(np.round(t * 1e7).astype(int).tolist())

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_T = 22.0
    HOP_END = HARD_END - FINAL_T - 1.0
    rng = np.random.default_rng(987654321)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    best_top, _ = _topk(best_X, 50)
    cur_X = best_X.copy(); cur_score = best_score; cur_top = best_top.copy()
    elite = [(best_score, best_X.copy())]
    seen = set()
    seen.add(_fingerprint_top(best_X))
    iu_all = np.triu_indices(n, 1)
    it = 0
    # quick diversification: quench a few heavily perturbed copies first
    # (cheap, escapes inherited basin immediately)
    for _ in range(3):
        if time.time() > HOP_END:
            break
        sig = 0.25
        X0 = best_X.copy()
        N = rng.standard_normal((n, d)) * sig
        N -= X0 * np.sum(N * X0, axis=1, keepdims=True)
        X0 = _norm_rows(X0 + N)
        Xq, sq = _quench(X0, steps=500, T0=0.004, T1=0.00015, lr=0.25, mu=0.88)
        tq, _ = _topk(Xq, 50)
        if sq < best_score - 1e-12 or (sq < best_score + 3e-4 and _lex_better(tq, best_top)):
            best_score = sq; best_X = Xq.copy(); best_top = tq.copy()
            elite.append((sq, Xq.copy())); elite.sort(key=lambda t: t[0]); elite = elite[:5]
        if sq < cur_score:
            cur_X = Xq.copy(); cur_score = sq; cur_top = tq.copy()
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 1.5:
            break
        frac = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        Tacc = 0.004 * (1 - frac) + 0.0002 * frac
        # choose base
        rr = rng.random()
        if rr < 0.45:
            base = best_X
        elif rr < 0.75:
            base = cur_X
        else:
            base = elite[int(rng.integers(0, len(elite)))][1]
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        pmx = Gb.max(axis=1)
        mglob = float(Gb.max())
        band = 8e-4
        cnt = (Gb > mglob - band).sum(axis=1)
        order_worst = np.argsort(-pmx)
        r = rng.random()
        X0 = None
        if r < 0.38:
            # hole re-insertion: kill k worst, refill at holes
            k = int(rng.integers(2, 6))
            kill = order_worst[:k].tolist()
            # occasionally kill a random rattler too for diversity
            if rng.random() < 0.4:
                kill[-1] = int(rng.integers(0, n))
                kill = sorted(set(kill))
                k = len(kill)
            keep = [i for i in range(n) if i not in set(kill)]
            R = base[keep]
            newpts = []
            for _ in range(k):
                Rall = np.vstack([R] + newpts) if newpts else R
                s, _ = _find_hole(Rall, rng, n_sample=3500)
                newpts.append(s[None, :])
            X0 = base.copy()
            for idx_k, kk in enumerate(kill):
                X0[kk] = newpts[idx_k][0]
        elif r < 0.65:
            # biased large subset shake
            k = int(rng.integers(8, 26))
            # half worst-biased, half random
            hw = k // 2
            worst_pool = order_worst[:28].tolist()
            sel1 = list(rng.choice(worst_pool, size=min(hw, len(worst_pool)), replace=False)) if hw > 0 else []
            rest = [i for i in range(n) if i not in sel1]
            sel2 = list(rng.choice(rest, size=k - len(sel1), replace=False))
            movable = np.array(sel1 + sel2, dtype=int)
            sigma = float(rng.uniform(0.10, 0.38))
            X0 = base.copy()
            N = rng.standard_normal((len(movable), d)) * sigma
            Xm = X0[movable]
            N -= Xm * np.sum(N * Xm, axis=1, keepdims=True)
            X0[movable] = _norm_rows(Xm + N)
        elif r < 0.80:
            # cluster rotation: preserve intra-cluster Gram
            k = int(rng.integers(6, 22))
            # pick spatially coherent cluster: seed + nearest neighbours
            seed = int(rng.integers(0, n))
            dists = Gb[seed].copy()
            neigh = np.argsort(-dists)[:k]
            Q = _rand_orth(d, rng)
            X0 = base.copy()
            X0[neigh] = _norm_rows(base[neigh] @ Q.T)
            # add small jitter after rotation
            N = rng.standard_normal((k, d)) * 0.02
            N -= X0[neigh] * np.sum(N * X0[neigh], axis=1, keepdims=True)
            X0[neigh] = _norm_rows(X0[neigh] + N)
        elif r < 0.90:
            # global moderate jitter
            sigma = float(rng.uniform(0.02, 0.07))
            X0 = base.copy()
            N = rng.standard_normal((n, d)) * sigma
            N -= X0 * np.sum(N * X0, axis=1, keepdims=True)
            X0 = _norm_rows(X0 + N)
        else:
            # fresh random replacement of few points
            k = int(rng.integers(3, 9))
            idxs = rng.choice(n, size=k, replace=False)
            X0 = base.copy()
            X0[idxs] = _norm_rows(rng.standard_normal((k, d)))
        # quench params random
        steps = int(rng.choice([450, 600, 800]))
        T0c = float(rng.choice([0.006, 0.003, 0.0015]))
        T1c = float(rng.choice([0.00025, 0.00012, 0.00006]))
        lrc = float(rng.choice([0.28, 0.22, 0.16]))
        muc = 0.88
        nz = float(rng.choice([0.0, 0.0, 0.0005]))
        Xq, sq = _quench(X0, steps=steps, T0=T0c, T1=T1c, lr=lrc, mu=muc, seed_noise=nz, rng=rng)
        tq, _ = _topk(Xq, 50)
        fp = _fingerprint_top(Xq)
        is_new = fp not in seen
        if is_new:
            seen.add(fp)
        # occasional quick LP when promising
        if sq < best_score + 8e-4 and (it % 7 == 0) and (HOP_END - time.time() > 8):
            Xlp, slp = _lp_polish(Xq, seconds=1.2, deadline=HOP_END)
            tlp, _ = _topk(Xlp, 50)
            if slp < sq - 1e-12 or (slp <= sq + 1e-12 and _lex_better(tlp, tq)):
                Xq, sq, tq = Xlp, slp, tlp
        improved_best = False
        if sq < best_score - 1e-12:
            improved_best = True
        elif sq < best_score + 2e-4 and _lex_better(tq, best_top) and is_new:
            improved_best = True
        if improved_best:
            best_score = sq; best_X = Xq.copy(); best_top = tq.copy()
            elite.append((sq, Xq.copy())); elite.sort(key=lambda t: t[0]); elite = elite[:5]
            cur_X = Xq.copy(); cur_score = sq; cur_top = tq.copy()
            continue
        # SA acceptance for cur
        if sq < cur_score + Tacc and (is_new or sq < cur_score):
            # lexicographic tie-break: don't wander to much worse topk
            cur_X = Xq.copy(); cur_score = sq; cur_top = tq.copy()
        elif rng.random() < 0.05:
            # random restart of walker to best to avoid drift
            cur_X = best_X.copy(); cur_score = best_score; cur_top = best_top.copy()
    # final polish: alternate low-T quench + LP on elite
    fdead = HARD_END - 0.5
    # sort elite
    elite.sort(key=lambda t: t[0])
    # also include cur
    try:
        cs, _ = _topk(cur_X, 50)
        elite.append((cur_score, cur_X.copy()))
    except Exception:
        pass
    elite.sort(key=lambda t: t[0])
    # dedup roughly
    final_cands = elite[:3]
    bestF = best_X.copy(); bestFs = best_score
    for _, Xe in final_cands:
        if time.time() > fdead - 2.0:
            break
        Xc = Xe.copy()
        # low-T tightening
        remain = fdead - time.time()
        if remain < 3:
            break
        qs = min(900, max(300, int(remain * 60)))
        Xc, sc = _quench(Xc, steps=qs, T0=0.0012, T1=0.00004, lr=0.18, mu=0.9)
        if sc < bestFs:
            bestFs = sc; bestF = Xc.copy()
        # LP
        remain2 = fdead - time.time()
        if remain2 > 2.0:
            Xl, sl = _lp_polish(Xc, seconds=min(6.0, remain2 - 0.5), deadline=fdead, r0=2.0e-3)
            if sl < bestFs:
                bestFs = sl; bestF = Xl.copy()
            # one more micro quench
            if fdead - time.time() > 2.0:
                Xc2, sc2 = _quench(Xl, steps=300, T0=0.0004, T1=0.00003, lr=0.10, mu=0.9)
                if sc2 < bestFs:
                    bestFs = sc2; bestF = Xc2.copy()
    bestF = _norm_rows(bestF)
    return bestF
