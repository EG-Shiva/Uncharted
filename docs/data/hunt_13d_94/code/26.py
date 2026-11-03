import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_smooth_single(X, steps, k0, k1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        W = np.exp(k * (G - mv))
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        grad = W @ X / sw
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(x, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
            p = 2500
        else:
            cc = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= t_end:
            break
    return x, best

def _batched_cluster_opt(F, V0, rng, B=32, steps=1100, k0=25.0, k1=350.0, lr0=0.012, lr1=0.0012, deadline=None):
    M, d = F.shape
    K = V0.shape[0]
    # init batch
    V = np.empty((B, K, d))
    V[0] = V0
    sigmas = [0.15, 0.3, 0.5, 0.8]
    for b in range(1, B):
        if b < 9:
            s = sigmas[(b - 1) % len(sigmas)]
            nb = V0 + s * rng.standard_normal(size=(K, d))
            nb /= np.linalg.norm(nb, axis=1, keepdims=True)
            V[b] = nb
        else:
            # random uniform on sphere (inside cavity after optimisation)
            # half biased toward cavity centre to stay in cavity
            rnd = rng.standard_normal(size=(K, d))
            if b % 2 == 0:
                # bias: add small pull toward mean of V0 complement? use random fixed point hole
                # pick random direction from V0 mean
                centre = V0.mean(axis=0)
                cn = np.linalg.norm(centre)
                if cn > 1e-9:
                    centre /= cn
                    rnd = rnd + 0.6 * centre[None, :]
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            V[b] = rnd
    m1 = np.zeros_like(V)
    m2 = np.zeros_like(V)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute fixed gram not needed
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        # dots
        # D: B x K x M
        D = np.einsum('bkd,md->bkm', V, F)
        Gv = np.einsum('bkd,bld->bkl', V, V)
        # per-batch max for stability (over relevant pairs)
        # Gv diag is 1.0, must be considered but we will zero weight later; max over off-diag + D
        # compute max excluding diag
        # efficient: max of D and off-diag Gv
        tmp = Gv.copy()
        for b in range(B):
            np.fill_diagonal(tmp[b], -10.0)
        mD = D.max(axis=(1, 2))
        mG = tmp.max(axis=(1, 2))
        mv = np.maximum(mD, mG)  # (B,)
        Wvf = np.exp(kk * (D - mv[:, None, None]))
        Wvv = np.exp(kk * (tmp - mv[:, None, None]))
        # zero diag already? tmp diag -10 gives tiny weight, force 0
        for b in range(B):
            np.fill_diagonal(Wvv[b], 0.0)
        S = Wvf.sum(axis=(1, 2)) + Wvv.sum(axis=(1, 2))  # (B,)
        S[~np.isfinite(S)] = 1.0
        S[S == 0] = 1.0
        Gvf = np.einsum('bkm,md->bkd', Wvf, F)
        Gvv = np.einsum('bkl,bld->bkd', Wvv, V)
        grad = (Gvf + Gvv) / S[:, None, None]
        dot = (grad * V).sum(axis=2, keepdims=True)
        gt = grad - dot * V
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        V = V - upd
        nr = np.linalg.norm(V, axis=2, keepdims=True)
        nr[nr == 0] = 1.0
        V = V / nr
        if not np.all(np.isfinite(V)):
            break
    return V

def solve():
    T0 = time.time()
    BUDGET = 172.0
    T_END = T0 + BUDGET
    HOP_END = T0 + 118.0
    rng = np.random.default_rng(7)
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = 94, 13
    cur = best.copy(); cur_val = best_val
    Ks = [16, 12, 8, 12, 16, 8]
    it = 0
    while time.time() < HOP_END and time.time() < T_END - 55:
        it += 1
        K = Ks[(it - 1) % len(Ks)]
        # contact degrees of cur
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        mx = Gc.max()
        # contact threshold tight (ties are 1e-14)
        deg = (Gc > mx - 1e-7).sum(axis=1)
        # seed: high-degree with small random tie-break, deterministic via rng
        jitter = rng.random(n) * 2.0
        seed = int(np.argmax(deg + jitter))
        dots = cur @ cur[seed]
        order = np.argsort(-dots)
        cluster = order[:K]
        # ensure seed included (it is, dots[seed]=1 max)
        mask = np.ones(n, dtype=bool); mask[cluster] = False
        F = cur[mask]  # (n-K, d) frozen
        V0 = cur[~mask]  # (K, d) original
        # fixed-fixed max
        Gf = F @ F.T
        np.fill_diagonal(Gf, -2.0)
        fixed_max = float(Gf.max())
        # batched joint reinsertion
        steps = 1100
        if T_END - time.time() < 85:
            steps = 700
        B = 32
        t_dead = min(HOP_END + 3, T_END - 45)
        Vb = _batched_cluster_opt(F, V0, rng, B=B, steps=steps, k0=22.0, k1=380.0,
                                  lr0=0.012, lr1=0.0012, deadline=t_dead)
        # exact scoring per batch
        Dmax = np.einsum('bkd,md->bkm', Vb, F).max(axis=(1, 2))
        Gvv_all = np.einsum('bkd,bld->bkl', Vb, Vb)
        for b in range(B):
            np.fill_diagonal(Gvv_all[b], -2.0)
        Vmax = Gvv_all.max(axis=(1, 2))
        overall = np.maximum(Dmax, Vmax)
        overall = np.maximum(overall, fixed_max)
        # distance from original cluster (permutation-invariant via Hungarian)
        dists = np.zeros(B)
        for b in range(B):
            C = -(Vb[b] @ V0.T)  # cost
            try:
                r_i, c_i = linear_sum_assignment(C)
                s = (-C[r_i, c_i]).sum()
                # chordal-ish distance: K - sum dots
                dists[b] = float(K - s)
            except Exception:
                dists[b] = 0.0
        # rank: primary exact max, secondary distance
        idx_sorted = np.argsort(overall)
        # take top 10 pool
        pool = idx_sorted[:min(10, B)]
        # choose best 4: best overall + diversity swap
        cand_idx = list(idx_sorted[:4])
        # if most distant in pool not in cand and within 0.004 of best, swap in
        if len(pool) >= 5:
            best_o = overall[idx_sorted[0]]
            # find most distant among pool with overall < best_o + 0.004
            feas = [i for i in pool if overall[i] < best_o + 0.004]
            if feas:
                most = max(feas, key=lambda i: dists[i])
                if most not in cand_idx:
                    cand_idx[-1] = most
        # tie-break: prefer larger dist among near-equal
        # unfreeze + refine best 3-5
        refined = []
        for ci, bi in enumerate(cand_idx):
            if time.time() > min(HOP_END, T_END - 45):
                break
            Xfull = np.empty((n, d))
            Xfull[mask] = F
            Xfull[~mask] = Vb[bi]
            v0 = float(overall[bi])
            # skip terrible
            if v0 > cur_val + 0.008 and ci >= 2:
                continue
            # short full Adam relax (unfrozen)
            Xfull = _adam_smooth_single(Xfull, steps=550, k0=30.0, k1=350.0, lr0=0.007, lr1=0.0012,
                                        deadline=min(HOP_END, T_END - 40))
            v1 = _max_cos(Xfull)
            # short LP polish
            Xfull, v2 = _lp_polish(Xfull, min(time.time() + 1.1, HOP_END, T_END - 40))
            refined.append((v2, Xfull, dists[bi]))
        if not refined:
            continue
        refined.sort(key=lambda t: t[0])
        bv, bX, bd = refined[0]
        # acceptance with annealing
        elapsed = time.time() - T0
        Temp = 0.0012 * max(0.08, 1.0 - elapsed / 120.0)
        if bv < cur_val:
            cur, cur_val = bX, bv
            if bv < best_val:
                best, best_val = bX.copy(), bv
        elif rng.random() < np.exp((cur_val - bv) / max(Temp, 1e-6)):
            cur, cur_val = bX, bv
        else:
            # opportunistic: if a distant candidate is close in value, jump to it to change basin
            for (vv, XX, dd) in refined[1:]:
                if dd > 1.0 and vv < cur_val + 0.002 and rng.random() < 0.25:
                    cur, cur_val = XX, vv
                    break
        if best_val < cur_val - 0.002 and rng.random() < 0.5:
            cur, cur_val = best.copy(), best_val
    # final intensive polish of best
    if time.time() < T_END - 5:
        best, best_val = _lp_polish(best, min(T_END - 6, time.time() + 20))
    while time.time() < T_END - 2:
        y = best + 0.002 * rng.standard_normal(size=best.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _adam_smooth_single(y, 350, 40.0, 400.0, 0.004, 0.001, deadline=T_END - 1.5)
        y, v = _lp_polish(y, min(T_END - 1, time.time() + 2.0))
        if v < best_val:
            best, best_val = y, v
        else:
            if time.time() > T_END - 6:
                break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
