import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_full(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        lb = 0.0; ll = 0.0
    else:
        lb = np.log(beta1 / beta0) / (steps - 1)
        ll = np.log(lr1 / lr0) / (steps - 1)
    lbb0 = np.log(beta0); llr0 = np.log(lr0)
    b1t = 1.0; b2t = 1.0
    eye_n = n
    for t in range(1, steps + 1):
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _adam_masked(Y0, idx, steps, beta0, beta1, lr0, lr1):
    # only rows idx move; others frozen
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        lb = 0.0; ll = 0.0
    else:
        lb = np.log(beta1 / beta0) / (steps - 1)
        ll = np.log(lr1 / lr0) / (steps - 1)
    lbb0 = np.log(beta0); llr0 = np.log(lr0)
    b1t = 1.0; b2t = 1.0
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    idx = np.asarray(idx, dtype=int)
    for t in range(1, steps + 1):
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        # mask frozen
        # zero grad for frozen rows
        # use in-place masking
        # build full Gr then mask: faster to zero rows not in idx
        # Since n=59 small, just zero and update only idx
        b1t *= b1; b2t *= b2
        # update moments only for moving rows to keep frozen exactly fixed
        M[idx] *= b1
        M[idx] += (1.0 - b1) * Gr[idx]
        V[idx] *= b2
        V[idx] += (1.0 - b2) * (Gr[idx] * Gr[idx])
        Mh = M[idx] / (1.0 - b1t)
        Vh = V[idx] / (1.0 - b2t)
        Y[idx] -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y[idx], axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y[idx] /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0

    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best_val0:
                best_val0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best_val0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = float(best_val0)

    k_cycle = [2, 3, 4, 6, 3, 2, 4, 3, 6, 2]
    it = 0
    # reserve time for final polish
    while time.time() < deadline - 18.0:
        k = int(k_cycle[it % len(k_cycle)])
        it += 1
        # adapt steps to remaining time
        t_rem = deadline - time.time()
        if t_rem < 20.0:
            break
        # contact counts with margin
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            C = (G > (m - 0.004)).sum(axis=1)
        except Exception:
            C = np.zeros(59)
        # choose k among highest-contact points
        order = np.argsort(-C, kind="stable")
        pool = order[:max(15, 3 * k)]
        if len(pool) < k:
            pool = order[:k]
        del_idx = np.sort(rng.choice(pool, size=k, replace=False))
        keep_mask = np.ones(59, dtype=bool)
        keep_mask[del_idx] = False
        F = x_best[keep_mask]  # (59-k, 22)
        mf = F.shape[0]

        # ---- generate ~20000 candidates ----
        n_gauss = 8000
        n_sum = 6000
        n_pert = 6000
        Cg = _normalize(rng.standard_normal((n_gauss, 22)))
        # sums / differences of frozen points
        Cs = np.empty((n_sum, 22), dtype=np.float64)
        for i in range(n_sum):
            a_i = int(rng.integers(0, mf))
            b_i = int(rng.integers(0, mf))
            if rng.random() < 0.5:
                v = F[a_i] + F[b_i]
            else:
                v = F[a_i] - F[b_i]
            nv = float(np.linalg.norm(v))
            if nv < 1e-9 or not np.isfinite(nv):
                v = rng.standard_normal(22)
                nv = float(np.linalg.norm(v))
            Cs[i] = v / nv
        # score gauss+sum to find holes
        C0 = np.vstack([Cg, Cs])  # 14000
        # batched scoring vs frozen
        # C0 @ F.T -> max
        S0 = (C0 @ F.T).max(axis=1)
        # pick hole centres: lowest S0
        nh = 30
        hidx = np.argpartition(S0, nh)[:nh]
        centres = C0[hidx]
        Cp = np.empty((n_pert, 22), dtype=np.float64)
        sigmas = np.array([0.05, 0.08, 0.12, 0.2])
        for i in range(n_pert):
            ci = centres[int(rng.integers(0, nh))]
            sg = float(sigmas[int(rng.integers(0, len(sigmas)))])
            v = ci + sg * rng.standard_normal(22)
            nv = float(np.linalg.norm(v))
            if nv < 1e-9 or not np.isfinite(nv):
                v = rng.standard_normal(22)
                nv = float(np.linalg.norm(v))
            Cp[i] = v / nv
        C_all = np.vstack([C0, Cp])  # 20000
        # frozen scores
        sf = (C_all @ F.T).max(axis=1)
        # greedy insertion accounting for among-inserted cosines
        cur = sf.copy()
        picked = []
        used = np.zeros(C_all.shape[0], dtype=bool)
        for s in range(k):
            # argmin among unused
            # set used to inf temporarily
            cur_masked = cur.copy()
            cur_masked[used] = np.inf
            bi = int(np.argmin(cur_masked))
            picked.append(bi)
            used[bi] = True
            piv = C_all[bi]
            dots = C_all @ piv
            # update cur = max(cur, dots) for unused
            # for used entries irrelevant
            np.maximum(cur, dots, out=cur)
            # keep picked at inf to avoid re-pick (already used)
        ins_vecs = C_all[np.array(picked)]
        # build full config
        Y = np.empty((59, 22), dtype=np.float64)
        Y[keep_mask] = F
        Y[del_idx] = ins_vecs
        Y = _normalize(Y)

        # masked optimisation of inserted rows only (~800 steps)
        msteps = 800
        # shorten if low on time
        if t_rem < 40.0:
            msteps = 500
        Ym, vm = _adam_masked(Y, del_idx, steps=msteps, beta0=60.0, beta1=600.0, lr0=0.01, lr1=0.0008)
        # unfreeze short polish 1500-2500 steps
        if t_rem < 40.0:
            fsteps = 1200
        else:
            fsteps = 1800 if (it % 3 != 0) else 2400
        # vary beta slightly per iteration for diversity
        b0 = 80.0 if (it % 2 == 0) else 50.0
        Yf, vf = _adam_full(Ym, steps=fsteps, beta0=b0, beta1=700.0, lr0=0.006, lr1=0.0006)
        if vf < best_val - 1e-12:
            best_val = float(vf)
            # re-evaluate true max of returned best (Ym vs Yf best tracking already)
            x_best = _normalize(Yf.copy())
        # also consider intermediate vm in case full polish drifted
        if vm < best_val - 1e-12:
            # re-polish vm lightly? just accept if better than current
            best_val = float(vm)
            x_best = _normalize(Ym.copy())
        if time.time() > deadline - 18.0:
            break

    # final polish of best
    if time.time() < deadline - 12.0:
        try:
            Yp, vp = _adam_full(x_best, steps=4000, beta0=120.0, beta1=900.0, lr0=0.004, lr1=0.0004)
            if vp < best_val:
                best_val = vp
                x_best = Yp.copy()
        except Exception:
            pass

    x_best = _normalize(x_best)
    return x_best
