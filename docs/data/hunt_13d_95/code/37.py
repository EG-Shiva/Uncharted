import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3, max_c=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos_single(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_c:
            idx = np.argpartition(c, -max_c)[-max_c:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        v = max_cos_single(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def batch_maxs(X):
    # X: (B,N,D)
    G = np.einsum('bik,bjk->bij', X, X)
    B, N, _ = G.shape
    for b in range(B):
        np.fill_diagonal(G[b], -2.0)
    return G.max(axis=(1, 2)), G


def fingerprints(X, Gm, maxs, tol=0.002):
    B, N, _ = Gm.shape
    eigs = np.empty((B, N))
    cc_sorted = np.empty((B, N), dtype=int)
    for b in range(B):
        # symmetrize
        Gb = (Gm[b] + Gm[b].T) * 0.5
        # for eig, restore diagonal 1
        np.fill_diagonal(Gb, 1.0)
        try:
            ev = np.linalg.eigvalsh(Gb)
        except Exception:
            ev = np.zeros(N)
        eigs[b] = np.sort(ev)
        c = (Gm[b] > maxs[b] - tol).sum(axis=1)
        cc_sorted[b] = np.sort(c)
    return eigs, cc_sorted


def crossover_child(base, donor, rng, K=None):
    n, d = base.shape
    if K is None:
        K = int(rng.integers(15, 26))
        K = min(K, n // 2)
    # contacts
    Gb = base @ base.T
    np.fill_diagonal(Gb, -2.0)
    mb = float(Gb.max())
    ccb = (Gb > mb - 0.002).sum(axis=1)
    ppm_b = Gb.max(axis=1)
    Gd = donor @ donor.T
    np.fill_diagonal(Gd, -2.0)
    md = float(Gd.max())
    ccd = (Gd > md - 0.002).sum(axis=1)
    replace_idx = np.argsort(-ppm_b - 0.1 * ccb)[:K]
    donor_idx = np.argsort(ccd)[:K]
    D = donor[donor_idx]  # K x d
    # voids of base: farthest random dirs
    C = rng.normal(size=(4000, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    mdots = (C @ base.T).max(axis=1)
    void_idx = np.argpartition(mdots, K)[:K]
    V = C[void_idx]
    # procrustes D -> V
    try:
        M = D.T @ V
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        Dr = D @ R
    except Exception:
        Dr = D
    Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)
    child = base.copy()
    child[replace_idx] = Dr
    # tiny jitter to break symmetry
    child[replace_idx] += 1e-4 * rng.normal(size=(K, d))
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    N, D, B = 95, 13, 24

    # fallback best
    try:
        x_best_file = np.load("best.npy").astype(float)
        x_best_file /= np.linalg.norm(x_best_file, axis=1, keepdims=True)
        best_file_val = max_cos_single(x_best_file)
    except Exception:
        x_best_file = None
        best_file_val = 1e9

    # fresh-start population: random Gaussian rows + low-discrepancy-ish
    # half pure Gaussian, half Gaussian with different scale / orthogonalized blocks
    X = np.empty((B, N, D))
    for b in range(B):
        if b < 12:
            Z = rng.normal(size=(N, D))
        else:
            # low-discrepancy: random rotation of a fixed quasi-uniform draw
            # use fresh gaussian but normalized with variance correction + shuffle
            Z = rng.normal(size=(N, D))
            # add deterministic Halton-like offset via sin to diversify
            off = 0.3 * np.sin(np.arange(N * D, dtype=float).reshape(N, D) + b * 1.7)
            Z = Z + off
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        X[b] = Z

    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    POP_T = 70.0
    t_pop_end = t0 + POP_T
    last_fp = t0
    # anneal schedule by elapsed fraction
    while time.time() < t_pop_end:
        remain_pop = t_pop_end - time.time()
        if remain_pop < 0.2:
            break
        frac = 1.0 - remain_pop / POP_T
        if frac < 0.25:
            beta, lr = 20.0, 0.01
        elif frac < 0.50:
            beta, lr = 80.0, 0.006
        elif frac < 0.75:
            beta, lr = 250.0, 0.004
        else:
            beta, lr = 700.0, 0.002
        # chunk of steps
        CH = 120
        for _ in range(CH):
            if time.time() >= t_pop_end:
                break
            step += 1
            G = np.einsum('bik,bjk->bij', X, X)
            # per-batch max for stability (off-diag)
            # set diag temporarily low to get max
            mx = np.empty(B)
            for b in range(B):
                # max off-diag: use triu
                Gb = G[b]
                # quick max ignoring diag by filling
                # copy not needed: compute max of G with diag -2 effect via:
                # max = max(max of G, need exclude diag which is 1 -> offdiag <1 so fine unless all 1)
                # but diag=1 is max, so subtract
                np.fill_diagonal(Gb, -2.0)
                mx[b] = float(Gb.max())
                np.fill_diagonal(Gb, 1.0)
            # exp
            # G in [-1,1], (G-mx)*beta
            E = np.exp((G - mx[:, None, None]) * beta)
            for b in range(B):
                np.fill_diagonal(E[b], 0.0)
            sum_all = E.sum(axis=(1, 2))  # (B,)
            sum_all = np.maximum(sum_all, 1e-300)
            W = E / sum_all[:, None, None] * 2.0  # so upper sums to 1
            Grad = np.einsum('bij,bjd->bid', W, X)
            dot = np.sum(Grad * X, axis=2, keepdims=True)
            Gt = Grad - dot * X
            m = b1 * m + (1 - b1) * Gt
            v = b2 * v + (1 - b2) * (Gt * Gt)
            bc1 = 1 - b1 ** step
            bc2 = 1 - b2 ** step
            mh = m / bc1
            vh = v / bc2
            upd = lr * mh / (np.sqrt(vh) + eps)
            # tangent projection of update not strictly needed; renormalize handles it
            X = X - upd
            X /= np.linalg.norm(X, axis=2, keepdims=True)
        # every ~10s fingerprint + crossover
        if time.time() - last_fp >= 9.0 or time.time() >= t_pop_end - 0.5:
            last_fp = time.time()
            try:
                maxs, Gm = batch_maxs(X.copy())
                eigs, ccs = fingerprints(X, Gm, maxs, tol=0.002)
                order = np.argsort(maxs)
                # survivors greedy distinct
                surv = []
                is_dup = np.zeros(B, dtype=bool)
                for bi in order:
                    dup = False
                    for s in surv:
                        ed = float(np.linalg.norm(eigs[bi] - eigs[s]))
                        if ed < 0.02 and abs(float(maxs[bi] - maxs[s])) < 2e-3:
                            dup = True
                            break
                    if dup:
                        is_dup[bi] = True
                    else:
                        surv.append(bi)
                # replace duplicates + worst with crossover children
                # identify worst 4
                worst = order[-4:]
                to_replace = set(np.where(is_dup)[0].tolist()) | set(worst.tolist())
                # keep best survivor never replaced
                if len(surv) > 0 and order[0] in to_replace:
                    to_replace.discard(order[0])
                if len(surv) >= 2:
                    for bi in list(to_replace):
                        # base = random survivor among top, donor = another
                        a = surv[int(rng.integers(0, min(len(surv), 6)))]
                        bb = surv[int(rng.integers(0, len(surv)))]
                        if bb == a:
                            bb = surv[(surv.index(a) + 1) % len(surv)]
                        child = crossover_child(X[a], X[bb], rng, K=None)
                        X[bi] = child
                        m[bi] = 0
                        v[bi] = 0
            except Exception:
                last_fp = time.time()
                continue

    # evaluate and pick top three distinct
    try:
        maxs, Gm = batch_maxs(X.copy())
    except Exception:
        maxs = np.array([max_cos_single(X[b]) for b in range(B)])
        Gm = None
    order = np.argsort(maxs)
    try:
        eigs, ccs = fingerprints(X, Gm, maxs, tol=0.002)
        elites = []
        for bi in order:
            ok = True
            for s in elites:
                if float(np.linalg.norm(eigs[bi] - eigs[s])) < 0.02:
                    ok = False
                    break
            if ok:
                elites.append(bi)
            if len(elites) >= 3:
                break
        if len(elites) < 3:
            for bi in order:
                if bi not in elites:
                    elites.append(bi)
                if len(elites) >= 3:
                    break
    except Exception:
        elites = order[:3].tolist()

    # LP-polish top three ~12-15s each within remaining budget
    polished = []
    for k, bi in enumerate(elites[:3]):
        rem = BUDGET - (time.time() - t0)
        # reserve ~28s for final long polish + 2s margin
        if rem < 35:
            break
        sec = min(13.0, max(5.0, (rem - 30.0) / (3 - k)))
        try:
            xp = lp_polish(X[bi].copy(), seconds=sec, radius_init=1e-3)
            polished.append(xp)
        except Exception:
            polished.append(X[bi].copy())
    if not polished:
        # fallback to best member
        polished = [X[order[0]].copy()]

    vals = [max_cos_single(p) for p in polished]
    wi = int(np.argmin(vals))
    winner = polished[wi].copy()
    wval = float(vals[wi])

    # long-polish winner
    rem = BUDGET - (time.time() - t0) - 1.5
    if rem > 3.0:
        try:
            sec1 = min(rem * 0.6, 20.0)
            winner = lp_polish(winner, seconds=sec1, radius_init=5e-4)
            wval = max_cos_single(winner)
            rem2 = BUDGET - (time.time() - t0) - 1.0
            if rem2 > 3.0:
                winner = lp_polish(winner, seconds=min(rem2, 15.0), radius_init=2e-4)
                wval = max_cos_single(winner)
        except Exception:
            pass
    winner /= np.linalg.norm(winner, axis=1, keepdims=True)

    # return only if beats best.npy, else fallback
    if x_best_file is not None and wval >= best_file_val - 1e-12:
        return x_best_file
    return winner
