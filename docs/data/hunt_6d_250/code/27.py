import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_refine(X_init, steps, b0, b1, lr):
    n, d = X_init.shape
    X = X_init.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    tt = 0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = b0 * ((b1 / b0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(np.max(G))
        G -= cur_max
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s_sum = float(G.sum() * 0.5)
        if s_sum < 1e-100:
            continue
        grad = (G @ X) / s_sum
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        grad -= dot * X
        tt += 1
        m *= b1m
        m += (1.0 - b1m) * grad
        v *= b2m
        v += (1.0 - b2m) * (grad * grad)
        c1 = 1.0 - (b1m ** tt)
        c2 = 1.0 - (b2m ** tt)
        vhat = v / c2
        step = (lr / c1) * m / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr
        scale = np.minimum(1.0, cap / sn)
        step = step * scale
        X -= step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
        md = np.einsum('ij,ij->i', m, X)[:, None]
        m -= md * X
    return X

def _lp_polish(x, deadline, k_top=5000, radius_init=2.0e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0, iu1 = np.triu_indices(n, 1)
    ntri = iu0.shape[0]
    K = min(k_top, ntri)
    while time.time() < deadline - 1.0 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        if K < ntri:
            top = np.argpartition(-c_all, K)[:K]
            I = iu0[top]; J = iu1[top]; c_sel = c_all[top]
        else:
            I = iu0; J = iu1; c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols_blocks = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols_blocks)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        val = _max_cos(y)
        if val < best - 1e-12:
            x, best = y, val
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _deep_hole(X, exclude_set, rng, nc=2048):
    n, d = X.shape
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ X.T
    for w in exclude_set:
        S[:, int(w)] = -2.0
    # for already-replaced holes we also want far from new picks: handled by caller
    mx = S.max(axis=1)
    bi = int(np.argmin(mx))
    return cand[bi], float(mx[bi])

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    HOP_END = t0 + 122.0
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(f), dtype=float)
            if a.shape == (n, d):
                a = np.nan_to_num(a)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    # normalize
    best /= np.linalg.norm(best, axis=1, keepdims=True)

    # quick settle of starting point
    try:
        Xc = _adam_refine(best, 800, 80.0, 2500.0, 0.0012)
        vc = _max_cos(Xc)
        if vc < best_val:
            best, best_val = Xc, vc
    except Exception:
        pass

    cur = best.copy()
    cur_val = best_val
    Temp = 6e-5
    trial = 0
    stall = 0
    while time.time() < HOP_END and trial < 200:
        trial += 1
        if time.time() > HARD - 55:
            break
        try:
            rm, Gmat = _rowmax(cur)
            order = np.argsort(-rm)
        except Exception:
            order = np.arange(n)
            Gmat = cur @ cur.T
        mode = trial % 6
        X0 = cur.copy()
        if mode == 0:
            # block kick of worst-loaded block + light global noise (jam snap)
            K = 18 + (trial % 17)  # 18..34
            idx = order[:K]
            # also add random companions to change topology
            extra = rng.choice(n, size=8, replace=False)
            idx = np.unique(np.concatenate([idx, extra]))
            sigma = 0.07 + 0.03 * ((trial // 6) % 3)  # 0.07,0.10,0.13
            X0[idx] += sigma * rng.standard_normal((len(idx), d))
            mask = np.ones(n, dtype=bool); mask[idx] = False
            X0[mask] += 0.008 * rng.standard_normal((n - len(idx), d))
        elif mode == 1:
            # multi-void reinsertion of worst points into deepest holes
            nw = 3 + (trial % 4)  # 3..6
            worst = list(order[:nw])
            # greedy sequential holes
            keep = [i for i in range(n) if i not in worst]
            Xref = cur[keep]
            newpts = []
            for k, w in enumerate(worst):
                # exclude already placed? compute against keep+newpts
                if newpts:
                    ref = np.vstack([Xref, np.vstack(newpts)])
                else:
                    ref = Xref
                # _deep_hole expects exclude handling; just search vs ref
                cand = rng.standard_normal((2048, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                S = cand @ ref.T
                mx = S.max(axis=1)
                bi = int(np.argmin(mx))
                newpts.append(cand[bi])
            for w, pt in zip(worst, newpts):
                X0[w] = pt + 0.005 * rng.standard_normal(d)
            # jitter rest lightly
            X0 += 0.004 * rng.standard_normal((n, d))
        elif mode == 2:
            # pair-split: separate current max pair and its shell
            try:
                Gtmp = cur @ cur.T
                np.fill_diagonal(Gtmp, -2.0)
                flat = int(np.argmax(Gtmp))
                ii, jj = divmod(flat, n)
                mid = cur[ii] + cur[jj]
                mn = np.linalg.norm(mid)
                if mn < 1e-12:
                    mid = rng.standard_normal(d)
                    mn = np.linalg.norm(mid)
                mid /= mn
                # push apart along difference direction
                diff = cur[ii] - cur[jj]
                dn = np.linalg.norm(diff)
                if dn < 1e-12:
                    diff = rng.standard_normal(d)
                    dn = np.linalg.norm(diff)
                diff /= dn
                s = 0.10 + 0.04 * (trial % 3)
                X0[ii] += s * diff + 0.02 * rng.standard_normal(d)
                X0[jj] -= s * diff + 0.02 * rng.standard_normal(d)
                # kick common neighbors (shell collision set)
                dots_i = cur @ cur[ii]
                dots_j = cur @ cur[jj]
                neigh = np.where((dots_i > best_val - 2e-3) | (dots_j > best_val - 2e-3))[0]
                neigh = neigh[(neigh != ii) & (neigh != jj)][:25]
                if len(neigh) > 0:
                    X0[neigh] += 0.05 * rng.standard_normal((len(neigh), d))
                X0 += 0.005 * rng.standard_normal((n, d))
            except Exception:
                X0 += 0.05 * rng.standard_normal((n, d))
        elif mode == 3:
            # slab shear: coherent displacement of half-space
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            vdir = rng.standard_normal(d)
            vdir -= (vdir @ u) * u
            vn = np.linalg.norm(vdir)
            if vn < 1e-12:
                vdir = rng.standard_normal(d)
                vn = np.linalg.norm(vdir)
            vdir /= vn
            proj = cur @ u
            med = np.median(proj)
            upper = proj > med
            alpha = 0.08 + 0.04 * ((trial // 6) % 2)
            X0[upper] += alpha * vdir
            X0[~upper] -= 0.3 * alpha * vdir
            X0 += 0.006 * rng.standard_normal((n, d))
        elif mode == 4:
            # random-subset big shake (basin jump)
            K = 40 + (trial % 20)
            idx = rng.choice(n, size=K, replace=False)
            X0[idx] += 0.12 * rng.standard_normal((K, d))
            X0 += 0.01 * rng.standard_normal((n, d))
        else:
            # global moderate shake
            X0 += 0.025 * rng.standard_normal((n, d))

        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # anneal: alternate schedules to sometimes equalize plateau, sometimes squeeze
        try:
            if trial % 3 == 0:
                Xc = _adam_refine(X0, 850, 60.0, 3000.0, 0.0014)
            elif trial % 3 == 1:
                Xc = _adam_refine(X0, 1000, 200.0, 6000.0, 0.0009)
            else:
                Xc = _adam_refine(X0, 700, 40.0, 1200.0, 0.0018)
        except Exception:
            continue
        try:
            vc = _max_cos(Xc)
        except Exception:
            continue
        if vc < best_val - 1e-12:
            best, best_val = Xc.copy(), vc
            cur, cur_val = Xc.copy(), vc
            stall = 0
        else:
            # Metropolis walk on cur to visit new basins
            if vc < cur_val - 1e-12:
                cur, cur_val = Xc.copy(), vc
                stall = 0
            else:
                stall += 1
                delta = vc - cur_val
                prob = np.exp(-delta / Temp) if Temp > 0 else 0.0
                if rng.random() < prob:
                    cur, cur_val = Xc.copy(), vc
                # periodic restart from best if wandering too far / stuck
                if stall > 18:
                    # reheat from best with big kick to force new basin
                    cur = best.copy()
                    cur_val = best_val
                    stall = 0
                    Temp = min(1.2e-4, Temp * 1.15)
                if trial % 25 == 0:
                    # anneal temperature down slowly
                    Temp = max(2e-5, Temp * 0.95)
        if time.time() >= HOP_END:
            break

    # final tie + squeeze with remaining time
    try:
        if time.time() < HARD - 6:
            best = _lp_polish(best, min(HARD - 1.0, time.time() + 26.0), k_top=5000, radius_init=2.0e-3)
            best_val = _max_cos(best)
    except Exception:
        pass
    try:
        if time.time() < HARD - 8:
            Xc = _adam_refine(best, 1000, 500.0, 8000.0, 0.0006)
            vc = _max_cos(Xc)
            if vc < best_val:
                best, best_val = Xc, vc
    except Exception:
        pass
    try:
        if time.time() < HARD - 4:
            best = _lp_polish(best, HARD - 1.0, k_top=5000, radius_init=3.0e-4)
            best_val = _max_cos(best)
    except Exception:
        pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    return best
