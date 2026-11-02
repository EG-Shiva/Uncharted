import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    base0 = best.copy()

    def lse_polish(X0, tau, maxiter):
        nn, dd = X0.shape
        IU = np.triu_indices(nn, 1)
        I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(nn, dd)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(nn, dd)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        return G.max(axis=1), G

    def fingerprint(X):
        rm, G = rowmax(X)
        # sorted per-point maxima rounded + top global values
        a = tuple(np.sort(rm).round(4).tolist())
        tri = G[np.triu_indices(n, 1)]
        b = tuple(np.sort(tri)[-30:].round(4).tolist())
        return (a, b)

    def hole_point(Xcur, n_cand=2000):
        cand = _normalize(rng.standard_normal((n_cand, d)))
        S = cand @ Xcur.T
        sc = S.max(axis=1)
        return cand[int(np.argmin(sc))]

    def repair_dup(X):
        # replace near-duplicate rows (cos>0.93) by hole points
        for _ in range(6):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            v = float(G.max())
            if v < 0.93:
                break
            k = int(np.argmax(G))
            i = k // n; j = k % n
            # replace j with hole wrt rest
            tmp = np.delete(X, j, axis=0)
            X[j] = hole_point(tmp, n_cand=1500)
        return _normalize(X)

    def align_to(Xa, Xb):
        # Hungarian on dots then SVD rotation, return Xb aligned to Xa
        S = Xa @ Xb.T
        rr, cc = linear_sum_assignment(-S)
        perm = np.empty(n, dtype=int)
        perm[rr] = cc
        Xbp = Xb[perm]
        M = Xbp.T @ Xa
        try:
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt
        except Exception:
            R = np.eye(d)
        return Xbp @ R, perm, R

    def greedy_minimax(X, deadline, lr0=3e-5):
        cur = _max_cos(X)
        lr = float(lr0)
        it = 0
        while time.time() < deadline:
            it += 1
            if it > 4000:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-13 or nj < 1e-13:
                break
            gi /= ni; gj /= nj
            improved = False
            # try a few step sizes / modes
            for mode in (0, 1, 2):
                Y = X.copy()
                s = lr
                if mode == 0:
                    Y[i] = xi - s * gi; Y[j] = xj - s * gj
                elif mode == 1:
                    Y[i] = xi - s * 2.0 * gi
                else:
                    Y[j] = xj - s * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                G2 = Y @ Y.T
                np.fill_diagonal(G2, -2.0)
                v = float(G2.max())
                if v < cur - 1e-15:
                    X[:] = Y; cur = v; improved = True
                    break
            if improved:
                lr = min(lr * 1.15, 8e-5)
            else:
                lr *= 0.5
                if lr < 2e-9:
                    break
        return X, cur

    def full_polish(X):
        # two-stage fine LSE + greedy
        X = lse_polish(X, 5e-4, 70)
        if time.time() - t0 > LIM - 6:
            return X, _max_cos(X)
        X = lse_polish(X, 1.2e-4, 110)
        if time.time() - t0 > LIM - 4:
            return X, _max_cos(X)
        X, v = greedy_minimax(X, min(time.time() + 2.5, t0 + LIM - 2.5), lr0=8e-6)
        return X, v

    # ---------- build archive of parents ----------
    parents = []
    # keep base as parent 0 (will be polished later only as reference)
    # generate kicks
    sigmas = [0.03, 0.08, 0.15, 0.30, 0.03, 0.08, 0.15, 0.30]
    for si, sig in enumerate(sigmas):
        if time.time() - t0 > 62:
            break
        Xn = _normalize(base0 + sig * rng.standard_normal((n, d)))
        # coarse polish time-boxed
        Xn = lse_polish(Xn, 0.008, 45)
        if time.time() - t0 > 64:
            # keep even if coarse
            pass
        else:
            Xn = lse_polish(Xn, 0.002, 35)
        v = _max_cos(Xn)
        # keep only roughly reasonable (0.209-0.215 range or better)
        if v < 0.222:
            parents.append(Xn)
        if len(parents) >= 8 and time.time() - t0 > 55:
            break

    # 70-point homotopy variant: drop 4 hubs, insert 4 holes
    if time.time() - t0 < 75:
        try:
            rm, _ = rowmax(base0)
            drop = np.argsort(-rm)[:4]
            keep = np.array([i for i in range(n) if i not in set(drop.tolist())])
            X70 = base0[keep].copy()
            # coarse anneal 70 pts
            X70 = lse_polish(X70, 0.006, 30)
            # grow back to 74 by hole insertion
            Xg = X70.copy()
            for _ in range(4):
                hp = hole_point(Xg, n_cand=2000)
                Xg = np.vstack([Xg, hp.reshape(1, d)])
            Xg = lse_polish(Xg, 0.006, 40)
            if _max_cos(Xg) < 0.222:
                parents.append(_normalize(Xg))
        except Exception:
            pass

    # 78-point homotopy variant: add 4 holes then prune 4 worst
    if time.time() - t0 < 85:
        try:
            X78 = base0.copy()
            for _ in range(4):
                hp = hole_point(X78, n_cand=2000)
                X78 = np.vstack([X78, hp.reshape(1, d)])
            X78 = lse_polish(X78, 0.006, 30)
            rm78 = (X78 @ X78.T)
            np.fill_diagonal(rm78, -10.0)
            r78 = rm78.max(axis=1)
            worst = np.argsort(-r78)[:4]
            keep78 = np.array([i for i in range(78) if i not in set(worst.tolist())])
            Xh = _normalize(X78[keep78])
            Xh = lse_polish(Xh, 0.004, 40)
            if _max_cos(Xh) < 0.222:
                parents.append(Xh)
        except Exception:
            pass

    # one or two random starts, coarsely polished
    for _ in range(2):
        if time.time() - t0 > 95:
            break
        Xr = _normalize(rng.standard_normal((n, d)))
        Xr = lse_polish(Xr, 0.012, 60)
        Xr = lse_polish(Xr, 0.004, 40)
        if _max_cos(Xr) < 0.225:
            parents.append(Xr)

    # update global best with parents (coarse may already beat)
    for P_ in parents:
        v = _max_cos(P_)
        if v < best_val:
            best_val, best = float(v), P_.copy()

    # ---------- deduplicate via fingerprints + Procrustes ----------
    uniq = []
    seen_fp = set()
    for P_ in parents:
        try:
            fp = fingerprint(P_)
        except Exception:
            continue
        if fp in seen_fp:
            continue
        # Procrustes duplicate check
        dup = False
        for Q_ in uniq:
            try:
                Qa, _, _ = align_to(Q_, P_)
                diff = float(np.linalg.norm(Q_ - Qa) / np.sqrt(n))
                if diff < 0.02:
                    dup = True
                    break
            except Exception:
                continue
        if dup:
            continue
        seen_fp.add(fp)
        uniq.append(P_.copy())
        if len(uniq) >= 12:
            break
    parents = uniq

    # if too few parents, add perturbed copies of best
    while len(parents) < 4 and time.time() - t0 < LIM - 30:
        Xn = _normalize(base0 + 0.10 * rng.standard_normal((n, d)))
        Xn = lse_polish(Xn, 0.006, 40)
        parents.append(Xn)

    # ---------- crossover unlike pairs ----------
    # dissimilarity by sorted rowmax L2
    def dissim(A, B):
        ra, _ = rowmax(A); rb, _ = rowmax(B)
        return float(np.linalg.norm(np.sort(ra) - np.sort(rb)))
    pairs = []
    L = len(parents)
    for i in range(L):
        for j in range(i + 1, L):
            try:
                dd = dissim(parents[i], parents[j])
            except Exception:
                dd = 0.0
            pairs.append((dd, i, j))
    pairs.sort(reverse=True)
    # take up to 7 most unlike pairs
    pairs = pairs[:7]

    child_fps = set(seen_fp)
    nchild_done = 0
    for (_, ii, jj) in pairs:
        if time.time() - t0 > LIM - 28:
            break
        Xa = parents[ii]; Xb = parents[jj]
        try:
            Xb_al, _, _ = align_to(Xa, Xb)
            Xa_al_back, _, _ = align_to(Xb, Xa)
        except Exception:
            continue
        # two directions of half-half + hub replacement => up to 4 children
        children = []
        try:
            rm_a, _ = rowmax(Xa)
            rm_b, _ = rowmax(Xb_al)
            low_a = np.argsort(rm_a)[:37]
            mask_a = np.zeros(n, dtype=bool); mask_a[low_a] = True
            c1 = Xb_al.copy(); c1[mask_a] = Xa[mask_a]
            children.append(c1)
            low_b = np.argsort(rm_b)[:37]
            mask_b = np.zeros(n, dtype=bool); mask_b[low_b] = True
            c2 = Xa.copy()
            # complementary: take low of B into A background
            c2b = Xa.copy(); c2b[mask_b] = Xb_al[mask_b]
            children.append(c2b)
            # hub-to-hole transplants both ways
            hubs_a = np.argsort(-rm_a)[:8]
            lows_b = np.argsort(rm_b)[:8]
            c3 = Xa.copy()
            for k in range(8):
                h = int(hubs_a[k]); l = int(lows_b[k])
                if h == l:
                    continue
                c3[h] = Xb_al[l]
            children.append(c3)
            rm_xb, _ = rowmax(Xb)
            # for other direction need rowmax in Xa_al_back frame
            rm_a2, _ = rowmax(Xa_al_back)
            hubs_b = np.argsort(-rm_xb)[:8]
            lows_a2 = np.argsort(rm_a2)[:8]
            c4 = Xb.copy()
            # map: c4 hub position h (in Xb coords) gets Xa_al_back[l]
            for k in range(8):
                h = int(hubs_b[k]); l = int(lows_a2[k])
                if h == l:
                    continue
                c4[h] = Xa_al_back[l]
            children.append(c4)
        except Exception:
            continue
        for ch in children:
            if time.time() - t0 > LIM - 26:
                break
            if nchild_done >= 12:
                break
            try:
                ch = _normalize(ch)
                ch = repair_dup(ch)
                fp = fingerprint(ch)
                if fp in child_fps:
                    continue
                child_fps.add(fp)
            except Exception:
                continue
            # coarse pre-check: skip if terrible
            v0 = _max_cos(ch)
            if v0 > 0.26:
                continue
            # fully polish only fingerprint-new children
            try:
                chp, vp = full_polish(ch)
            except Exception:
                continue
            nchild_done += 1
            if vp < best_val - 1e-13:
                best_val, best = float(vp), chp.copy()
            # add polished child to archive for further recombination if good
            if vp < best_val + 0.002 and len(parents) < 14:
                parents.append(chp.copy())

    # ---------- final intensive squeeze of global best ----------
    if time.time() - t0 < LIM - 1.0:
        try:
            Xf = best.copy()
            Xf = lse_polish(Xf, 6e-5, 120)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            else:
                Xf = best.copy()
            Xf, vf = greedy_minimax(Xf, t0 + LIM - 0.6, lr0=7e-6)
            if vf < best_val:
                best = Xf.copy(); best_val = float(vf)
        except Exception:
            pass
    best = _normalize(best)
    return best
