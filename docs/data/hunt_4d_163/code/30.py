import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def _gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x = _norm(x)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.002, 0.0006), iters=(70, 90)):
    x = _norm(x_init.copy())
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = _norm(y.reshape(n, d))
    return Yf, _max_cos(Yf)

def _lp_polish(x, deadline, r0=1e-3):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _find_hole(x_fixed, rng, n_samples=6000):
    Y = rng.randn(n_samples, 4)
    Y = _norm(Y)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    y = Y[idx].copy()
    vy = float(vals[idx])
    for _ in range(120):
        dots = x_fixed @ y
        j = int(np.argmax(dots))
        g = x_fixed[j] - dots[j] * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        improved = False
        for step in (0.02, 0.005):
            yn = y - step * g / ng
            yn /= max(np.linalg.norm(yn), 1e-12)
            vn = float((x_fixed @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if not improved:
            break
    return y, vy

def _contact_deg(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - tol).sum(axis=1)
    return m, deg

def _graph_hash(x):
    _, deg = _contact_deg(x, tol=1e-6)
    s = tuple(sorted(deg.tolist()))
    return s

def _align_Q(A, B, iters=4):
    n = A.shape[0]
    B0 = B.copy()
    Q = np.eye(4)
    for _ in range(iters):
        Bc = B0 @ Q
        S = A @ Bc.T
        flat = np.argsort(-S, axis=None, kind='stable')
        usedA = np.zeros(n, bool)
        usedB = np.zeros(n, bool)
        perm = np.empty(n, dtype=int)
        cnt = 0
        for f in flat:
            ai = int(f // n); bj = int(f % n)
            if not usedA[ai] and not usedB[bj]:
                perm[ai] = bj
                usedA[ai] = True; usedB[bj] = True
                cnt += 1
                if cnt >= n:
                    break
        Bass = Bc[perm]
        M = Bass.T @ A
        try:
            U, _, Vt = np.linalg.svd(M)
            Qd = U @ Vt
        except Exception:
            break
        Q = Q @ Qd
        if np.linalg.norm(Qd - np.eye(4)) < 1e-10:
            break
    return Q

def _union_maximin(U, k, rng):
    M = U.shape[0]
    G = U @ U.T
    best_start = None
    best_sel = None
    best_score = 2.0
    for trial in range(2):
        start = int(rng.randint(M)) if trial == 0 else int(np.argmax(np.linalg.norm(U - U[best_sel[0]], axis=1))) if best_sel is not None else int(rng.randint(M))
        sel_mask = np.zeros(M, bool)
        selected = [start]
        sel_mask[start] = True
        curbest = G[:, start].copy()
        curbest[start] = np.inf
        for _ in range(1, k):
            tmp = np.where(sel_mask, np.inf, curbest)
            idx = int(np.argmin(tmp))
            selected.append(idx)
            sel_mask[idx] = True
            col = G[:, idx]
            np.maximum(curbest, col, out=curbest)
            curbest[idx] = np.inf
        sel_arr = np.array(selected, dtype=int)
        # score = max cos inside selected
        Gs = G[np.ix_(sel_arr, sel_arr)]
        np.fill_diagonal(Gs, -2.0)
        sc = float(Gs.max())
        if trial == 0 or sc < best_score:
            best_score = sc
            best_sel = sel_arr
            best_start = start
    # local swaps
    sel = best_sel.copy()
    unsel = np.array([i for i in range(M) if i not in set(sel.tolist())], dtype=int)
    for _ in range(25):
        Gs = G[np.ix_(sel, sel)]
        np.fill_diagonal(Gs, -2.0)
        mi = int(np.argmax(Gs.max(axis=1)))
        # worst position mi (index into sel)
        rest = np.delete(sel, mi)
        Grest = G[:, rest] if len(rest) > 0 else np.zeros((M, 0))
        if Grest.shape[1] > 0:
            cand_score = Grest[unsel].max(axis=1) if len(unsel) > 0 else np.array([])
        else:
            cand_score = np.zeros(len(unsel))
        if len(cand_score) == 0:
            break
        bi = int(np.argmin(cand_score))
        # current max of sel
        cur_max = float(Gs.max())
        # new max if swap
        new_sel = sel.copy(); new_sel[mi] = unsel[bi]
        Gn = G[np.ix_(new_sel, new_sel)]
        np.fill_diagonal(Gn, -2.0)
        new_max = float(Gn.max())
        if new_max < cur_max - 1e-12:
            sel = new_sel
            sset = set(sel.tolist())
            unsel = np.array([i for i in range(M) if i not in sset], dtype=int)
        else:
            break
    return U[sel]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    rng = np.random.RandomState(20261102)
    nT = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(nT, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (nT, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(nT, 4)
    bx = _norm(bx)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    archive = [(best_val, best_x.copy())]
    seen_hashes = set()
    try:
        seen_hashes.add(_graph_hash(best_x))
    except Exception:
        pass

    def add_archive(x, v):
        nonlocal best_x, best_val
        x = _norm(x.copy())
        try:
            h = _graph_hash(x)
        except Exception:
            h = None
        if v < best_val - 1e-12:
            best_val = float(v); best_x = x.copy()
        # diversity keep
        if v > best_val + 0.006:
            return False
        if h is not None and h in seen_hashes:
            # allow if it improves best
            if v >= best_val - 1e-12:
                return False
        if h is not None:
            seen_hashes.add(h)
        archive.append((float(v), x.copy()))
        archive.sort(key=lambda t: t[0])
        # keep top 12
        del archive[12:]
        return True

    def light_polish(x, gd_steps=1800, T0=0.009, T1=0.0005, lr0=0.038, mu=0.86, noise=0.0015):
        dl = time.time() + 2.2
        xb, vb = _gd(x, steps=gd_steps, T0=T0, T1=T1, lr0=lr0, mu=mu, rng=rng, noise0=noise, deadline=min(dl, SEARCH_END))
        try:
            xl, vl = _lbfgs_smooth(xb, deadline=min(dl, SEARCH_END))
            if vl < vb:
                xb, vb = xl, vl
        except Exception:
            pass
        return xb, vb

    def worst_index(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = (G > m - 0.003).sum(axis=1)
        mx = G.max(axis=1)
        # lexicographic: mx primary
        return int(np.lexsort((deg, mx))[-1])

    # ---- N-continuation path down->up ----
    try:
        cur = best_x.copy()
        # down to 155
        for N in range(162, 154, -1):
            if time.time() > SEARCH_END - 20:
                break
            if time.time() > DEADLINE - 60:
                break
            wi = worst_index(cur)
            cur = np.delete(cur, wi, axis=0)
            cur, _ = light_polish(cur, gd_steps=1500, T0=0.008, T1=0.0005, lr0=0.036, mu=0.86, noise=0.0012)
        # up back to 163
        for N in range(cur.shape[0] + 1, 164):
            if time.time() > SEARCH_END - 18:
                break
            h, _ = _find_hole(cur, rng, n_samples=6000)
            cur = np.vstack([cur, h.reshape(1, 4)])
            cur, cv = light_polish(cur, gd_steps=1700, T0=0.009, T1=0.0005, lr0=0.038, mu=0.86, noise=0.0012)
            if N == 163:
                add_archive(cur, cv)
            elif N >= 160:
                pass
    except Exception:
        pass
    # ---- N-continuation path up->down ----
    try:
        cur2 = best_x.copy()
        # small perturbation for distinct basin
        cur2 = _norm(cur2 + rng.randn(*cur2.shape) * 0.015)
        cur2, cv2 = light_polish(cur2, gd_steps=1200, T0=0.008, T1=0.0006, lr0=0.036, mu=0.86, noise=0.001)
        for N in range(164, 173):
            if time.time() > SEARCH_END - 14:
                break
            h, _ = _find_hole(cur2, rng, n_samples=6000)
            cur2 = np.vstack([cur2, h.reshape(1, 4)])
            cur2, _ = light_polish(cur2, gd_steps=1500, T0=0.008, T1=0.0005, lr0=0.036, mu=0.86, noise=0.0012)
            if cur2.shape[0] != N:
                break
        for N in range(cur2.shape[0] - 1, 162, -1):
            if time.time() > SEARCH_END - 10:
                break
            wi = worst_index(cur2)
            cur2 = np.delete(cur2, wi, axis=0)
            cur2, cv = light_polish(cur2, gd_steps=1700, T0=0.009, T1=0.0005, lr0=0.038, mu=0.86, noise=0.0012)
            if N == 163:
                add_archive(cur2, cv)
    except Exception:
        pass

    # ensure archive has at least 2 entries: add mutants
    tries = 0
    while len(archive) < 3 and time.time() < SEARCH_END - 8 and tries < 4:
        tries += 1
        try:
            sig = 0.06 + 0.05 * tries
            cur = _norm(best_x + rng.randn(*best_x.shape) * sig)
            cur, cv = light_polish(cur, gd_steps=1800, T0=0.011, T1=0.0006, lr0=0.040, mu=0.85, noise=0.002)
            add_archive(cur, cv)
        except Exception:
            pass

    # ---- orthogonal crossover loop ----
    cx_iter = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 45:
            break
        cx_iter += 1
        try:
            if len(archive) < 2:
                base = best_x
                cur = _norm(base + rng.randn(*base.shape) * 0.08)
                cur_b, cur_v = _gd(cur, steps=2500, T0=0.012, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.002, deadline=SEARCH_END)
                try:
                    lbx, lbv = _lbfgs_smooth(cur_b, deadline=min(SEARCH_END, time.time() + 2.0))
                    if lbv < cur_v:
                        cur_b, cur_v = lbx, lbv
                except Exception:
                    pass
                add_archive(cur_b, cur_v)
                continue
            # pick most distant pair by degree-vector L1
            archive_sorted = sorted(archive, key=lambda t: t[0])
            # consider top up to 6
            cand = archive_sorted[:min(6, len(archive_sorted))]
            best_pair = None
            best_dist = -1
            for i in range(len(cand)):
                hi = _graph_hash(cand[i][1])
                for j in range(i + 1, len(cand)):
                    hj = _graph_hash(cand[j][1])
                    # L1 on sorted degree (pad)
                    L = max(len(hi), len(hj))
                    d = sum(abs((hi[k] if k < len(hi) else 0) - (hj[k] if k < len(hj) else 0)) for k in range(L))
                    # prefer pairs with good values but distant
                    score = d - 200.0 * max(0.0, (cand[i][0] - best_val) + (cand[j][0] - best_val))
                    if score > best_dist:
                        best_dist = score
                        best_pair = (i, j)
            if best_pair is None:
                best_pair = (0, 1)
            A = cand[best_pair[0]][1]
            B = cand[best_pair[1]][1]
            # alternate direction
            if cx_iter % 2 == 1:
                A, B = B, A
            Q = _align_Q(A, B, iters=4)
            Brot = _norm(B @ Q)
            U = np.vstack([A, Brot])
            # de-duplicate very close points slightly jitter
            child0 = _union_maximin(U, nT, rng)
            child0 = _norm(child0 + rng.randn(*child0.shape) * 0.003)
            cur_b, cur_v = _gd(child0, steps=2800, T0=0.011 if cx_iter % 3 else 0.016, T1=0.0005, lr0=0.040, mu=0.86, rng=rng, noise0=0.0018, deadline=SEARCH_END)
            if cur_v < best_val + 0.003 and time.time() < SEARCH_END - 1.0:
                try:
                    lbx, lbv = _lbfgs_smooth(cur_b, deadline=min(SEARCH_END, time.time() + 2.5))
                    if lbv < cur_v:
                        cur_b, cur_v = lbx, lbv
                except Exception:
                    pass
            added = add_archive(cur_b, cur_v)
            # if not diverse, try swapped start (different greedy seed consumes rng) next iter will differ
            if not added and (cx_iter % 4 == 0):
                # aggressive topology jump: cap rotation on best
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = best_x @ u
                q = 0.80 + 0.12 * rng.rand()
                thresh = np.quantile(dots, q)
                idx = np.where(dots > thresh)[0]
                if len(idx) < 3:
                    idx = np.argsort(dots)[-12:]
                Ar = rng.randn(4, 4) * (0.10 + 0.25 * rng.rand())
                Ar = Ar - Ar.T
                Qr, _ = np.linalg.qr(np.eye(4) + Ar)
                curm = best_x.copy()
                curm[idx] = (Qr @ curm[idx].T).T
                curm = _norm(curm + rng.randn(*curm.shape) * 0.004)
                cb, cv = _gd(curm, steps=2500, T0=0.013, T1=0.0006, lr0=0.040, mu=0.86, rng=rng, noise0=0.0015, deadline=SEARCH_END)
                try:
                    lbx, lbv = _lbfgs_smooth(cb, deadline=min(SEARCH_END, time.time() + 2.0))
                    if lbv < cv:
                        cb, cv = lbx, lbv
                except Exception:
                    pass
                add_archive(cb, cv)
        except Exception:
            continue

    # ---- final LP polish on top candidates ----
    archive.sort(key=lambda t: t[0])
    final_cands = archive[:3]
    for k, (v, x) in enumerate(final_cands):
        if time.time() > DEADLINE - 4:
            break
        try:
            # extra LBFGS first
            if time.time() < DEADLINE - 12:
                x2, v2 = _lbfgs_smooth(x, deadline=min(DEADLINE - 3, time.time() + 4.0), taus=(0.0012, 0.0005), iters=(100, 140))
                if v2 < v:
                    x, v = x2, v2
                    if v < best_val:
                        best_val, best_x = float(v), x.copy()
            xp, vp = _lp_polish(x, deadline=min(DEADLINE - 1.5, time.time() + 10.0), r0=1e-3)
            if vp < best_val - 1e-12:
                best_val, best_x = float(vp), xp.copy()
            # keep
            archive.append((float(vp), xp.copy()))
        except Exception:
            continue
    archive.sort(key=lambda t: t[0])
    best_x = archive[0][1].copy() if len(archive) > 0 else best_x
    best_x = _norm(best_x)
    return best_x
