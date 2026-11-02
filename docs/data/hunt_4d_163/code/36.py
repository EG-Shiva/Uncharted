import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
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
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(100, 100, 140)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
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
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf, _max_cos(Yf)

def _lp(x, deadline, r0=1e-3, cap=1400):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
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
            if int(np.count_nonzero(sel)) == 0:
                sel[int(np.argmax(c))] = True
            if int(np.count_nonzero(sel)) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
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

def _hopf_code(n, K, alloc, thetas, twists, sh1, sh2, pattern, jitter, rng):
    X = np.zeros((n, 4))
    pos = 0
    for k in range(K):
        nk = int(alloc[k])
        th = float(thetas[k])
        tw = float(twists[k]); s1 = float(sh1[k]); s2 = float(sh2[k])
        ct, st = np.cos(th), np.sin(th)
        if nk <= 0:
            continue
        if pattern == 0:
            nx = max(1, int(round(np.sqrt(nk))))
            ny = max(1, int(np.ceil(nk / nx)))
        elif pattern == 1:
            nx = max(1, int(round(np.sqrt(nk * 1.15))))
            ny = max(1, int(np.ceil(nk / nx)))
        else:
            nx = max(1, int(round(np.sqrt(nk * 0.87))))
            ny = max(1, int(np.ceil(nk / nx)))
        for s in range(nk):
            ix = s % nx
            iy = (s // nx) % ny
            off = 0.0
            if pattern == 1 and (iy & 1):
                off = 0.5
            p1 = 2 * np.pi * (ix + off + s1 + tw * iy / max(1, ny)) / max(1, nx)
            p2 = 2 * np.pi * (iy + s2) / max(1, ny)
            if jitter > 0:
                p1 += rng.randn() * jitter
                p2 += rng.randn() * jitter
            X[pos, 0] = ct * np.cos(p1)
            X[pos, 1] = ct * np.sin(p1)
            X[pos, 2] = st * np.cos(p2)
            X[pos, 3] = st * np.sin(p2)
            pos += 1
            if pos >= n:
                break
        if pos >= n:
            break
    while pos < n:
        v = rng.randn(4); v /= np.linalg.norm(v)
        X[pos] = v; pos += 1
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _hurwitz_pool(rng, B=4, n_rand=1200):
    pts = []
    seen = set()
    rngs = range(-B, B + 1)
    for a in rngs:
        for b in rngs:
            for c_ in rngs:
                for d in rngs:
                    if a == 0 and b == 0 and c_ == 0 and d == 0:
                        continue
                    v = np.array([a, b, c_, d], dtype=float)
                    nv = float(v @ v)
                    if nv <= 0 or nv > 2 * B * B:
                        continue
                    v = v / np.sqrt(nv)
                    key = tuple(np.round(v, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    pts.append(v)
    # half-integer Hurwitz
    for a in rngs:
        for b in rngs:
            for c_ in rngs:
                for d in rngs:
                    v = np.array([a + 0.5, b + 0.5, c_ + 0.5, d + 0.5], dtype=float)
                    nv = float(v @ v)
                    if nv > 2 * B * B + 1:
                        continue
                    v = v / np.sqrt(nv)
                    key = tuple(np.round(v, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    pts.append(v)
    P = np.array(pts)
    # add random points to enrich
    R = rng.randn(n_rand, 4)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    if len(P) > 0:
        P = np.vstack([P, R])
    else:
        P = R
    # cap size for speed
    if len(P) > 4200:
        idx = rng.choice(len(P), 4200, replace=False)
        P = P[idx]
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P

def _greedy_select(P, n, rng, n_starts=3, n_swaps=3000):
    C = P @ P.T
    np.fill_diagonal(C, -2.0)
    mP = len(P)
    best_S = None
    best_m = 2.0
    for st in range(n_starts):
        s0 = int(rng.randint(mP))
        S = [s0]
        inS = np.zeros(mP, dtype=bool); inS[s0] = True
        bd = C[:, s0].copy()
        bd[s0] = 2.0
        for _ in range(1, n):
            # pick unselected with minimal bd
            cand = np.where(~inS)[0]
            j = cand[int(np.argmin(bd[cand]))]
            S.append(int(j)); inS[j] = True
            row = C[:, j]
            bd = np.maximum(bd, row)
            bd[inS] = 2.0
        S = np.array(S, dtype=int)
        # current max
        sub = C[np.ix_(S, S)]
        cur = float(sub.max())
        # swaps
        for _ in range(n_swaps):
            i_pos = int(rng.randint(n))
            j = int(rng.randint(mP))
            if inS[j]:
                continue
            i_old = int(S[i_pos])
            # max without i_old
            others = np.delete(S, i_pos)
            # max among others
            m1 = float(C[np.ix_(others, others)].max()) if len(others) > 1 else -2.0
            m2 = float(C[j, others].max()) if len(others) > 0 else -2.0
            nm = m1 if m1 > m2 else m2
            if nm < cur - 1e-12:
                S[i_pos] = j
                inS[i_old] = False; inS[j] = True
                cur = nm
                if cur < best_m - 1e-12 and st > 0:
                    pass
        if cur < best_m:
            best_m = cur
            best_S = S.copy()
    X = P[best_S]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, best_m

def solve():
    t0 = time.time()
    BUDGET = 172.0
    n = 163
    rng = np.random.RandomState(12345)
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    inits = []
    if bx is not None:
        try:
            bx = np.asarray(bx, dtype=float)
            if bx.shape == (n, 4):
                bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
                inits.append(bx)
        except Exception:
            pass
    if rec is not None:
        try:
            rec = np.asarray(rec, dtype=float)
            if rec.shape == (n, 4):
                rec = rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12)
                inits.append(rec)
        except Exception:
            pass
    # ---- Hopf structured search from scratch ----
    hopf_best = []
    try:
        htrials = 500
        scored = []
        for t in range(htrials):
            K = int(rng.randint(3, 9))
            w = rng.rand(K) + 0.3
            w = w / w.sum()
            alloc = np.floor(w * n).astype(int)
            while alloc.sum() < n:
                alloc[int(rng.randint(K))] += 1
            while alloc.sum() > n:
                k = int(rng.randint(K))
                if alloc[k] > 5:
                    alloc[k] -= 1
            th = np.sort(rng.uniform(0.22, 1.35, size=K))
            # push apart thetas a bit
            tw = rng.uniform(-0.5, 0.5, size=K)
            s1 = rng.rand(K); s2 = rng.rand(K)
            pat = int(rng.randint(3))
            jit = float(rng.choice([0.0, 0.01, 0.03]))
            X = _hopf_code(n, K, alloc, th, tw, s1, s2, pat, jit, rng)
            v = _max_cos(X)
            scored.append((v, X, (K, alloc.copy(), th.copy(), tw.copy(), s1.copy(), s2.copy(), pat, jit)))
            if time.time() - t0 > 14:
                break
        scored.sort(key=lambda z: z[0])
        for v, X, meta in scored[:4]:
            hopf_best.append(X.copy())
            inits.append(X.copy())
        # torus-position / twist local tweak on the very best hopf: hill climb thetas+twists
        if len(scored) > 0 and time.time() - t0 < 20:
            v0, X0, meta = scored[0]
            K, alloc, th, tw, s1, s2, pat, jit = meta
            cur_th, cur_tw = th.copy(), tw.copy()
            cur_s1, cur_s2 = s1.copy(), s2.copy()
            cur_v = v0
            cur_X = X0
            for it in range(120):
                if time.time() - t0 > 22:
                    break
                k = int(rng.randint(K))
                prop_th, prop_tw = cur_th.copy(), cur_tw.copy()
                prop_s1, prop_s2 = cur_s1.copy(), cur_s2.copy()
                prop_th[k] += rng.randn() * 0.03
                prop_tw[k] += rng.randn() * 0.08
                if rng.rand() < 0.5:
                    prop_s1[k] += rng.randn() * 0.03
                    prop_s2[k] += rng.randn() * 0.03
                prop_th[k] = float(np.clip(prop_th[k], 0.12, 1.45))
                Xp = _hopf_code(n, K, alloc, prop_th, prop_tw, prop_s1, prop_s2, pat, 0.0, rng)
                vp = _max_cos(Xp)
                if vp < cur_v:
                    cur_th, cur_tw, cur_s1, cur_s2 = prop_th, prop_tw, prop_s1, prop_s2
                    cur_v, cur_X = vp, Xp
            inits.append(cur_X.copy())
    except Exception:
        pass
    # ---- Hurwitz lattice pool + greedy maximin ----
    try:
        if time.time() - t0 < 40:
            P = _hurwitz_pool(rng, B=4, n_rand=1200)
            Xg, _ = _greedy_select(P, n, rng, n_starts=3, n_swaps=2500)
            inits.append(Xg)
            # second greedy with different seed pool jitter
            Pj = P + rng.randn(*P.shape) * 0.012
            Pj /= np.linalg.norm(Pj, axis=1, keepdims=True)
            Xg2, _ = _greedy_select(Pj, n, rng, n_starts=2, n_swaps=1500)
            inits.append(Xg2)
    except Exception:
        pass
    if len(inits) == 0:
        c = rng.randn(n, 4); c /= np.linalg.norm(c, axis=1, keepdims=True)
        inits.append(c)
    vals = [_max_cos(c) for c in inits]
    bi = int(np.argmin(vals))
    best_x = inits[bi].copy()
    best_val = float(vals[bi])
    # quick anneal screening of structured inits (short) to find promising basin
    try:
        scr_deadline = t0 + 55
        screened = [(best_val, best_x)]
        for X in inits:
            if time.time() > scr_deadline:
                break
            v0 = _max_cos(X)
            if v0 > best_val + 0.02 and X is not best_x:
                continue
            Xs, vs = _anneal(X, steps=2500, T0=0.02, T1=0.0008, lr0=0.035, mu=0.88, rng=rng, noise0=0.001, deadline=min(time.time() + 3.0, scr_deadline))
            screened.append((vs, Xs))
            if vs < best_val:
                best_val, best_x = vs, Xs.copy()
        screened.sort(key=lambda z: z[0])
        # keep top 3 distinct basins
        tops = [s[1] for s in screened[:3]]
    except Exception:
        tops = [best_x]
    # main basin hopping mixing structured tops + perturbations
    SEARCH_END = t0 + 132.0
    trial = 0
    sigmas = [0.02, 0.06, 0.13, 0.008, 0.05]
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 38:
            break
        # cycle base among tops and best
        if len(tops) > 0 and trial % 4 != 0:
            base = tops[trial % len(tops)]
        else:
            base = best_x
        sigma = sigmas[trial % len(sigmas)]
        cur = base.copy()
        mode = trial % 6
        if mode == 0:
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            rowm = G.max(axis=1)
            order = np.argsort(rowm + 0.0004 * deg)
            worst = order[-10:]
            noise = rng.randn(*cur.shape) * (sigma * 0.3)
            noise[worst] += rng.randn(len(worst), 4) * (sigma * 2.0)
            cur = cur + noise
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            # Clifford-torus re-projection shake: convert to Hopf, jitter torus angle
            W = cur.copy()
            # random orthogonal mix to explore different Hopf fibrations
            if rng.rand() < 0.6:
                A = rng.randn(4, 4)
                Q, _ = np.linalg.qr(A)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                W = W @ Q.T
            # extract Hopf coords, quantize theta toward K tori then jitter
            r1 = np.sqrt(W[:, 0] ** 2 + W[:, 1] ** 2)
            r2 = np.sqrt(W[:, 2] ** 2 + W[:, 3] ** 2)
            th = np.arctan2(np.maximum(r2, 1e-12), np.maximum(r1, 1e-12))
            K = int(rng.randint(3, 9))
            edges = np.linspace(th.min(), th.max(), K + 1)
            for k in range(K):
                msk = (th >= edges[k]) & (th <= edges[k + 1])
                if int(msk.sum()) == 0:
                    continue
                shift = rng.randn() * sigma * 2.0
                th[msk] += shift * 0.3
            th = np.clip(th, 0.05, 1.52)
            p1 = np.arctan2(W[:, 1], W[:, 0]) + rng.randn(n) * sigma * 1.5
            p2 = np.arctan2(W[:, 3], W[:, 2]) + rng.randn(n) * sigma * 1.5
            cur = np.stack([np.cos(th) * np.cos(p1), np.cos(th) * np.sin(p1),
                            np.sin(th) * np.cos(p2), np.sin(th) * np.sin(p2)], axis=1)
            if 'Q' in dir():
                try:
                    cur = cur @ Q
                except Exception:
                    pass
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 3:
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = 0.45 + 0.35 * rng.rand()
            dots = cur @ u
            mask = dots > thr
            cnt = int(mask.sum())
            if cnt == 0 or cnt > n // 2:
                idx = rng.choice(n, 14, replace=False)
                cur[idx] += rng.randn(14, 4) * sigma
            else:
                vdir = rng.randn(4); vdir -= float(vdir @ u) * u
                nv = float(np.linalg.norm(vdir))
                if nv < 1e-9:
                    vdir = rng.randn(4); nv = float(np.linalg.norm(vdir))
                vdir /= nv
                cur[mask] += vdir[None, :] * (sigma * 1.5) + rng.randn(cnt, 4) * (sigma * 0.6)
                cur += rng.randn(*cur.shape) * (sigma * 0.15)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 4:
            # pair-break top contacts
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            top = np.argsort(c)[-8:]
            for tt in top[:5]:
                i = int(iu0[tt]); j = int(iu1[tt])
                if rng.rand() < 0.7:
                    d = cur[i] - cur[j]
                    nd = float(np.linalg.norm(d))
                    if nd > 1e-9:
                        d /= nd
                        cur[i] += d * sigma * 0.8
                        cur[j] -= d * sigma * 0.8
            cur += rng.randn(*cur.shape) * (sigma * 0.3)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        else:
            cur = rng.randn(n, 4)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        dl = min(time.time() + 4.5, SEARCH_END)
        T0a = float(rng.choice([0.02, 0.03, 0.012]))
        cur_b, v = _anneal(cur, steps=6000, T0=T0a, T1=0.0007, lr0=0.038, mu=0.88, rng=rng, noise0=0.0015, deadline=dl)
        if v < best_val + 0.004 and time.time() < t0 + BUDGET - 36:
            cur_b2, v2 = _lbfgs(cur_b, deadline=time.time() + 2.0)
            if v2 < v:
                cur_b, v = cur_b2, v2
            if v < best_val:
                pol, vp = _lp(cur_b, deadline=time.time() + 2.5)
                if vp < best_val:
                    best_x, best_val = pol, vp
                    tops.append(pol.copy())
                    if len(tops) > 5:
                        tops = tops[-5:]
                elif v < best_val:
                    best_x, best_val = cur_b, v
                    tops.append(cur_b.copy())
                    if len(tops) > 5:
                        tops = tops[-5:]
            elif v < best_val + 0.0005 and rng.rand() < 0.15:
                tops.append(cur_b.copy())
                if len(tops) > 5:
                    tops = tops[-5:]
        trial += 1
    # final intensive polish of champion
    try:
        if time.time() < t0 + BUDGET - 4:
            Xf, vf = _anneal(best_x, steps=9000, T0=0.008, T1=0.00025, lr0=0.03, mu=0.9, rng=rng, noise0=0.0, deadline=t0 + BUDGET - 22)
            if vf < best_val:
                best_x, best_val = Xf, vf
            Xf2, vf2 = _lbfgs(best_x, deadline=t0 + BUDGET - 14)
            if vf2 < best_val:
                best_x, best_val = Xf2, vf2
            Xp, vp = _lp(best_x, deadline=t0 + BUDGET - 2, r0=1e-3)
            if vp < best_val:
                best_x, best_val = Xp, vp
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
