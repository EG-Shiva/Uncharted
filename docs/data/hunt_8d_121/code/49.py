import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normed(A):
    A = np.asarray(A, dtype=float)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    T_DISC_END = t0 + 40.0
    T_CONT_END = t0 + 130.0
    rng = np.random.RandomState(0)
    n, d = 121, 8

    # incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands = [_normed(rng.randn(n, d))]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(np.min(vals0))

    def lbfgs_lse(Xinit, beta, maxiter):
        nn = Xinit.shape[0]
        xf0 = Xinit.ravel().copy()
        b = float(beta)
        def fg(xf):
            X = xf.reshape(nn, d)
            ns = np.sqrt(np.sum(X * X, axis=1))
            ns = np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            m = float(np.max(G - 2.0 * np.eye(nn)))
            A = b * (G - m)
            np.clip(A, -700, 0, out=A)
            E = np.exp(A)
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(np.triu(E, 1)))
            if not np.isfinite(S) or S <= 0:
                S = 1e-300
            F = m + np.log(S) / b
            EY = E.dot(Y)
            GradY = EY / S
            dot = np.sum(GradY * Y, axis=1)
            GradX = (GradY - dot[:, None] * Y) / ns[:, None]
            return float(F), np.nan_to_num(GradX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            res = minimize(fg, xf0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8), "maxls": 20})
            Xo = np.asarray(res.x, dtype=float).reshape(nn, d)
        except Exception:
            Xo = Xinit.copy()
        Xo = _normed(np.nan_to_num(Xo, nan=0.0, posinf=0.0, neginf=0.0) + 1e-16)
        return Xo

    def lp_polish(x_init, seconds):
        x = _normed(x_init.copy())
        try:
            cur = _max_cos(x)
        except Exception:
            return x
        Iall, Jall = np.triu_indices(n, 1)
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.5 and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = float(c_all.max())
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2500:
                idx = np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if getattr(res, "status", 1) != 0:
                radius *= 0.5
                continue
            try:
                y = _normed(x + res.x[:-1].reshape(n, d))
                v = _max_cos(y)
            except Exception:
                radius *= 0.5
                continue
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    # ---------- Stage 1: algebraic pools + discrete selection ----------
    disc_X = []
    try:
        # E8 roots (240)
        E = []
        # D8 part
        for i in range(d):
            for j in range(i + 1, d):
                for si in (1.0, -1.0):
                    for sj in (1.0, -1.0):
                        v = np.zeros(d); v[i] = si; v[j] = sj
                        E.append(v)
        # spinor part: 128 even-parity (+-0.5)^8
        for mask in range(256):
            bits = [(mask >> k) & 1 for k in range(d)]
            if sum(bits) % 2 == 1:
                continue
            v = np.array([0.5 if b == 0 else -0.5 for b in bits])
            E.append(v)
        E = np.array(E, dtype=float) / np.sqrt(2.0)  # (240,8)
        # hypercube 256
        H = np.array([[1.0 if (mask >> k) & 1 else -1.0 for k in range(d)] for mask in range(256)], dtype=float) / np.sqrt(8.0)
        # cross 16
        C = np.zeros((16, d))
        for i in range(d):
            C[2*i, i] = 1.0; C[2*i+1, i] = -1.0
        # random pool
        R1 = _normed(rng.randn(2500, d))
        # one random rotation of E+H union
        Qm, _ = np.linalg.qr(rng.randn(d, d))
        if np.linalg.det(Qm) < 0:
            Qm[:, 0] *= -1
        Er = E @ Qm.T
        Qm2, _ = np.linalg.qr(rng.randn(d, d))
        if np.linalg.det(Qm2) < 0:
            Qm2[:, 0] *= -1
        Hr = H @ Qm2.T
        R2 = _normed(rng.randn(500, d))
        P = np.vstack([E, H, C, R1, Er, Hr, R2])
        # dedup lightly not needed
        Np = P.shape[0]
        Gpool = P @ P.T
        # greedy from several seeds
        nseed = 10
        seeds = [int(rng.randint(Np)) for _ in range(nseed)]
        # also force seeds spread: include 0 and Np//2
        seeds[0] = 0; seeds[1] = Np // 3
        greedy_sets = []
        for s in seeds:
            if time.time() > T_DISC_END - 5:
                break
            cur = Gpool[:, s].copy()
            cur[s] = np.inf
            in_set = np.zeros(Np, dtype=bool); in_set[s] = True
            sel = [s]
            for _ in range(n - 1):
                # argmin among not in set
                # use masked array trick
                cur_masked = np.where(in_set, np.inf, cur)
                q = int(np.argmin(cur_masked))
                sel.append(q)
                in_set[q] = True
                cur[q] = np.inf
                # update: cur[c] = max(cur[c], Gpool[c,q])
                gcol = Gpool[:, q]
                # only for not-in-set
                np.maximum(cur, gcol, out=cur)
                cur[in_set] = np.inf
            sel = np.array(sel, dtype=int)
            # current max
            sub = Gpool[np.ix_(sel, sel)]
            np.fill_diagonal(sub, -2.0)
            M = float(sub.max())
            greedy_sets.append((M, sel))
        # SA swaps on each greedy set
        best_disc = []
        for (M0, sel0) in greedy_sets:
            if time.time() > T_DISC_END:
                break
            sel = sel0.copy()
            in_set = np.zeros(Np, dtype=bool); in_set[sel] = True
            sub = Gpool[np.ix_(sel, sel)]
            np.fill_diagonal(sub, -2.0)
            M = float(sub.max())
            # locate max pair positions in sel-index space
            am, bm = divmod(int(np.argmax(sub)), n)
            T0, T1 = 0.004, 0.00005
            iters = 6000
            # adjust iters to fit budget
            for it in range(iters):
                if it % 1000 == 0 and time.time() > T_DISC_END:
                    break
                frac = it / float(iters)
                T = T0 * (1 - frac) + T1 * frac
                p = int(rng.randint(n))
                q = int(rng.randint(Np))
                if in_set[q]:
                    continue
                # max from q to sel\{p}
                row = Gpool[q, sel]
                # max excluding p
                if p == 0:
                    max2 = float(np.max(row[1:]))
                elif p == n - 1:
                    max2 = float(np.max(row[:p]))
                else:
                    a1 = float(np.max(row[:p])); a2 = float(np.max(row[p+1:])); max2 = a1 if a1 > a2 else a2
                if max2 >= M:
                    newM = max2
                else:
                    if p != am and p != bm:
                        newM = M
                    else:
                        # recompute M without p
                        keep = [k for k in range(n) if k != p]
                        s2 = sel[keep]
                        ssub = Gpool[np.ix_(s2, s2)]
                        np.fill_diagonal(ssub, -2.0)
                        Mrem = float(ssub.max())
                        newM = Mrem if Mrem > max2 else max2
                if newM < M or rng.rand() < np.exp((M - newM) / max(T, 1e-12)):
                    # accept
                    old = int(sel[p])
                    in_set[old] = False; in_set[q] = True
                    sel[p] = q
                    if newM >= M - 1e-12 and max2 < M:
                        # need full recompute (accepted sideways); recompute M,a,b
                        ssub = Gpool[np.ix_(sel, sel)]
                        np.fill_diagonal(ssub, -2.0)
                        M = float(ssub.max())
                        am, bm = divmod(int(np.argmax(ssub)), n)
                    else:
                        # newM is correct max (either max2 or M)
                        M = float(newM)
                        if max2 >= M - 1e-12:
                            # max now involves q at p; find partner
                            prow = Gpool[q, sel]
                            # argmax excluding p==self (G=1)
                            prow2 = prow.copy(); prow2[p] = -2
                            bm2 = int(np.argmax(prow2))
                            am, bm = (p, bm2) if p < bm2 else (bm2, p)
                            # M already max2
                        # else M unchanged, am,bm unchanged (p not in max pair)
                        pass
            best_disc.append((M, sel.copy()))
        best_disc.sort(key=lambda z: z[0])
        for _, sel in best_disc[:3]:
            disc_X.append(_normed(P[sel].copy()))
    except Exception:
        pass

    # ---------- Stage 2: continuous release ----------
    # budget: until T_CONT_END
    scales = [1e-3, 1e-2, 3e-2]
    betas = [60.0, 150.0, 350.0]
    cand_starts = []
    for Xd in disc_X:
        cand_starts.append(Xd)
    # add incumbent-based safety start (small perturb) - cheap, guarantees competitiveness
    try:
        cand_starts.append(Xbest.copy())
    except Exception:
        pass
    # limit to 4 starts
    cand_starts = cand_starts[:4]
    for si, Xs0 in enumerate(cand_starts):
        if time.time() > T_CONT_END:
            break
        is_incumbent = (si == len(cand_starts) - 1)
        try:
            if is_incumbent:
                # light refine: single LSE + keep
                Xc = lbfgs_lse(Xs0, 200.0, 300)
                v = _max_cos(Xc)
                if v < Vbest and np.all(np.isfinite(Xc)):
                    Vbest = float(v); Xbest = _normed(Xc.copy())
                continue
            Xc = Xs0.copy()
            # staged symmetry breaking
            for sc in scales:
                if time.time() > T_CONT_END - 3:
                    break
                # deterministic perturbation
                rs = np.random.RandomState(1000 + si * 100 + int(sc * 1e5))
                Xp = _normed(Xc + rs.randn(n, d) * float(sc))
                # small LSE
                Xp = lbfgs_lse(Xp, 60.0, 120)
                # keep progression if reasonable (not too bad)
                v = _max_cos(Xp)
                # allow moving even if slightly worse to break symmetry; keep best track
                if v < Vbest and np.all(np.isfinite(Xp)):
                    Vbest = float(v); Xbest = _normed(Xp.copy())
                Xc = Xp
                if time.time() > T_CONT_END - 3:
                    break
            # annealing with increasing beta
            for b in betas:
                if time.time() > T_CONT_END - 2:
                    break
                Xc = lbfgs_lse(Xc, b, 250)
                v = _max_cos(Xc)
                if v < Vbest and np.all(np.isfinite(Xc)):
                    Vbest = float(v); Xbest = _normed(Xc.copy())
        except Exception:
            continue
        if time.time() > T_CONT_END:
            break
        # immediate return guard
        if time.time() > DEADLINE - 5:
            break

    # ---------- Stage 3: LP polish remainder ----------
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 3:
            # polish best; use half budget, then try one kicked variant if time left
            Xp = lp_polish(Xbest, min(remain * 0.7, 30.0))
            v = _max_cos(Xp)
            if np.all(np.isfinite(Xp)) and v < Vbest:
                Vbest = float(v); Xbest = _normed(Xp.copy())
        remain = DEADLINE - time.time() - 1.0
        if remain > 5:
            # one diversified kick of best + quick LSE + polish
            try:
                rk = np.random.RandomState(7)
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G)); a = k // n; b_ = k % n
                Y = Xbest.copy()
                Y[a] += rk.randn(d) * 0.02
                Y[b_] += rk.randn(d) * 0.02
                Y = _normed(Y)
                Y = lbfgs_lse(Y, 200.0, 200)
                if time.time() < DEADLINE - 4:
                    Y = lp_polish(Y, max(2.0, remain - 4))
                v = _max_cos(Y)
                if np.all(np.isfinite(Y)) and v < Vbest:
                    Vbest = float(v); Xbest = _normed(Y.copy())
            except Exception:
                pass
    except Exception:
        pass

    Xbest = _normed(Xbest)
    # final safety: no NaN, correct shape
    if Xbest.shape != (121, 8) or not np.all(np.isfinite(Xbest)):
        try:
            Xbest = _normed(np.load("best.npy").astype(float))
        except Exception:
            Xbest = _normed(rng.randn(n, d))
    return Xbest
