import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lex_metrics(X, iu0=None, tau=2e-4):
    n = X.shape[0]
    if iu0 is None:
        iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0[0], iu0[1]]
    m = float(np.max(c))
    # mean top100, lse top500
    if len(c) >= 500:
        part = np.argpartition(-c, 500 - 1)[:500]
        top500 = c[part]
        p2 = np.argpartition(-top500, 100 - 1)[:100]
        mean100 = float(np.mean(top500[p2]))
        e = np.exp((top500 - m) / tau)
        lse = float(m + tau * np.log(float(np.mean(e)) + 1e-300))
    else:
        mean100 = float(np.mean(c))
        lse = m
    return m, mean100, lse, c

def _lex_less(a, b, eps=1e-12):
    if a[0] < b[0] - eps:
        return True
    if abs(a[0] - b[0]) <= eps:
        if a[1] < b[1] - eps:
            return True
        if abs(a[1] - b[1]) <= eps:
            if a[2] < b[2] - 1e-11:
                return True
    return False

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    HARD = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 82, 14
    dm = d - 1
    Npar = n * dm  # 1066

    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = _norm(rng.standard_normal((n, d)))
    else:
        iu_tmp = np.triu_indices(n, 1)
        def mc(a):
            G = a @ a.T
            np.fill_diagonal(G, -2.0)
            return float(np.max(G))
        cands.sort(key=mc)
        X0 = cands[0].copy()
    X0 = _norm(X0)
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    m0, mean0, lse0, _ = _lex_metrics(X0, iu0)
    best_tup = (m0, mean0, lse0)
    Xbest = X0.copy()
    best_max = m0

    def keep_if_better(Y):
        nonlocal Xbest, best_tup, best_max
        Y = _norm(Y)
        m, mn, ls, _ = _lex_metrics(Y, iu0)
        tup = (m, mn, ls)
        if m < best_max - 1e-15:
            best_max = m
            Xbest = Y.copy()
            best_tup = tup
            return True
        # also track lex without worsening max (for return we only care max, but keep lex best if equal)
        if abs(m - best_max) <= 1e-12 and _lex_less(tup, best_tup):
            Xbest = Y.copy()
            best_tup = tup
        return False

    # ---- 1. tangent bases ----
    # orthonormal basis of orthogonal complement of each x_i
    T = np.empty((n, d, dm))
    try:
        for i in range(n):
            x = X0[i]
            A = rng.standard_normal((d, dm))
            proj = x @ A  # (dm,)
            A = A - np.outer(x, proj)
            Q, _ = np.linalg.qr(A)
            # ensure tangent: Q orthogonal to x (sign fix not needed)
            T[i] = Q
    except Exception:
        # fallback: return best
        return _norm(Xbest)

    if time.time() > DEADLINE - 60:
        return _norm(Xbest)

    # ---- 2. rigidity Jacobian of top pairs ----
    G0 = X0 @ X0.T
    c0 = G0[I0, J0]
    PJ = 800
    if PJ > len(c0):
        PJ = len(c0)
    selJ = np.argpartition(-c0, PJ - 1)[:PJ]
    # sort tightest first for determinism
    o = np.argsort(-c0[selJ])
    selJ = selJ[o]
    IJ = I0[selJ]; JJ = J0[selJ]
    Jmat = np.zeros((PJ, Npar))
    try:
        for k in range(PJ):
            i = int(IJ[k]); j = int(JJ[k])
            Jmat[k, i * dm:(i + 1) * dm] = T[i].T @ X0[j]
            Jmat[k, j * dm:(j + 1) * dm] = T[j].T @ X0[i]
    except Exception:
        return _norm(Xbest)

    # ---- 3. SVD, floppy directions ----
    Kflop = 30
    try:
        # Jmat is 800x1066
        _, S, Vt = np.linalg.svd(Jmat, full_matrices=False)
        # Vt shape (800,1066); lowest are last rows
        Vflop = Vt[-Kflop:, :].copy()  # (Kflop,Npar)
    except Exception:
        # fallback random subspace
        Vflop = rng.standard_normal((Kflop, Npar))
        # orthonormalize rows
        Vflop, _ = np.linalg.qr(Vflop.T)
        Vflop = Vflop.T

    if time.time() > DEADLINE - 55:
        return _norm(Xbest)

    # ---- 4. search along random combos of floppy directions ----
    Nprobe = 900
    log_lo = np.log(1e-4); log_hi = np.log(3e-2)
    # store probes metrics
    probe_list = []  # (m,mean,lse,Y or z?)
    # we cannot store all Y (900*82*14 ~ 1M floats, fine) but store only top few
    # first evaluate on the fly, keep top B
    Bkeep = 6
    top_cands = []  # list of (tup, Y)
    # precompute for speed: use local vars
    for t in range(Nprobe):
        if time.time() > DEADLINE - 45:
            break
        alpha = rng.standard_normal(Kflop)
        na = float(np.linalg.norm(alpha))
        if na == 0 or not np.isfinite(na):
            continue
        alpha /= na
        ls = float(rng.uniform(log_lo, log_hi))
        s = float(np.exp(ls))
        z = alpha @ Vflop  # (Npar,)
        # z has norm ~1 (since Vflop rows orthonormal); scale by s
        z = z * s
        Zblk = z.reshape(n, dm)
        # D = einsum
        D = np.einsum('ndk,nk->nd', T, Zblk)
        Y = _norm(X0 + D)
        if not np.all(np.isfinite(Y)):
            continue
        # fast metrics
        G = Y @ Y.T
        c = G[I0, J0]
        m = float(np.max(c))
        # quick prune: if far worse than best, skip detailed tail (still need mean? skip)
        # allow tolerance 8e-4 above best for exploration
        if m > best_max + 8e-4:
            continue
        # detailed tail
        part = np.argpartition(-c, 500 - 1)[:500]
        top500 = c[part]
        p2 = np.argpartition(-top500, 100 - 1)[:100]
        mean100 = float(np.mean(top500[p2]))
        e = np.exp((top500 - m) / 2e-4)
        lse = float(m + 2e-4 * np.log(float(np.mean(e)) + 1e-300))
        tup = (m, mean100, lse)
        # insert into top
        top_cands.append((tup, Y.copy()))
        # keep sorted small: prune periodically
        if len(top_cands) > 60:
            top_cands.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
            top_cands = top_cands[:40]

    if len(top_cands) == 0:
        return _norm(Xbest)
    top_cands.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
    # deduplicate roughly by keeping diverse steps? just take first Bkeep
    Qcands = [yc for _, yc in top_cands[:Bkeep]]

    # ---- 5. quench only best few with short annealed LogSumExp Adam ----
    def adam_quench(Xinit, steps, lr, tau0, tau1):
        X = _norm(Xinit.copy())
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        # track best snapshot lex
        mb, mnb, lsb, _ = _lex_metrics(X, iu0)
        best_loc = X.copy()
        best_t = (mb, mnb, lsb)
        for st in range(1, steps + 1):
            if time.time() > DEADLINE - 18:
                break
            frac = st / max(1, steps)
            tau = tau0 * ((tau1 / tau0) ** frac)
            G = X @ X.T
            # max for stable exp
            # use triu values
            c = G[I0, J0]
            m = float(np.max(c))
            # weights softmax
            # clip exponent to avoid underflow overhead: only top matters but full is cheap (3321)
            ex = np.exp((c - m) / tau)
            ssum = float(np.sum(ex))
            if not np.isfinite(ssum) or ssum <= 0:
                break
            w = ex / ssum
            # build W matrix
            W = np.zeros((n, n))
            W[I0, J0] = w
            W[J0, I0] = w
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            # Adam
            m_ad = b1 * m_ad + (1 - b1) * Gt
            v_ad = b2 * v_ad + (1 - b2) * (Gt * Gt)
            mhat = m_ad / (1 - b1 ** st)
            vhat = v_ad / (1 - b2 ** st)
            delta = lr * mhat / (np.sqrt(vhat) + eps)
            # limit step norm to avoid blowup
            X = _norm(X - delta)
            if st % 10 == 0 or st == steps:
                mm, mn, ls, _ = _lex_metrics(X, iu0)
                tup = (mm, mn, ls)
                if _lex_less(tup, best_t):
                    best_t = tup
                    best_loc = X.copy()
                # also global best by max
                if mm < best_max - 1e-15:
                    # update global (via outer var)
                    pass
        return best_loc, best_t

    quenched = []
    # quench at most 3 candidates to stay cheap
    Nq = min(3, len(Qcands))
    # order already best-first
    for qi in range(Nq):
        if time.time() > DEADLINE - 18:
            break
        Y0 = Qcands[qi]
        # short anneal: 220 steps
        Yq, tq = adam_quench(Y0, steps=220, lr=1.5e-3, tau0=2e-3, tau1=2e-4)
        quenched.append((tq, Yq))
        # retention: if improved max, record
        if tq[0] < best_max - 1e-15:
            best_max = tq[0]
            Xbest = _norm(Yq.copy())
            best_tup = tq

    # also consider quenched lex ordering for LP start
    if len(quenched) > 0:
        quenched.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))
        Xcur = _norm(quenched[0][1].copy())
    else:
        Xcur = _norm(Qcands[0].copy())
        # check if it beats best
        m, mn, ls, _ = _lex_metrics(Xcur, iu0)
        if m < best_max - 1e-15:
            best_max = m
            Xbest = Xcur.copy()
            best_tup = (m, mn, ls)

    # ---- 6. one capped LP polish ----
    if time.time() < DEADLINE - 12:
        try:
            Xp = _norm(Xcur.copy())
            Gp = Xp @ Xp.T
            cp = Gp[I0, J0]
            mp = float(np.max(cp))
            # active set: top 1100
            Kact = 1100
            if Kact > len(cp):
                Kact = len(cp)
            act = np.argpartition(-cp, Kact - 1)[:Kact]
            Ia = I0[act]; Ja = J0[act]; ca = cp[act]
            p = len(act)
            nd = n * d
            nv = nd + 1
            radius = 5e-4
            rows = np.repeat(np.arange(p), 2 * d + 1)
            C = np.empty((p, 2 * d + 1), dtype=int)
            cd = np.arange(d)
            C[:, :d] = Ia[:, None] * d + cd
            C[:, d:2 * d] = Ja[:, None] * d + cd
            C[:, 2 * d] = nd
            cols = C.ravel()
            M = np.empty((p, 2 * d + 1))
            M[:, :d] = Xp[Ja]
            M[:, d:2 * d] = Xp[Ia]
            M[:, 2 * d] = -1.0
            vals = M.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((Xp.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            tl = max(0.5, min(6.0, DEADLINE - 4 - time.time()))
            if tl >= 0.5:
                res = linprog(obj, A_ub=A_ub, b_ub=-ca, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * nd + [(None, None)],
                              method="highs", options={"time_limit": float(tl)})
                if res is not None and getattr(res, "status", 1) == 0 and res.x is not None:
                    dx = np.asarray(res.x[:-1]).reshape(n, d)
                    if np.all(np.isfinite(dx)):
                        Y = _norm(Xp + dx)
                        mY, mnY, lsY, _ = _lex_metrics(Y, iu0)
                        # accept if lex better than current
                        cur_m, cur_mn, cur_ls, _ = _lex_metrics(Xp, iu0)
                        if _lex_less((mY, mnY, lsY), (cur_m, cur_mn, cur_ls)):
                            Xcur = Y
                            if mY < best_max - 1e-15:
                                best_max = mY
                                Xbest = Y.copy()
                                best_tup = (mY, mnY, lsY)
        except Exception:
            pass

    # final safety: ensure Xbest is at least as good as start
    Xbest = _norm(Xbest)
    # verify finite
    if not np.all(np.isfinite(Xbest)):
        Xbest = _norm(X0)
    return Xbest
