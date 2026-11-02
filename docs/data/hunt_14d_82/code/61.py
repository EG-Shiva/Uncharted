import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _full_metrics(X, iu0):
    G = X @ X.T
    c = G[iu0[0], iu0[1]]
    m = float(c.max())
    part = np.argpartition(-c, 499)[:500]
    t500 = c[part]
    p2 = np.argpartition(-t500, 99)[:100]
    mean100 = float(t500[p2].mean())
    beta = 2000.0
    e = np.exp(np.clip(beta * (t500 - m), -30.0, 0.0))
    lse = m + float(np.log(np.mean(e) + 1e-300)) / beta
    return m, mean100, lse, c

def _lex_less(a, b, eps=1e-12):
    if a[0] < b[0] - eps:
        return True
    if abs(a[0] - b[0]) <= eps:
        if a[1] < b[1] - eps:
            return True
        if abs(a[1] - b[1]) <= eps:
            if a[2] < b[2] - eps:
                return True
    return False

def _adam_quench(X0, iters, t_end, seed):
    # short annealed LogSumExp projected Adam on sphere
    n, d = X0.shape
    X = X0.copy()
    iu = np.triu_indices(n, 1)
    I0, J0 = iu
    M = len(I0)
    rng = np.random.default_rng(seed)
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # tau schedule
    tau0, tau1 = 5e-3, 2e-4
    taus = np.exp(np.linspace(np.log(tau0), np.log(tau1), iters))
    lr0 = 0.008
    best = X.copy()
    G0 = X @ X.T
    c0 = G0[I0, J0]
    bm = float(c0.max())
    for it in range(iters):
        if time.time() >= t_end:
            break
        tau = float(taus[it])
        lr = lr0 * (0.3 + 0.7 * (1.0 - it / max(1, iters)))
        G = X @ X.T
        C = G[I0, J0] / tau
        cmax = float(C.max())
        E = np.exp(np.clip(C - cmax, -40.0, 0.0))
        S = float(E.sum()) + 1e-300
        W = E / S
        # grad_k = sum_j w_kj x_j
        Grad = np.zeros_like(X)
        # accumulate via bincount-like loop in vectorized form
        # use np.add.at
        np.add.at(Grad, I0, (W[:, None] * X[J0]))
        np.add.at(Grad, J0, (W[:, None] * X[I0]))
        # project to tangent
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # adam
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** (it + 1)
        c2 = 1 - b2 ** (it + 1)
        step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        # cap step
        sn = float(np.linalg.norm(step)) + 1e-300
        if sn > 0.02:
            step *= 0.02 / sn
        Y = _norm(X + step)
        # evaluate max cheap
        Gy = Y @ Y.T
        cy = Gy[I0, J0]
        my = float(cy.max())
        X = Y
        if my < bm:
            bm = my
            best = Y.copy()
    return best, bm

def solve():
    t0 = time.time()
    DEAD = t0 + 130.0
    HARD = t0 + 138.0
    n, d = 82, 14
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0

    X0 = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                v = float(G.max())
                if v < bv:
                    bv = v
                    X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        rng = np.random.default_rng(0)
        X0 = _norm(rng.standard_normal((n, d)))
    Xcur = X0.copy()
    m_cur, mean_cur, lse_cur, _ = _full_metrics(Xcur, iu0)
    cur_trip = (m_cur, mean_cur, lse_cur)
    Xbest = Xcur.copy()
    best_trip = cur_trip

    # ---- rigidity Jacobian of top-K near contacts ----
    try:
        K = 800
        G = Xcur @ Xcur.T
        call = G[I0, J0]
        topK = np.argpartition(-call, K - 1)[:K]
        Ik = I0[topK]; Jk = J0[topK]
        # tangent bases T_i (14x13)
        rngT = np.random.default_rng(7)
        Tall = np.zeros((n, d, d - 1))
        for i in range(n):
            xi = Xcur[i]
            R = rngT.standard_normal((d, d - 1))
            coef = xi @ R  # (13,)
            R = R - xi[:, None] * coef[None, :]
            Q, _ = np.linalg.qr(R)
            # ensure orthogonality to xi (qr of projected should be); re-orthogonalize once
            Q = Q - xi[:, None] * (xi @ Q)[None, :]
            # renormalize columns
            qn = np.linalg.norm(Q, axis=0, keepdims=True) + 1e-300
            Q = Q / qn
            Tall[i] = Q
        Dt = n * (d - 1)  # 1066
        J = np.zeros((K, Dt))
        # T_i^T x_j
        for k in range(K):
            i = int(Ik[k]); j = int(Jk[k])
            J[k, i * 13:(i + 1) * 13] = Tall[i].T @ Xcur[j]
            J[k, j * 13:(j + 1) * 13] = Tall[j].T @ Xcur[i]
        # bottom eigenvectors of J^T J
        C = J.T @ J
        # symmetrize
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)  # ascending
        mflop = 40
        Vb = V[:, :mflop]  # 1066 x 40 floppy
    except Exception:
        Vb = None
        Tall = None

    cands = []  # list of (trip, X)
    if Vb is not None and time.time() < DEAD - 10.0:
        try:
            rngS = np.random.default_rng(12345)
            ncomb = 110
            Coef = rngS.standard_normal((ncomb, mflop))
            Coef = Coef / (np.linalg.norm(Coef, axis=1, keepdims=True) + 1e-300)
            steps = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
            # precompute Delta fields: (ncomb, n, d)
            # Vb rows grouped per point
            for li in range(ncomb):
                if time.time() > DEAD - 10.0:
                    break
                v = Vb @ Coef[li]  # 1066
                Delta = np.empty((n, d))
                for i in range(n):
                    Delta[i] = Tall[i] @ v[i * 13:(i + 1) * 13]
                dn = float(np.linalg.norm(Delta)) + 1e-300
                Delta = Delta / dn
                for s in steps:
                    Y = _norm(Xcur + float(s) * Delta)
                    mY, meanY, lseY, _ = _full_metrics(Y, iu0)
                    # keep if max no worse (tiny tol) and tail improved
                    if mY <= m_cur + 5e-9:
                        trip = (mY, meanY, lseY)
                        # require some tail improvement to be interesting
                        if meanY < mean_cur - 1e-12 or mY < m_cur - 1e-12 or lseY < lse_cur - 1e-12:
                            cands.append((trip, Y.copy()))
                    # also track global best if strictly better max
                    if mY < best_trip[0] - 1e-13:
                        Xbest = Y.copy()
                        mm, me, ls, _ = mY, meanY, lseY
                        best_trip = (mm, me, ls)
        except Exception:
            pass

    # rank candidates by lexicographic trip
    cands.sort(key=lambda t: (t[0][0], t[0][1], t[0][2]))
    # deduplicate / keep top few diverse
    Kq = min(3, len(cands))
    qlist = []
    for k in range(Kq):
        qlist.append(cands[k][1])
    # if no candidates, create small random floppy kicks as fallback
    if len(qlist) == 0 and Vb is not None:
        try:
            rngF = np.random.default_rng(999)
            for k in range(2):
                v = Vb @ rngF.standard_normal(mflop)
                v = v / (np.linalg.norm(v) + 1e-300)
                Delta = np.empty((n, d))
                for i in range(n):
                    Delta[i] = Tall[i] @ v[i * 13:(i + 1) * 13]
                Delta = Delta / (np.linalg.norm(Delta) + 1e-300)
                Y = _norm(Xcur + 3e-3 * Delta)
                qlist.append(Y)
        except Exception:
            pass

    # quench only best few
    per_quench = 450
    for qi, Q0 in enumerate(qlist):
        if time.time() > DEAD - 12.0:
            break
        try:
            Qb, _ = _adam_quench(Q0, per_quench, min(DEAD - 8.0, HARD - 10.0), seed=1000 + qi)
            mQ, meanQ, lseQ, _ = _full_metrics(Qb, iu0)
            tripQ = (mQ, meanQ, lseQ)
            if _lex_less(tripQ, best_trip):
                # check max improvement or tail improvement with max tie
                Xbest = Qb.copy()
                best_trip = tripQ
                if mQ < m_cur - 1e-12:
                    Xcur = Qb.copy()
                    cur_trip = tripQ
                    m_cur, mean_cur, lse_cur = tripQ
        except Exception:
            continue

    # one capped LP polish on current best
    try:
        if time.time() < DEAD - 8.0:
            Xp = Xbest.copy()
            Gp = Xp @ Xp.T
            cp = Gp[I0, J0]
            mp = float(cp.max())
            # wide but capped set
            CAP = 1200
            if len(cp) > CAP:
                sel_idx = np.argpartition(-cp, CAP - 1)[:CAP]
                I = I0[sel_idx]; J = J0[sel_idx]; c = cp[sel_idx]
            else:
                I, J, c = I0, J0, cp
            p_ = len(I)
            nv = n * d + 1
            radius = 3e-4
            rows = np.repeat(np.arange(p_), 2 * d + 1)
            cd = np.arange(d)
            Cmat = np.empty((p_, 2 * d + 1), dtype=int)
            Cmat[:, :d] = I[:, None] * d + cd
            Cmat[:, d:2 * d] = J[:, None] * d + cd
            Cmat[:, 2 * d] = n * d
            cols = Cmat.ravel()
            Mmat = np.empty((p_, 2 * d + 1))
            Mmat[:, :d] = Xp[J]
            Mmat[:, d:2 * d] = Xp[I]
            Mmat[:, 2 * d] = -1.0
            vals = Mmat.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
            A_eq = coo_matrix((Xp.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 4.0})
            if res.status == 0:
                Y = _norm(Xp + res.x[:-1].reshape(n, d))
                mY, meanY, lseY, _ = _full_metrics(Y, iu0)
                if (mY, meanY, lseY)[0] < best_trip[0] - 1e-13 or _lex_less((mY, meanY, lseY), best_trip):
                    # accept only if not worse in max beyond tol
                    if mY <= best_trip[0] + 1e-12:
                        Xbest = Y.copy()
                        best_trip = (mY, meanY, lseY)
    except Exception:
        pass

    Xbest = _norm(np.asarray(Xbest, dtype=float))
    if not np.all(np.isfinite(Xbest)):
        Xbest = _norm(X0.copy())
    return Xbest
