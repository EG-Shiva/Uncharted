import time
import numpy as np
from scipy.optimize import minimize, linprog

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    TLIM = 172.0
    t_end = t0 + TLIM
    n, d = 68, 12
    NS, K = 4, 17
    rng = np.random.default_rng(41717)
    # load incumbent
    x_inc = None
    v_inc = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                s = _max_cos(an)
                if s < v_inc:
                    v_inc = s
                    x_inc = an.copy()
        except Exception:
            pass
    if x_inc is None:
        x_inc = rng.standard_normal((n, d))
        x_inc /= np.linalg.norm(x_inc, axis=1, keepdims=True)
        v_inc = _max_cos(x_inc)

    TWO_PI_OVER_17 = 2.0 * np.pi / 17.0

    def build_CS(mults):
        mults = np.asarray(mults, dtype=int) % 17
        C = np.empty((K, 6))
        S = np.empty((K, 6))
        for k in range(K):
            for j in range(6):
                ang = TWO_PI_OVER_17 * int(mults[j]) * k
                C[k, j] = np.cos(ang)
                S[k, j] = np.sin(ang)
        return C, S

    def expand_X(A, C, S):
        # A (4,12) -> X (68,12)
        X = np.empty((NS * K, d))
        for k in range(K):
            c = C[k]; s = S[k]
            for si in range(NS):
                a = A[si]
                row = si * K + k
                # 6 planes
                for j in range(6):
                    u = a[2 * j]; v = a[2 * j + 1]
                    X[row, 2 * j] = u * c[j] - v * s[j]
                    X[row, 2 * j + 1] = u * s[j] + v * c[j]
        return X

    def make_sym_fg(C, S, beta):
        IU = np.triu_indices(n, 1)
        def fg(yf):
            Y = yf.reshape(NS, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            A = Y / nr
            X = expand_X(A, C, S)
            g = (X @ X.T)[IU]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            Sm = float(e.sum()) + 1e-300
            F = m + np.log(Sm) / beta
            w = e / Sm
            P = np.zeros((n, n))
            P[IU] = w
            P = P + P.T
            GX = P @ X  # (68,12)
            GX4 = GX.reshape(NS, K, d)
            GA = np.zeros((NS, d))
            for k in range(K):
                c = C[k]; s = S[k]
                Gk = GX4[:, k, :]  # (4,12)
                for j in range(6):
                    g0 = Gk[:, 2 * j]; g1 = Gk[:, 2 * j + 1]
                    cj = c[j]; sj = s[j]
                    GA[:, 2 * j] += g0 * cj + g1 * sj
                    GA[:, 2 * j + 1] += -g0 * sj + g1 * cj
            dot = np.sum(GA * A, axis=1, keepdims=True)
            GY = (GA - dot * A) / nr
            return F, GY.ravel()
        return fg

    def run_sym(seeds0, C, S, betas, iters):
        y0 = (seeds0 / np.linalg.norm(seeds0, axis=1, keepdims=True)).ravel().copy()
        y = y0
        for beta, mi in zip(betas, iters):
            fg = make_sym_fg(C, S, float(beta))
            try:
                r = minimize(fg, y, jac=True, method='L-BFGS-B',
                             options={'maxiter': int(mi), 'ftol': 1e-15, 'gtol': 1e-7,
                                      'maxfun': int(mi) * 3})
                y = r.x
            except Exception:
                break
            if time.time() > t_end - 8:
                break
        Y = y.reshape(NS, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        X = expand_X(Y, C, S)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return Y, X

    def sym_lp_polish(A0, mults, C, S, deadline, r0=6e-4, cap_p=2500):
        A = A0 / np.linalg.norm(A0, axis=1, keepdims=True)
        X = expand_X(A, C, S)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        best = _max_cos(X)
        radius = r0
        iu0 = np.triu_indices(n, 1)
        # map flat idx -> (s,k)
        Ss = np.repeat(np.arange(NS), K)
        Ks = np.tile(np.arange(K), NS)
        nv = NS * d + 1
        while time.time() < deadline and radius > 1e-14:
            if time.time() > deadline - 0.5:
                break
            Gc = X @ X.T
            cfull = Gc[iu0]
            m = float(cfull.max())
            sel = cfull > m - max(20.0 * radius, 1e-9)
            I = iu0[0][sel]; J = iu0[1][sel]
            csel = cfull[sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            if p > cap_p:
                od = np.argsort(-csel)[:cap_p]
                I, J = I[od], J[od]
                csel = csel[od]
                p = cap_p
            A_ub = np.zeros((p, nv))
            for r in range(p):
                i = I[r]; j = J[r]
                s1, k1 = int(Ss[i]), int(Ks[i])
                s2, k2 = int(Ss[j]), int(Ks[j])
                Xi = X[i]; Xj = X[j]
                # v1 = R_{k1}^T Xj
                c1 = C[k1]; s1v = S[k1]
                c2 = C[k2]; s2v = S[k2]
                v1 = np.empty(d); v2 = np.empty(d)
                for jj in range(6):
                    v1[2*jj] = Xj[2*jj]*c1[jj] + Xj[2*jj+1]*s1v[jj]
                    v1[2*jj+1] = -Xj[2*jj]*s1v[jj] + Xj[2*jj+1]*c1[jj]
                    v2[2*jj] = Xi[2*jj]*c2[jj] + Xi[2*jj+1]*s2v[jj]
                    v2[2*jj+1] = -Xi[2*jj]*s2v[jj] + Xi[2*jj+1]*c2[jj]
                if s1 == s2:
                    A_ub[r, s1*d:(s1+1)*d] = v1 + v2
                else:
                    A_ub[r, s1*d:(s1+1)*d] = v1
                    A_ub[r, s2*d:(s2+1)*d] = v2
                A_ub[r, -1] = -1.0
            b_ub = -csel
            A_eq = np.zeros((NS, nv))
            for si in range(NS):
                A_eq[si, si*d:(si+1)*d] = A[si]
            try:
                res = linprog(np.concatenate([np.zeros(NS*d), [1.0]]),
                              A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(NS),
                              bounds=[(-radius, radius)]*(NS*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Dseed = res.x[:-1].reshape(NS, d)
            An = A + Dseed
            An /= np.linalg.norm(An, axis=1, keepdims=True)
            Xn = expand_X(An, C, S)
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            v = _max_cos(Xn)
            if v < best - 1e-15:
                A, X, best = An, Xn, v
                radius = min(radius * 1.5, 0.02)
            else:
                improved = False
                for sh in (0.3, 0.1):
                    At = A + Dseed * sh
                    At /= np.linalg.norm(At, axis=1, keepdims=True)
                    Xt = expand_X(At, C, S)
                    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                    vt = _max_cos(Xt)
                    if vt < best:
                        A, X, best = At, Xt, vt
                        improved = True
                        break
                if not improved:
                    radius *= 0.5
        return A, X, best

    def full_lp_polish(x, deadline, r0=5e-4, cap_p=2000):
        from scipy.sparse import coo_matrix
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        best = _max_cos(x)
        radius = r0
        iu = np.triu_indices(nn, 1)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > deadline - 0.4:
                break
            c = (x @ x.T)[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            I = iu[0][sel]; J = iu[1][sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            csel = c[sel]
            if p > cap_p:
                od = np.argsort(-csel)[:cap_p]
                I, J = I[od], J[od]
                csel = csel[od]
                p = cap_p
            nvv = nn * dd + 1
            rows = np.repeat(np.arange(p), 2 * dd + 1)
            cols = np.concatenate([I[:, None]*dd+np.arange(dd), J[:, None]*dd+np.arange(dd), np.full((p,1), nn*dd)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nvv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn*dd))), shape=(nn, nvv)).tocsr()
                obj = np.zeros(nvv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)]*(nn*dd)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(nn, dd)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best - 1e-15:
                x, best = y, v
                radius = min(radius*1.5, 0.02)
            else:
                improved = False
                for sh in (0.3, 0.1):
                    yy = x + res.x[:-1].reshape(nn, dd)*sh
                    yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                    vv = _max_cos(yy)
                    if vv < best:
                        x, best = yy, vv
                        improved = True
                        break
                if not improved:
                    radius *= 0.5
        return x

    # build ~20 multiplier sets (m1=1 fixed, up to scale)
    mults_list = []
    mults_list.append((1,1,1,1,1,1))
    mults_list.append((1,2,3,4,5,6))
    mults_list.append((1,2,4,8,16,9))
    mults_list.append((1,3,5,7,9,11))
    while len(mults_list) < 20:
        v = [1] + [int(rng.integers(1, 17)) for _ in range(5)]
        # random permutation of 6 entries and random sign flips (folded to 0..16)
        perm = rng.permutation(6)
        vv = [v[i] for i in perm]
        # random signs: map m -> -m mod17 with prob 0.5
        for j in range(6):
            if rng.random() < 0.5:
                vv[j] = (-vv[j]) % 17
        tup = tuple(int(x) % 17 for x in vv)
        # canonical: scale so first nonzero is 1
        tup_arr = list(tup)
        # find first nonzero
        f = None
        for x in tup_arr:
            if x % 17 != 0:
                f = x; break
        if f is not None:
            inv = pow(int(f), -1, 17)
            tup_arr = [(x*inv) % 17 for x in tup_arr]
            tup = tuple(tup_arr)
        if tup not in mults_list:
            mults_list.append(tup)
    # search
    cands = []  # (val, mults, seeds, X)
    phase1_end = t0 + 120.0
    betas_seq = [100.0, 300.0, 1000.0, 3000.0]
    iters_seq = [120, 120, 120, 120]
    for mi, mults in enumerate(mults_list):
        if time.time() > phase1_end:
            break
        if time.time() > t_end - 30:
            break
        C, S = build_CS(mults)
        # one trial per mult (two for first few to use budget)
        ntrials = 2 if mi < 4 else 1
        for tr in range(ntrials):
            if time.time() > phase1_end:
                break
            seeds0 = rng.standard_normal((NS, d))
            seeds0 /= np.linalg.norm(seeds0, axis=1, keepdims=True)
            Aopt, Xopt = run_sym(seeds0, C, S, betas_seq, iters_seq)
            v = _max_cos(Xopt)
            cands.append((v, mults, Aopt.copy(), Xopt.copy()))
            if time.time() > t_end - 30:
                break
    cands.sort(key=lambda z: z[0])
    # polish best two
    polished = []
    for idx in range(min(2, len(cands))):
        if time.time() > t_end - 25:
            break
        v, mults, Aopt, Xopt = cands[idx]
        C, S = build_CS(mults)
        # extra high-beta seed refine
        try:
            Aopt2, Xopt2 = run_sym(Aopt, C, S, [3000.0, 6000.0], [150, 120])
            v2 = _max_cos(Xopt2)
            if v2 < v:
                v, Aopt, Xopt = v2, Aopt2, Xopt2
        except Exception:
            pass
        # symmetry-preserving LP
        try:
            A3, X3, v3 = sym_lp_polish(Aopt, mults, C, S, min(t_end - 12, time.time()+12), r0=5e-4)
            if v3 < v:
                v, Aopt, Xopt = v3, A3, X3
        except Exception:
            pass
        polished.append((v, mults, Aopt, Xopt))
    if polished:
        polished.sort(key=lambda z: z[0])
        best_sym_val, best_sym_mults, best_sym_A, best_sym_X = polished[0]
    elif cands:
        best_sym_val, best_sym_mults, best_sym_A, best_sym_X = cands[0]
    else:
        return x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)

    # decide release: only if within 5e-4 of incumbent
    if best_sym_val <= v_inc + 5e-4:
        # final asymmetric release: LSE + full LP
        X = best_sym_X.copy()
        # short full LSE polish
        try:
            IUf = np.triu_indices(n, 1)
            def fg_full(yf, beta=3000.0):
                Y = yf.reshape(n, d)
                nr = np.linalg.norm(Y, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Xn = Y / nr
                g = (Xn @ Xn.T)[IUf]
                m = float(g.max())
                e = np.exp(beta*(g-m))
                Sm = float(e.sum())+1e-300
                F = m + np.log(Sm)/beta
                w = e/Sm
                P = np.zeros((n,n)); P[IUf]=w; P=P+P.T
                GX = P@Xn
                dot = np.sum(GX*Xn, axis=1, keepdims=True)
                GY = (GX-dot*Xn)/nr
                return F, GY.ravel()
            for beta, mi_ in [(2000.0, 80), (8000.0, 60)]:
                if time.time() > t_end - 10:
                    break
                def clo(yf, b=beta):
                    return fg_full(yf, beta=b)
                y0 = (X/np.linalg.norm(X,axis=1,keepdims=True)).ravel().copy()
                try:
                    r = minimize(clo, y0, jac=True, method='L-BFGS-B',
                                 options={'maxiter': mi_, 'ftol':1e-15,'gtol':1e-8,'maxfun':mi_*3})
                    Yn = r.x.reshape(n,d)
                    Yn/=np.linalg.norm(Yn,axis=1,keepdims=True)
                    if _max_cos(Yn) < _max_cos(X):
                        X = Yn
                except Exception:
                    pass
        except Exception:
            pass
        if time.time() < t_end - 3:
            try:
                X = full_lp_polish(X, t_end-0.8, r0=4e-4)
            except Exception:
                pass
        vX = _max_cos(X)
        if vX < min(best_sym_val, v_inc):
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            return X
        if best_sym_val < v_inc:
            best_sym_X /= np.linalg.norm(best_sym_X, axis=1, keepdims=True)
            return best_sym_X
        return x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)
    else:
        # not competitive: return incumbent
        return x_inc / np.linalg.norm(x_inc, axis=1, keepdims=True)
