import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _perm_sign_to_mat(p, s):
    M = np.zeros((6, 6))
    for i in range(6):
        M[i, p[i]] = s[i]
    return M

def _mat_to_perm_sign(M):
    p = np.zeros(6, dtype=int)
    s = np.zeros(6)
    for i in range(6):
        j = int(np.argmax(np.abs(M[i])))
        p[i] = j
        s[i] = float(np.sign(M[i, j]) if abs(M[i, j]) > 1e-9 else 1.0)
        if s[i] == 0:
            s[i] = 1.0
    return p, s

def _build_base_group(block4, kind):
    # block4: tuple/list of 4 indices, block2 = rest
    block4 = list(block4)
    block2 = [i for i in range(6) if i not in block4]
    # S4 on block4
    mats = []
    for perm in itertools.permutations(block4):
        p = np.arange(6)
        # perm gives new order: position k in block4 gets perm[k]?
        # we need all bijections: map block4[i] -> perm[i]? Actually perm is ordering of block4 values.
        # Build p such that p[block4[i]] = perm[i]
        for i, b in enumerate(block4):
            p[b] = perm[i]
        s = np.ones(6)
        mats.append(_perm_sign_to_mat(p, s))
    # block2 part
    b2mats = []
    if kind == 48:  # swap only
        p0 = np.arange(6); s0 = np.ones(6)
        b2mats.append(_perm_sign_to_mat(p0, s0))
        p1 = np.arange(6); p1[block2[0]], p1[block2[1]] = p1[block2[1]], p1[block2[0]]
        b2mats.append(_perm_sign_to_mat(p1, np.ones(6)))
    elif kind == 96:  # Klein flips on block2 (no swap)
        for f0 in (1.0, -1.0):
            for f1 in (1.0, -1.0):
                p = np.arange(6); s = np.ones(6)
                s[block2[0]] = f0; s[block2[1]] = f1
                b2mats.append(_perm_sign_to_mat(p, s))
    elif kind == 192:  # full B2 on block2
        for swap in (False, True):
            for f0 in (1.0, -1.0):
                for f1 in (1.0, -1.0):
                    p = np.arange(6); s = np.ones(6)
                    if swap:
                        p[block2[0]], p[block2[1]] = p[block2[1]], p[block2[0]]
                        # signs attached to rows: need care: row block2[0] gets flip f0 etc.
                        s[block2[0]] = f0; s[block2[1]] = f1
                    else:
                        s[block2[0]] = f0; s[block2[1]] = f1
                    b2mats.append(_perm_sign_to_mat(p, s))
    else:
        raise ValueError
    # direct product (blocks disjoint, commute)
    # A in mats (S4), B in b2mats: combined = A @ B (=B@A)
    G = []
    for A in mats:
        for B in b2mats:
            G.append(A @ B)
    # deduplicate
    uniq = []
    seen = set()
    for M in G:
        key = tuple(np.round(M.ravel(), 6))
        if key not in seen:
            seen.add(key)
            uniq.append(M)
    return np.array(uniq)  # (m,6,6)

def _conjugate_group(Gmats, C):
    Ci = C.T  # orthogonal
    out = []
    for M in Gmats:
        Mp = C @ M @ Ci
        # clean tiny
        Mp = np.round(Mp, 10)
        out.append(Mp)
    return np.array(out)

def _random_signed_perm_mat(rng):
    p = rng.permutation(6)
    s = rng.choice(np.array([-1.0, 1.0]), size=6)
    M = np.zeros((6, 6))
    for i in range(6):
        M[i, p[i]] = s[i]
    # this corresponds to action y_i = s_i x_{p_i} but as matrix with row i col p_i.
    # For conjugation we need matrix that acts as y=Cx, same form. ok.
    return M

def solve():
    t0 = time.time()
    BUDGET = 170.0
    SYM_END = t0 + 52.0
    rng = np.random.RandomState(0)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    # load incumbent
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)
    global_best_val = best_val
    global_best_X = x_best.copy()

    # ---------- symmetric helpers ----------
    def sym_fun_factory(Ms, q, r, beta):
        m = Ms.shape[0]
        # Ms: (m,6,6)
        def fun(th):
            th = np.asarray(th, dtype=float)
            R = th[:q*6].reshape(q, 6) if q > 0 else np.zeros((0, 6))
            F = th[q*6:].reshape(r, 6) if r > 0 else np.zeros((0, 6))
            rnR = np.linalg.norm(R, axis=1, keepdims=True) if q > 0 else np.ones((0, 1))
            rnR = np.maximum(rnR, 1e-12)
            rnF = np.linalg.norm(F, axis=1, keepdims=True) if r > 0 else np.ones((0, 1))
            rnF = np.maximum(rnF, 1e-12)
            U = R / rnR if q > 0 else np.zeros((0, 6))
            V = F / rnF if r > 0 else np.zeros((0, 6))
            if q > 0:
                # O[q,g,i] = sum_j Ms[g,i,j]*U[q,j]
                O = np.einsum('gij,qj->qgi', Ms, U)  # q,m,6
                core = O.reshape(q*m, 6)
                X = np.concatenate([core, V], axis=0) if r > 0 else core
            else:
                X = V
            G = X @ X.T
            g = G[iu0, iu1]
            m0 = float(g.max())
            e = np.exp(beta * (g - m0))
            S = float(e.sum()) + 1e-300
            loss = m0 + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X  # dL/dX
            # split
            if q > 0:
                dCore = GX[:q*m].reshape(q, m, 6)
                # dU[q,j] = sum_g sum_i Ms[g,i,j]*dCore[q,g,i]
                dU = np.einsum('gij,qgi->qj', Ms, dCore)
                dotU = (dU * U).sum(axis=1, keepdims=True)
                dR = (dU - dotU * U) / rnR
            else:
                dR = np.zeros((0, 6))
            if r > 0:
                dV = GX[q*m:] if q > 0 else GX
                dotV = (dV * V).sum(axis=1, keepdims=True)
                dF = (dV - dotV * V) / rnF
            else:
                dF = np.zeros((0, 6))
            grad = np.concatenate([dR.reshape(-1), dF.reshape(-1)]) if (q*6 + r*6) > 0 else np.zeros(0)
            return float(loss), grad
        return fun

    def run_sym(Ms, q, r, th0, beta, maxiter):
        fun = sym_fun_factory(Ms, q, r, beta)
        try:
            res = minimize(fun, th0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "maxfun": 10000})
            th = res.x
        except Exception:
            th = th0
        # build X
        R = th[:q*6].reshape(q, 6) if q > 0 else np.zeros((0, 6))
        F = th[q*6:].reshape(r, 6) if r > 0 else np.zeros((0, 6))
        rnR = np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12) if q > 0 else np.ones((0, 1))
        rnF = np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12) if r > 0 else np.ones((0, 1))
        U = R / rnR if q > 0 else np.zeros((0, 6))
        V = F / rnF if r > 0 else np.zeros((0, 6))
        if q > 0:
            O = np.einsum('gij,qj->qgi', Ms, U)
            core = O.reshape(q*m, 6) if False else O.reshape(q*Ms.shape[0], 6)
            X = np.concatenate([core, V], axis=0) if r > 0 else core
        else:
            X = V
        return _norm(X)

    # ---------- symmetric sampling ----------
    best_sym_X = None
    best_sym_val = 1e9
    sym_trials = 0
    kinds = [48, 96, 192]
    while time.time() < SYM_END and time.time() - t0 < BUDGET - 110:
        sym_trials += 1
        kind = kinds[int(rng.randint(0, 3))]
        # random block4
        block4 = tuple(sorted(rng.choice(6, 4, replace=False).tolist()))
        try:
            G0 = _build_base_group(block4, kind)
        except Exception:
            continue
        m = G0.shape[0]
        if m not in (48, 96, 192):
            continue
        # random conjugate
        C = _random_signed_perm_mat(rng)
        try:
            Ms = _conjugate_group(G0, C)
        except Exception:
            Ms = G0
        q = n // m
        r = n - q * m
        if q == 0:
            continue
        # init representatives + free tail
        th0_core = rng.randn(q * 6)
        # free tail init: random + occasionally from incumbent voids
        if rng.rand() < 0.4 and global_best_X is not None:
            # pick r random rows of incumbent + noise
            idx = rng.choice(n, r, replace=False)
            Ft = global_best_X[idx] + 0.05 * rng.randn(r, 6)
            th0_tail = Ft.reshape(-1)
        else:
            th0_tail = rng.randn(r * 6)
        th0 = np.concatenate([th0_core, th0_tail])
        beta = float(rng.choice([40.0, 60.0, 90.0]))
        maxiter = 60 if time.time() - t0 < 30 else 40
        try:
            Xsym = run_sym(Ms, q, r, th0, beta, maxiter)
        except Exception:
            continue
        v = _max_cos(Xsym)
        if v < best_sym_val:
            best_sym_val = v
            best_sym_X = Xsym.copy()
        # quick second beta sharpen on promising
        if v < global_best_val + 0.02 and time.time() < SYM_END - 5:
            try:
                # continue from th with higher beta: re-extract th? just re-run with higher beta starting from Xsym reps
                # build th1 from Xsym
                # core reps: average? take first element of each orbit as rep
                reps = []
                for k in range(q):
                    reps.append(Xsym[k*m])
                reps = np.array(reps)
                Ft2 = Xsym[q*m:]
                th1 = np.concatenate([reps.reshape(-1), Ft2.reshape(-1)])
                Xsym2 = run_sym(Ms, q, r, th1, beta + 60.0, 30)
                v2 = _max_cos(Xsym2)
                if v2 < best_sym_val:
                    best_sym_val = v2
                    best_sym_X = Xsym2.copy()
            except Exception:
                pass
        if best_sym_val < global_best_val:
            global_best_val = best_sym_val
            global_best_X = best_sym_X.copy()
        # limit trials
        if sym_trials > 60:
            break

    # ---------- free optimisation + LP ----------
    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy() * 1.0
        # scale: use unnormalised vars = x0 (already unit) -> rn=1
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m0 = float(g.max())
            e = np.exp(beta * (g - m0))
            S = float(e.sum()) + 1e-300
            loss = m0 + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=10):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps and time.time() - t0 < BUDGET - 1.0:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m0 = float(c.max())
            thr = m0 - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            pnum = len(I)
            if pnum == 0:
                break
            if pnum > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2-1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                pnum = k2
            rows = np.repeat(np.arange(pnum), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((pnum, 1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((pnum, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pnum, n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    def add_tangent_noise(X, eps, rng):
        Z = rng.randn(*X.shape)
        Z -= (Z * X).sum(axis=1, keepdims=True) * X
        zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        Z = Z / zn
        Y = X + eps * Z
        return _norm(Y)

    # candidate pool for release
    pool = [(global_best_val, global_best_X.copy())]
    if best_sym_X is not None:
        # avoid duplicate
        if _max_cos(best_sym_X) < 0.85:
            pool.append((best_sym_val, best_sym_X.copy()))
    # also incumbent x_best
    try:
        if x_best is not None:
            pool.append((_max_cos(x_best), x_best.copy()))
    except Exception:
        pass

    # sort pool by val
    pool = sorted(pool, key=lambda t: t[0])[:3]

    noises = [0.0, 1e-4, 1e-3, 1e-2]
    # time-aware loop
    for (pv, px) in list(pool):
        for eps in noises:
            if time.time() - t0 > BUDGET - 18:
                break
            rng2 = np.random.RandomState(int(1000 + eps*1e6) + int(pv*1e9) % 100000)
            if eps == 0.0:
                X0 = px.copy()
            else:
                X0 = add_tangent_noise(px, eps, rng2)
            # free L-BFGS schedule: low then high beta
            remain = BUDGET - (time.time() - t0)
            if remain < 12:
                break
            # choose iters based on remain
            X1 = run_free_lbfgs(X0, 50.0, 80)
            v1 = _max_cos(X1)
            if v1 < global_best_val:
                global_best_val = v1
                global_best_X = X1.copy()
            # second stage higher beta if time
            if time.time() - t0 < BUDGET - 14:
                X2 = run_free_lbfgs(X1, 120.0, 80)
                v2 = _max_cos(X2)
                if v2 < global_best_val:
                    global_best_val = v2
                    global_best_X = X2.copy()
                # LP polish best of the two
                dl = min(6.0, max(2.0, BUDGET - (time.time() - t0) - 2.0))
                if dl > 1.0:
                    Xp, vp = lp_polish(X2 if v2 < v1 else X1, time.time() + dl)
                    if vp < global_best_val:
                        global_best_val = vp
                        global_best_X = Xp.copy()
            # early exit if improved a lot? continue to try other noises
    # final LP on global best if time
    if time.time() - t0 < BUDGET - 3:
        try:
            Xp, vp = lp_polish(global_best_X, time.time() + max(1.0, BUDGET - (time.time() - t0) - 1.0), r_init=8e-4)
            if vp < global_best_val:
                global_best_val = vp
                global_best_X = Xp.copy()
        except Exception:
            pass

    global_best_X = _norm(global_best_X)
    return global_best_X
