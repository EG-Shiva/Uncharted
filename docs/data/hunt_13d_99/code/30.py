import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def contacts_balanced(x, tol=1e-6, need=5):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    return int(np.sum(cnt >= need)), cnt, m

def equi_objective(Xflat, n, d, alpha, lam, mu):
    X = Xflat.reshape(n, d)
    r = np.linalg.norm(X, axis=1, keepdims=True)
    r[r == 0] = 1.0
    x = X / r
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    # global LSE over i<j
    iu = np.triu_indices(n, 1)
    Gu = G[iu]
    mG = float(np.max(Gu))
    E = np.exp(alpha * (Gu - mG))
    S = float(np.sum(E))
    L = mG + np.log(S) / alpha
    # pair weights
    # full matrix of exp for gradient
    Ef = np.exp(alpha * (G - mG))
    np.fill_diagonal(Ef, 0.0)
    Wpair = Ef / S  # w_ij for ordered? sum_full/2=S*? check: S=sum over upper, sum_full=2S, Wpair_full=E/ S => gradient uses E/S (which is 2* normalized) -> correct because w_upper=E/S
    # per-point softmax
    rowmax = np.max(G, axis=1)  # (n,)
    # stable
    Erow = np.exp(alpha * (G - rowmax[:, None]))
    np.fill_diagonal(Erow, 0.0)
    Si = np.sum(Erow, axis=1) + 1e-300
    s = rowmax + np.log(Si) / alpha
    R = Erow / Si[:, None]  # row weights
    mean_s = float(np.mean(s))
    var_s = float(np.mean((s - mean_s) ** 2))
    std_s = np.sqrt(max(var_s, 0.0))
    if std_s < 1e-14:
        c = np.zeros(n)
    else:
        c = (s - mean_s) / (n * std_s)
    y = s - L
    ay = alpha * y
    # sigmoid stable
    sig = np.zeros(n)
    t = np.zeros(n)
    for i in range(n):
        v = ay[i]
        if v >= 0:
            sig[i] = 1.0 / (1.0 + np.exp(-v))
            t[i] = y[i] + np.log1p(np.exp(-v)) / alpha
        elif v > -40:
            e = np.exp(v)
            sig[i] = e / (1.0 + e)
            t[i] = np.log1p(e) / alpha
        else:
            sig[i] = np.exp(v)
            t[i] = np.exp(v) / alpha
    mean_sig = float(np.mean(sig))
    mean_t = float(np.mean(t))
    F = L + lam * std_s + mu * mean_t
    coeff_L = 1.0 - mu * mean_sig
    coeff = lam * c + mu * sig / n  # per s_i
    # grad_x
    # term1: coeff_L * Wpair @ x
    G1 = Wpair @ x  # (n,d)
    # term2: per-row part
    # for each k: coeff[k]* sum_j R[k,j] x_j = (diag(coeff) @ R @ x)
    G2 = (coeff[:, None] * R) @ x  # this gives for k the k==i part
    # term3: sum_{i !=k} coeff[i] R[i,k] x_i = ?
    # = for each k: sum_i (coeff[i]*R[i,k]) * x_i
    # Let A_ik = coeff[i]*R[i,k], then grad3[k] = sum_i A_ik x_i = (A^T @ x)[k]? A^T[k,i]=A[i,k], so (A^T @ x)[k]=sum_i A[i,k] x_i correct.
    A = coeff[:, None] * R  # A[i,k]
    G3 = A.T @ x
    # careful: G2 double counts? G2[k]=coeff[k]*sum_j R[k,j]x_j, G3[k]=sum_i coeff[i]R[i,k]x_i, distinct
    grad_x = coeff_L * G1 + G2 + G3
    # project to X
    # grad_X[k] = (grad_x[k] - (grad_x[k].x[k]) x[k]) / r[k]
    dot = np.sum(grad_x * x, axis=1, keepdims=True)
    grad_X = (grad_x - dot * x) / r
    return F, grad_X.ravel()

def run_stage(x0, alpha, lam, mu, maxiter):
    n, d = x0.shape
    X0 = x0.ravel().copy()
    def fun(Xf):
        F, g = equi_objective(Xf, n, d, alpha, lam, mu)
        return F, g
    try:
        res = minimize(fun, X0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter*3,
                                'ftol': 1e-14, 'gtol': 1e-7})
        Xopt = res.x.reshape(n, d)
    except Exception:
        return x0
    nr = np.linalg.norm(Xopt, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xopt /= nr
    # guard nan
    if not np.all(np.isfinite(Xopt)):
        return x0
    return Xopt

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            # keep tightest 4000 to bound LP size
            order = np.argsort(-c[sel])[:4000]
            I, J = I[order], J[order]
            p = len(I)
            csv = c[sel][order]
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    base_bal, _, _ = contacts_balanced(best_x, 1e-6, 5)
    # homotopy schedule
    alphas = [40.0, 80.0, 160.0, 320.0, 640.0]
    lams = [1.0, 0.5, 0.2, 0.1, 0.05]
    mu = 0.5
    maxiters = [80, 80, 70, 60, 50]
    # build inits
    inits = []
    for sigma in [0.01, 0.03, 0.08]:
        x0 = xb + rng.standard_normal(size=(n, d)) * sigma
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        inits.append(x0)
    for _ in range(5):
        x0 = rng.standard_normal(size=(n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        inits.append(x0)
    cands = []  # (score, bal, x)
    # time allocation: explore until ~125s
    EXPLORE_UNTIL = 125.0
    for xi, xinit in enumerate(inits):
        if time.time() - t0 > EXPLORE_UNTIL:
            break
        cur = xinit.copy()
        for a, l, mi in zip(alphas, lams, maxiters):
            if time.time() - t0 > EXPLORE_UNTIL:
                break
            cur = run_stage(cur, a, l, mu, mi)
            if not np.all(np.isfinite(cur)):
                break
        sc = max_cos(cur)
        # short polish to evaluate basin (Direction-1 style, 1.0s)
        try:
            pol = lp_polish(cur, 1.0)
            scp = max_cos(pol)
        except Exception:
            pol = cur
            scp = sc
        bal, _, _ = contacts_balanced(pol, 1e-6, 5)
        cands.append((scp, bal, pol.copy()))
        # update global best by score
        if scp < best_val - 1e-12:
            best_val = scp
            best_x = pol.copy()
            base_bal = bal
        elif bal > base_bal and scp < best_val + 1e-6:
            # balanced basin within 1e-6 worse: keep as alternative target, also track
            # if it is balanced breakthrough, allow walker to keep it for final polish
            pass
        # also if raw cur (unpolished) is better, consider
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = cur.copy()
            b2, _, _ = contacts_balanced(cur, 1e-6, 5)
            base_bal = b2
    # rank candidates: prefer balanced then score? For final polish pick top by score + balanced bonus
    # sort by score, but boost balanced
    def keyfn(t):
        sc, bal, _ = t
        return sc - 1e-7 * bal
    cands_sorted = sorted(cands, key=keyfn)
    remain = BUDGET - (time.time() - t0) - 2.0
    # final polish: spend on best_x and top 2 candidates
    pool = []
    pool.append(best_x.copy())
    for _, _, xx in cands_sorted[:2]:
        pool.append(xx.copy())
    # deduplicate roughly
    best_final_val = max_cos(pool[0])
    best_final_x = pool[0].copy()
    for xx in pool:
        v = max_cos(xx)
        if v < best_final_val:
            best_final_val = v
            best_final_x = xx.copy()
    if remain > 5:
        per = remain / min(len(pool), 3)
        for xx in pool[:3]:
            if time.time() - t0 > BUDGET - 2.0:
                break
            try:
                yp = lp_polish(xx, per * 0.8)
                vp = max_cos(yp)
                if vp < best_final_val:
                    best_final_val = vp
                    best_final_x = yp.copy()
            except Exception:
                continue
            # extra high-alpha smooth touch before LP for one candidate
            if time.time() - t0 < BUDGET - 8:
                try:
                    tmp = run_stage(xx, 320.0, 0.02, 0.2, 40)
                    tmp2 = lp_polish(tmp, 1.0)
                    v2 = max_cos(tmp2)
                    if v2 < best_final_val:
                        best_final_val = v2
                        best_final_x = tmp2.copy()
                except Exception:
                    pass
    # ensure at least as good as input
    if best_final_val > best_val + 1e-12:
        best_final_x = best_x.copy()
    best_final_x /= np.linalg.norm(best_final_x, axis=1, keepdims=True)
    return best_final_x
