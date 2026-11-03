import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _edge_set(X, slack):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gmax = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    sel = (G[iu] > gmax - slack)
    return set(zip(iu[0][sel].tolist(), iu[1][sel].tolist())), gmax

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d-1), dtype=np.float64)
    E = np.eye(d)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        B = np.delete(E, k, axis=1)  # 8x7
        # project out x
        c = x @ B  # 7
        B = B - np.outer(x, c)
        Q, _ = np.linalg.qr(B)
        # ensure tangent: Q^T x ~ 0 ; fix sign deterministically
        T[i] = Q
    return T

def _rigidity_step(X, slack, rng, amp):
    n, d = X.shape
    N = n*(d-1)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gmax = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    order = np.where(c > gmax - slack)[0]
    if len(order) < 50:
        # fall back to larger slack
        order = np.argsort(c)[-max(1200, N//2):]
    vals = c[order]
    srt = np.argsort(vals)  # ascending, tightest last
    order = order[srt[::-1]]
    # cap constraints
    MAXP = 2500
    if len(order) > MAXP:
        order = order[:MAXP]
    I = iu[0][order]; J = iu[1][order]
    p = len(I)
    T = _tangent_bases(X)
    # build rigidity matrix p x N
    R = np.empty((p, N), dtype=np.float64)
    # vectorize partially: loop is fine (p<=2500)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        xi = X[i]; xj = X[j]
        vi = T[i].T @ xj  # 7
        vj = T[j].T @ xi  # 7
        row = np.zeros(N, dtype=np.float64)
        row[i*(d-1):(i+1)*(d-1)] = vi
        row[j*(d-1):(j+1)*(d-1)] = vj
        R[k] = row
    # smallest ~50 singular vectors
    try:
        if p >= N:
            _, S, Vt = np.linalg.svd(R, full_matrices=False)
            Vsmall = Vt[-50:].T  # N x 50
        elif p >= 50:
            _, S, Vt = np.linalg.svd(R, full_matrices=False)
            # Vt is p x N, smallest among them + random complement
            # complement basis via QR of random
            Qc = rng.randn(N, 50)
            # orthogonalize against Vt
            # Vt rows orthonormal; project out
            Qc = Qc - Vt.T @ (Vt @ Qc)
            Qc, _ = np.linalg.qr(Qc)
            # combine last 25 of Vt + complement
            take = min(25, Vt.shape[0])
            A = Vt[-take:].T  # N x take
            Vsmall = np.concatenate([A, Qc[:, :(50-take)]], axis=1)
        else:
            Vsmall = rng.randn(N, 50)
            Vsmall /= np.linalg.norm(Vsmall, axis=0, keepdims=True) + 1e-18
    except Exception:
        Vsmall = rng.randn(N, 50)
        Vsmall /= np.linalg.norm(Vsmall, axis=0, keepdims=True) + 1e-18
    coeff = rng.randn(50)
    coeff /= np.linalg.norm(coeff) + 1e-18
    v = Vsmall @ coeff  # N, unit in coeff sense but not normalized
    nv = np.linalg.norm(v)
    if nv == 0 or not np.finite(nv):
        v = rng.randn(N); nv = np.linalg.norm(v)
    v /= nv
    # scale: amp is desired RMS angular displacement
    # total norm s = amp*sqrt(n)
    s = float(amp) * np.sqrt(float(n))
    v = v * s
    U = v.reshape(n, d-1)  # tangent coords
    Xn = np.empty_like(X)
    for i in range(n):
        u = U[i]
        th = float(np.linalg.norm(u))
        if th < 1e-15:
            Xn[i] = X[i]
        else:
            direction = T[i] @ (u / th)
            # exponential map
            Xn[i] = np.cos(th) * X[i] + np.sin(th) * direction
    # renormalize
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    return Xn

def _quench(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps+1):
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0 or not np.isfinite(Z):
            continue
        Grad = (W @ X)/Z
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        m *= b1; m += (1.0-b1)*Gt
        v *= b2; v += (1.0-b2)*(Gt*Gt)
        mhat = m/(1.0-b1**t); vhat = v/(1.0-b2**t)
        step = mhat/(np.sqrt(vhat)+eps)
        X = X - lr*step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds, r0=1e-3, cap=2500):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            idx = np.argsort(c[sel])[-cap:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _jaccard(a, b):
    if len(a)==0 and len(b)==0:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter/union

def solve():
    t0 = time.time()
    BUDGET = 174.0
    n, d = 168, 8
    best = None; best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n,d):
                continue
            a = a/np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(n,d); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy(); cur_val = best_val
    global_best = best.copy(); global_val = best_val
    rng = np.random.RandomState(777)
    slacks = [1e-7, 1e-5, 1e-4]
    it = 0
    # initial LP squeeze a touch to get baseline graph
    # main loop: rigidity floppy steps
    while True:
        it += 1
        now = time.time()
        remain = t0+BUDGET-now
        if remain < 6.0:
            break
        if now > t0+150.0:
            break
        # pick slack threshold cyclically + random
        slack = slacks[rng.randint(len(slacks))]
        # amplitude log-uniform 0.02..0.15
        la, lb = np.log(0.02), np.log(0.15)
        amp = float(np.exp(rng.uniform(la, lb)))
        # parent graph at 1e-5 for Jaccard
        try:
            parent_edges, _ = _edge_set(cur, 1e-5)
        except Exception:
            parent_edges = set()
        try:
            Xt = _rigidity_step(cur, slack, rng, amp)
        except Exception:
            continue
        # soft-max annealing beta 50->600, 1500 steps (shorten if late)
        steps = 1500
        if remain < 25:
            steps = 800
        # slight randomization of beta/lr
        try:
            Xt = _quench(Xt, steps=steps, beta0=50.0, beta1=600.0, lr0=0.005, lr1=0.0003)
        except Exception:
            continue
        # short LP polish
        lp_sec = 1.5
        if remain < 20:
            lp_sec = 0.8
        try:
            Xt = _lp_polish(Xt, seconds=lp_sec, r0=1e-3)
        except Exception:
            pass
        try:
            v = _max_cos(Xt)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        try:
            child_edges, _ = _edge_set(Xt, 1e-5)
            jac = _jaccard(parent_edges, child_edges)
        except Exception:
            jac = 1.0
        if v < global_val:
            global_val = v; global_best = Xt.copy()
            cur = Xt.copy(); cur_val = v
        elif jac < 0.9 and v < cur_val + 0.003:
            # topology changed: move to new basin
            cur = Xt.copy(); cur_val = v
        elif v < cur_val:
            cur = Xt.copy(); cur_val = v
        # else discard
        # occasional restart to global best if cur drifts bad
        if cur_val > global_val + 0.002 and rng.rand() < 0.5:
            cur = global_best.copy(); cur_val = global_val
    # final polish of global best with remaining time
    try:
        remain = t0+BUDGET-time.time()-1.0
        if remain > 2.0:
            global_best = _quench(global_best, steps=600, beta0=100.0, beta1=800.0, lr0=0.003, lr1=0.0002)
        remain = t0+BUDGET-time.time()-0.5
        if remain > 1.0:
            global_best = _lp_polish(global_best, seconds=min(remain, 8.0), r0=8e-4)
            gv = _max_cos(global_best)
            if np.isfinite(gv):
                global_val = gv
    except Exception:
        pass
    global_best /= np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
