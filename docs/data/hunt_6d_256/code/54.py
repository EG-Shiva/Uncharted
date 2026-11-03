import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _base_cube():
    # 64 x 6 of +/-1 / sqrt6
    n = 64
    H = np.empty((64, 6), dtype=np.float64)
    for i in range(64):
        for k in range(6):
            H[i, k] = 1.0 if ((i >> k) & 1) == 0 else -1.0
    H /= np.sqrt(6.0)
    return H

# skew packing: 15 params -> 6x6 skew
_TRIL = []
for _i in range(6):
    for _j in range(_i + 1, 6):
        _TRIL.append((_i, _j))

def _vec_to_skew(v):
    S = np.zeros((6, 6), dtype=np.float64)
    for k, (i, j) in enumerate(_TRIL):
        S[i, j] = v[k]
        S[j, i] = -v[k]
    return S

def _cayley(S):
    I = np.eye(6)
    A = I - S
    B = I + S
    # R = A^{-1} B
    try:
        R = np.linalg.solve(A, B)
    except Exception:
        R = np.linalg.lstsq(A, B, rcond=None)[0]
    return R, A

def _params_to_X(p, H):
    # p size 45 -> R1,R2,R3, A1,A2,A3
    Rlist = []
    Alist = []
    for b in range(3):
        v = p[b * 15:(b + 1) * 15]
        S = _vec_to_skew(v)
        R, A = _cayley(S)
        Rlist.append(R)
        Alist.append(A)
    X0 = H
    X1 = H @ Rlist[0].T
    X2 = H @ Rlist[1].T
    X3 = H @ Rlist[2].T
    X = np.vstack([X0, X1, X2, X3])
    # renormalize (should already be unit, Cayley orthogonal up to numerics)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, Rlist, Alist

def _cross_mask():
    n = 256
    blk = np.arange(n) // 64
    M = blk[:, None] != blk[None, :]
    np.fill_diagonal(M, False)
    return M

_MASK = None

def _sym_eval_and_grad(p, H, K, mask):
    X, Rlist, Alist = _params_to_X(p, H)
    G = X @ X.T
    # only cross pairs i<j
    # compute LSE over cross upper triangle
    iu = np.triu_indices(256, 1)
    # cross selector
    sel = mask[iu]
    g = G[iu][sel]  # m2 entries
    gmax = float(g.max())
    # weights
    E = np.exp(K * (g - gmax))
    Ssum = float(E.sum())
    f = gmax + np.log(Ssum) / K - np.log(float(g.size)) / K if False else gmax + np.log(Ssum) / K
    # Build full W symmetric (256x256) sparse-ish dense is fine
    W = np.zeros((256, 256), dtype=np.float64)
    wvec = E / Ssum  # sums to 1 over unordered pairs
    # scatter
    I = iu[0][sel]
    J = iu[1][sel]
    W[I, J] = wvec
    W[J, I] = wvec
    GradX = W @ X  # 256x6, gradient of f w.r.t X
    # split
    grad_Rs = []
    for b in range(3):
        Gk = GradX[(b + 1) * 64:(b + 2) * 64, :]  # 64x6
        gradR = Gk.T @ H  # 6x6
        grad_Rs.append(gradR)
    # backprop through Cayley to skew params
    grad_p = np.zeros(45, dtype=np.float64)
    I6 = np.eye(6)
    for b in range(3):
        gradR = grad_Rs[b]
        R = Rlist[b]
        A = Alist[b]
        # M = (R+I) gradR^T A^{-1}  ; grad_skew_full = M^T with skew projection
        # Compute AinvT etc without explicit inverse: solve
        # M = (R+I) @ gradR.T @ inv(A)
        # M^T = inv(A)^T @ gradR @ (R+I)^T
        # Let Y = gradR @ (R+I)^T  (6x6), then M^T = inv(A)^T @ Y = solve(A^T, Y)
        Y = gradR @ (R + I6).T
        try:
            Mt = np.linalg.solve(A.T, Y)
        except Exception:
            Mt = np.linalg.lstsq(A.T, Y, rcond=None)[0]
        # skew effective
        Ms = 0.5 * (Mt - Mt.T)
        # extract upper triangle
        for k, (i, j) in enumerate(_TRIL):
            grad_p[b * 15 + k] = Ms[i, j] * 2.0  # because S[i,j]=v, S[j,i]=-v => d = 2*Ms[i,j]? check: tr(Mt^T? ) Let's use 2 factor.
            # Actually df = tr(M dS) with M = (R+I)gradR^T A^{-1} ... we computed Mt = M^T.
            # df = tr(M dS) = sum_ij M_ij dS_ij = sum_{i<j} (M_ij - M_ji) dv_k = sum 2*Ms_ij dv
            # Ms as defined, so grad = 2*Ms_ij. Above Ms already (Mt-MtT)/2, so 2*Ms = Mt-MtT.
        # The above double counts? We set Ms=(Mt-MtT)/2 then *2 => Mt-MtT. ok.
    return f, float(G[iu].max()), grad_p, X

def _adam_sym(H, p0, steps, K0, K1, lr, rng=None):
    p = p0.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    mask = _MASK
    best_val = np.inf
    best_p = p.copy()
    best_X = None
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        f, true_max, g, X = _sym_eval_and_grad(p, H, K, mask)
        if true_max < best_val:
            best_val = float(true_max)
            best_p = p.copy()
            best_X = X.copy()
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        # mild clip
        p = p - step
        # keep bounded to avoid huge S (near singular)
        np.clip(p, -3.0, 3.0, out=p)
    # re-eval best
    return best_p, best_X, best_val

def _adam_lse_full(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum() * 0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        grad = (W @ x) / Ssum
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 200 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (mval - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            pnum = cap
        rows = np.repeat(np.arange(pnum), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((pnum, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pnum, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pnum, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            best = val
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 172.0
    global _MASK
    _MASK = _cross_mask()
    H = _base_cube()
    rng = np.random.default_rng(12345)
    # load incumbent
    inc = None
    inc_val = np.inf
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < inc_val:
                    inc_val = float(vv)
                    inc = a.copy()
        except Exception:
            pass
    if inc is None:
        inc = rng.standard_normal((256, 6))
        inc /= np.linalg.norm(inc, axis=1, keepdims=True)
        inc_val = _max_cos(inc)
    # ---- symmetry search: 10 inits ----
    # time budget for sym: ~55s max
    sym_deadline = t0 + 55.0
    n_restart = 10
    best_sym_X = None
    best_sym_val = np.inf
    best_sym_p = None
    # first init zero
    inits = []
    inits.append(np.zeros(45))
    for _ in range(n_restart - 1):
        inits.append(rng.standard_normal(45) * 0.7)
    # steps per restart adapt to time: ~1600 steps each
    for ri, p0 in enumerate(inits):
        if time.time() > sym_deadline:
            break
        # vary steps slightly
        steps = 1600
        # last restarts shorter if time low
        remain = sym_deadline - time.time()
        left = len(inits) - ri
        if remain / left < 4.0:
            steps = max(600, int(remain / left * 250))
        try:
            bp, bX, bv = _adam_sym(H, p0, steps=steps, K0=200.0, K1=5000.0, lr=0.02)
        except Exception:
            continue
        if bv < best_sym_val:
            best_sym_val = float(bv)
            best_sym_X = bX.copy()
            best_sym_p = bp.copy()
    # if sym failed, use incumbent
    if best_sym_X is None:
        best_sym_X = inc.copy()
        best_sym_val = inc_val
    # ---- release symmetry ----
    # perturb 1e-3 tangent
    rel = best_sym_X.copy()
    pert = rng.standard_normal(rel.shape) * 1e-3
    # project to tangent
    dot = np.sum(pert * rel, axis=1, keepdims=True)
    pert -= dot * rel
    rel = rel + pert
    rel /= np.linalg.norm(rel, axis=1, keepdims=True)
    # full LSE-Adam: allocate time
    # leave ~25s for LP + final
    cur_dead = t0 + LIMIT - 30.0
    # run 2 rounds of LSE with different K/lr if time allows
    released = rel
    released_val = _max_cos(released)
    try:
        # round 1
        rx, rv = _adam_lse_full(released, steps=2500, K0=150.0, K1=2500.0, lr=0.02)
        if rv < released_val:
            released, released_val = rx, rv
        if time.time() < cur_dead - 8:
            rx2, rv2 = _adam_lse_full(released, steps=1500, K0=800.0, K1=6000.0, lr=0.008)
            if rv2 < released_val:
                released, released_val = rx2, rv2
    except Exception:
        pass
    # LP polish released with available time (up to ~20s)
    lp_dead = min(time.time() + 20.0, t0 + LIMIT - 6.0)
    try:
        released = _lp_polish(released, lp_dead)
        released_val = _max_cos(released)
    except Exception:
        pass
    # opportunistic: also try short polish of incumbent if time remains and released not better
    # use remaining time to try 1-2 perturbed incumbent runs
    best_final = released
    best_final_val = released_val
    if inc_val < best_final_val:
        best_final = inc.copy()
        best_final_val = inc_val
    # if time left >10s, try quick incumbent-based improvement
    while time.time() < t0 + LIMIT - 8.0:
        # small perturbation of current best_final and quick Adam
        try:
            sig = 0.01
            trial = best_final + sig * rng.standard_normal(best_final.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            tx, tv = _adam_lse_full(trial, steps=800, K0=300.0, K1=4000.0, lr=0.015)
            if tv < best_final_val:
                # quick LP squeeze 1.5s
                tx = _lp_polish(tx, min(time.time() + 1.5, t0 + LIMIT - 5.0))
                tv = _max_cos(tx)
                if tv < best_final_val:
                    best_final = tx
                    best_final_val = tv
            else:
                # no improvement, break if released already beats inc and time low
                if time.time() > t0 + LIMIT - 12.0:
                    break
        except Exception:
            break
        # only a couple iterations
        if time.time() > t0 + LIMIT - 9.0:
            break
        # avoid infinite loop: at most 3 tries
        break
    # final safety: ensure best vs inc
    if inc_val < best_final_val:
        best_final = inc.copy()
    best_final = best_final / np.linalg.norm(best_final, axis=1, keepdims=True)
    return best_final
