import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, r0=2e-3, cap=1300):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 300:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    T = np.empty((n, 4, 3))
    E = np.eye(4)
    for i in range(n):
        xi = x[i]
        m = int(np.argmax(np.abs(xi)))
        cols = [k for k in range(4) if k != m]
        M = np.stack([E[:, k] - xi[k] * xi for k in cols], axis=1)
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _rotation_modes(T, x):
    n = x.shape[0]
    N = n * 3
    R = []
    for a in range(4):
        for b in range(a + 1, 4):
            A = np.zeros((4, 4))
            A[a, b] = 1.0
            A[b, a] = -1.0
            V = np.empty(N)
            for i in range(n):
                dxi = A @ x[i]
                V[i * 3:(i + 1) * 3] = T[i].T @ dxi
            R.append(V)
    R = np.stack(R, axis=1)
    Q, _ = np.linalg.qr(R)
    return Q

def _build_rigidity(x, T, I, J):
    p = len(I)
    n = x.shape[0]
    N = n * 3
    Rmat = np.empty((p, N))
    for r in range(p):
        a = int(I[r]); b = int(J[r])
        ca = T[a].T @ x[b]
        cb = T[b].T @ x[a]
        row = np.zeros(N)
        row[a * 3:(a + 1) * 3] = ca
        row[b * 3:(b + 1) * 3] = cb
        Rmat[r] = row
    return Rmat

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    HARD = 168.0
    FINAL_RESERVE = 20.0
    TOL_EXCEED = 2e-4
    n, d = 163, 4
    xb = None; xr = None
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            if c is not None:
                c = np.asarray(c, dtype=float)
                if c.shape == (163, 4):
                    c = _norm(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
                    if np.all(np.isfinite(c)) and np.all(np.linalg.norm(c, axis=1) > 1e-9):
                        cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.normal(size=(163, 4))))
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    iu0, iu1 = np.triu_indices(n, 1)

    def flex_modes_for(x, m0):
        G = x @ x.T
        c_all = G[iu0, iu1]
        slack = m0 - c_all
        sel9 = np.where(slack < 1e-9)[0]
        sel7 = np.where(slack < 1e-7)[0]
        # use 1e-9 primarily, fall back to 1e-7 if too few
        use = sel9 if sel9.size >= 250 else sel7
        if use.size < 10:
            # take tightest 400
            use = np.argsort(slack)[:400]
        I = iu0[use]; J = iu1[use]
        T = _tangent_bases(x)
        Qrot = _rotation_modes(T, x)
        Rmat = _build_rigidity(x, T, I, J)
        N = n * 3
        # eigen of H = R^T R
        try:
            H = Rmat.T @ Rmat
            H = 0.5 * (H + H.T)
            w, V = np.linalg.eigh(H)
        except Exception:
            return None, None, None, None
        # w ascending, V columns
        K = min(30, N)
        cand_modes = V[:, :K].T.copy()  # K x N
        flex = []
        for k in range(K):
            v = cand_modes[k]
            # remove rotation component
            coeff = Qrot.T @ v
            v2 = v - Qrot @ coeff
            nv = float(np.linalg.norm(v2))
            if nv < 1e-9:
                continue
            v2 = v2 / nv
            # skip if it is essentially a rotation (small residual) and eigenvalue large?
            # keep near-null: eigenvalue small or residual large
            # eigenvalue:
            ev = float(w[k])
            # keep if ev < 0.5 or residual significant; be permissive for bottom modes
            if ev > 2.0 and k < 5:
                pass
            if nv < 0.15:
                continue
            flex.append(v2)
        if len(flex) < 4:
            # take more modes
            for k in range(K, min(K+30, N)):
                try:
                    v = V[:, k]
                except Exception:
                    break
                coeff = Qrot.T @ v
                v2 = v - Qrot @ coeff
                nv = float(np.linalg.norm(v2))
                if nv < 0.2:
                    continue
                flex.append(v2 / nv)
                if len(flex) >= 12:
                    break
        if len(flex) == 0:
            return None, None, None, None
        flex = np.stack(flex, axis=0)  # F x N
        return flex, T, Qrot, (I, J)

    # initial flex set from incumbent
    incumbent = best.copy()
    incumbent_val = best_val
    flex, Tcur, _, _ = flex_modes_for(incumbent, incumbent_val)
    if flex is None:
        # fallback: just polish
        left = HARD - (time.time() - t0) - 1.0
        if left > 2:
            best = _lp_polish(best, seconds=min(left, 25.0))
        return _norm(best)
    Fnum = flex.shape[0]
    n_modes = 32
    alphas = np.array([0.004, 0.010, 0.022, 0.045, 0.080, 0.120, 0.170], dtype=float)
    # precompute second-shell indices for incumbent
    G0 = incumbent @ incumbent.T
    c0 = G0[iu0, iu1]
    slack0 = incumbent_val - c0
    second_mask = (slack0 >= 1e-9) & (slack0 < 5e-4)
    # if too few, widen
    if int(np.count_nonzero(second_mask)) < 30:
        second_mask = (slack0 >= 1e-9) & (slack0 < 3e-3)
    second_idx = np.where(second_mask)[0]
    # also keep second shell up to 1e-7 variant: include 1e-7..2e-4 explicitly
    # line-search cycles
    for trial in range(n_modes):
        if time.time() - t0 > HARD - FINAL_RESERVE - 2.0:
            break
        # sample random combination of remaining near-null modes
        coef = rng.normal(size=(Fnum,))
        coef /= (np.linalg.norm(coef) + 1e-12)
        v = coef @ flex  # N,
        # normalize to max per-point norm = 1
        Vmat = v.reshape(n, 3)
        pn = np.linalg.norm(Vmat, axis=1).max()
        if pn < 1e-12:
            continue
        v = v / pn
        Vmat = v.reshape(n, 3)
        # displacement in ambient coords: Dx[i] = T[i] @ Vmat[i]
        # Tcur is n x 4 x 3
        Dx = np.einsum('nij,nj->ni', Tcur, Vmat)
        # also try opposite sign on alternate trials to cover both directions
        if trial % 2 == 1:
            Dx = -Dx
        found_alpha = None
        found_y = None
        last_feasible_y = None
        last_feasible_a = None
        for a in alphas:
            if time.time() - t0 > HARD - FINAL_RESERVE - 1.0:
                break
            y = _norm(incumbent + a * Dx)
            Gy = y @ y.T
            cy = Gy[iu0, iu1]
            m_a = float(np.max(cy))
            if m_a > incumbent_val + TOL_EXCEED:
                break
            last_feasible_y = y
            last_feasible_a = a
            # check activation of second shell
            if second_idx.size > 0:
                sec_max = float(np.max(cy[second_idx]))
                # becomes active: within 1e-5 of incumbent or within 2e-6 of new max
                if sec_max >= incumbent_val - 2e-5 or sec_max >= m_a - 3e-6:
                    found_alpha = a
                    found_y = y
                    break
        cand_y = found_y if found_y is not None else None
        # if no activation but feasible, try the largest feasible as forced exchange (only every 4th trial to save time)
        if cand_y is None and last_feasible_y is not None and (trial % 4 == 0):
            # only if it moved enough: check change in max pairs? just take mid alpha
            cand_y = last_feasible_y
        if cand_y is None:
            continue
        # short trust-region LP polish, retain only if improves
        remain = HARD - FINAL_RESERVE - (time.time() - t0)
        if remain < 1.0:
            break
        try:
            yp = _lp_polish(cand_y, seconds=min(1.2, remain), r0=2e-3)
        except Exception:
            continue
        vpol = _max_cos(yp)
        if vpol < best_val - 1e-12:
            best = yp.copy()
            best_val = vpol
            incumbent = yp.copy()
            incumbent_val = vpol
            # rebuild flex around new incumbent if time permits
            if time.time() - t0 < HARD - FINAL_RESERVE - 8.0:
                try:
                    nflex, nT, _, _ = flex_modes_for(incumbent, incumbent_val)
                    if nflex is not None:
                        flex, Tcur = nflex, nT
                        Fnum = flex.shape[0]
                        G0 = incumbent @ incumbent.T
                        c0 = G0[iu0, iu1]
                        slack0 = incumbent_val - c0
                        second_mask = (slack0 >= 1e-9) & (slack0 < 5e-4)
                        if int(np.count_nonzero(second_mask)) < 30:
                            second_mask = (slack0 >= 1e-9) & (slack0 < 3e-3)
                        second_idx = np.where(second_mask)[0]
                except Exception:
                    pass
        # strict time check
        if time.time() - t0 > HARD - FINAL_RESERVE:
            break
    # final reserve: polish best
    try:
        left = HARD - (time.time() - t0) - 0.5
        if left > 1.5:
            best = _lp_polish(best, seconds=min(left, FINAL_RESERVE - 1.0), r0=1.5e-3)
    except Exception:
        pass
    best = _norm(np.nan_to_num(best))
    return best
