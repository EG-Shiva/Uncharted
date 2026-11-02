import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _sparsify_quench(X, cap, steps, lr0, lr1, tau, s_sig, w_cnt, mu=0.85, clip=0.03, rng=None, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    thr = cap - 0.002
    for it in range(steps):
        f = it / max(1, steps - 1)
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # high-temp soft-max weights
        # subtract max for stability
        E = np.exp(np.clip((G - m) / tau, -60, 0.0))
        np.fill_diagonal(E, 0.0)
        se = float(E.sum())
        if not np.isfinite(se) or se <= 0:
            Ws = np.zeros_like(G)
        else:
            Ws = E / se
        # smooth count weights: derivative of sigmoid
        z = (G - thr) / s_sig
        z = np.clip(z, -30, 30)
        sig = 1.0 / (1.0 + np.exp(-z))
        dSig = sig * (1.0 - sig) / s_sig
        np.fill_diagonal(dSig, 0.0)
        W = Ws + w_cnt * dSig
        # symmetrize (already symmetric)
        Gr = W @ X
        Gr = Gr + Gr  # factor 2, folded into lr
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        if noise > 0 and rng is not None:
            X = X + noise * (1.0 - f) * rng.standard_normal(X.shape)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itlp = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        itlp += 1
        if itlp > 25:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 500:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 500:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return X

def _lp_duals(X, window=0.002, maxp=400):
    n, d = X.shape
    iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    m = float(c.max())
    order = np.argsort(-c)
    sel_list = []
    for idx in order:
        if c[idx] > m - window and len(sel_list) < maxp:
            sel_list.append(int(idx))
        elif c[idx] <= m - window:
            break
    if len(sel_list) < 5:
        return None, None, None
    sel = np.array(sel_list)
    I = iu0[0][sel]; J = iu0[1][sel]
    p = len(sel); nv = n * d + 1
    radius = 5e-4
    try:
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                      options={"time_limit": 2.0})
        if res.status != 0:
            return I, J, None
        duals = None
        try:
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                duals = np.asarray(res.ineqlin.marginals, dtype=float)
            elif isinstance(res.get('ineqlin', None), dict):
                duals = np.asarray(res['ineqlin'].get('marginals', None), dtype=float)
        except Exception:
            duals = None
        if duals is None or duals.shape[0] != p:
            # fallback: use constraint activity as stress proxy
            duals = -(c[sel] - m)  # larger for tighter
            duals = np.abs(duals) + 1e-6
        else:
            duals = np.abs(duals)
        return I, J, duals
    except Exception:
        return I, J, None

def _low_stress_perturb(X, I, J, duals, eps, rng):
    # Build tangential Jacobian and take smallest singular vectors
    try:
        n, d = X.shape
        p = len(I)
        # tangential projected blocks
        # Jmat p x (n*d)
        N = n * d
        Jmat = np.zeros((p, N), dtype=float)
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            xi = X[i]; xj = X[j]
            # project xj onto tangent at xi and vice versa
            bi = xj - float(xj @ xi) * xi
            bj = xi - float(xi @ xj) * xj
            Jmat[k, i*d:(i+1)*d] = bi
            Jmat[k, j*d:(j+1)*d] = bj
        # SVD: smallest right singular vectors
        # use economy SVD
        U, S, Vh = np.linalg.svd(Jmat, full_matrices=False)
        # Vh shape (p, N) when p<N; last rows are smallest
        nvec = min(3, Vh.shape[0])
        D = np.zeros(N)
        for t in range(nvec):
            v = Vh[-(t+1)]
            # random sign
            if rng.random() < 0.5:
                v = -v
            D += (1.0/(t+1)) * v
        D /= (np.linalg.norm(D) + 1e-12)
        Dm = D.reshape(n, d)
        # make tangent
        dot = np.sum(Dm * X, axis=1, keepdims=True)
        Dm = Dm - dot * X
        dn = np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12
        Dm = Dm / dn
        # identify heavily loaded contacts
        if duals is not None:
            order = np.argsort(-duals)
            K = min(8, p)
            hot_pairs = [(int(I[order[k]]), int(J[order[k]])) for k in range(K)]
        else:
            hot_pairs = [(int(I[k]), int(J[k])) for k in range(min(8, p))]
        hot_idx = sorted(set([a for a, b in hot_pairs] + [b for a, b in hot_pairs]))
        Xn = X.copy()
        mask = np.full(n, 0.25)
        mask[hot_idx] = 1.0
        Xn = Xn + eps * mask[:, None] * Dm
        # extra small random tangent jitter on hot endpoints
        jr = rng.standard_normal((len(hot_idx), d)) * (eps * 0.3)
        # project jitter
        for ti, gi in enumerate(hot_idx):
            v = jr[ti] - float(jr[ti] @ Xn[gi]) * Xn[gi]
            Xn[gi] = Xn[gi] + v
        Xn /= (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
        return Xn
    except Exception:
        return X

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    deadline = t0 + TIME_LIMIT
    rng = np.random.default_rng(777)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        best = z
        best_val = _max_cos(best)
    base_val = float(best_val)
    base_X = best.copy()

    # quick initial polish to have clean reference
    try:
        if deadline - time.time() > 20:
            best = _lp_polish(best, 2.0, deadline - 5.0)
            best_val = _max_cos(best)
            base_val = float(best_val)
            base_X = best.copy()
    except Exception:
        pass

    deltas = [1e-3, 3e-4, 1e-4, 0.0]
    # two homotopy chains with different seeds/noise if time permits
    chain_inits = [0.004, 0.007]
    tried = 0
    for chain_noise in chain_inits:
        if time.time() > deadline - 30:
            break
        # start from base plus noise
        X = base_X + chain_noise * rng.standard_normal((n, d))
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        for di, delta in enumerate(deltas):
            if time.time() > deadline - 25:
                break
            cap = base_val + delta
            # choose sparsification strength per delta
            if delta >= 9e-4:
                steps, lr0, lr1, tau, s_sig, w_cnt = 550, 0.012, 0.0012, 0.005, 2.5e-4, 1.2e-6
                eps_pert = 0.008
            elif delta >= 2e-4:
                steps, lr0, lr1, tau, s_sig, w_cnt = 500, 0.009, 0.0009, 0.004, 2.0e-4, 1.0e-6
                eps_pert = 0.006
            elif delta > 0:
                steps, lr0, lr1, tau, s_sig, w_cnt = 450, 0.007, 0.0007, 0.003, 1.8e-4, 8e-7
                eps_pert = 0.005
            else:
                steps, lr0, lr1, tau, s_sig, w_cnt = 500, 0.006, 0.0005, 0.0025, 1.5e-4, 6e-7
                eps_pert = 0.0
            try:
                X = _sparsify_quench(X, cap, steps, lr0, lr1, tau, s_sig, w_cnt, mu=0.85, clip=0.03, rng=rng, noise=0.0)
            except Exception:
                continue
            # diagnostics (counts) - not strictly needed
            try:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                mcur = float(G.max())
                # near-max pairs within 0.002 of current max
                cnt_pairs = int((G[np.triu_indices(n,1)] > mcur - 0.002).sum())
                _ = cnt_pairs
            except Exception:
                pass
            # LP-dual stress + low-stress perturbation (except final zero stage before polish)
            if delta > 0 and time.time() < deadline - 20:
                try:
                    I, J, duals = _lp_duals(X, window=0.002, maxp=350)
                    if I is not None and len(I) >= 5:
                        X = _low_stress_perturb(X, I, J, duals, eps_pert, rng)
                        # short re-quench after perturbation at same cap
                        X = _sparsify_quench(X, cap, 150, lr0*0.5, lr1*0.5, tau, s_sig, w_cnt, mu=0.82, clip=0.02, rng=None)
                except Exception:
                    pass
            # LP polish after each stage
            try:
                remain = deadline - time.time() - 1.0
                if remain > 4:
                    Xp = _lp_polish(X, min(2.5, remain-1.0), deadline - 0.5)
                    vp = _max_cos(Xp)
                    # keep any intermediate below incumbent
                    if vp < best_val - 1e-12:
                        best_val = vp
                        best = Xp.copy()
                        X = Xp.copy()
                    else:
                        # even if not better, continue chain from polished point if it didn't blow up
                        # allow cap rise: keep Xp if within cap+0.002
                        vraw = _max_cos(X)
                        vpol = vp
                        if vpol < vraw and vpol < base_val + delta + 0.002:
                            X = Xp.copy()
            except Exception:
                pass
            tried += 1
        # end deltas; if we improved, update base
        if best_val < base_val - 1e-12:
            base_val = float(best_val)
            base_X = best.copy()

    # final sharpen
    remain = deadline - time.time() - 0.8
    if remain > 4:
        try:
            polished = _lp_polish(best, min(remain-0.5, 14.0), deadline - 0.4)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
