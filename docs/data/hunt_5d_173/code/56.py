import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= m - thr
    cnt = mask.sum(axis=1)
    return m, cnt, G

def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            G = x @ x.T
            cfull = G[iu0, iu1]
            m = float(cfull.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = cfull > thresh
            if int(sel.sum()) > 3000:
                idx = np.where(sel)[0]
                order = np.argsort(cfull[idx])[-3000:]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu0[sel]; J = iu1[sel]
            c = cfull[sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            nv = n * d + 1
            cols = np.concatenate([I[:, None]*d + np.arange(d),
                                   J[:, None]*d + np.arange(d),
                                   np.full((p,1), n*d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2*d+1)
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            try:
                radius *= 0.5
            except Exception:
                break
            continue
    return x

def _greedy_void(frozen, k, rng, n_cand=20000):
    d = frozen.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    s_max = (cand @ frozen.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)

def _lse_block_refine(Y0, frozen, alpha=80.0, maxiter=150):
    k, d = Y0.shape
    Nf = frozen.shape[0]
    iu0, iu1 = np.triu_indices(k, 1) if k > 1 else (np.array([],dtype=int), np.array([],dtype=int))
    ntri = len(iu0)
    Z0 = Y0.copy().ravel()
    # scale to avoid zero
    def fun_grad(Zf):
        Z = Zf.reshape(k, d)
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1e-12
        Y = Z / nr
        C1 = Y @ frozen.T  # k x Nf
        if ntri > 0:
            G2 = Y @ Y.T
            C2 = G2[iu0, iu1]
            m0 = max(float(C1.max()), float(C2.max()))
        else:
            C2 = None
            m0 = float(C1.max())
        E1 = np.exp(alpha*(C1 - m0))
        S = float(E1.sum())
        if C2 is not None:
            E2 = np.exp(alpha*(C2 - m0))
            S += float(E2.sum())
        S = max(S, 1e-300)
        val = m0 + np.log(S)/alpha
        W1 = E1 / S
        gradY = W1 @ frozen  # k x d
        if C2 is not None:
            W2 = E2 / S
            # accumulate
            for t in range(ntri):
                a = iu0[t]; b = iu1[t]; w = W2[t]
                gradY[a] += w * Y[b]
                gradY[b] += w * Y[a]
        # chain through normalization
        dot = (gradY * Y).sum(axis=1, keepdims=True)
        gradZ = (gradY - dot * Y) / nr
        return float(val), gradZ.ravel()
    try:
        res = minimize(lambda z: fun_grad(z), Z0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8, "maxfun": 10000})
        Z = res.x.reshape(k, d)
        return _norm(Z)
    except Exception:
        return Y0.copy()

def _lse_full_refine(X0, alpha=90.0, maxiter=60):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Z0 = X0.copy().ravel()
    def fun_grad(Zf):
        Z = Zf.reshape(n, d)
        nr = np.linalg.norm(Z, axis=1, keepdims=True)
        nr[nr < 1e-12] = 1e-12
        Y = Z / nr
        # compute only upper tri via full matrix (173x173 fine)
        G = Y @ Y.T
        C = G[iu0, iu1]
        m0 = float(C.max())
        E = np.exp(alpha*(C - m0))
        S = float(E.sum())
        S = max(S, 1e-300)
        val = m0 + np.log(S)/alpha
        W = np.zeros((n, n))
        W[iu0, iu1] = E / S
        W = W + W.T
        gradY = W @ Y
        dot = (gradY * Y).sum(axis=1, keepdims=True)
        gradZ = (gradY - dot * Y) / nr
        return float(val), gradZ.ravel()
    try:
        res = minimize(lambda z: fun_grad(z), Z0, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7, "maxfun": 5000})
        Z = res.x.reshape(n, d)
        return _norm(Z)
    except Exception:
        return X0.copy()

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    ENUM_END = t0 + 120.0
    rng_global = np.random.default_rng(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (173, 5):
                an = _norm(a)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            continue
    if best is None:
        best = _norm(rng_global.standard_normal((173, 5)))
        best_val = _max_cos(best)
    base = best.copy()
    base_val = best_val
    try:
        m0, cnt, G0 = _contact_counts(base, thr=1e-9)
    except Exception:
        return _norm(best)
    n, d = 173, 5
    # identify rattler(s)
    try:
        order_asc = np.argsort(cnt)
        rattler = int(order_asc[0])
        # nearest neighbours in cosine to rattler
        sim = G0[rattler].copy()
        sim[rattler] = -2.0
        nn_order = np.argsort(-sim)
        # highest contact order
        hc_order = np.argsort(-cnt)
        hc_order = [int(i) for i in hc_order if int(i) != rattler]
        removal_sets = []
        for k in (4, 8, 12):
            # A: rattler + nearest
            try:
                sA = [rattler] + [int(nn_order[t]) for t in range(k-1)]
                removal_sets.append((k, sA))
            except Exception:
                pass
            # B: rattler + highest-contact
            try:
                sB = [rattler] + hc_order[:k-1]
                removal_sets.append((k, sB))
            except Exception:
                pass
            # C: mix half-half
            try:
                h1 = (k-1)//2
                h2 = (k-1)-h1
                cand = [rattler]
                for t in range(h1):
                    cand.append(int(nn_order[t]))
                # add high-contact not already in cand
                cc = 0
                while len(cand) < k and cc < len(hc_order):
                    v = hc_order[cc]; cc += 1
                    if v not in cand:
                        cand.append(v)
                removal_sets.append((k, cand))
            except Exception:
                pass
        # deduplicate
        seen = set()
        uniq = []
        for k, s in removal_sets:
            key = tuple(sorted(s))
            if key not in seen and len(key) == k:
                seen.add(key)
                uniq.append((k, list(key)))
        removal_sets = uniq
    except Exception:
        removal_sets = []

    idx_counter = 0
    for k, rem in removal_sets:
        if time.time() > ENUM_END or time.time() > DEADLINE - 25.0:
            break
        try:
            rem_arr = np.array(rem, dtype=int)
            mask = np.ones(n, dtype=bool)
            mask[rem_arr] = False
            frozen = base[mask]
            rng = np.random.default_rng(1000 + idx_counter*17 + k*131)
            idx_counter += 1
            # 20k samples + greedy void filling
            try:
                Y0 = _greedy_void(frozen, k, rng, n_cand=20000)
            except Exception:
                continue
            if time.time() > ENUM_END:
                continue
            # SLSQP/log-sum-exp refinement on removed block alone
            try:
                Y1 = _lse_block_refine(Y0, frozen, alpha=50.0, maxiter=120)
                # second sharper stage
                if time.time() < ENUM_END and time.time() < DEADLINE - 20:
                    Y1 = _lse_block_refine(Y1, frozen, alpha=120.0, maxiter=80)
            except Exception:
                Y1 = Y0
            try:
                Xcand = base.copy()
                Xcand[rem_arr] = _norm(Y1)
                # frozen-block quality check: max among rebuilt interactions
                Gc = Xcand @ Xcand.T
                np.fill_diagonal(Gc, -2.0)
                # only check rebuilt-involved pairs to decide to proceed
                # compute max over pairs with at least one in rem
                mv = -2.0
                for a in rem_arr:
                    row = Gc[a]
                    rmax = float(row.max())
                    if rmax > mv:
                        mv = rmax
                # Only proceed if below or near record (allow small slack)
                if mv > base_val + 0.004:
                    # still try one more random greedy with different seed quickly? skip polish
                    continue
            except Exception:
                continue
            # short smooth anneal on all points (unfrozen)
            try:
                if time.time() < DEADLINE - 18:
                    Xcand = _lse_full_refine(Xcand, alpha=90.0, maxiter=50)
            except Exception:
                pass
            # final LP polish short
            try:
                rem_time = DEADLINE - time.time()
                pt = 3.0
                if rem_time < 25:
                    pt = max(0.8, rem_time - 8.0)
                pt = min(pt, 4.0)
                if pt > 0.4:
                    Xcand = _lp_polish(Xcand, pt)
            except Exception:
                pass
            try:
                v = _max_cos(_norm(Xcand))
                if v < best_val - 1e-12:
                    best_val = v
                    best = _norm(Xcand).copy()
            except Exception:
                continue
        except Exception:
            continue
    # final safety polish on best if time remains
    try:
        if time.time() < DEADLINE - 10.0:
            rem = DEADLINE - time.time() - 2.0
            pt = min(6.0, max(1.0, rem*0.2))
            cand = _lp_polish(best, pt)
            v = _max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass
    try:
        best = _norm(best)
    except Exception:
        try:
            best = _norm(np.load("best.npy").astype(float))
        except Exception:
            best = _norm(np.random.default_rng(1).standard_normal((173,5)))
    return best
