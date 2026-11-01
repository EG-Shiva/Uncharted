import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contact_loads(x, m=None, beta=300.0):
    G = x @ x.T
    np.fill_diagonal(G, -10.0)
    if m is None:
        m = float(np.max(G))
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    loads = np.sum(E, axis=1)
    return loads, G, m

def _lp_polish(x, deadline, r0=8e-4, cap_p=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.25:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argpartition(-csel, cap_p)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        improved = False
        for scale in (1.0, 0.3, 0.1):
            y = x + step * scale
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best - 1e-15:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return x

def _init_k_from_holes(core, k, rng, ncand=50000, pool_top=2000):
    d = core.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_core = cand @ core.T
    smax = s_core.max(axis=1)
    order = np.argsort(smax)
    take = min(pool_top, ncand)
    pool = cand[order[:take]]
    spool = smax[order[:take]]
    # randomize first pick among top 20 for diversity
    ntop = min(20, take)
    first = int(rng.integers(ntop))
    chosen = [first]
    chosen_set = set(chosen)
    # greedy completion
    dots_vs_chosen = None
    for _ in range(1, k):
        C = pool[chosen]  # (c,d)
        D = pool @ C.T  # (take, c)
        dm = D.max(axis=1)
        score = np.maximum(spool, dm)
        # mask already chosen
        for c in chosen:
            score[c] = 2.0
        # pick among best few randomly to diversify
        kbest = min(5, take - len(chosen))
        part = np.argpartition(score, kbest - 1)[:kbest]
        # choose best with small prob bias, else random among them
        if rng.random() < 0.7:
            nxt = int(part[np.argmin(score[part])])
        else:
            nxt = int(rng.choice(part))
        chosen.append(nxt)
    Y0 = pool[np.array(chosen)]
    # tiny jitter so L-BFGS not exactly symmetric
    Y0 = Y0 + 1e-4 * rng.standard_normal(Y0.shape)
    Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
    return Y0

def _opt_k_points(core, Yinit, betas=(100.0, 300.0, 600.0), maxiter=250):
    k, d = Yinit.shape
    nc = core.shape[0]
    Ycur = Yinit.copy()
    for beta in betas:
        def fg(yf):
            Y = yf.reshape(k, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G1 = X @ core.T  # k x nc
            G2 = X @ X.T
            m1 = float(np.max(G1)) if G1.size else -2.0
            # max of off-diagonal G2
            m2 = -2.0
            if k > 1:
                tmp = G2.copy()
                np.fill_diagonal(tmp, -2.0)
                m2 = float(np.max(tmp))
            m = m1 if m1 > m2 else m2
            E1 = np.exp(beta * (G1 - m))
            E2full = np.exp(beta * (G2 - m))
            # zero diagonal for E2
            np.fill_diagonal(E2full, 0.0)
            # for k>1, E2 includes both (i,j) and (j,i); sum counts twice, fine
            # to avoid double counting bias, halve? LSE wants each unordered pair once.
            # Use half weight: divide E2 by 2 in sum and gradient
            S = float(np.sum(E1) + 0.5 * float(np.sum(E2full))) + 1e-300
            F = m + np.log(S) / beta
            W1 = E1 / S
            W2 = (0.5 * E2full) / S
            # symmetrize (already symmetric)
            W2 = W2 + W2.T  # now each unordered counted with full weight split? careful
            # Actually 0.5*E2 sym + transpose gives E2/S for off-diag. Correct.
            # Gradient
            GX = W1 @ core + W2 @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return F, GY.ravel()
        y0 = Ycur.ravel().copy()
        try:
            r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-7,
                                  'maxfun': maxiter * 3})
            Ycur = r.x.reshape(k, d)
            Ycur /= np.linalg.norm(Ycur, axis=1, keepdims=True)
        except Exception:
            break
    return Ycur

def solve():
    t0 = time.time()
    DEADLINE = 178.0
    rng = np.random.default_rng(7)
    X0 = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (68, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal((68, 12))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val = _max_cos(X0)
    global_best = X0.copy()
    global_val = best_val
    n, d = 68, 12
    k_choices = [3, 4, 5, 6, 8]
    it = 0
    # reserve final polish ~12s
    HOP_END = t0 + 165.0
    while time.time() < HOP_END:
        it += 1
        remain = HOP_END - time.time()
        if remain < 1.0:
            break
        k = k_choices[int(rng.integers(len(k_choices)))]
        # occasionally bias to smaller k early, larger later
        # rank contact loads
        try:
            loads, Gfull, mcur = _contact_loads(global_best, beta=300.0)
        except Exception:
            loads = np.ones(n)
            Gfull = global_best @ global_best.T
            mcur = _max_cos(global_best)
        rsel = rng.random()
        if rsel < 0.40:
            del_idx = np.argsort(-loads)[:k]
        elif rsel < 0.75:
            # point + near-contact neighbours
            # pick seed among top 15 weighted by load
            top15 = np.argsort(-loads)[:15]
            w = loads[top15] - loads[top15].min() + 1e-9
            w = w / w.sum()
            seed = int(rng.choice(top15, p=w))
            # neighbours: largest inner products
            row = Gfull[seed].copy()
            row[seed] = -10
            nbrs = np.argsort(-row)[:k-1]
            del_idx = np.array([seed] + list(nbrs), dtype=int)
        else:
            # sample proportional to load
            w = loads - loads.min() + 0.05 * (loads.max() - loads.min() + 1e-9)
            w = np.maximum(w, 1e-12)
            w = w / w.sum()
            del_idx = rng.choice(n, size=k, replace=False, p=w)
        del_idx = np.asarray(del_idx, dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[del_idx] = False
        core = global_best[mask]
        # dense hole search init
        try:
            Yinit = _init_k_from_holes(core, k, rng, ncand=50000, pool_top=2000)
        except Exception:
            continue
        try:
            Yopt = _opt_k_points(core, Yinit, betas=(100.0, 300.0, 600.0), maxiter=220)
        except Exception:
            continue
        Xtry = np.vstack([core, Yopt])
        v_try = _max_cos(Xtry)
        # quick reject: if far worse, skip polish
        if v_try > global_val + 0.004:
            continue
        # decide polish budget
        if v_try < global_val - 2e-6:
            budget = min(3.0, max(1.2, remain * 0.15))
        elif v_try < global_val + 0.0012:
            budget = min(1.6, max(0.8, remain * 0.06))
            # only probe every few times to save time
            if (it % 2 == 0) and v_try > global_val:
                continue
        else:
            # occasional probe
            if it % 7 != 0:
                continue
            budget = 1.0
        if time.time() + budget > HOP_END + 2:
            budget = max(0.6, HOP_END - time.time())
        try:
            Xpol = _lp_polish(Xtry, deadline=time.time() + budget)
        except Exception:
            continue
        v_pol = _max_cos(Xpol)
        if v_pol < global_val - 1e-12:
            global_best = Xpol
            global_val = v_pol
        # extra: if surgery found better raw but polish didn't beat, try one more short full LSE? skip
    # final squeeze
    time_left = (t0 + DEADLINE) - time.time()
    if time_left > 3.0:
        try:
            global_best = _lp_polish(global_best, deadline=time.time() + min(time_left - 1.0, 14.0))
        except Exception:
            pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    global_best = np.nan_to_num(global_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(global_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    global_best = global_best / nr
    return global_best
