import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts_stress(X, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    np.fill_diagonal(G, 1.0)
    cm = G > (m - tol)
    contacts = np.sum(cm, axis=1) - 1
    contacts = contacts.astype(int)
    # summed stress: sum of dots of contacting neighbours
    stress = np.sum(np.where(cm, G, 0.0), axis=1)
    return m, contacts, stress, G

def _lp_polish(X, seconds, init_radius=0.015, cap=2500):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.15 > seconds and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > (m - thresh)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            kth = np.partition(c_all, -cap)[-cap]
            sel = c_all >= kth
            cnt = int(np.sum(sel))
            if cnt > cap + 500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:cap]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_anneal(X, iters=350, p_start=40, p_end=250, lr=0.04):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = float(Gp.max())
        if m <= 1e-12:
            break
        # avoid overflow: clip ratio
        with np.errstate(over='ignore', under='ignore'):
            W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        nm = float(np.mean(np.linalg.norm(grad_t, axis=1))) + 1e-12
        step = lr * (1.0 - 0.4 * frac) / nm
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_init(F, k, rng, ncand=8000, pool=600):
    d = F.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ F.T
    smax = s.max(axis=1)
    order = np.argsort(smax)
    # deepest holes first
    pool_idx = order[:pool]
    Pc = cand[pool_idx]
    Ps = smax[pool_idx]
    # greedy farthest among pool for diversity (joint-biased, not one-at-a-time final)
    chosen = []
    chosen_idx = []
    # first is deepest
    chosen.append(Pc[0])
    chosen_idx.append(0)
    # precompute pool-pool dots for speed
    Gpp = Pc @ Pc.T
    for i in range(1, k):
        # score for each pool candidate: max(ps, max dot to chosen)
        # compute max dot to chosen set
        if len(chosen) == 1:
            dc = Gpp[:, 0]
        else:
            dc = Gpp[:, chosen_idx].max(axis=1)
        score = np.maximum(Ps, dc)
        # mask already chosen
        score[chosen_idx] = 1e9
        bi = int(np.argmin(score))
        chosen.append(Pc[bi])
        chosen_idx.append(bi)
        if len(chosen) >= k:
            break
    # if k > pool (never), pad random
    V = np.stack(chosen[:k], axis=0)
    if V.shape[0] < k:
        extra = rng.standard_normal((k - V.shape[0], d))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        V = np.concatenate([V, extra], axis=0)
    return V

def _joint_hinge(V0, F, rng, steps=2200, tau0=0.02, tau1=0.0025, lr=0.05):
    V = V0 / np.linalg.norm(V0, axis=1, keepdims=True)
    k, d = V.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        Gvf = V @ F.T
        Gvv = V @ V.T
        np.fill_diagonal(Gvv, -1.0)
        m = float(max(Gvf.max(), Gvv.max()))
        # softmax weights
        Wvf = np.exp(np.clip((Gvf - m) / tau, -50, 0))
        Wvv = np.exp(np.clip((Gvv - m) / tau, -50, 0))
        np.fill_diagonal(Wvv, 0.0)
        grad = Wvf @ F + Wvv @ V
        dot = np.sum(grad * V, axis=1, keepdims=True)
        grad_t = grad - dot * V
        nm = float(np.mean(np.linalg.norm(grad_t, axis=1))) + 1e-12
        step = lr * (1.0 - 0.3 * frac) / nm
        V = V - step * grad_t
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        V /= nrm
        # DE-style joint move every 550 steps: try to replace worst point
        if (s + 1) % 550 == 0 and (s + 1) < steps:
            # worst point = largest max dot
            mvf = Gvf.max(axis=1)
            mvv = Gvv.max(axis=1)
            mtot = np.maximum(mvf, mvv)
            wi = int(np.argmax(mtot))
            # generate trial candidates biased to holes
            nt = 1500
            dd = F.shape[1]
            T = rng.standard_normal((nt, dd))
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            St = T @ F.T
            stm = St.max(axis=1)
            # also vs other V (excluding wi)
            Vrest = np.delete(V, wi, axis=0)
            if Vrest.shape[0] > 0:
                Sr = T @ Vrest.T
                srm = Sr.max(axis=1)
                stm = np.maximum(stm, srm)
            bi = int(np.argmin(stm))
            if float(stm[bi]) < float(mtot[wi]) - 1e-4:
                V[wi] = T[bi]
    return V / np.linalg.norm(V, axis=1, keepdims=True)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cands = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    scores = [_max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    xb = cands[bi].copy()
    bb = float(scores[bi])
    # light initial polish only if time allows and not already polished
    # do 2s to ensure baseline
    rem = BUDGET - (time.time() - t0)
    if rem > 30:
        xb = _lp_polish(xb, seconds=2.0, init_radius=1e-3)
        bb = _max_cos(xb)

    # deletion size schedule: 8,14,22 repeated with variations
    schedule = [22, 14, 8, 18, 12, 22, 8, 14, 20, 10, 16, 22, 14, 8]
    si = 0
    # keep a second-best for diversity occasionally
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 9.0:
            break
        K = schedule[si % len(schedule)]
        si += 1
        # estimate LP budget for this cycle
        # reserve final 6s polish
        cycles_left = 3
        lp_budget = 2.2
        if rem < 20:
            lp_budget = 1.2
        # --- analyse jammed core on xb ---
        try:
            m0, contacts, stress, _ = _contacts_stress(xb, tol=1e-7)
        except Exception:
            continue
        # rattlers: contacts<=0 ; also treat 1-contact as loose
        ratt = np.where(contacts <= 0)[0].tolist()
        # order by contacts desc, stress desc
        order = np.argsort(-contacts * 1000.0 - stress)
        # build delete set of size K containing all rattlers if possible
        D = set(ratt)
        for idx in order:
            if len(D) >= K:
                break
            D.add(int(idx))
        # if rattlers more than K, truncate to K worst? keep K
        D = list(D)[:K]
        # ensure exactly K (pad if needed)
        if len(D) < K:
            for idx in order:
                if int(idx) not in D:
                    D.append(int(idx))
                if len(D) >= K:
                    break
        D = np.array(sorted(D), dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[D] = False
        Ridx = np.where(keep_mask)[0]
        F0 = xb[Ridx].copy()
        # briefly relax remainder (~160 pts) to open holes
        # small smooth anneal on remainder alone
        try:
            Fr = _smooth_anneal(F0, iters=120, p_start=30, p_end=80, lr=0.03)
        except Exception:
            Fr = F0
        # check remainder internal max stays reasonable; if it blew up, keep F0
        try:
            mr = _max_cos(Fr)
            if mr > bb + 0.005:
                Fr = F0
        except Exception:
            Fr = F0
        # jointly optimize deleted points against frozen remainder
        k = len(D)
        try:
            V0 = _hole_init(Fr, k, rng, ncand=8000, pool=600)
            Vopt = _joint_hinge(V0, Fr, rng, steps=2000, tau0=0.02, tau1=0.0025, lr=0.05)
        except Exception:
            continue
        # assemble
        Xnew = np.empty_like(xb)
        Xnew[Ridx] = Fr
        Xnew[D] = Vopt
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        # unfreeze: smooth exponential-repulsion anneal 40->250
        try:
            Xnew = _smooth_anneal(Xnew, iters=350, p_start=40, p_end=250, lr=0.04)
        except Exception:
            pass
        # time-capped LP polish
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 7.5:
            # no LP, just evaluate
            try:
                v = _max_cos(Xnew)
            except Exception:
                continue
            if v < bb:
                xb, bb = Xnew, v
            continue
        try:
            Xp = _lp_polish(Xnew, seconds=float(min(lp_budget, rem2 - 6.5)), init_radius=0.02)
            v = _max_cos(Xp)
        except Exception:
            continue
        if v < bb - 1e-12:
            xb, bb = Xp, v
            # extra squeeze on improvement if time allows
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 12:
                try:
                    Xp2 = _lp_polish(xb, seconds=1.2, init_radius=8e-4)
                    v2 = _max_cos(Xp2)
                    if v2 < bb:
                        xb, bb = Xp2, v2
                except Exception:
                    pass
        # else discard (stay in best basin, try different K next time)
        # add small random perturbation to rng state even on failure (already via rng draws)
    # final safety polish
    remf = BUDGET - (time.time() - t0)
    if remf > 2.5:
        try:
            Xf = _lp_polish(xb, seconds=min(remf - 0.8, 5.0), init_radius=8e-4)
            vf = _max_cos(Xf)
            if vf < bb:
                xb, bb = Xf, vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
