import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU = None
def _get_iu(n):
    global _IU
    if _IU is None or _IU[0].shape[0] != n*(n-1)//2:
        _IU = np.triu_indices(n, 1)
    return _IU

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _refine_void(v0, others, steps=160, tau0=0.008, tau1=0.001, lr0=0.06):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    lr = float(lr0)
    ltau = np.log(tau1 / tau0)
    best = float((others @ v).max())
    bestv = v.copy()
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        if tau < 1e-12:
            tau = 1e-12
        dots = others @ v
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ others
        gv = float(g @ v)
        gt = g - gv * v
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-13:
            break
        gt /= nrm
        v2 = v - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            v = v2
            lr = min(lr * 1.06, 0.09)
            if m2 < best:
                best = m2
                bestv = v.copy()
        else:
            lr *= 0.5
            if lr < 1e-7:
                break
    return bestv, best

def _find_void(others, rng, n_cand=60000, n_ref=20):
    d = others.shape[1]
    C = rng.standard_normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    worst = np.empty(n_cand, dtype=np.float64)
    chunk = 10000
    for a in range(0, n_cand, chunk):
        b = min(a + chunk, n_cand)
        D = C[a:b] @ others.T
        worst[a:b] = D.max(axis=1)
    if n_ref >= n_cand:
        idx = np.argsort(worst)
    else:
        idx = np.argpartition(worst, n_ref)[:n_ref]
    bestv = None
    bestm = 1e9
    for k in idx:
        v0 = C[int(k)]
        vr, mr = _refine_void(v0, others)
        if mr < bestm:
            bestm = mr
            bestv = vr.copy()
    bestv /= np.linalg.norm(bestv)
    return bestv

def _shifted_adam(X0, p_list, shift, steps_total, lr0, lr1, deadline=1e18):
    n, d = X0.shape
    iu0, iu1 = _get_iu(n)
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tglob = 0
    total = max(1, steps_total)
    per = []
    L = len(p_list)
    base = steps_total // L
    rem = steps_total - base * L
    for li in range(L):
        per.append(base + (1 if li < rem else 0))
    for li, p in enumerate(p_list):
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        pm1 = float(p - 1.0)
        for s in range(per[li]):
            if time.time() > deadline:
                break
            frac = tglob / max(1, total - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            G = X @ X.T
            c = G[iu0, iu1]
            pos = c > shift
            if not np.any(pos):
                tglob += 1
                continue
            sc = c[pos] - shift
            smax = float(sc.max())
            if not np.isfinite(smax) or smax <= 0:
                tglob += 1
                continue
            ratio = sc / smax
            np.clip(ratio, 1e-300, 1.0, out=ratio)
            epos = np.exp(pm1 * np.log(ratio))
            esum = float(epos.sum())
            if not np.isfinite(esum) or esum <= 0:
                tglob += 1
                continue
            w = np.zeros_like(c)
            w[pos] = epos / esum
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            grad = W @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-15:
                tglob += 1
                continue
            tglob += 1
            tt += 1
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            X = X - step
            X = _norm(X)
        if time.time() > deadline:
            break
    return X

def _hardK(X0, iters, K, step0, deadline=1e18):
    n, d = X0.shape
    iu0, iu1 = _get_iu(n)
    P = len(iu0)
    K = int(max(1, min(K, P)))
    X = _norm(X0.copy())
    best = _maxcos(X)
    bestX = X.copy()
    step = float(step0)
    grad = np.empty_like(X)
    for it in range(iters):
        if (it % 25 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        if K < P:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(P)
        I = iu0[idx]; J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.einsum('ij,ij->i', grad, X)[:, None]
        gt = grad - dot * X
        Xt = _norm(X - step * gt)
        vv = _maxcos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.25, 0.008)
        else:
            step *= 0.55
            if step < 1e-12:
                step = float(step0) * 0.3
                if time.time() > deadline:
                    break
    return bestX, best

def _lp_polish(x0, seconds, r_init=8e-4):
    x = _norm(x0.copy())
    n, d = x.shape
    best = _maxcos(x)
    xb = x.copy()
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        Gf = x @ x.T
        call = Gf[iu]
        mm = float(call.max())
        sel = call > mm - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4000:
            idx = np.argpartition(-call, 4000)[:4000]
            msk = np.zeros_like(sel)
            msk[idx] = True
            sel = msk
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        csub = call[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _maxcos(y)
        if vv < best:
            best = vv
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)

    X0 = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=float)
            if A.shape != (93, 13):
                continue
            A = _norm(A)
            vv = _maxcos(A)
            if vv < best_val:
                best_val = vv
                X0 = A.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal(size=(93, 13))
        X0 = _norm(X0)
        best_val = _maxcos(X0)
    x_best = X0.copy()
    gbest = best_val

    # diagnostics: contacts at 1e-9, 1e-7, 1e-5
    G = x_best @ x_best.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    cmax = float(G.max())
    contact9 = (G > cmax - 1e-9)
    # per-point contacts (upper count *2 approx via full)
    n = 93
    contacts = np.zeros(n, dtype=int)
    iu0, iu1 = _get_iu(n)
    c = G[iu0, iu1]
    tight = c > cmax - 1e-9
    for a, b, t in zip(iu0, iu1, tight):
        if t:
            contacts[int(a)] += 1
            contacts[int(b)] += 1
    ratt = np.where(contacts == 0)[0]
    # sort rattlers by increasing row-max (loosest first = smallest rowmax)
    if len(ratt) == 0:
        order = np.argsort(rowmax)[:19]
    else:
        order = ratt[np.argsort(rowmax[ratt])]
    hub = int(np.argmax(contacts))

    trials = []
    for k in [1, 2, 3, 5, 8, 13, 19]:
        # variant A: k loosest rattlers
        if len(order) >= k:
            trials.append(list(map(int, order[:k])))
        elif len(order) > 0:
            # pad with next loosest overall
            need = k
            base = list(map(int, order))
            idx_sorted = np.argsort(rowmax)
            for ii in idx_sorted:
                ii = int(ii)
                if ii not in base and ii != hub:
                    base.append(ii)
                if len(base) >= need:
                    break
            trials.append(base[:need])
        # variant B: hub + (k-1) loosest (for k>=1)
        if k == 1:
            trials.append([hub])
        else:
            base2 = [hub]
            for ii in order:
                ii = int(ii)
                if ii == hub:
                    continue
                base2.append(ii)
                if len(base2) >= k:
                    break
            if len(base2) < k:
                idx_sorted = np.argsort(rowmax)
                for ii in idx_sorted:
                    ii = int(ii)
                    if ii not in base2:
                        base2.append(ii)
                    if len(base2) >= k:
                        break
            if len(base2) == k and base2 not in trials:
                trials.append(base2)

    # time budget per trial
    for ti, dell in enumerate(trials):
        now = time.time()
        remain = deadline - now
        left_trials = len(trials) - ti
        if remain < 6:
            break
        # allocate proportional to k
        k = len(dell)
        per_trial = min(remain / max(1, left_trials), 6.0 + k * 1.1)
        per_trial = max(per_trial, 3.0)
        trial_end = min(now + per_trial, deadline - 4.0)
        try:
            keep = np.ones(n, dtype=bool)
            keep[np.array(dell, dtype=int)] = False
            Xcur = x_best[keep].copy()
            ok = True
            # greedy reinsert
            for ins in range(k):
                if time.time() > trial_end - 2.0:
                    ok = False
                    break
                vnew = _find_void(Xcur, rng, n_cand=60000, n_ref=20)
                Xcur = np.vstack([Xcur, vnew])
                Xcur = _norm(Xcur)
                # shifted p-norm Adam few hundred steps
                sdead = min(trial_end - 1.0, time.time() + 1.2)
                if time.time() < sdead:
                    Xcur = _shifted_adam(Xcur, [32, 64, 128], 0.20, 210, 3e-3, 3e-4, deadline=sdead)
                else:
                    Xcur = _norm(Xcur)
            if not ok:
                continue
            if Xcur.shape[0] != 93:
                continue
            # top-K hard-max descent K=150
            hdead = min(trial_end - 0.6, deadline - 3.0)
            Xcur, hv = _hardK(Xcur, 320, 150, 0.004, deadline=hdead)
            # short LP polish
            lsec = max(0.4, min(1.6, trial_end - time.time() - 0.2))
            if lsec > 0.25 and time.time() < trial_end:
                Xcur, hv2 = _lp_polish(Xcur, lsec)
                hv = min(hv, hv2)
            else:
                hv = _maxcos(Xcur)
            if hv < gbest - 1e-12 and np.all(np.isfinite(Xcur)):
                gbest = float(hv)
                x_best = _norm(Xcur.copy())
        except Exception:
            continue
        # early exit if big win? keep going to use time
        if time.time() > deadline - 4.0:
            break

    x_best = _norm(x_best)
    # tiny shrink guard not needed; ensure finite
    return x_best
