import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

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

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    # per point counts (symmetric, count once per row excluding diag)
    cnt = C.sum(axis=1)
    return m, cnt, G

def _adam_softmax(X0, steps, tau0, tau1, lr0, lr1, deadline=1e18):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 and tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 and lr0 != lr1 else 0.0
    tt = 0
    for s in range(steps):
        if (s % 80 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        if not np.all(np.isfinite(gt)):
            break
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** tt)
        vh = v / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - d2 * X
        X = X - step_t
        X = _norm(X)
    return X

def _topK(X0, iters, K, step0, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    L = len(iu0)
    K = int(min(max(1, K), L))
    best = _maxcos(X)
    bestX = X.copy()
    step = float(step0)
    grad = np.empty_like(X)
    for it in range(iters):
        if (it % 50 == 0) and time.time() > deadline:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        if K < L:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(L)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if not np.isfinite(ng) or ng < 1e-15:
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = _norm(X - step * gt / (ng / np.sqrt(float(n)) + 1e-12) * 0.05 - step * 0.0 * gt)
        # simpler: normalized direction scaling
        # fallback to plain step if above explodes: use plain gt step
        # recompute plain candidate if worse handling below
        vv = _maxcos(Xt)
        if vv < best:
            best = vv
            bestX = Xt.copy()
            X = Xt
            step = min(step * 1.2, 0.01)
        else:
            # try smaller plain step
            Xt2 = _norm(X - step * 0.3 * gt)
            vv2 = _maxcos(Xt2)
            if vv2 < best:
                best = vv2
                bestX = Xt2.copy()
                X = Xt2
                step = min(step * 1.1, 0.01)
            else:
                step *= 0.6
                if step < 5e-7:
                    step = float(step0) * 0.3
                    # random tangent kick on worst point to escape
                    rm = _rowmax(X)
                    wi = int(np.argmax(rm))
                    rr = np.random.standard_normal(d)
                    rr -= float(rr @ X[wi]) * X[wi]
                    rn = float(np.linalg.norm(rr))
                    if rn > 1e-12:
                        rr /= rn
                        X = X.copy()
                        X[wi] = _norm((X[wi] + 1e-3 * rr)[None, :])[0]
    return bestX, best

def _find_void(others, rng, n_cand=4000, iters=120):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    bi = int(np.argmin(worst))
    vv = C[bi].copy()
    lr = 0.06
    tau = 0.006
    for _ in range(iters):
        dots = others @ vv
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ vv) * vv
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        gt /= nrm
        v2 = vv - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            vv = v2
            lr = min(lr * 1.05, 0.09)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
        tau *= 0.995
        if tau < 0.0015:
            tau = 0.0015
    vv /= float(np.linalg.norm(vv))
    return vv

def _lp_polish(x, seconds, r_init=3e-4):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _maxcos(x)
    except Exception:
        return x
    radius = float(r_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            Gfull = x @ x.T
            c_all = Gfull[iu]
            mm = float(c_all.max())
            thresh = max(25.0 * radius, 5e-8)
            sel = c_all > mm - thresh
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                idx = np.argpartition(-c_all, 2500)[:2500]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
            I = iu[0][sel]
            J = iu[1][sel]
            p = len(I)
            csub = c_all[sel]
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            r_eq = np.repeat(np.arange(n), d)
            c_eq = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
            y = _norm(y)
            v = _maxcos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(7)
    np.random.seed(7)
    # load
    best_val = 1e9
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.asarray(np.load(fn), dtype=float)
            if A.shape != (93, 13):
                continue
            A = _norm(A)
            v = _maxcos(A)
            if v < best_val:
                best_val = v
                x_best = A.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.standard_normal((93, 13))
        x_best = _norm(x_best)
        best_val = _maxcos(x_best)
    n, d = 93, 13
    # quick LP on loaded best to ensure tight baseline (short)
    if time.time() < t0 + 20:
        try:
            y = _lp_polish(x_best, seconds=2.0)
            vy = _maxcos(y)
            if vy < best_val:
                best_val = vy
                x_best = y.copy()
        except Exception:
            pass
    x_cur = x_best.copy()
    cur_val = best_val
    pool = [(best_val, x_best.copy())]
    hop_end = t0 + 118.0
    trial = 0
    # basin hopping with rattler-to-void surgery
    while time.time() < hop_end and time.time() < deadline - 50:
        trial += 1
        remain = min(hop_end, deadline - 50) - time.time()
        if remain < 4:
            break
        try:
            m_glob, cnt, G = _contacts(x_cur, tol=1e-9)
            rm = _rowmax(x_cur)
        except Exception:
            x_cur = x_best.copy()
            cur_val = best_val
            continue
        rattlers = np.where(cnt == 0)[0]
        low = np.where(cnt <= 1)[0]
        # decide how many to replace: 1 mostly, sometimes 2-3 to change topology
        if trial % 7 == 0:
            k = 3
        elif trial % 3 == 0:
            k = 2
        else:
            k = 1
        # choose indices
        cand_idx = []
        if len(rattlers) > 0:
            # prefer rattlers with smallest rowmax (most wasted) + random
            order = np.argsort(rm[rattlers])
            # take worst wasted first, then random
            nr = min(k, len(rattlers))
            # mix: first from ordered, rest random
            perm = rng.permutation(len(rattlers))
            chosen = []
            # deterministic pick of most wasted for first slot
            chosen.append(int(rattlers[order[0]]))
            for pi in perm:
                if len(chosen) >= nr:
                    break
                ci = int(rattlers[pi])
                if ci not in chosen:
                    chosen.append(ci)
            cand_idx.extend(chosen)
        # fill up to k with low-contact or random or overloaded hub perturbation
        while len(cand_idx) < k:
            if len(low) > 0 and rng.random() < 0.6:
                ci = int(rng.choice(low))
            else:
                ci = int(rng.integers(0, n))
            if ci not in cand_idx:
                cand_idx.append(ci)
        Xtry = x_cur.copy()
        # occasionally kick the overloaded hub slightly instead of replacing
        if trial % 5 == 0:
            try:
                hub = int(np.argmax(cnt))
                if cnt[hub] >= 6:
                    rr = rng.standard_normal(d)
                    rr -= float(rr @ Xtry[hub]) * Xtry[hub]
                    rn = float(np.linalg.norm(rr))
                    if rn > 1e-12:
                        rr /= rn
                        amp = 0.02 + 0.03 * rng.random()
                        Xtry[hub] = _norm((Xtry[hub] + amp * rr)[None, :])[0]
            except Exception:
                pass
        # sequential void reinsertion
        for ci in cand_idx:
            try:
                others = np.delete(Xtry, ci, axis=0)
                vv = _find_void(others, rng, n_cand=3500, iters=110)
                Xtry[ci] = vv
            except Exception:
                continue
        # small global jitter to break plateau degeneracy
        if rng.random() < 0.5:
            sig = 0.008 + 0.012 * rng.random()
            Xtry = Xtry + sig * rng.standard_normal(size=Xtry.shape)
            Xtry = _norm(Xtry)
        # relax: softmax Adam short + shorter when low time
        dl = min(hop_end, deadline - 50)
        try:
            Xtry = _adam_softmax(Xtry, steps=320, tau0=0.015, tau1=0.002, lr0=2.5e-3, lr1=6e-4, deadline=dl)
            # light topK to push max (K large to equalize plateau)
            Xtry, v_try = _topK(Xtry, iters=120, K=220, step0=0.004, deadline=dl)
            v_try = _maxcos(Xtry)
        except Exception:
            continue
        pool.append((float(v_try), Xtry.copy()))
        if len(pool) > 14:
            pool = sorted(pool, key=lambda t: t[0])[:14]
        # acceptance: intensify on improvement, else side-step with tolerance
        frac = (time.time() - t0) / BUDGET
        tol = 0.004 * (1.0 - frac) + 0.0003
        if v_try < best_val - 1e-12:
            best_val = float(v_try)
            x_best = Xtry.copy()
            x_cur = Xtry.copy()
            cur_val = float(v_try)
        elif v_try < cur_val + tol:
            x_cur = Xtry.copy()
            cur_val = float(v_try)
        else:
            # occasionally restart from best with perturbation
            if rng.random() < 0.25:
                sig2 = 0.05 + 0.15 * rng.random()
                x_cur = _norm(x_best + sig2 * rng.standard_normal(size=x_best.shape))
                cur_val = _maxcos(x_cur)
        # early exit if big win then intensify
        if best_val < 0.2425:
            break
    # select top candidates for polish
    pool = sorted(pool, key=lambda t: t[0])[:8]
    # add current best if missing
    cands = []
    for v, X in pool:
        if len(cands) >= 4:
            break
        cands.append((v, X.copy()))
    # polish phase
    polished = []
    polish_end = deadline - 6.0
    for ci, (v0, X0) in enumerate(cands):
        if time.time() > polish_end:
            break
        Xp = _norm(X0.copy())
        try:
            # stronger softmax then topK narrowing
            rem = polish_end - time.time()
            if rem < 3:
                break
            Xp = _adam_softmax(Xp, steps=260, tau0=0.008, tau1=0.0012, lr0=1.8e-3, lr1=4e-4, deadline=polish_end)
            Xp, _ = _topK(Xp, iters=250, K=120, step0=0.003, deadline=polish_end)
            Xp, _ = _topK(Xp, iters=250, K=30, step0=0.002, deadline=polish_end)
            # LP finish with remaining time slice
            rem2 = polish_end - time.time()
            if rem2 > 2.5:
                Xp = _lp_polish(Xp, seconds=min(4.0, rem2 - 0.5))
            vp = _maxcos(Xp)
        except Exception:
            vp = _maxcos(Xp)
        polished.append((float(vp), Xp.copy()))
        if vp < best_val:
            best_val = float(vp)
            x_best = Xp.copy()
    if len(polished) > 0:
        polished = sorted(polished, key=lambda t: t[0])
        best_val, x_best = polished[0][0], polished[0][1].copy()
    # safety: if we somehow got worse than input, reload input best
    try:
        for fn in ("best.npy",):
            A = _norm(np.asarray(np.load(fn), dtype=float))
            va = _maxcos(A)
            if va < best_val:
                best_val = va
                x_best = A.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    # final tiny shrink to enforce strict feasibility / avoid NaN
    x_best = np.asarray(x_best, dtype=float)
    # ensure no zero rows
    nr = np.linalg.norm(x_best, axis=1)
    nr[nr == 0] = 1.0
    x_best = x_best / nr[:, None]
    return x_best
