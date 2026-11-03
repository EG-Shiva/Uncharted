import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(X, seconds, deadline=1e18):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu0, iu1]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:4000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        cfull = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            bx = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return bx

def _adam_shift(X, freeze, delta, iters, tau0, tau1, lr, deadline=1e18):
    n, d = X.shape
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    mmt = np.zeros_like(X)
    vmt = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0, l1 = np.log(tau0), np.log(tau1)
    for it in range(1, iters + 1):
        if time.time() > deadline - 0.5:
            break
        tau = float(np.exp(l0 + (l1 - l0) * (it / iters)))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if freeze is not None and delta != 0.0:
            # shifted max: r-pairs shifted by +delta
            r = freeze
            mr = float(G[r].max())
            # core max (exclude r)
            Gc = G.copy()
            Gc[r, :] = -2.0
            Gc[:, r] = -2.0
            mc = float(Gc.max())
            mp = mc if mc >= mr + delta else mr + delta
            Gs = (G - mp) / tau
            # add shift to r row/col
            sh = delta / tau
            Gs[r, :] += sh
            Gs[:, r] += sh
            # diag was -2, fix it to very negative
            Gs[r, r] = -50.0
            # need to correct double add on (r,:) vs (:,r) intersection? row and col same pairs counted twice but symmetric; we added sh twice for non-r? Actually Gs symmetric, adding to row and col doubles. So subtract one.
            Gs[r, :] -= 0.0  # keep as is? we added twice? Let's fix: we added sh to row and col, so off-diag r-pairs got 2*sh. Correct to 1*sh:
            Gs[r, :] -= sh / 1.0  # net: row gets sh, col gets sh? simpler recompute
            # redo cleanly:
            Gs = (G - mp) / tau
            Gs[r, :] += sh
            Gs[:, r] += 0.0  # row already covers symmetric? W symmetric needs both but E symmetric; adding to row only breaks symmetry. Add half each:
            # To keep symmetry, set both row and col to same shifted value:
            # shifted value = (G[r,j]+delta-mp)/tau
            # So set Gs[r,j]=Gs[j,r]=that
            # Gs currently (G-mp)/tau, so need +sh on both
            # We already did row +sh, need col +sh but diagonal double; do:
            Gs[:, r] += sh
            Gs[r, r] = -50.0
            # off-diag r pairs now have +2sh, remove one sh:
            Gs[r, :] -= sh
            # now row has +sh? Wait: row had +sh then -sh =0, col has +sh. Asymmetric.
            # Simplest symmetric: after double add, subtract sh/ ? Let's just set explicitly:
            # (avoid confusion) recompute r-line:
            row = (G[r] + delta - mp) / tau
            row = np.clip(row, -50.0, 0.0)
            Gs[r, :] = row
            Gs[:, r] = row
            Gs[r, r] = -50.0
            np.clip(Gs, -50.0, 0.0, out=Gs)
        else:
            mp = float(G.max())
            Gs = (G - mp) / tau
            np.clip(Gs, -50.0, 0.0, out=Gs)
        E = np.exp(Gs)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        grad = W @ X
        g = grad - (np.sum(grad * X, axis=1, keepdims=True)) * X
        if freeze is not None:
            g[freeze, :] = 0.0
        mmt = b1 * mmt + (1.0 - b1) * g
        vmt = b2 * vmt + (1.0 - b2) * (g * g)
        mhat = mmt / (1.0 - b1 ** it)
        vhat = vmt / (1.0 - b2 ** it)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        if freeze is not None:
            step[freeze, :] = 0.0
        X = X - step
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        X = X / xn
    return X

def _largest_void(core, rng, ncand=20000):
    # core: (m,6) normalized
    cand = rng.standard_normal(size=(ncand, 6))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    D = cand @ core.T
    scores = D.max(axis=1)
    bi = int(np.argmin(scores))
    best = cand[bi].copy()
    # refine top few
    topk = np.argpartition(scores, 4)[:5]
    refined_best = best
    refined_val = float(scores[bi])
    for t in topk:
        y = cand[int(t)].copy()
        # local descent on smooth max
        for it in range(300):
            tau = 0.02 * (0.002 / 0.02) ** (it / 300.0)
            d = core @ y
            m = float(d.max())
            e = np.exp(np.clip((d - m) / tau, -50, 0))
            w = e / (float(e.sum()) + 1e-300)
            grad = w @ core
            gt = grad - float(grad @ y) * y
            lr = 0.05 * (1.0 - 0.5 * it / 300.0)
            y = y - lr * gt
            y = y / (np.linalg.norm(y) + 1e-15)
        v = float((core @ y).max())
        if v < refined_val:
            refined_val = v
            refined_best = y.copy()
    return refined_best, refined_val

def solve():
    t_start = time.time()
    deadline = t_start + 172.0
    rng = np.random.default_rng(7)
    n, d = 219, 6
    x_best = None
    v_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < v_best:
                    v_best = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal(size=(n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        v_best = _max_cos(x_best)

    def remaining():
        return deadline - time.time()

    # baseline polish a little
    if remaining() > 40:
        xp = _lp_polish(x_best, seconds=min(6.0, remaining() - 30.0), deadline=deadline)
        vp = _max_cos(xp)
        if vp < v_best:
            x_best, v_best = xp, vp

    # identify rattlers: smallest rowmax (most isolated)
    rm, _ = _row_max(x_best)
    order = np.argsort(rm)  # ascending: most isolated first
    rattlers = list(order[:4])

    deltas = [1e-5, 2e-5, 5e-5, 1e-4]
    # alternate rattler-first attempts
    for r in rattlers:
        for delta in deltas:
            if remaining() < 28:
                break
            X = x_best.copy()
            # re-evaluate r still rattler? keep same r
            # Phase 1: freeze r, void-expansion on core
            X = _adam_shift(X, freeze=int(r), delta=float(delta),
                            iters=1100, tau0=0.004, tau1=0.0005, lr=0.006,
                            deadline=deadline)
            if remaining() < 20:
                continue
            # Phase 2: move r to refined largest void of core
            core = np.delete(X, int(r), axis=0)
            try:
                yv, vv = _largest_void(core, rng, ncand=20000)
            except Exception:
                continue
            X[int(r)] = yv
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            # Phase 3: release, joint relax
            X = _adam_shift(X, freeze=None, delta=0.0,
                            iters=700, tau0=0.003, tau1=0.0004, lr=0.006,
                            deadline=deadline)
            v = _max_cos(X)
            if v > v_best + 0.0015:
                continue
            # count contacts of r within 1e-7
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            vm = float(G.max())
            cnt_r = int(np.sum(G[int(r)] > vm - 1e-7))
            # core max (pairs not involving r)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            # mask pairs involving r
            mask = (iu0 == int(r)) | (iu1 == int(r))
            core_max = float(c[~mask].max()) if np.any(~mask) else vm
            # require rattler integrated (>=5 contacts) and core dropped to accept as new topology
            # but always LP-polish if promising and keep if strictly better (fallback safe)
            if remaining() < 12:
                if v < v_best:
                    # quick accept without LP if time low
                    if cnt_r >= 5 or v < v_best - 1e-7:
                        x_best, v_best = X, v
                continue
            Xp = _lp_polish(X, seconds=min(5.0, remaining() - 8.0), deadline=deadline)
            vp = _max_cos(Xp)
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            vpm = float(Gp.max())
            cnt_rp = int(np.sum(Gp[int(r)] > vpm - 1e-7))
            if vp < v_best and (cnt_rp >= 5 or core_max < v_best):
                x_best, v_best = Xp, vp
                # refresh rattler list after success
                rm2, _ = _row_max(x_best)
                order2 = np.argsort(rm2)
                # update remaining rattlers to new most isolated (avoid reusing same)
                # keep outer loop consistent by just continuing
            elif vp < v_best - 2e-7:
                # strict improvement even without contact condition: still take (keeps incumbent fallback)
                x_best, v_best = Xp, vp
            # else discard
        if remaining() < 28:
            break

    if remaining() > 4:
        x_best = _lp_polish(x_best, seconds=max(1.0, remaining() - 2.0), deadline=deadline)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
