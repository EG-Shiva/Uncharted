import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk(X, k=50):
    G = X @ X.T
    iu = np.triu_indices(G.shape[0], 1)
    c = G[iu]
    if len(c) > k:
        idx = np.argpartition(-c, k)[:k]
        top = c[idx]
        top.sort()
        return top[::-1].copy()
    top = np.sort(c)[::-1].copy()
    return top

def _lex_better(new_top, old_top, tol=1e-11):
    for a, b in zip(new_top, old_top):
        if a < b - tol:
            return True
        if a > b + tol:
            return False
    return False

def _quench_lse(X_init, steps, T0, T1, lr, mu=0.9):
    X = X_init.copy()
    n, d = X.shape
    V = np.zeros_like(X)
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bm = float(G0.max())
    bX = X.copy()
    btop = _topk(X, 50)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
        if (s % 120 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bm - 1e-12:
                bm = vc
                bX = X.copy()
                btop = _topk(X, 50)
    return bX, bm, btop

def _lp_trust_step(x, I, J, csel, radius):
    n, d = x.shape
    p = len(I)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d),
                           J[:, None] * d + np.arange(d),
                           np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                  bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    return res

def _lp_polish_eps(x, eps_active, radius0, iters, deadline=None, cap=1400):
    x = _norm_rows(x.copy())
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    cur_m = _max_cos(x)
    cur_top = _topk(x, 50)
    radius = radius0
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        G = x @ x.T
        cuv = G[iu]
        m = float(cuv.max())
        sel = cuv > m - eps_active
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            order = np.argsort(-cuv[sel])[:cap] if False else np.argpartition(-cuv[sel], cap)[:cap]
            # need tightest first; argpartition then sort that subset
            # simpler: argsort on subset is fine for 2485 size
            order = np.argsort(-cuv[sel])[:cap]
            I = iu[0][sel][order]; J = iu[1][sel][order]; csel = cuv[sel][order]
        else:
            I = iu[0][sel]; J = iu[1][sel]; csel = cuv[sel]
        try:
            res = _lp_trust_step(x, I, J, csel, radius)
        except Exception:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        dx = res.x[:-1].reshape(n, d)
        # exact-max line search over alpha
        best_a = 0.0
        best_v = m
        best_y = None
        for alpha in (1.0, 0.6, 0.3, 0.15):
            y = _norm_rows(x + alpha * dx)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_a = alpha
                best_y = y
        if best_y is not None and best_v < cur_m - 2e-13:
            x = best_y
            cur_m = best_v
            cur_top = _topk(x, 50)
            radius = min(radius * 1.4, 0.02)
        else:
            # try lexicographic improvement even if max ties: check alpha=1 topk
            y = _norm_rows(x + dx)
            yt = _topk(y, 50)
            ym = float(yt[0])
            if ym <= cur_m + 5e-12 and _lex_better(yt, cur_top):
                x = y
                cur_m = ym
                cur_top = yt
                radius = min(radius * 1.2, 0.02)
            else:
                radius *= 0.5
                if radius < 1e-13:
                    break
    return x, cur_m, cur_top

def _rigidity_svd(x, eps_active, cap=700):
    n, d = x.shape
    N = n * d
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    cuv = G[iu]
    m = float(cuv.max())
    sel = cuv > m - eps_active
    I0 = iu[0][sel]; J0 = iu[1][sel]; c0 = cuv[sel]
    order = np.argsort(-c0)
    if len(order) > cap:
        order = order[:cap]
    I = I0[order]; J = J0[order]; c = c0[order]
    p = len(I)
    Jmat = np.zeros((p, N), dtype=float)
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        cij = float(c[k])
        vi = x[j] - cij * x[i]
        vj = x[i] - cij * x[j]
        Jmat[k, i*d:(i+1)*d] = vi
        Jmat[k, j*d:(j+1)*d] = vj
    try:
        _, S, Vt = np.linalg.svd(Jmat, full_matrices=False)
    except Exception:
        return None, None, I, J, c, m
    return S, Vt, I, J, c, m

def _null_kick(x, S, Vt, step, rng, kbot=36):
    n, d = x.shape
    K = min(kbot, Vt.shape[0])
    coeff = rng.standard_normal(K)
    Dvec = coeff @ Vt[-K:, :]
    D = Dvec.reshape(n, d)
    # tangent projection (remove radial)
    D = D - x * np.sum(D * x, axis=1, keepdims=True)
    # normalize
    fn = float(np.linalg.norm(D))
    if not np.isfinite(fn) or fn < 1e-12:
        N = rng.standard_normal((n, d)) * 0.01
        N -= x * np.sum(N * x, axis=1, keepdims=True)
        D = N
        fn = float(np.linalg.norm(D))
    D = D / (fn + 1e-30) * step * np.sqrt(float(n))
    # flip sign to help current max pair
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    kflat = int(np.argmax(G))
    wi, wj = kflat // n, kflat % n
    cij = float(x[wi] @ x[wj])
    vi = x[wj] - cij * x[wi]
    vj = x[wi] - cij * x[wj]
    lin = float(vi @ D[wi] + vj @ D[wj])
    if lin > 0:
        D = -D
    y = _norm_rows(x + D)
    return y

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    HOP_END = HARD_END - 16.0
    rng = np.random.default_rng(20260607)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_m, best_X = cands[0][0], cands[0][1].copy()
    best_top = _topk(best_X, 50)
    cur_X, cur_m, cur_top = best_X.copy(), best_m, best_top.copy()
    # initial epsilon-active polish sweep (plateau, not single contact)
    for eps, rad, its in [(2e-4, 3e-3, 6), (5e-5, 2e-3, 6), (1e-5, 1.2e-3, 6)]:
        if time.time() > HOP_END - 40:
            break
        nx, nm, nt = _lp_polish_eps(cur_X, eps, rad, its, deadline=HOP_END)
        if nm < cur_m - 1e-13 or (nm <= cur_m + 5e-12 and _lex_better(nt, cur_top)):
            cur_X, cur_m, cur_top = nx, nm, nt
            if nm < best_m - 1e-13 or (nm <= best_m + 5e-12 and _lex_better(nt, best_top)):
                best_X, best_m, best_top = nx.copy(), nm, nt.copy()
    eps_cycle = [2e-4, 5e-5, 1e-5, 2e-6, 5e-7]
    it = 0
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 1.5:
            break
        eps = eps_cycle[it % len(eps_cycle)]
        # rigidity analysis on current best/cur (alternate base)
        base = cur_X if (it % 3 != 0) else best_X
        try:
            S, Vt, Iact, Jact, cact, mm = _rigidity_svd(base, eps, cap=650)
        except Exception:
            S, Vt = None, None
        # occasional diagnostics
        if S is not None and it % 12 == 1:
            try:
                rank = int(np.sum(S > 1e-5))
                contacts = int(np.sum(cact > mm - 5e-8))
            except Exception:
                rank, contacts = -1, -1
            # print kept minimal; verifier ignores stdout
            # print(f"it{it} eps{eps:.0e} p{len(cact)} rank{rank} contacts5e-8 {contacts} smin{S[-1]:.2e} smax{S[0]:.2e} best{best_m:.10f}")
            pass
        # trust-region LP on plateau set
        nx, nm, nt = _lp_polish_eps(base, eps, 1.5e-3, 4, deadline=HOP_END)
        if nm < best_m - 1e-13 or (nm <= best_m + 5e-12 and _lex_better(nt, best_top)):
            best_X, best_m, best_top = nx.copy(), nm, nt.copy()
            cur_X, cur_m, cur_top = nx.copy(), nm, nt.copy()
            continue
        if nm < cur_m - 1e-13 or (nm <= cur_m + 5e-12 and _lex_better(nt, cur_top)):
            cur_X, cur_m, cur_top = nx, nm, nt
        # null-space kick + re-polish (moves max pair apart almost for free)
        if S is None or Vt is None:
            # fallback random tangent kick
            sigma = float(rng.uniform(0.004, 0.016))
            N = rng.standard_normal((n, d)) * sigma
            N -= base * np.sum(N * base, axis=1, keepdims=True)
            kicked = _norm_rows(base + N)
        else:
            step = float(rng.uniform(0.006, 0.022))
            kb = int(rng.integers(24, 48))
            try:
                kicked = _null_kick(base, S, Vt, step, rng, kbot=kb)
            except Exception:
                sigma = 0.01
                N = rng.standard_normal((n, d)) * sigma
                N -= base * np.sum(N * base, axis=1, keepdims=True)
                kicked = _norm_rows(base + N)
        # quench kicked point
        T0 = float(rng.uniform(0.0012, 0.0035))
        T1 = float(rng.uniform(0.00008, 0.0004))
        lr = float(rng.uniform(0.10, 0.22))
        steps = int(rng.integers(220, 480))
        qX, qm, qt = _quench_lse(kicked, steps, T0, T1, lr)
        # LP squeeze on two epsilon levels successively
        qX2, qm2, qt2 = _lp_polish_eps(qX, 5e-5, 1.2e-3, 4, deadline=HOP_END)
        qX3, qm3, qt3 = _lp_polish_eps(qX2, 5e-6, 8e-4, 4, deadline=HOP_END)
        # lexicographic acceptance on sorted top-50
        if qm3 < best_m - 1e-13 or (qm3 <= best_m + 5e-12 and _lex_better(qt3, best_top)):
            best_X, best_m, best_top = qX3.copy(), qm3, qt3.copy()
            cur_X, cur_m, cur_top = qX3.copy(), qm3, qt3.copy()
        elif qm3 < cur_m - 1e-13 or (qm3 <= cur_m + 5e-12 and _lex_better(qt3, cur_top)):
            cur_X, cur_m, cur_top = qX3.copy(), qm3, qt3.copy()
        else:
            # small chance to random-walk on plateau to change basin (lex tie tolerant)
            if rng.random() < 0.08:
                # accept if within 2e-6 of cur to keep diversity
                if qm3 < cur_m + 2e-6:
                    cur_X, cur_m, cur_top = qX3.copy(), qm3, qt3.copy()
        # extra rattler surgery every ~7 iters: reinsert worst point into hole
        if it % 7 == 0 and time.time() < HOP_END - 5:
            try:
                Gc = cur_X @ cur_X.T
                np.fill_diagonal(Gc, -2.0)
                rm = Gc.max(axis=1)
                w = int(np.argmax(rm))
                R = np.delete(cur_X, w, axis=0)
                Z = _norm_rows(rng.standard_normal((4000, d)))
                sc = Z @ R.T
                ms = sc.max(axis=1)
                k0 = int(np.argmin(ms))
                s = Z[k0].copy()
                # quick refine s against R
                v = np.zeros_like(s)
                for _ in range(150):
                    g = R @ s
                    m_ = float(g.max())
                    e = np.exp(np.clip((g - m_) / 0.004, -50, 0))
                    den = float(e.sum()) + 1e-30
                    Gr = (R.T @ e) / den
                    Gt = Gr - float(Gr @ s) * s
                    v = 0.8 * v + Gt
                    s = s - 0.12 * v
                    s = s / (np.linalg.norm(s) + 1e-12)
                trial = cur_X.copy()
                trial[w] = s / (np.linalg.norm(s) + 1e-12)
                tX, tm, tt = _lp_polish_eps(trial, 5e-5, 1.0e-3, 3, deadline=HOP_END)
                if tm < best_m - 1e-13 or (tm <= best_m + 5e-12 and _lex_better(tt, best_top)):
                    best_X, best_m, best_top = tX.copy(), tm, tt.copy()
                    cur_X, cur_m, cur_top = tX.copy(), tm, tt.copy()
            except Exception:
                pass
    # final squeeze on best
    try:
        fx, fm, ft = _lp_polish_eps(best_X, 1e-4, 2e-3, 10, deadline=HARD_END - 1.0)
        if fm <= best_m + 1e-12:
            # keep lex-best among fx/best
            if fm < best_m - 1e-13 or _lex_better(ft, best_top):
                best_X, best_m = fx, fm
            else:
                # if LP tied max but lex worse, keep original best
                pass
        fx2, fm2, ft2 = _lp_polish_eps(best_X, 1e-5, 1e-3, 10, deadline=HARD_END - 0.5)
        if fm2 < best_m - 1e-13 or (fm2 <= best_m + 5e-12 and _lex_better(ft2, _topk(best_X, 50))):
            best_X, best_m = fx2, fm2
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    # tiny shrink guard (no-op, keeps unit rows)
    return np.asarray(best_X, dtype=float)
