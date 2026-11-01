import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_mom(X, steps, T0, T1, lr, mu=0.9, deadline=None, track=True):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(max(T0,1e-12)); lt1 = np.log(max(T1,1e-12))
    if track:
        G0 = X @ X.T; np.fill_diagonal(G0, -2.0)
        bv = float(G0.max()); bX = X.copy()
    else:
        bv = 1e100; bX = None
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
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
        if track and ((s % 150 == 0) or s == steps - 1):
            Gc = X @ X.T; np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < bv:
                bv = vc; bX = X.copy()
    if track:
        return bX, bv
    return X, _max_cos(X)

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1.2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cv = G[iu]
        m = float(cv.max())
        sel = cv > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cs = cv[sel]
        p = len(I)
        if p > 1500:
            od = np.argsort(-cs)[:1500]
            I, J, cs = I[od], J[od], cs[od]
            p = 1500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

# ---------- lattice pool ----------
def _lattice_pool(rng, target=2200):
    pts = []
    seen = set()
    tries = 0
    while len(pts) < target and tries < target * 40:
        tries += 1
        r = rng.random()
        if r < 0.55:
            v = rng.integers(-2, 3, size=12).astype(np.float64)
            if int(v.sum()) % 2 != 0:
                k = int(rng.integers(0, 12))
                # fix parity by +-1
                v[k] += 1.0 if v[k] < 2 else -1.0
            n2 = float((v * v).sum())
            if n2 < 4 or n2 > 8 or n2 == 0:
                continue
        else:
            # half-integer shell
            v = rng.integers(-2, 3, size=12).astype(np.float64) + 0.5
            # randomly zero out some coords to vary norm
            zm = rng.random(12) < 0.25
            v[zm] = 0.0
            # need at least some half entries
            n2 = float((v * v).sum())
            if n2 < 3.0 or n2 > 8.0 or n2 == 0:
                continue
        key = tuple(np.round(v / np.sqrt(n2), 6))
        if key in seen:
            continue
        seen.add(key)
        pts.append(v / np.sqrt(n2))
    if len(pts) == 0:
        return _norm_rows(rng.standard_normal((target, 12)))
    P = np.array(pts)
    # pad with random if short
    if P.shape[0] < 500:
        extra = _norm_rows(rng.standard_normal((500, 12)))
        P = np.concatenate([P, extra], axis=0)
    return P

def _greedy_select(pool, rng):
    P = pool.shape[0]
    K = 71
    start = int(rng.integers(0, P))
    sel = [start]
    is_sel = np.zeros(P, dtype=bool); is_sel[start] = True
    # iterative
    for _ in range(1, K):
        S = pool[sel]  # k x 12
        dots = pool @ S.T  # P x k
        # max over k
        mx = dots.max(axis=1)
        mx[is_sel] = 1e100
        nxt = int(np.argmin(mx))
        sel.append(nxt); is_sel[nxt] = True
    return np.array(sel, dtype=int)

def _sa_swaps(pool, sel, rng, iters=3500, t0=0.004, t1=0.00015):
    P = pool.shape[0]
    K = len(sel)
    S = pool[sel].copy()
    G = S @ S.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(K, 1)
    def stats_from_G(Gm):
        v = Gm[iu]
        m = float(v.max())
        cnt = int(np.count_nonzero(v > 0.22))
        # sum exp
        ss = float(np.exp(np.clip((v - 0.22) / 0.002, -50, 50)).sum())
        return m, cnt, ss
    bm, bcnt, bss = stats_from_G(G)
    best_sel = sel.copy()
    cur_m, cur_cnt, cur_ss = bm, bcnt, bss
    # need G with diag -2 for excl max
    is_sel = np.zeros(P, dtype=bool); is_sel[sel] = True
    # precompute mask helper
    for it in range(iters):
        frac = it / max(1, iters - 1)
        T = t0 * (1 - frac) + t1 * frac
        i = int(rng.integers(0, K))
        j = int(rng.integers(0, P))
        if is_sel[j]:
            continue
        cand = pool[j]
        d = S @ cand  # K dots to current (includes self i)
        # excl stats: mask i
        mask = np.ones(K, dtype=bool); mask[i] = False
        Gsub = G[np.ix_(mask, mask)]
        np.fill_diagonal(Gsub, -2.0)
        # max excl
        m_excl = float(Gsub.max())
        v_excl = Gsub[np.triu_indices(K - 1, 1)]
        cnt_excl = int(np.count_nonzero(v_excl > 0.22))
        ss_excl = float(np.exp(np.clip((v_excl - 0.22) / 0.002, -50, 50)).sum())
        d_excl = np.delete(d, i)
        m_new_cand = float(d_excl.max()) if len(d_excl) else -2.0
        m_new = m_excl if m_excl > m_new_cand else m_new_cand
        cnt_new = cnt_excl + int(np.count_nonzero(d_excl > 0.22))
        ss_new = ss_excl + float(np.exp(np.clip((d_excl - 0.22) / 0.002, -50, 50)).sum())
        # lexicographic better?
        better = False
        if m_new < cur_m - 1e-12:
            better = True
        elif abs(m_new - cur_m) <= 1e-12:
            if cnt_new < cur_cnt:
                better = True
            elif cnt_new == cur_cnt and ss_new < cur_ss - 1e-9:
                better = True
        if better:
            # accept
            old_idx = int(sel[i])
            is_sel[old_idx] = False; is_sel[j] = True
            sel[i] = j
            S[i] = cand
            G = S @ S.T; np.fill_diagonal(G, -2.0)
            cur_m, cur_cnt, cur_ss = m_new, cnt_new, ss_new
            # update best lexicographically
            is_best = False
            if m_new < bm - 1e-12:
                is_best = True
            elif abs(m_new - bm) <= 1e-12:
                if cnt_new < bcnt:
                    is_best = True
                elif cnt_new == bcnt and ss_new < bss:
                    is_best = True
            if is_best:
                bm, bcnt, bss = m_new, cnt_new, ss_new
                best_sel = sel.copy()
        else:
            # annealing on primary
            dm = m_new - cur_m
            if dm < 0.02 and rng.random() < np.exp(-max(0.0, dm) / max(T, 1e-9)):
                # also allow small cnt worsening? accept
                old_idx = int(sel[i])
                is_sel[old_idx] = False; is_sel[j] = True
                sel[i] = j
                S[i] = cand
                G = S @ S.T; np.fill_diagonal(G, -2.0)
                cur_m, cur_cnt, cur_ss = m_new, cnt_new, ss_new
    return best_sel, (bm, bcnt, bss)

# ---------- group orbits ----------
def _expand_cyclic(seeds):
    ns = seeds.shape[0]
    r = seeds.reshape(ns, 6, 2)
    out = np.empty((ns * 6, 12))
    for m in range(6):
        rr = np.roll(r, shift=m, axis=1).reshape(ns, 12)
        out[m::6 if False else None] = None if False else None
    # simpler: stack
    blocks = []
    for m in range(6):
        blocks.append(np.roll(r, shift=m, axis=1).reshape(ns, 12))
    # interleave per seed: (ns,6,12) -> (ns*6,12)
    arr = np.stack(blocks, axis=1)  # ns x 6 x 12
    return arr.reshape(ns * 6, 12)

def _pullback_cyclic(grad_X, ns):
    # grad_X: (ns*6,12)
    arr = grad_X.reshape(ns, 6, 12)
    g = np.zeros((ns, 12))
    for m in range(6):
        gm = arr[:, m, :].reshape(ns, 6, 2)
        back = np.roll(gm, shift=-m, axis=1).reshape(ns, 12)
        g += back
    return g / 6.0

def _expand_rot(seeds):
    ns = seeds.shape[0]
    ang = 2 * np.pi / 6.0
    r = seeds.reshape(ns, 6, 2)
    outs = []
    for m in range(6):
        c = np.cos(m * ang); s = np.sin(m * ang)
        x = r[:, :, 0]; y = r[:, :, 1]
        xr = c * x - s * y; yr = s * x + c * y
        rr = np.stack([xr, yr], axis=-1)  # ns,6,2
        outs.append(rr.reshape(ns, 12))
    arr = np.stack(outs, axis=1)
    return arr.reshape(ns * 6, 12)

def _pullback_rot(grad_X, ns):
    arr = grad_X.reshape(ns, 6, 12)
    ang = 2 * np.pi / 6.0
    g = np.zeros((ns, 6, 2))
    for m in range(6):
        c = np.cos(m * ang); s = np.sin(m * ang)
        gm = arr[:, m, :].reshape(ns, 6, 2)
        gx = gm[:, :, 0]; gy = gm[:, :, 1]
        # R(-m)
        bx = c * gx + s * gy; by = -s * gx + c * gy
        g[:, :, 0] += bx; g[:, :, 1] += by
    return (g.reshape(ns, 12)) / 6.0

def _seed_quench(seeds_init, expand_fn, pullback_fn, steps, T0, T1, lr, mu, deadline=None):
    seeds = _norm_rows(seeds_init.copy())
    ns = seeds.shape[0]
    V = np.zeros_like(seeds)
    lt0 = np.log(max(T0, 1e-12)); lt1 = np.log(max(T1, 1e-12))
    Xb = expand_fn(seeds)
    bv = _max_cos(Xb); bS = seeds.copy()
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        X = expand_fn(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        GradX = (E @ X) / rs
        dotx = np.sum(GradX * X, axis=1, keepdims=True)
        GtX = GradX - dotx * X
        Gs = pullback_fn(GtX, ns)
        # project to tangent of seeds
        dot = np.sum(Gs * seeds, axis=1, keepdims=True)
        Gt = Gs - dot * seeds
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        seeds = seeds - lr * V
        seeds = _norm_rows(seeds)
        if (s % 150 == 0) or s == steps - 1:
            Xc = expand_fn(seeds)
            vc = _max_cos(Xc)
            if vc < bv:
                bv = vc; bS = seeds.copy()
    return bS, bv

def _drop_worst(X, kdrop=1):
    # drop kdrop points with largest row-max to reach 71
    while X.shape[0] > 71:
        G = X @ X.T; np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        idx = int(np.argmax(rm))
        X = np.delete(X, idx, axis=0)
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    HOP_END = HARD_END - 28.0
    rng = np.random.default_rng(20260711)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                cands.append(_norm_rows(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal((n, d))))
    scored = [(_max_cos(a), a) for a in cands]
    scored.sort(key=lambda t: t[0])
    best_score, best_X = scored[0][0], scored[0][1].copy()
    # ---- (a) lattice pool branch ----
    try:
        pool = _lattice_pool(rng, target=2200)
        for rep in range(2):
            if time.time() > HOP_END - 60:
                break
            sel0 = _greedy_select(pool, rng)
            sel1, _ = _sa_swaps(pool, sel0.copy(), rng, iters=3000)
            X0 = _norm_rows(pool[sel1].copy())
            # release to quench
            T0 = float(10 ** rng.uniform(-2.6, -1.7))
            T1 = float(10 ** rng.uniform(-4.0, -3.2))
            lr = float(10 ** rng.uniform(-2.6, -1.9))
            Xq, sq = _quench_mom(X0.copy(), steps=900, T0=T0, T1=T1, lr=lr, deadline=HOP_END)
            Xq = _lp_polish(Xq, seconds=2.0, deadline=HOP_END)
            vq = _max_cos(Xq)
            if vq < best_score:
                best_score = vq; best_X = Xq.copy()
    except Exception:
        pass
    # ---- (b) group-orbit branch ----
    try:
        configs = [
            ("cyc12", 12, _expand_cyclic, _pullback_cyclic),
            ("rot12", 12, _expand_rot, _pullback_rot),
            ("cyc6s", 6, _expand_cyclic, _pullback_cyclic),
        ]
        for ci, (nm, ns, expf, pullf) in enumerate(configs):
            for rep in range(2):
                if time.time() > HOP_END - 25:
                    break
                S0 = _norm_rows(rng.standard_normal((ns, 12)))
                # for ns=6 signed trick: expand gives 36; need 72 -> duplicate with sign?
                if ns == 6 and nm == "cyc6s":
                    # seed quench on cyclic (36 pts) then double with sign flip to 72
                    Sq, _ = _seed_quench(S0, expf, pullf, steps=700, T0=0.02, T1=0.0015, lr=0.02, mu=0.88, deadline=HOP_END)
                    Xf = expf(Sq)  # 36
                    Xf = np.concatenate([Xf, -Xf], axis=0)  # 72
                else:
                    Sq, _ = _seed_quench(S0, expf, pullf, steps=800, T0=0.02, T1=0.0012, lr=0.02, mu=0.88, deadline=HOP_END)
                    Xf = expf(Sq)
                Xf = _drop_worst(_norm_rows(Xf), kdrop=Xf.shape[0] - 71)
                # break symmetry slightly
                sig = 0.004 if rng.random() < 0.5 else 0.012
                Xb = _norm_rows(Xf + rng.standard_normal(Xf.shape) * sig)
                Xq, sq = _quench_mom(Xb, steps=700, T0=0.008, T1=0.0006, lr=0.03, deadline=HOP_END)
                vq = _max_cos(Xq)
                if vq < best_score:
                    best_score = vq; best_X = Xq.copy()
                # extra tighten
                Xq2 = _lp_polish(Xq, seconds=1.5, deadline=HOP_END)
                vq2 = _max_cos(Xq2)
                if vq2 < best_score:
                    best_score = vq2; best_X = Xq2.copy()
    except Exception:
        pass
    # ---- final intensive polish on best ----
    try:
        # short random quenches around best + LP
        while time.time() < HARD_END - 6.0:
            tleft = HARD_END - 6.0 - time.time()
            if tleft < 2.0:
                break
            sig = float(10 ** rng.uniform(-3.0, -1.2)) if rng.random() < 0.6 else 0.0
            if sig > 0:
                X0 = _norm_rows(best_X + rng.standard_normal(best_X.shape) * sig)
            else:
                X0 = best_X.copy()
            T0 = float(10 ** rng.uniform(-2.8, -1.8))
            T1 = float(10 ** rng.uniform(-4.2, -3.1))
            lr = float(10 ** rng.uniform(-2.7, -1.8))
            steps = 700 if tleft > 12 else 400
            Xq, sq = _quench_mom(X0, steps=steps, T0=T0, T1=T1, lr=lr, deadline=HARD_END - 5.0)
            if sq < best_score + 0.002:
                Xq = _lp_polish(Xq, seconds=min(2.5, max(0.8, tleft - 1.5)), deadline=HARD_END - 4.0)
                vq = _max_cos(Xq)
                if vq < best_score:
                    best_score = vq; best_X = Xq.copy()
            else:
                if sq < best_score:
                    best_score = sq; best_X = Xq.copy()
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    # tiny shrink guard
    return best_X
