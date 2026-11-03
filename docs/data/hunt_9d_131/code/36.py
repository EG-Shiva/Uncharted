import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _fast_polish(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish_deadline(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gall = (x @ x.T)[iu]
        m = float(Gall.max())
        sel = Gall > m - max(20 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 6000:
            idx = np.where(sel)[0]
            order = np.argsort(-Gall[idx])[:6000]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = Gall[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _contact_info(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = G >= (m - tol)
    deg = C.sum(axis=1)
    iu = np.triu_indices(X.shape[0], 1)
    sel = C[iu]
    edges = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return m, deg, edges

def _build_ternary():
    d = 9
    N = 3 ** d
    P = np.empty((N - 1, d), dtype=float)
    k = 0
    # enumerate base-3
    for num in range(N):
        tmp = num
        zero = True
        v = np.empty(d, dtype=float)
        for j in range(d):
            r = tmp % 3
            tmp //= 3
            if r == 0:
                v[j] = -1.0
            elif r == 1:
                v[j] = 0.0
            else:
                v[j] = 1.0
            if r != 1:
                zero = False
        if zero:  # all zeros -> digit 1 everywhere means num corresponds to 0 vector; skip
            # need check: all r==1 => zero vector
            continue
        # actually zero flag inverted: we set zero=False if any non-1? above: if r!=1 zero=False, init zero=True
        # so if still True => all r==1 => skip (already continued? we need to detect)
        # The continue above is wrong place; handle here
        P[k] = v
        k += 1
    # The loop above skipped nothing correctly because 'zero' logic flawed for num=...?
    # Fix: the all-zero vector corresponds to all r==1. For that num, zero stays True, we should skip.
    # But we already appended? Let's rebuild correctly: we appended before checking.
    # Instead filter: remove zero rows (norm 0)
    # Simpler: norms
    nrm = np.linalg.norm(P, axis=1)
    # P currently has N entries? Let's recount: we filled N slots but skipped none, k==N.
    # Actually we never skipped, so P has N rows with one zero row inside. Remove it.
    mask = nrm > 1e-12
    P = P[mask]
    P /= nrm[mask][:, None]
    return P

def _greedy_select(P, start_idx):
    M = P.shape[0]
    sel = [int(start_idx)]
    sel_mask = np.zeros(M, dtype=bool)
    sel_mask[int(start_idx)] = True
    cur = P @ P[int(start_idx)]
    cur[int(start_idx)] = np.inf
    for _ in range(1, 131):
        # argmin among unselected
        # use where
        best = -1
        # np.argmin with masked inf
        idx = int(np.argmin(np.where(sel_mask, np.inf, cur)))
        sel.append(idx)
        sel_mask[idx] = True
        d = P @ P[idx]
        # update
        # numba-free: maximum
        np.maximum(cur, d, out=cur)
        cur[idx] = np.inf
        cur[sel_mask] = np.inf
    return sel

def _swap_improve(P, sel, max_iters=120):
    sel = list(map(int, sel))
    n = len(sel)
    M = P.shape[0]
    # map pool idx -> position for forbidden check via boolean mask
    sel_mask = np.zeros(M, dtype=bool)
    for s in sel:
        sel_mask[s] = True
    X = P[np.array(sel)].copy()
    for _ in range(max_iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        row_max = G.max(axis=1)
        order = np.argsort(-row_max)
        improved = False
        for iw in order[:8]:  # try worst few
            iw = int(iw)
            old_score = float(row_max[iw])
            # others
            mask = np.ones(n, dtype=bool); mask[iw] = False
            Xo = X[mask]
            Go = Xo @ Xo.T
            np.fill_diagonal(Go, -2.0)
            m_others = float(np.max(Go))
            # scores of all pool vs Xo
            D = P @ Xo.T
            scores = D.max(axis=1)
            # forbid vectors already present in Xo
            # build forbidden set: sel without iw
            forb = [s for k, s in enumerate(sel) if k != iw]
            scores[forb] = np.inf
            bj = int(np.argmin(scores))
            bs = float(scores[bj])
            # old pool index at iw
            old_pool = sel[iw]
            if bj == old_pool:
                continue
            new_max = bs if bs > m_others else m_others
            if (new_max < m - 1e-12) or (new_max <= m + 1e-12 and bs < old_score - 1e-12):
                # accept
                sel[iw] = bj
                sel_mask[old_pool] = False
                sel_mask[bj] = True
                X[iw] = P[bj]
                improved = True
                break
        if not improved:
            break
    return sel

def solve():
    t0 = time.time()
    DEAD = t0 + 178.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy(); inc_val = best_val
    _, inc_deg, inc_edges = _contact_info(incumbent)

    # ---- algebraically structured pool ----
    try:
        P3 = _build_ternary()
    except Exception:
        P3 = None
    # subsampled quinary pool
    try:
        qN = 12000
        Q = rng.integers(-2, 3, size=(qN, d)).astype(float)
        qn = np.linalg.norm(Q, axis=1)
        keep = qn > 1e-12
        Q = Q[keep]; qn = qn[keep]
        Q /= qn[:, None]
        # dedup roughly: round and unique
        Qr = np.round(Q, 8)
        # use unique rows
        _, uidx = np.unique(Qr, axis=0, return_index=True)
        Q = Q[np.sort(uidx)]
        Pall = np.vstack([P3, Q]) if P3 is not None else Q
    except Exception:
        Pall = P3
        Q = None
    if Pall is None or len(Pall) < 500:
        # fallback: just basin hop from incumbent
        Pall = None

    disc_seeds = []
    if Pall is not None and time.time() < t0 + 20:
        P = P3 if P3 is not None else Pall
        M = P.shape[0]
        # find indices for specific structured vectors
        # coordinate e0
        e0 = np.zeros(d); e0[0] = 1.0
        dots = P @ e0
        i_e0 = int(np.argmax(dots))
        dall = np.ones(d) / np.sqrt(d)
        i_all = int(np.argmax(P @ dall))
        vdd = np.zeros(d); vdd[0] = 1/np.sqrt(2); vdd[1] = 1/np.sqrt(2)
        i_dd = int(np.argmax(P @ vdd))
        starts = [i_e0, i_all, i_dd, int(rng.integers(M)), int(rng.integers(M))]
        for si in starts:
            if time.time() > t0 + 25:
                break
            try:
                sel = _greedy_select(P, si)
                # swap on combined pool for power
                sel2 = _swap_improve(Pall, sel, max_iters=60)
                X0 = Pall[np.array(sel2)]
                disc_seeds.append(X0)
            except Exception:
                continue

    # ---- polish discrete seeds 40-60s total ----
    polished = []
    div_best = None; div_val = 1e9
    if disc_seeds:
        t_disc_end = min(t0 + 62, DEAD - 110)
        # ensure at least some time
        if t_disc_end < time.time() + 5:
            t_disc_end = time.time() + 5
        si = 0
        # also create rotated variants
        rot_seeds = []
        for X0 in list(disc_seeds):
            try:
                A = rng.standard_normal((d, d))
                Qm, _ = np.linalg.qr(A)
                if np.linalg.det(Qm) < 0:
                    Qm[:, 0] *= -1
                Xr = X0 @ Qm
                rot_seeds.append(Xr)
            except Exception:
                pass
        all_seeds = disc_seeds + rot_seeds[:3]
        for X0 in all_seeds:
            if time.time() > t_disc_end:
                break
            for rep in range(2):
                if time.time() > t_disc_end:
                    break
                Y = X0.copy()
                # small jitter to break symmetry (rotated seeds already rotated)
                Y += rng.standard_normal(Y.shape) * (0.004 if rep == 0 else 0.015)
                Y = _norm(Y)
                lr0 = float(rng.uniform(0.012, 0.028))
                lr1 = float(rng.uniform(0.0002, 0.0008))
                m0 = float(rng.uniform(0.02, 0.05))
                m1 = float(rng.uniform(0.0015, 0.004))
                steps = 2200
                if time.time() > t_disc_end - 8:
                    steps = 1200
                Y = _fast_polish(Y, steps, lr0, lr1, m0, m1)
                Y = _fast_polish(Y, 350, 0.0006, 0.00015, 0.002, 0.001)
                v = _max_cos(Y)
                polished.append((v, Y))
                # diversity metrics
                try:
                    _, deg, edges = _contact_info(Y)
                    nz = int(np.sum(deg == 0))
                    inter = len(edges.intersection(inc_edges))
                    union = len(edges.union(inc_edges)) + 1
                    overlap = inter / union
                except Exception:
                    nz = 99; overlap = 1.0
                # retain if no rattlers and different topology even if worse
                if nz == 0 and overlap < 0.70:
                    if v < div_val:
                        div_val = v; div_best = Y.copy()
                if v < best_val - 1e-12:
                    best_val = v; best = Y.copy()
                si += 1
        # quick LP on top 2 discrete (capped)
        polished_sorted = sorted(polished, key=lambda t: t[0])[:2]
        for v0, Y0 in polished_sorted:
            if time.time() > t_disc_end + 8 or time.time() > DEAD - 100:
                break
            Y1 = _lp_polish_deadline(Y0, min(time.time() + 4, DEAD - 95))
            v1 = _max_cos(Y1)
            if v1 < best_val:
                best_val = v1; best = Y1.copy()
            if v1 < div_val:
                try:
                    _, deg1, edges1 = _contact_info(Y1)
                    if int(np.sum(deg1 == 0)) == 0:
                        div_val = v1; div_best = Y1.copy()
                except Exception:
                    pass

    if div_best is None:
        div_best = best.copy(); div_val = best_val

    # ---- elitist basin hopping for remaining time ----
    def _probs(X, mv):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = G >= (mv - 1e-7)
        deg = C.sum(axis=1).astype(float)
        w = (deg + 1.0) ** 1.5
        return w / w.sum()

    try:
        probs_best = _probs(best, best_val)
    except Exception:
        probs_best = np.full(n, 1.0 / n)
    try:
        probs_div = _probs(div_best, div_val)
    except Exception:
        probs_div = np.full(n, 1.0 / n)
    it = 0
    hop_end = DEAD - 32
    while time.time() < hop_end:
        it += 1
        if it % 10 == 1:
            try:
                probs_best = _probs(best, best_val)
                probs_div = _probs(div_best, div_val)
            except Exception:
                pass
        # alternate basins: mostly best, sometimes diverse
        use_div = (it % 5 == 0) and div_best is not None
        src = div_best if use_div else best
        src_val = div_val if use_div else best_val
        probs = probs_div if use_div else probs_best
        Y = src.copy()
        cyc = it % 7
        if cyc == 0:
            Y += 0.008 * rng.standard_normal(Y.shape)
        elif cyc == 1:
            Y += 0.022 * rng.standard_normal(Y.shape)
        elif cyc == 2:
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.10 * rng.standard_normal((1, d))
        elif cyc == 3:
            idx = rng.choice(n, 2, replace=False, p=probs)
            Y[idx] += 0.10 * rng.standard_normal((2, d))
        elif cyc == 4:
            idx = rng.choice(n, 3, replace=False, p=probs)
            Y[idx] += 0.16 * rng.standard_normal((3, d))
        elif cyc == 5:
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.35 * rng.standard_normal((1, d))
        else:
            Y += 0.012 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, 1, replace=False, p=probs)
            Y[idx] += 0.12 * rng.standard_normal((1, d))
        Y = _norm(Y)
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        steps = int(rng.integers(1800, 2600))
        if cyc in (4, 5):
            steps = int(steps * 1.2)
        # cap steps if time low
        if hop_end - time.time() < 15:
            steps = min(steps, 1200)
        Y = _fast_polish(Y, steps, lr0, lr1, m0, m1)
        Y = _fast_polish(Y, 300, 0.0006, 0.00015, 0.002, 0.001)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best_val = v; best = Y.copy()
            try:
                probs_best = _probs(best, best_val)
            except Exception:
                pass
        elif use_div and v < div_val - 1e-12:
            div_val = v; div_best = Y.copy()
        # also accept diverse improvements
        if not use_div and v < div_val and v > best_val:
            try:
                _, degt, edgest = _contact_info(Y)
                if int(np.sum(degt == 0)) == 0:
                    inter = len(edgest.intersection(inc_edges))
                    union = len(edgest.union(inc_edges)) + 1
                    if inter / union < 0.70 and v < div_val - 1e-9:
                        div_val = v; div_best = Y.copy()
            except Exception:
                pass
        if time.time() > hop_end:
            break

    # final LP squeeze
    remain = DEAD - time.time()
    if remain > 6:
        best = _lp_polish_deadline(best, time.time() + min(remain - 1, 28))
        best_val = _max_cos(best)
    best = _norm(best)
    return best
