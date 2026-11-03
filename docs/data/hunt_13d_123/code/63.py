import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x


def _find_holes(X, k, rng, n_cand=60000, n_ref=20, it_ref=200):
    m, d = X.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ X.T
    md = dots.max(axis=1)
    idx = np.argpartition(md, n_ref)[:n_ref]
    base = cand[idx]
    refined = np.empty((n_ref, d))
    for bi in range(n_ref):
        y = base[bi].copy()
        for it in range(it_ref):
            ds = X @ y
            mv = float(ds.max())
            w = np.exp((ds - mv) / 0.005)
            s = w.sum()
            if s == 0:
                break
            w /= s
            grad = w @ X
            gt = grad - float(grad @ y) * y
            lr = 0.05 * (1.0 - it / it_ref) + 0.005
            y = y - lr * gt
            nrm = np.linalg.norm(y)
            if nrm == 0:
                break
            y /= nrm
        refined[bi] = y
    scores = (refined @ X.T).max(axis=1)
    order = np.argsort(scores)
    chosen = []
    for oi in order:
        y = refined[oi]
        ok = True
        for c in chosen:
            if float(y @ c) > 0.95:
                ok = False
                break
        if ok:
            chosen.append(y)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for oi in order:
            if len(chosen) >= k:
                break
            y = refined[oi]
            dup = False
            for c in chosen:
                if np.allclose(y, c):
                    dup = True
                    break
            if not dup:
                chosen.append(y)
    H = np.array(chosen[:k])
    cover = float(scores[order[0]])
    return H, cover


def _optimize_retained(y, k_need, deadline, rng, lam=0.45):
    m, d = y.shape
    try:
        yh, _ = _find_holes(y, k_need, rng, n_cand=60000)
    except Exception:
        yh = None
    if yh is None or yh.shape[0] == 0:
        yh = rng.standard_normal(size=(max(1, k_need), d))
        yh /= np.linalg.norm(yh, axis=1, keepdims=True)
    outer = 0
    while time.time() < deadline - 0.6:
        outer += 1
        steps = 300
        if outer % 3 == 1:
            T0, T1, lr0, lr1 = 0.02, 0.003, 0.40, 0.05
        elif outer % 3 == 2:
            T0, T1, lr0, lr1 = 0.012, 0.002, 0.30, 0.03
        else:
            T0, T1, lr0, lr1 = 0.028, 0.005, 0.45, 0.06
        Th = 0.01
        for s in range(steps):
            if time.time() >= deadline - 0.4:
                break
            frac = s / max(steps - 1, 1)
            T = T0 + (T1 - T0) * frac
            lr = lr0 + (lr1 - lr0) * frac
            G = y @ y.T
            np.fill_diagonal(G, -10.0)
            mv = float(G.max())
            E = np.exp((G - mv) / T)
            np.fill_diagonal(E, 0.0)
            row = E.sum(axis=1, keepdims=True) + 1e-20
            Avg = (E @ y) / row
            dot = (Avg * y).sum(axis=1, keepdims=True)
            Gp = Avg - dot * y
            try:
                D = y @ yh.T
                maxj = D.max(axis=0, keepdims=True)
                Eh = np.exp((D - maxj) / Th)
                colsum = Eh.sum(axis=0, keepdims=True) + 1e-20
                Ehn = Eh / colsum
                A = Ehn @ yh
                sdot = (A * y).sum(axis=1, keepdims=True)
                Gh = A - sdot * y
                Gh = Gh * (m / max(1, yh.shape[0]))
                Gt = Gp + lam * Gh
            except Exception:
                Gt = Gp
            y = y - lr * 0.2 * Gt
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y /= yn
        if time.time() >= deadline - 2.0:
            break
        try:
            yh2, _ = _find_holes(y, k_need, rng, n_cand=50000, n_ref=12, it_ref=120)
            if yh2.shape[0] == k_need:
                yh = yh2
        except Exception:
            pass
    return y


def solve():
    t0 = time.time()
    DEADLINE = t0 + 160.0
    HARD = t0 + 177.0
    n, d = 123, 13
    rng = np.random.default_rng(0)
    np.random.seed(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # contact analysis for removal sets
    try:
        G0 = best @ best.T
        np.fill_diagonal(G0, -2.0)
        mx = float(G0.max())
        deg = (G0 > mx - 1e-9).sum(axis=1)
        order = np.argsort(deg, kind="stable")
    except Exception:
        order = np.arange(n)
        deg = np.zeros(n)
    rem_sets = []
    rem_sets.append([int(order[0])])
    for k in (2, 3, 4):
        rem_sets.append([int(x) for x in order[:k]])
    # dedup
    uniq = []
    for s in rem_sets:
        if s not in uniq:
            uniq.append(s)
    rem_sets = uniq
    # time allocation: split remaining time among sets + final
    for si, rem in enumerate(rem_sets):
        if time.time() >= DEADLINE - 8:
            break
        k = len(rem)
        nsets_left = len(rem_sets) - si
        time_left = DEADLINE - time.time()
        # reserve ~6s per remaining set for LP + reinsert
        budget = time_left / nsets_left
        budget = min(max(budget, 8.0), 32.0)
        sub_deadline = time.time() + max(3.0, budget - 7.0)
        if sub_deadline >= DEADLINE - 6:
            sub_deadline = DEADLINE - 6
        mask = np.ones(n, dtype=bool)
        mask[np.array(rem, dtype=int)] = False
        y0 = best[mask].copy()
        # optimize retained with hole reward
        try:
            yopt = _optimize_retained(y0, k, min(sub_deadline, HARD - 8), rng)
        except Exception:
            yopt = y0
        if time.time() >= HARD - 16:
            # still try reinsert quickly
            pass
        # sequential reinsertion at refined deepest holes
        try:
            cur = yopt.copy()
            for j in range(k):
                if time.time() >= HARD - 14:
                    break
                need = 1
                H, _ = _find_holes(cur, 1, rng, n_cand=60000, n_ref=16, it_ref=200)
                h = H[0]
                h = h / np.linalg.norm(h)
                cur = np.vstack([cur, h[None, :]])
            full = cur
            if full.shape[0] != n:
                # pad/truncate fallback
                if full.shape[0] > n:
                    full = full[:n]
                else:
                    extra = n - full.shape[0]
                    e = rng.standard_normal(size=(extra, d))
                    e /= np.linalg.norm(e, axis=1, keepdims=True)
                    full = np.vstack([full, e])
            full /= np.linalg.norm(full, axis=1, keepdims=True)
        except Exception:
            continue
        # strictly time-capped LP polish (at most 15s, here <=6s)
        lp_sec = min(6.0, max(1.5, DEADLINE - time.time() - 2.0))
        lp_sec = min(lp_sec, 15.0)
        if lp_sec < 1.0:
            v = _max_cos(full)
            if v < best_val:
                best_val = v
                best = full.copy()
            continue
        try:
            pol = _lp_polish(full, lp_sec)
            v = _max_cos(pol)
        except Exception:
            pol = full
            v = _max_cos(pol)
        if v < best_val:
            best_val = v
            best = pol.copy()
            # update order for later sets based on new best
            try:
                G0 = best @ best.T
                np.fill_diagonal(G0, -2.0)
                mx = float(G0.max())
                deg = (G0 > mx - 1e-9).sum(axis=1)
                order = np.argsort(deg, kind="stable")
            except Exception:
                pass
        if time.time() >= DEADLINE - 4:
            break
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
