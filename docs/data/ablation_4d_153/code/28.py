import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(rm))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _degrees(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    mask = G >= m - tol
    deg = np.sum(mask, axis=1)
    return m, deg, int(np.sum(mask) // 2)


def _graph_hash(x):
    try:
        m, deg, p = _degrees(x, tol=1e-7)
        return (p, tuple(sorted(deg.tolist())))
    except Exception:
        return (0, ())


def _largest_hole(cand_rng, x, n_cand=50000, batch=10000):
    n, d = x.shape
    best_val = 1e9
    best_pt = None
    remaining = n_cand
    while remaining > 0:
        B = batch if remaining >= batch else remaining
        remaining -= B
        C = cand_rng.standard_normal(size=(B, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        G = C @ x.T
        rm = np.max(G, axis=1)
        k = int(np.argmin(rm))
        if float(rm[k]) < best_val:
            best_val = float(rm[k])
            best_pt = C[k].copy()
    return best_pt, best_val


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    EXPLORE_END = HARD_END - 30.0
    rng = np.random.default_rng(7)
    hole_rng = np.random.default_rng(12345)
    n, d = 153, 4

    base = None
    base_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < base_v:
                    base_v = v
                    base = a.copy()
        except Exception:
            pass
    if base is None:
        base = _random_sphere(n, d, rng)
        base_v = _max_cos(base)
    record_hash = _graph_hash(base)
    record_v = base_v

    candidates = [(base_v, base.copy(), record_hash)]

    def add_candidate(x, v=None):
        if x.shape[0] != 153:
            return
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        if v is None:
            v = _max_cos(x)
        h = _graph_hash(x)
        candidates.append((v, x.copy(), h))

    # ---------- overshoot buildup 153 -> 156 ----------
    try:
        x_cur = base.copy()
        snapshots = {}  # N -> array
        n_inserted_total = 0
        for Nt in (154, 155, 156):
            if time.time() > EXPLORE_END - 25.0:
                break
            # insert one point at largest hole (50k candidates)
            pt, _ = _largest_hole(hole_rng, x_cur, n_cand=50000, batch=10000)
            x_cur = np.vstack([x_cur, pt.reshape(1, d)])
            x_cur = x_cur / np.linalg.norm(x_cur, axis=1, keepdims=True)
            n_inserted_total += 1
            # joint soft-max optimise
            x_cur = _smooth_opt(x_cur, steps=500, tau0=0.04, tau1=0.004, lr0=0.04, lr1=0.004)
            # LP polish ~4 sec each
            dl = min(time.time() + 4.0, EXPLORE_END - 20.0)
            if time.time() < dl:
                x_cur, _ = _lp_polish(x_cur, dl)
            snapshots[Nt] = x_cur.copy()
    except Exception:
        snapshots = {}

    # ---------- prune snapshots back to 153 ----------
    def prune_to_153(x_start, n_inserted):
        x = x_start.copy()
        N = x.shape[0]
        inserted_idx = list(range(N - n_inserted, N)) if n_inserted > 0 else []
        while x.shape[0] > 153:
            if time.time() > EXPLORE_END - 8.0:
                break
            Ncur = x.shape[0]
            m, deg, _ = _degrees(x, tol=1e-7)
            # candidate deletion set: inserted points + high-degree points
            order = np.argsort(-deg)
            top_high = [int(i) for i in order[:8]]
            cand_set = set()
            for ii in inserted_idx:
                if 0 <= ii < Ncur:
                    # inserted indices shift after deletions; approximate by keeping last n_inserted
                    pass
            # use last k indices as proxies for inserted
            k_ins = max(0, Ncur - 153)
            # actually keep track: last n_inserted map; simpler: include last 4 indices
            for ii in range(max(0, Ncur - 4), Ncur):
                cand_set.add(ii)
            for ii in top_high:
                cand_set.add(ii)
            cand_list = sorted(cand_set)
            # raw scores without polish
            raw = []
            for c in cand_list:
                xd = np.delete(x, c, axis=0)
                raw.append((_max_cos(xd), c))
            raw.sort(key=lambda t: t[0])
            # polish top 4
            Kpol = min(4, len(raw))
            best_lx = None
            best_lv = 1e9
            for k in range(Kpol):
                if time.time() > EXPLORE_END - 5.0:
                    break
                _, c = raw[k]
                xd = np.delete(x, c, axis=0)
                # short 0.8s LP polish for scoring
                dl = time.time() + 0.8
                try:
                    yd, vd = _lp_polish(xd, dl)
                except Exception:
                    yd, vd = xd, _max_cos(xd)
                if vd < best_lv:
                    best_lv = vd
                    best_lx = yd
            if best_lx is None:
                # fallback: delete raw best
                _, c = raw[0]
                best_lx = np.delete(x, c, axis=0)
                best_lv = _max_cos(best_lx)
            x = best_lx
            # update inserted tracking: just decrement
        # brief joint soft-max + short LP after prune
        if x.shape[0] == 153 and time.time() < EXPLORE_END - 3.0:
            try:
                x = _smooth_opt(x, steps=250, tau0=0.02, tau1=0.004, lr0=0.02, lr1=0.004)
                dl = min(time.time() + 1.5, EXPLORE_END - 2.0)
                if time.time() < dl:
                    x, _ = _lp_polish(x, dl)
            except Exception:
                pass
        return x

    for Nt in (154, 155, 156):
        if Nt not in snapshots:
            continue
        if time.time() > EXPLORE_END - 10.0:
            break
        try:
            xs = snapshots[Nt]
            n_ins = Nt - 153
            xp = prune_to_153(xs, n_ins)
            if xp.shape[0] == 153:
                add_candidate(xp)
        except Exception:
            continue

    # ---------- shrink-and-regrow 153 -> 142 -> 153 ----------
    shrink_trials = 0
    while time.time() < EXPLORE_END - 12.0:
        shrink_trials += 1
        try:
            # choose source: best diverse so far or base
            # alternate between base and current best
            if len(candidates) > 1 and (shrink_trials % 2 == 0):
                # pick current best by score
                candidates.sort(key=lambda t: t[0])
                src = candidates[0][1].copy()
            else:
                src = base.copy()
            rm, _ = _row_max(src)
            rattler = int(np.argmin(rm))
            # delete rattler + 10 random points
            others = [i for i in range(153) if i != rattler]
            # deterministic-ish random via rng
            sel10 = rng.choice(others, size=10, replace=False)
            delset = set([rattler] + sel10.tolist())
            keep = [i for i in range(153) if i not in delset]
            x142 = src[keep].copy()
            # fully re-optimise at 142
            x142 = _smooth_opt(x142, steps=700, tau0=0.05, tau1=0.003, lr0=0.05, lr1=0.003)
            dl = min(time.time() + 3.0, EXPLORE_END - 10.0)
            if time.time() < dl:
                x142, _ = _lp_polish(x142, dl)
            # greedily regrow to 153
            xg = x142
            # use fewer candidates per insertion to save time except last ones
            for step in range(11):
                if time.time() > EXPLORE_END - 4.0:
                    break
                nc = 30000 if step < 10 else 50000
                pt, _ = _largest_hole(hole_rng, xg, n_cand=nc, batch=10000)
                xg = np.vstack([xg, pt.reshape(1, d)])
                xg = _smooth_opt(xg, steps=150, tau0=0.02, tau1=0.005, lr0=0.02, lr1=0.005)
                dl2 = time.time() + 0.4
                # only LP-polish every other step to save time, plus final steps
                if step >= 8 or (step % 2 == 1):
                    if time.time() < dl2 and dl2 < EXPLORE_END:
                        xg, _ = _lp_polish(xg, dl2)
            if xg.shape[0] == 153 and time.time() < EXPLORE_END:
                # brief polish
                try:
                    dl3 = min(time.time() + 1.0, EXPLORE_END)
                    if time.time() < dl3:
                        xg, _ = _lp_polish(xg, dl3)
                except Exception:
                    pass
                add_candidate(xg)
            # only do a few trials; break if time low
            if shrink_trials >= 3:
                # continue if plenty of time left
                if time.time() > EXPLORE_END - 25.0:
                    break
                if shrink_trials >= 6:
                    break
        except Exception:
            continue

    # ---------- pick best, diversity-aware ----------
    candidates.sort(key=lambda t: t[0])
    best_v, best_x, best_h = candidates[0]
    # if best ties record but hash identical, prefer a diverse near-record
    if abs(best_v - record_v) < 1e-9 and best_h == record_hash and len(candidates) > 1:
        for v, x, h in candidates[1:]:
            if h != record_hash and v < record_v + 5e-6:
                best_v, best_x = v, x.copy()
                break
    # ---------- final full LP polish (reserve 30s) ----------
    try:
        dl = HARD_END - 1.0
        best_x, best_v = _lp_polish(best_x, dl)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
