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
    return np.max(G, axis=1), float(np.max(G))


def _smooth(x0, steps, tau0, tau1, lr0, lr1, beta=0.9):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gm = float(G.max())
        if Gm < bv:
            bv = Gm
            bx = x.copy()
        E = np.exp((G - Gm) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-30)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x /= nrm
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        np.fill_diagonal(Gfull, -2.0)
        m = float(Gfull.max())
        c_full = (x @ x.T)[iu]
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        # cap pairs
        idx = np.where(sel)[0]
        if len(idx) > 3500:
            order = np.argpartition(c_full[idx], -3500)[-3500:]
            idx = idx[order]
            I = iu[0][idx]
            J = iu[1][idx]
            c = c_full[idx]
        else:
            I, J = iu[0][sel], iu[1][sel]
            c = c_full[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def _largest_hole(x, rng, n_cand=50000, batch=10000):
    n, d = x.shape
    best_c = None
    best_v = 1e9
    nb = n_cand // batch
    for _ in range(nb):
        C = rng.standard_normal(size=(batch, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ x.T
        rm = S.max(axis=1)
        i = int(np.argmin(rm))
        if float(rm[i]) < best_v:
            best_v = float(rm[i])
            best_c = C[i].copy()
    rem = n_cand - nb * batch
    if rem > 0:
        C = rng.standard_normal(size=(rem, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ x.T
        rm = S.max(axis=1)
        i = int(np.argmin(rm))
        if float(rm[i]) < best_v:
            best_v = float(rm[i])
            best_c = C[i].copy()
    return best_c


def _degrees(x, tol=1e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    B = G > (m - tol)
    deg = B.sum(axis=1)
    return deg, m


def _graph_hash(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    B = G > (m - tol)
    deg = sorted(B.sum(axis=1).tolist())
    cnt = int(B.sum() // 2)
    return (round(m, 9), cnt, tuple(deg))


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    FINAL_RESERVE = 30.0
    EXPLORE_END = HARD_END - FINAL_RESERVE - 2.0
    rng = np.random.default_rng(0)

    # load start
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        a = rng.standard_normal((153, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    base153 = cand[bi].copy()
    base_val = float(vals[bi])
    record_hash = _graph_hash(base153)

    endpoints = [(base_val, base153.copy(), record_hash)]

    # ---------- Overshoot 153 -> 154,155,156 ----------
    try:
        cur = base153.copy()
        inserted = []  # indices in cur
        targets = [154, 155, 156]
        for tgt in targets:
            if time.time() > EXPLORE_END - 25:
                break
            hole = _largest_hole(cur, rng, n_cand=50000)
            cur = np.vstack([cur, hole.reshape(1, 4)])
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            inserted.append(cur.shape[0] - 1)
            bx, _ = _smooth(cur, 700, 0.02, 0.0015, 0.05, 0.004, beta=0.9)
            cur = bx
            if time.time() > EXPLORE_END - 20:
                continue
            cur = _lp_polish(cur, 2.0, deadline=EXPLORE_END)
        # prune back to 153 greedily
        while cur.shape[0] > 153 and time.time() < EXPLORE_END - 5:
            ncur = cur.shape[0]
            deg, _ = _degrees(cur, tol=1e-4)
            order = np.argsort(-deg)
            cands = []
            for ii in inserted:
                if 0 <= ii < ncur and ii not in cands:
                    cands.append(ii)
            for o in order:
                if len(cands) >= 10:
                    break
                if int(o) not in cands:
                    cands.append(int(o))
            # evaluate each deletion with short LP polish
            best_del_val = 1e9
            best_del_x = None
            best_del_idx = cands[0]
            for di in cands:
                if time.time() > EXPLORE_END - 3:
                    break
                y = np.delete(cur, di, axis=0)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                # short polish ~0.7s
                yp = _lp_polish(y, 0.7, deadline=EXPLORE_END)
                v = _max_cos(yp)
                if v < best_del_val:
                    best_del_val = v
                    best_del_x = yp
                    best_del_idx = di
            # update inserted indices after deletion
            new_inserted = []
            for ii in inserted:
                if ii == best_del_idx:
                    continue
                elif ii > best_del_idx:
                    new_inserted.append(ii - 1)
                else:
                    new_inserted.append(ii)
            inserted = new_inserted
            cur = best_del_x
            # brief smooth to diversify slightly if stuck at record hash?
            # keep as is
        if cur.shape[0] == 153:
            # medium polish
            if time.time() < EXPLORE_END - 3:
                cur = _lp_polish(cur, min(2.0, max(0.5, EXPLORE_END - time.time() - 1.0)))
            v = _max_cos(cur)
            h = _graph_hash(cur)
            endpoints.append((v, cur.copy(), h))
    except Exception:
        pass

    # ---------- Shrink to 142 and regrow ----------
    n_regrow_trials = 0
    while time.time() < EXPLORE_END - 12 and n_regrow_trials < 2:
        try:
            n_regrow_trials += 1
            rng2 = np.random.default_rng(100 + n_regrow_trials)
            start = base153.copy()
            # also try from best endpoint so far
            if len(endpoints) > 1:
                # pick current best
                endpoints_sorted = sorted(endpoints, key=lambda e: e[0])
                if endpoints_sorted[0][0] < base_val - 1e-12:
                    start = endpoints_sorted[0][1].copy()
            rm, _ = _row_max(start)
            rattler = int(np.argmin(rm))
            perm = rng2.permutation(153)
            delset = {rattler}
            for p in perm:
                if len(delset) >= 11:
                    break
                delset.add(int(p))
            # delset size 11 -> 142 points
            keep = [i for i in range(153) if i not in delset]
            x142 = start[keep].copy()
            # fully re-optimise at 142
            bx, _ = _smooth(x142, 1200, 0.025, 0.0012, 0.06, 0.003, beta=0.9)
            x142 = bx
            if time.time() < EXPLORE_END - 10:
                x142 = _lp_polish(x142, min(4.0, EXPLORE_END - time.time() - 8.0))
            # greedily regrow to 153
            cur2 = x142
            for k in range(11):
                if time.time() > EXPLORE_END - 4:
                    break
                hole = _largest_hole(cur2, rng2, n_cand=30000)
                cur2 = np.vstack([cur2, hole.reshape(1, 4)])
                cur2 /= np.linalg.norm(cur2, axis=1, keepdims=True)
                bx2, _ = _smooth(cur2, 250, 0.015, 0.002, 0.04, 0.005, beta=0.9)
                cur2 = bx2
                # brief joint polish
                cur2 = _lp_polish(cur2, 0.35, deadline=EXPLORE_END)
            if cur2.shape[0] == 153:
                if time.time() < EXPLORE_END - 2:
                    cur2 = _lp_polish(cur2, min(2.0, EXPLORE_END - time.time() - 0.5))
                v2 = _max_cos(cur2)
                h2 = _graph_hash(cur2)
                endpoints.append((v2, cur2.copy(), h2))
                if v2 < base_val - 1e-9:
                    base153 = cur2.copy()
                    base_val = v2
        except Exception:
            continue

    # pick best endpoint (prefer diverse hash on near-tie)
    endpoints_sorted = sorted(endpoints, key=lambda e: e[0])
    best_v, best_x = endpoints_sorted[0][0], endpoints_sorted[0][1].copy()
    # diversity: if best equals record basin hash and second is within 2e-6 but diverse, still keep best (score matters)
    # no switch; just keep lowest

    # final full LP polish with reserved time
    remain = HARD_END - time.time() - 1.5
    if remain > 2:
        use = min(remain, FINAL_RESERVE)
        best_x = _lp_polish(best_x, use, deadline=HARD_END - 1.0)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
