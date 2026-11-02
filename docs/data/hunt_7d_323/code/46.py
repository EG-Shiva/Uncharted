import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _worst_vec(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _contacts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _smooth(x, steps, tau0, tau1, lr0, lr1, mu, deadline, track_best=True):
    n, d = x.shape
    vel = np.zeros_like(x)
    best = _max_cos(x) if track_best else np.inf
    best_x = x.copy() if track_best else None
    # check interval for best tracking to save matmuls
    for s in range(steps):
        if (s & 15) == 0 and time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        # Gram
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights, vectorized full matrix
        # E = exp((G-m)/tau), diag 0
        # for stability clamp
        E = np.exp((G - m) / tau, out=G)  # reuse G memory
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        # gradient: (E @ x)/ (S/2?) use normalized
        Wx = (E @ x) / (S * 0.5 + 1e-30)
        # tangent projection
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        # renormalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if track_best and ((s % 100 == 99) or s == steps - 1):
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    if track_best:
        # ensure we return best seen
        return best_x, best
    return x, _max_cos(x)

def _best_void(y, rng, n_cand=20000):
    n, d = y.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ y.T
    worst = np.max(dots, axis=1)
    # take top few distinct
    k = min(8, n_cand)
    idx = np.argpartition(worst, k - 1)[:k]
    # order by worst
    idx = idx[np.argsort(worst[idx])]
    best_p = cand[idx[0]]
    best_w = float(worst[idx[0]])
    # local refine around each of top 4
    for ti in idx[:4]:
        cur = cand[ti]
        cw = float(worst[ti])
        for rnd, sig, ns in ((0, 0.06, 1500), (1, 0.02, 1500)):
            trial = cur[None, :] + rng.standard_normal(size=(ns, d)) * sig
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            dw = np.max(trial @ y.T, axis=1)
            bi = int(np.argmin(dw))
            if float(dw[bi]) < cw:
                cw = float(dw[bi])
                cur = trial[bi]
        if cw < best_w:
            best_w = cw
            best_p = cur
    return best_p / np.linalg.norm(best_p)

def _lp_polish(x, deadline, init_radius=4e-3, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2 * d] = J[:, None] * d + ar
        Cm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 150.0
    rng = np.random.default_rng(20261101)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        x0 = rng.standard_normal(size=(323, 7))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = _max_cos(x0)
    x_best = x0.copy(); best = float(b0)
    n, d = 323, 7

    # ---------- cardinality homotopy forward ----------
    k_list = [12, 10, 14, 8, 11, 13]
    attempt = 0
    while time.time() < EXPLORE_END - 6 and time.time() < HARD - 25:
        k_remove = k_list[attempt % len(k_list)]
        # identify rattlers + hubs from current best
        mcur = best
        cc = _contacts(x_best, mcur, tol=1e-9)
        ratt = np.where(cc == 0)[0].tolist()
        order_desc = np.argsort(-cc)
        # build removal set: 2 rattlers (if exist) + hubs + random
        torep = []
        for r in ratt:
            torep.append(int(r))
            if len(torep) >= 2:
                break
        # add hubs
        for o in order_desc:
            if len(torep) >= k_remove - 2:
                break
            if int(o) not in torep:
                torep.append(int(o))
        # fill remainder randomly
        while len(torep) < k_remove:
            c = int(rng.integers(0, n))
            if c not in torep:
                torep.append(c)
        keep = np.ones(n, dtype=bool); keep[torep] = False
        y = x_best[keep].copy()
        # optimise subcode N=309..317 until well below record
        dl1 = min(time.time() + 2.2, EXPLORE_END - 4, HARD - 24)
        # annealed: start broad, end sharp
        y, vy = _smooth(y, steps=1100, tau0=0.022, tau1=0.0030, lr0=2.6, lr1=0.45, mu=0.82, deadline=dl1, track_best=True)
        # if subcode not well below record, still try regrow but with small budget; skip if terrible
        # regrow one at a time
        dl_regrow = min(time.time() + 4.5, EXPLORE_END - 2, HARD - 23)
        cur = y
        # need to grow back to 323
        need = n - cur.shape[0]
        ok = True
        for gi in range(need):
            if time.time() >= dl_regrow:
                ok = False
                break
            pnew = _best_void(cur, rng, n_cand=16000)
            cur = np.vstack([cur, pnew[None, :]])
            # allow all points to move after each insertion (short relax)
            dl_s = min(time.time() + 0.45, dl_regrow)
            # fewer steps per insertion, sharper tau
            cur, _ = _smooth(cur, steps=180, tau0=0.009, tau1=0.0025, lr0=1.6, lr1=0.35, mu=0.78, deadline=dl_s, track_best=False)
        if not ok or cur.shape[0] != n:
            attempt += 1
            continue
        # slightly longer joint relax before scoring (unpolished)
        dl_j = min(time.time() + 0.8, EXPLORE_END - 1, HARD - 22)
        cur, _ = _smooth(cur, steps=350, tau0=0.008, tau1=0.0020, lr0=1.4, lr1=0.30, mu=0.80, deadline=dl_j, track_best=True)
        v = _max_cos(cur)
        # retain only if competitive unpolished (within 4e-3 of best, since polish can gain ~1e-3 at most; be permissive but not too loose)
        if v < best - 1e-12 and np.all(np.isfinite(cur)):
            best = float(v); x_best = cur.copy()
        elif v < best + 0.002 and attempt % 3 == 2:
            # occasional near-miss: try a quick extra anneal to see if it dips
            dl_x = min(time.time() + 0.5, EXPLORE_END, HARD - 22)
            cur2, v2 = _smooth(cur, steps=300, tau0=0.006, tau1=0.0018, lr0=1.0, lr1=0.25, mu=0.75, deadline=dl_x, track_best=True)
            if v2 < best - 1e-12:
                best = float(v2); x_best = cur2.copy()
        attempt += 1
        if attempt >= 9:
            break
        if time.time() >= EXPLORE_END - 6:
            break

    # ---------- reverse route: overfull N=330 then delete 7 ----------
    if time.time() < EXPLORE_END - 4 and time.time() < HARD - 22:
        try:
            # build overfull from x_best + 7 voids
            base = x_best.copy()
            over = base
            for _ in range(7):
                pnew = _best_void(over, rng, n_cand=12000)
                over = np.vstack([over, pnew[None, :]])
            dl_o = min(time.time() + 2.0, EXPLORE_END - 2, HARD - 21)
            over, _ = _smooth(over, steps=800, tau0=0.018, tau1=0.0028, lr0=2.2, lr1=0.4, mu=0.80, deadline=dl_o, track_best=True)
            # delete 7 worst-contact points
            w = _worst_vec(over)
            del_idx = np.argpartition(w, -7)[-7:]
            keep2 = np.ones(over.shape[0], dtype=bool); keep2[del_idx] = False
            back = over[keep2].copy()
            dl_b = min(time.time() + 1.2, EXPLORE_END - 1, HARD - 20)
            back, vb = _smooth(back, steps=450, tau0=0.009, tau1=0.0020, lr0=1.5, lr1=0.30, mu=0.80, deadline=dl_b, track_best=True)
            vb = _max_cos(back)
            if vb < best - 1e-12 and np.all(np.isfinite(back)):
                best = float(vb); x_best = back.copy()
        except Exception:
            pass

    # ---------- one bounded LP polish before deadline ----------
    if time.time() < HARD - 2:
        try:
            dl_f = min(HARD - 1.0, time.time() + max(3.0, HARD - 1.0 - time.time()))
            # cap polish time to ~18s max to guarantee return
            dl_f = min(dl_f, time.time() + 18.0)
            x_best = _lp_polish(x_best, deadline=dl_f, init_radius=4e-3, max_pairs=2200)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize strictly
    bad = ~np.all(np.isfinite(x_best), axis=1)
    if np.any(bad):
        for i in np.where(bad)[0]:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
