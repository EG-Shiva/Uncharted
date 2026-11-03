import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x0, steps, a0, a1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_val = _max_cos(x)
    best_x = x.copy()
    la0, la1 = np.log(a0), np.log(a1)
    b1p, b2p = 1.0, 1.0
    arange_d = None
    for t in range(1, steps + 1):
        alpha = np.exp(la0 + (la1 - la0) * t / steps)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp((G - gmax) * alpha)
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr[:, 0] < 1e-12, :] = 1.0
        x /= nr
        if (t % 100 == 0) or (t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish_deadline(x, deadline, radius_init=0.002):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > 3000:
            part = np.argpartition(-c_all[sel_idx], 3000)[:3000]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        csel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        rows_I = np.repeat(np.arange(p), d)
        rows = np.concatenate([rows_I, np.repeat(np.arange(p), d), np.arange(p)], axis=0)
        cols = np.concatenate([(I[:, None] * d + ar).reshape(-1),
                               (J[:, None] * d + ar).reshape(-1),
                               np.full(p, n * d)], axis=0)
        vals = np.concatenate([x[J].reshape(-1), x[I].reshape(-1), -np.ones(p)], axis=0)
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn[:, 0] < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        val = _max_cos(y)
        if val < best:
            best = val
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if val <= _max_cos(x) + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    TOTAL = 176.0
    ANNEAL_END = t0 + 148.0
    FINAL_END = t0 + 174.5
    rng = np.random.default_rng(0)
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            x_best = np.load(fn)
            break
        except Exception:
            continue
    if x_best is None:
        x_best = rng.standard_normal((256, 6))
    x_best = np.asarray(x_best, dtype=float)
    if x_best.shape != (256, 6):
        tmp = rng.standard_normal((256, 6))
        try:
            r = min(x_best.shape[0], 256)
            c = min(x_best.shape[1], 6)
            tmp[:r, :c] = x_best[:r, :c]
        except Exception:
            pass
        x_best = tmp
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    xn = np.linalg.norm(x_best, axis=1, keepdims=True)
    xn[xn[:, 0] < 1e-12] = 1.0
    x_best = x_best / xn
    best_val = _max_cos(x_best)
    n, d = 256, 6
    cycle = 0
    while time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - time.time()
        if rem < 1.0:
            break
        r = rng.random()
        x = None
        kind = ""
        if r < 0.35:
            sd = float(rng.choice(np.array([0.008, 0.015, 0.03, 0.06])))
            x = x_best + sd * rng.standard_normal((n, d))
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            kind = "g"
        elif r < 0.55:
            k = int(rng.integers(4, 26))
            idx = rng.choice(n, size=k, replace=False)
            sd = float(rng.uniform(0.08, 0.38))
            x = x_best.copy()
            x[idx] = x[idx] + sd * rng.standard_normal((k, d))
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            kind = "s"
        elif r < 0.80:
            # void-targeted reinsert of worst/hub or rattler
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m0 = G.max()
            per = np.sum(G >= m0 - 1e-9, axis=1)
            if rng.random() < 0.35:
                ratt = np.where(per == 0)[0]
                if len(ratt) > 0:
                    k = min(len(ratt), int(rng.integers(1, 3)))
                    rep = rng.choice(ratt, size=k, replace=False)
                else:
                    k = int(rng.integers(1, 3))
                    rep = np.argsort(-per)[:k]
            else:
                k = int(rng.integers(1, 3))
                pool = np.argsort(-per)[:20]
                rep = rng.choice(pool, size=k, replace=False)
            x = x_best.copy()
            for w in rep:
                mask = np.ones(n, dtype=bool)
                mask[w] = False
                rest = x[mask]
                ncand = 9000
                cand = rng.standard_normal((ncand, d))
                cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ rest.T
                md = dots.max(axis=1)
                bi = int(np.argmin(md))
                bc = cand[bi].copy()
                for _ in range(35):
                    dd = rest @ bc
                    j = int(np.argmax(dd))
                    gv = rest[j] - (rest[j] @ bc) * bc
                    bc = bc - 0.05 * gv
                    nn = np.linalg.norm(bc)
                    if not np.isfinite(nn) or nn < 1e-12:
                        break
                    bc = bc / nn
                x[w] = bc
            x = x + 0.0025 * rng.standard_normal((n, d))
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            kind = "v"
        else:
            # cap rotation: preserve intra-cap geometry, change inter-cap contacts
            u = rng.standard_normal(d)
            u = u / np.linalg.norm(u)
            dots = x_best @ u
            mcap = int(rng.integers(15, 46))
            cap = np.argsort(-dots)[:mcap]
            x = x_best.copy()
            cap_pts = x[cap].copy()
            nrot = int(rng.integers(3, 6))
            thmax = float(rng.uniform(0.15, 0.55))
            for _ in range(nrot):
                a, b = rng.choice(d, size=2, replace=False)
                th = float(rng.uniform(-thmax, thmax))
                c, s = np.cos(th), np.sin(th)
                ci = cap_pts[:, a].copy()
                cj = cap_pts[:, b].copy()
                cap_pts[:, a] = c * ci - s * cj
                cap_pts[:, b] = s * ci + c * cj
            x[cap] = cap_pts
            x = x + 0.002 * rng.standard_normal((n, d))
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            kind = "c"
        # anneal params
        if kind == "g":
            steps = int(rng.integers(450, 800))
        elif kind == "s":
            steps = int(rng.integers(600, 1000))
        else:
            steps = int(rng.integers(700, 1150))
        a0 = float(rng.uniform(15.0, 40.0))
        a1 = float(rng.uniform(200.0, 520.0))
        lr = float(rng.uniform(0.007, 0.018))
        xa, va = _anneal(x, steps, a0, a1, lr)
        if va < best_val - 1e-12:
            dl = min(time.time() + 1.3, FINAL_END - 1.0)
            if dl > time.time() + 0.3:
                yp, vp = _lp_polish_deadline(xa, dl, radius_init=0.003)
            else:
                yp, vp = xa, va
            if vp < best_val - 1e-12:
                x_best = yp
                best_val = vp
        elif va < best_val + 0.0025:
            do_lp = False
            if va < best_val + 0.0004:
                do_lp = True
            elif (cycle % 3 == 0) and (va < best_val + 0.0015):
                do_lp = True
            if do_lp:
                dl = min(time.time() + 0.7, FINAL_END - 1.0)
                if dl > time.time() + 0.3:
                    yp, vp = _lp_polish_deadline(xa, dl, radius_init=0.0025)
                    if vp < best_val - 1e-12:
                        x_best = yp
                        best_val = vp
        if time.time() >= ANNEAL_END:
            break
    # final long polish
    remain = FINAL_END - time.time()
    if remain > 1.5:
        dl = time.time() + min(remain, 22.0)
        fp, fv = _lp_polish_deadline(x_best, dl, radius_init=0.0018)
        if fv < best_val:
            x_best = fp
            best_val = fv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
