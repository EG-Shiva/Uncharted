import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(0)

    # ---------- helpers ----------
    def max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def per_point_max(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    def contact_counts(x, thr=1e-7):
        n = x.shape[0]
        iu = np.triu_indices(n, 1)
        g = (x @ x.T)[iu]
        m = float(g.max())
        tight = g > m - thr
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        return m, cnt

    def smooth_full(x0, steps, beta0, beta1, lr0, lr1):
        n, d = x0.shape
        x = x0.copy()
        Iu, Ju = np.triu_indices(n, 1)
        m1 = np.zeros_like(x); v1 = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lb0, lb1 = np.log(beta0), np.log(beta1)
        la0, la1 = np.log(lr0), np.log(lr1)
        for s in range(steps):
            if time.time() - t0 > HARD - 3.0:
                break
            f = s / max(steps - 1, 1)
            beta = np.exp(lb0 + (lb1 - lb0) * f)
            lr = np.exp(la0 + (la1 - la0) * f)
            G = x @ x.T
            g = G[Iu, Ju]
            gm = g.max()
            e = np.exp(beta * (g - gm))
            w = e / e.sum()
            W = np.zeros((n, n))
            W[Iu, Ju] = w
            W = W + W.T
            grad = W @ x
            dot = (grad * x).sum(axis=1, keepdims=True)
            gt = grad - dot * x
            m1 = b1 * m1 + (1 - b1) * gt
            v1 = b2 * v1 + (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1 ** (s + 1))
            vhat = v1 / (1 - b2 ** (s + 1))
            x = x - lr * mhat / (np.sqrt(vhat) + eps)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def short_smooth(x0, steps, beta0, beta1, lr0, lr1):
        return smooth_full(x0, steps, beta0, beta1, lr0, lr1)

    def lp_polish(x, budget):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        n, d = x.shape
        try:
            best = max_cos(x)
        except Exception:
            return x, 9.0
        radius = 1e-3
        start = time.time()
        iu0 = np.triu_indices(n, 1)
        while time.time() - start < budget and radius > 1e-14:
            if time.time() - t0 > HARD - 2.0:
                break
            G = x @ x.T
            c_all = G[iu0]
            m = c_all.max()
            sel = c_all > m - max(20 * radius, 1e-9)
            I = iu0[0][sel]; J = iu0[1][sel]
            csel = c_all[sel]
            p = len(I)
            if p == 0:
                break
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
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, best

    def largest_void(x_keep, n_cand=20000):
        cand = rng.normal(size=(n_cand, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x_keep.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        v = cand[bi].copy()
        best_md = float(md[bi])
        sig = 0.03
        for it in range(120):
            prop = v[None, :] + rng.normal(0, sig, size=(32, 4))
            prop /= np.linalg.norm(prop, axis=1, keepdims=True)
            mds = (prop @ x_keep.T).max(axis=1)
            k = int(np.argmin(mds))
            if mds[k] < best_md:
                v = prop[k].copy()
                best_md = float(mds[k])
            sig *= 0.992
            if it % 40 == 39:
                sig = max(sig, 0.004)
        return v

    def opt_full(x0, budget):
        # split budget: smooth then LP then light smooth
        t_s = time.time()
        if budget <= 0.3:
            return x0, max_cos(x0)
        # first anneal
        x = smooth_full(x0, steps=650, beta0=50.0, beta1=1500.0, lr0=0.005, lr1=0.00035)
        remain = budget - (time.time() - t_s)
        if remain < 0.4:
            return x, max_cos(x)
        # LP with most of remain
        lp_bud = max(0.4, remain - 0.6)
        # keep a bit for final smoothing
        x, v = lp_polish(x, lp_bud)
        remain2 = budget - (time.time() - t_s)
        if remain2 > 1.0:
            x2 = smooth_full(x, steps=300, beta0=90.0, beta1=2500.0, lr0=0.002, lr1=0.0002)
            v2 = max_cos(x2)
            if v2 < v:
                x, v = x2, v2
                r3 = budget - (time.time() - t_s)
                if r3 > 0.5:
                    x, v = lp_polish(x, min(r3, 1.5))
        return x, max_cos(x)

    # ---------- load ----------
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best153 = max_cos(xb)
    x_best = xb.copy()

    def time_left():
        return HARD - (time.time() - t0)

    # small helper to choose removal index (rattler first, else highest per-point max)
    def choose_removal(x):
        _, cnt = contact_counts(x, thr=1e-7)
        ratt = np.where(cnt == 0)[0]
        if len(ratt) > 0:
            return int(ratt[0])
        ppm = per_point_max(x)
        return int(np.argmax(ppm))

    bank = {}

    # ---------- forward: descend 152..148 ----------
    try:
        x_cur = xb.copy()
        for N in [152, 151, 150, 149, 148]:
            if time_left() < 25:
                break
            idx = choose_removal(x_cur)
            mask = np.ones(x_cur.shape[0], dtype=bool)
            mask[idx] = False
            x_init = x_cur[mask].copy()
            bud = 7.0 if N != 148 else 6.0
            bud = min(bud, max(0.8, time_left() - 22.0))
            if bud < 0.8:
                break
            x_opt, v_opt = opt_full(x_init, bud)
            bank[N] = x_opt.copy()
            x_cur = x_opt.copy()
    except Exception:
        pass

    # ---------- forward: ascend 149..153 from 148 optimum ----------
    try:
        if 148 in bank:
            y_cur = bank[148].copy()
        else:
            # build 148 by removing 5 worst from record
            tmp = xb.copy()
            for _ in range(5):
                idx = choose_removal(tmp)
                m = np.ones(tmp.shape[0], dtype=bool); m[idx] = False
                tmp = tmp[m]
            y_cur = tmp
            y_cur, _ = opt_full(y_cur, min(5.0, max(1.0, time_left() - 60)))
        for N in [149, 150, 151, 152, 153]:
            if time_left() < 28:
                break
            v = largest_void(y_cur)
            v = v + rng.normal(0, 0.002, size=4)
            v /= np.linalg.norm(v)
            x_init = np.vstack([y_cur, v[None, :]])
            bud = 7.0
            if N == 153:
                bud = 8.0
            bud = min(bud, max(0.8, time_left() - 20.0))
            if bud < 0.8:
                break
            x_opt, v_opt = opt_full(x_init, bud)
            bank[N] = x_opt.copy()
            y_cur = x_opt.copy()
            if N == 153 and v_opt < best153 - 1e-12:
                best153 = float(v_opt)
                x_best = x_opt.copy()
    except Exception:
        pass

    # ---------- reverse: grow 154..160 from record ----------
    try:
        z_cur = xb.copy()
        # if forward 153 improved, seed from best? keep record basin for diversity: use xb
        for N in [154, 155, 156, 157, 158, 159, 160]:
            if time_left() < 22:
                break
            v = largest_void(z_cur)
            v = v + rng.normal(0, 0.002, size=4)
            v /= np.linalg.norm(v)
            x_init = np.vstack([z_cur, v[None, :]])
            bud = 3.5
            bud = min(bud, max(0.8, time_left() - 18.0))
            if bud < 0.7:
                break
            x_opt, v_opt = opt_full(x_init, bud)
            bank[N] = x_opt.copy()
            z_cur = x_opt.copy()
    except Exception:
        pass

    # ---------- reverse: delete 160..153 greedily ----------
    try:
        if 160 not in bank and 159 not in bank and 158 not in bank:
            raise RuntimeError("no high-N bank")
        # start from highest available
        for _N in [160, 159, 158, 157, 156, 155, 154]:
            if _N in bank:
                w_cur = bank[_N].copy()
                curN = _N
                break
        else:
            raise RuntimeError("empty")
        # descend until 153
        while curN > 153:
            if time_left() < 14:
                break
            n = w_cur.shape[0]
            ppm = per_point_max(w_cur)
            # candidate removals: top-8 per-point-max
            order = np.argsort(-ppm)[:min(8, n)]
            # quick score without reopt
            G = w_cur @ w_cur.T
            np.fill_diagonal(G, -2.0)
            quick = []
            for idx in order:
                m = np.ones(n, dtype=bool); m[idx] = False
                quick.append((float(G[np.ix_(m, m)].max()), int(idx)))
            quick.sort(key=lambda t: t[0])
            shortlist = [idx for _, idx in quick[:4]]
            best_cand_val = 9.0
            best_cand_x = None
            for idx in shortlist:
                m = np.ones(n, dtype=bool); m[idx] = False
                x_init = w_cur[m].copy()
                # short re-optimisation to recompute
                xs = short_smooth(x_init, steps=220, beta0=60.0, beta1=1200.0, lr0=0.004, lr1=0.0004)
                vs = max_cos(xs)
                if vs < best_cand_val:
                    best_cand_val = vs
                    best_cand_x = xs
            # full optimise the winner to N-1
            bud = 6.0
            bud = min(bud, max(0.8, time_left() - 10.0))
            if bud < 0.7:
                # accept short version without full opt
                w_cur = best_cand_x.copy()
                curN = w_cur.shape[0]
                bank[curN] = w_cur.copy()
                continue
            x_opt, v_opt = opt_full(best_cand_x, bud)
            w_cur = x_opt.copy()
            curN = w_cur.shape[0]
            bank[curN] = w_cur.copy()
            if curN == 153 and v_opt < best153 - 1e-12:
                best153 = float(v_opt)
                x_best = x_opt.copy()
    except Exception:
        pass

    # if we have a bank 153 not yet recorded, check it
    try:
        if 153 in bank:
            v = max_cos(bank[153])
            if v < best153 - 1e-12:
                best153 = float(v)
                x_best = bank[153].copy()
            elif best153 >= max_cos(xb) - 1e-12:
                # keep whichever is better between bank and current best
                if v < best153:
                    best153 = float(v)
                    x_best = bank[153].copy()
    except Exception:
        pass

    # ---------- final LP polish on best ----------
    try:
        fb = max(1.0, time_left() - 2.0)
        fb = min(fb, 18.0)
        if fb > 0.6:
            x_best, best153 = lp_polish(x_best, fb)
    except Exception:
        pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: reshape, remove nan
    x_best = np.asarray(x_best, dtype=float).reshape(153, 4)
    norms = np.linalg.norm(x_best, axis=1, keepdims=True)
    norms[~np.isfinite(norms)] = 1.0
    norms[norms < 1e-12] = 1.0
    x_best = x_best / norms
    x_best[~np.isfinite(x_best)] = 0.25
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
