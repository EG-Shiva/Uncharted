import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    iu = np.triu_indices(n, 1)

    # ---- load start ----
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # ---------- helpers ----------
    def lp_polish(x, seconds):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        cur_best = _max_cos(x)
        if seconds <= 0.05:
            return x, cur_best
        radius = 1e-3
        start = time.time()
        while True:
            if time.time() - start >= seconds:
                break
            if radius < 1e-14:
                break
            G = x @ x.T
            c_all = G[iu]
            m = float(c_all.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c_all > thresh
            if int(sel.sum()) > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(c_all[idx])[-3500:]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
            I = iu[0][sel]
            J = iu[1][sel]
            c = c_all[sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            nv = n * d + 1
            cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                                   (J[:, None] * d + np.arange(d)),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2 * d + 1)
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            if v < cur_best:
                x, cur_best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, cur_best

    def smooth_refine(x0, steps, a0, a1, s0, s1):
        cur = x0.copy()
        bv = _max_cos(cur)
        bx = cur.copy()
        for s in range(steps):
            if steps > 1:
                frac = s / (steps - 1)
            else:
                frac = 0.0
            alpha = a0 * ((a1 / a0) ** frac)
            step = s0 + (s1 - s0) * frac
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            cm = float(G.max())
            if cm < bv:
                bv = cm
                bx = cur.copy()
            # weights exp(alpha*(G-cm)) <=1
            # clip argument to avoid underflow cost (optional)
            W = np.exp(alpha * (G - cm))
            np.fill_diagonal(W, 0.0)
            Gw = W @ cur
            coeff = (W * G).sum(axis=1)
            grad = Gw - cur * coeff[:, None]
            med = float(np.median(np.linalg.norm(grad, axis=1))) + 1e-12
            cur = cur - (step / med) * grad
            nn = np.linalg.norm(cur, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cur = cur / nn
        fv = _max_cos(cur)
        if fv < bv:
            bv = fv
            bx = cur.copy()
        return bx, bv

    def find_void(X, exclude, nsamp=12000):
        # best empty spot given others
        others = np.delete(X, exclude, axis=0)
        best_y = None
        best_m = 1e9
        chunk = 3000
        cands = rng.normal(size=(nsamp, d))
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
        for s in range(0, nsamp, chunk):
            Y = cands[s:s + chunk]
            D = Y @ others.T
            m = D.max(axis=1)
            i = int(np.argmin(m))
            if float(m[i]) < best_m:
                best_m = float(m[i])
                best_y = Y[i].copy()
        # local hill climb: push away from closest
        y = best_y.copy()
        st = 0.02
        for _ in range(120):
            dots = others @ y
            j = int(np.argmax(dots))
            g = others[j] - dots[j] * y
            gn = float(np.linalg.norm(g)) + 1e-12
            if gn < 1e-12:
                break
            yn = y - st * g / gn
            yn /= float(np.linalg.norm(yn))
            nd = float((others @ yn).max())
            if nd < best_m:
                best_m = nd
                y = yn
            else:
                st *= 0.7
                if st < 1e-5:
                    break
        return y

    def contact_counts(X, val):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mask = G >= val - 1e-9
        return mask.sum(axis=1), G

    # ---------- main basin hopping ----------
    trial = 0
    # quick LP squeeze of start if time allows (already optimal, skip long)
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        trial += 1
        mode = trial % 8
        cnt, Gbest = contact_counts(best, best_val)
        order_desc = np.argsort(-cnt)
        order_asc = np.argsort(cnt)
        y = best.copy()

        if mode == 0:  # small global shake
            y = y + 0.022 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 350, 25.0, 220.0, 0.020, 0.002)
        elif mode == 1:  # kick jammed
            victims = order_desc[:4]
            y[victims] += 0.09 * rng.normal(size=(len(victims), d))
            # small shake of neighbours
            y += 0.008 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 400, 30.0, 250.0, 0.018, 0.002)
        elif mode == 2:  # replace worst 2
            victims = order_desc[:2]
            y[victims] = rng.normal(size=(len(victims), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 450, 20.0, 250.0, 0.022, 0.002)
        elif mode == 3:  # relocate rattlers / low-contact to voids
            lows = order_asc[:2]
            for vv in lows:
                y[int(vv)] = find_void(y, int(vv), nsamp=10000)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 350, 30.0, 220.0, 0.015, 0.002)
        elif mode == 4:  # medium global
            y = y + 0.06 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 450, 20.0, 200.0, 0.020, 0.002)
        elif mode == 5:  # split current worst pair
            # find max pair
            np.fill_diagonal(Gbest, -2.0)
            flat = int(np.argmax(Gbest))
            a = flat // n
            b = flat % n
            diff = y[a] - y[b]
            dn = float(np.linalg.norm(diff)) + 1e-12
            eps = 0.03 + 0.02 * rng.random()
            y[a] += eps * diff / dn
            y[b] -= eps * diff / dn
            y += 0.01 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 350, 30.0, 250.0, 0.015, 0.002)
        elif mode == 6:  # subset shake 20
            sub = rng.choice(n, 20, replace=False)
            y[sub] += 0.06 * rng.normal(size=(len(sub), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 400, 25.0, 220.0, 0.018, 0.002)
        else:  # random direction large for one jammed + rattler
            v1 = int(order_desc[0])
            v0 = int(order_asc[0])
            y[v1] += 0.12 * rng.normal(size=(d,))
            y[v0] = rng.normal(size=(d,))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y, v = smooth_refine(y, 450, 20.0, 250.0, 0.020, 0.002)

        if v < best_val - 1e-12:
            # squeeze further with LP
            rem2 = deadline - time.time()
            pt = 1.2 if rem2 > 20 else max(0.5, rem2 - 8.5)
            y2, v2 = lp_polish(y, min(pt, max(0.5, rem2 - 8.0)))
            if v2 < best_val:
                best, best_val = y2, v2
            else:
                best, best_val = y, v
            continue
        # promising but not beating: try short LP squeeze
        if v < best_val + 0.004:
            rem2 = deadline - time.time()
            if rem2 < 9.0:
                break
            # only LP every other such case to save time
            if (trial % 2 == 0) or (v < best_val + 0.0015):
                pt = 0.9 if rem2 > 25 else max(0.4, rem2 - 8.5)
                y2, v2 = lp_polish(y, pt)
                if v2 < best_val - 1e-12:
                    best, best_val = y2, v2

    # final squeeze of incumbent
    rem = deadline - time.time()
    if rem > 1.0:
        best, best_val = lp_polish(best, max(0.5, rem - 1.0))

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
