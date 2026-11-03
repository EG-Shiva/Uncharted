import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G))

def _refine(X, steps, b0, b1, lr0, lr1, mu=0.9, iso=0.0, cent=0.0, seed_V=None):
    n, d = X.shape
    Xw = X.copy()
    V = np.zeros_like(Xw) if seed_V is None else seed_V.copy()
    best_X = Xw.copy()
    # initial value
    G0 = Xw @ Xw.T
    np.fill_diagonal(G0, -2.0)
    best_v = float(np.max(G0))
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    eye_d = np.eye(d)
    for s in range(steps):
        f = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = Xw @ Xw.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights
        # clip exponent to avoid underflow cost; exp is fine
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Grad = E @ Xw
        if iso > 0.0:
            C = (Xw.T @ Xw) / n
            Em = C - eye_d / d
            Grad = Grad + iso * (Xw @ Em)
        if cent > 0.0:
            mu_vec = np.mean(Xw, axis=0)
            Grad = Grad + cent * mu_vec[None, :]
        # tangential projection
        dot = np.sum(Grad * Xw, axis=1, keepdims=True)
        Grad -= dot * Xw
        gn = float(np.max(np.linalg.norm(Grad, axis=1)))
        if gn > 1e-12:
            Grad /= gn
        else:
            continue
        V = mu * V + (1.0 - mu) * Grad
        Xw = Xw - lr * V
        nr = np.linalg.norm(Xw, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Xw /= nr
        # track best using current m (approx) every 25 steps with exact check
        if (s + 1) % 25 == 0 or s == steps - 1:
            Gc = Xw @ Xw.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(np.max(Gc))
            if vc < best_v:
                best_v = vc
                best_X = Xw.copy()
    return best_X, best_v

def _best_hole(others, rng, nsamp=2000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(nsamp, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = np.max(dots, axis=1)
    j = int(np.argmin(md))
    return cand[j]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    EXPL_END = t0 + 142.0
    rng = np.random.default_rng(0)
    n, d = 95, 13

    X0 = None
    bv = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn).astype(float)
            if A.shape == (95, 13):
                A /= np.linalg.norm(A, axis=1, keepdims=True)
                v = _max_cos(A)
                if v < bv:
                    bv = v
                    X0 = A.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv = _max_cos(X0)

    best = X0.copy()
    best_val = float(bv)
    cur = best.copy()
    cur_val = best_val

    trial = 0
    # temperature for Metropolis sideways moves
    T_accept = 0.0012
    sigmas = [0.015, 0.04, 0.10, 0.25]

    while time.time() < EXPL_END:
        if time.time() > DEADLINE - 35:
            break
        trial += 1
        mode = trial % 12

        # contact info of cur (cheap)
        try:
            rm, cm = _row_max(cur)
        except Exception:
            rm = np.zeros(n); cm = cur_val
        order_worst = np.argsort(-rm)
        # rattlers / low contact approx via row_max far below global max
        low = np.where(rm < cm - 1e-3)[0]

        Xt = cur.copy()
        if mode == 0:
            Xt = Xt + 0.015 * rng.standard_normal((n, d))
        elif mode == 1:
            Xt = Xt + 0.04 * rng.standard_normal((n, d))
        elif mode == 2:
            # subset shake
            k = int(rng.integers(4, 12))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] += 0.28 * rng.standard_normal((k, d))
            Xt += 0.005 * rng.standard_normal((n, d))
        elif mode == 3:
            # replace single worst by hole
            w = int(order_worst[0])
            others = np.delete(Xt, w, axis=0)
            Xt[w] = _best_hole(others, rng, nsamp=2500)
            Xt += 0.008 * rng.standard_normal((n, d))
        elif mode == 4:
            # replace two worst by holes (joint)
            ws = [int(order_worst[0]), int(order_worst[1])]
            # keep others fixed
            mask = np.ones(n, dtype=bool); mask[ws] = False
            others = Xt[mask]
            h1 = _best_hole(others, rng, nsamp=2500)
            # second hole against others + h1
            ext = np.vstack([others, h1[None, :]])
            h2 = _best_hole(ext, rng, nsamp=2500)
            Xt[ws[0]] = h1; Xt[ws[1]] = h2
            Xt += 0.008 * rng.standard_normal((n, d))
        elif mode == 5:
            Xt = Xt + 0.10 * rng.standard_normal((n, d))
        elif mode == 6:
            # shake hubs + lows strongly
            k = min(8, n)
            idx = order_worst[:k]
            Xt[idx] += 0.22 * rng.standard_normal((k, d))
            if len(low) > 0:
                lk = low[:min(len(low), 4)]
                Xt[lk] += 0.30 * rng.standard_normal((len(lk), d))
        elif mode == 7:
            Xt = Xt + 0.25 * rng.standard_normal((n, d))
        elif mode == 8:
            k = int(rng.integers(8, 16))
            idx = rng.choice(n, size=k, replace=False)
            Xt[idx] += 0.16 * rng.standard_normal((k, d))
            Xt += 0.005 * rng.standard_normal((n, d))
        elif mode == 9:
            # interpolate toward random to jump basin
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            a = 0.35
            Xt = (1 - a) * Xt + a * R
        elif mode == 10:
            # global medium + hub hole
            w = int(order_worst[int(rng.integers(0, 3))])
            others = np.delete(Xt, w, axis=0)
            Xt[w] = _best_hole(others, rng, nsamp=1500)
            Xt = Xt + 0.03 * rng.standard_normal((n, d))
        else:
            s = sigmas[(trial // 12) % len(sigmas)]
            Xt = Xt + s * rng.standard_normal((n, d))

        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)

        # fast screening refine
        # vary beta/lr slightly by mode for diversity
        if mode in (3, 4, 10):
            Xr, vr = _refine(Xt, steps=380, b0=30.0, b1=350.0, lr0=0.025, lr1=0.004, mu=0.88, iso=0.0, cent=0.5)
        elif mode == 9:
            Xr, vr = _refine(Xt, steps=420, b0=25.0, b1=250.0, lr0=0.03, lr1=0.005, mu=0.85, iso=0.0, cent=1.0)
        else:
            Xr, vr = _refine(Xt, steps=350, b0=30.0, b1=300.0, lr0=0.022, lr1=0.004, mu=0.9, iso=0.0, cent=0.0)

        improved_best = vr < best_val - 1e-12
        if improved_best:
            # extend polish immediately for promising point
            if time.time() < DEADLINE - 30:
                Xr2, v2 = _refine(Xr, steps=500, b0=300.0, b1=1200.0, lr0=0.008, lr1=0.0015, mu=0.9)
                if v2 < vr:
                    Xr, vr = Xr2, v2
            best = Xr.copy()
            best_val = float(vr)
            cur = Xr.copy()
            cur_val = float(vr)
            # cool acceptance a bit after improvement
            T_accept = max(0.0004, T_accept * 0.97)
        else:
            # Metropolis move of cur to allow topology wandering
            if vr < cur_val or rng.random() < np.exp((cur_val - vr) / T_accept):
                cur = Xr.copy()
                cur_val = float(vr)
            # if cur drifted far above best, pull back
            if cur_val > best_val + 0.006 and (trial % 8 == 0):
                cur = best.copy() + 0.06 * rng.standard_normal((n, d))
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                # quick value update
                cur_val = _max_cos(cur)
        # slow anneal of acceptance temp
        # keep it in range
        if trial % 50 == 0:
            T_accept = min(0.0015, max(0.0004, T_accept))

    # final squeeze of best with staged high-beta polish
    # also consider cur if very close
    cands = [(best, best_val)]
    try:
        cv = _max_cos(cur)
        if cv < best_val + 0.001:
            cands.append((cur.copy(), cv))
    except Exception:
        pass
    # pick best candidate
    cands.sort(key=lambda t: t[1])
    best, best_val = cands[0][0].copy(), float(cands[0][1])

    # staged polish, time-guarded
    while time.time() < DEADLINE - 2.0:
        remaining = DEADLINE - time.time()
        if remaining < 2.5:
            break
        # choose stage length to fit remaining
        if remaining > 25:
            steps = 1200
        elif remaining > 12:
            steps = 700
        else:
            steps = 350
        b0, b1 = 200.0, 1800.0
        nb, nv = _refine(best, steps=steps, b0=b0, b1=b1, lr0=0.006, lr1=0.0008, mu=0.92, iso=0.0, cent=0.2)
        if nv < best_val - 1e-14:
            best, best_val = nb, float(nv)
            continue
        # try alternative without centroid / different lr
        nb2, nv2 = _refine(best, steps=min(steps, 600), b0=400.0, b1=2500.0, lr0=0.004, lr1=0.0006, mu=0.9)
        if nv2 < best_val - 1e-14:
            best, best_val = nb2, float(nv2)
            continue
        break

    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
