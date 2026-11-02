import time
import numpy as np

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    rng = np.random.default_rng(12345)
    n, d = 323, 7

    def max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def worst_per_point(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1)

    # load start
    x_best = None
    best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal(size=(n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = max_cos(x_best)
    x_best = x_best.copy()

    def smooth_polish(x_init, steps, T0, T1, lr0, lr1, mu=0.8):
        x = x_init.copy()
        v = np.zeros_like(x)
        # current best in run
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        mb = float(np.max(G0))
        xb = x.copy()
        lT0 = np.log(T0); lT1 = np.log(T1)
        llr0 = np.log(lr0); llr1 = np.log(lr1)
        for s in range(steps):
            if s % 128 == 0 and time.time() > DEADLINE - 2.0:
                break
            frac = s / max(1, steps - 1)
            T = float(np.exp(lT0 + (lT1 - lT0) * frac))
            lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            if m < mb:
                mb = m
                xb = x.copy()
            # W = exp((G-m)/T) in place
            G -= m
            G /= T
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            Srow = np.sum(G, axis=1)
            A = G @ x
            A /= (Srow[:, None] + 1e-30)
            dot = np.sum(A * x, axis=1, keepdims=True)
            Gt = A - dot * x
            v *= mu
            v += Gt
            x -= lr * v
            nr = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            nr[nr == 0] = 1.0
            x /= nr
        return xb, mb, x

    def greedy_reinsert(y, to_replace, n_cand=8000):
        # sequential max-min
        for k in to_replace:
            cand = rng.standard_normal(size=(n_cand, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(y, k, axis=0)
            dots = cand @ others.T
            w = np.max(dots, axis=1)
            bi = int(np.argmin(w))
            y[k] = cand[bi]
        return y

    def find_void(x, n_cand=20000):
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ x.T
        w = np.max(dots, axis=1)
        bi = int(np.argmin(w))
        return cand[bi], float(w[bi])

    # short initial polish to set baseline quickly
    try:
        xb, mb, xf = smooth_polish(x_best, steps=800, T0=0.012, T1=0.002, lr0=0.015, lr1=0.003, mu=0.8)
        if mb < best:
            best = mb
            x_best = xb.copy()
    except Exception:
        pass

    it = 0
    # main basin hopping
    while time.time() < DEADLINE - 8.0:
        remain = DEADLINE - 8.0 - (time.time() - t0 + 0)  # not used
        s = it % 6
        x = x_best.copy()
        try:
            if s == 0:
                # global small jitter
                sig = 0.025
                x += rng.standard_normal(size=x.shape) * sig
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1100, T0=0.020, T1=0.0015, lr0=0.030, lr1=0.003, mu=0.8)
            elif s == 1:
                # worst-k jitter
                w = worst_per_point(x)
                k = 25
                idx = np.argpartition(w, -k)[-k:]
                x[idx] += rng.standard_normal(size=(k, d)) * 0.08
                x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1100, T0=0.022, T1=0.0015, lr0=0.030, lr1=0.003, mu=0.8)
            elif s == 2:
                # void reinsertion of worst + rattlers
                w = worst_per_point(x)
                # estimate contacts for rattler detection: use threshold best-1e-4
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                cnt = np.sum(G > best - 1e-4, axis=1)
                ratt = np.where(cnt <= 1)[0].tolist()
                k = 8
                idx_worst = np.argpartition(w, -k)[-k:].tolist()
                torep = []
                for r in ratt[:4]:
                    torep.append(int(r))
                for o in idx_worst:
                    if len(torep) >= 8:
                        break
                    if int(o) not in torep:
                        torep.append(int(o))
                # void-biased: also replace one by pure void optimum
                if len(torep) > 0:
                    # replace first with void centre
                    vv, _ = find_void(np.delete(x, torep[0], axis=0), n_cand=15000)
                    x[torep[0]] = vv
                    rest = torep[1:]
                    if len(rest) > 0:
                        x = greedy_reinsert(x, rest, n_cand=8000)
                x += rng.standard_normal(size=x.shape) * 0.004
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1300, T0=0.025, T1=0.0012, lr0=0.035, lr1=0.003, mu=0.8)
            elif s == 3:
                # cap-cluster coherent shift
                vv = rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
                proj = x @ vv
                k = int(rng.integers(12, 42))
                idx = np.argpartition(proj, -k)[-k:]
                sv = rng.standard_normal(size=(d,))
                sv = sv - (sv @ vv) * vv
                nv = np.linalg.norm(sv)
                if nv < 1e-12:
                    sv = rng.standard_normal(size=(d,))
                    nv = np.linalg.norm(sv)
                sv = sv / nv * float(rng.uniform(0.05, 0.16))
                x[idx] += sv
                x[idx] += rng.standard_normal(size=(k, d)) * 0.012
                x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1200, T0=0.025, T1=0.0015, lr0=0.032, lr1=0.003, mu=0.82)
            elif s == 4:
                # random subset large kick
                k = 32
                idx = rng.choice(n, size=k, replace=False)
                x[idx] += rng.standard_normal(size=(k, d)) * 0.12
                x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1100, T0=0.028, T1=0.0018, lr0=0.035, lr1=0.004, mu=0.8)
            else:
                # medium global + void reinsert small
                w = worst_per_point(x)
                k = 5
                idx = np.argpartition(w, -k)[-k:].tolist()
                x = greedy_reinsert(x, [int(vv2) for vv2 in idx], n_cand=6000)
                x += rng.standard_normal(size=x.shape) * 0.012
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                xb, mb, xf = smooth_polish(x, steps=1000, T0=0.018, T1=0.0015, lr0=0.028, lr1=0.003, mu=0.8)
        except Exception:
            it += 1
            continue
        if mb < best - 1e-12:
            best = mb
            x_best = xb.copy()
        else:
            # also consider final state if better (rare)
            try:
                vf = max_cos(xf)
                if vf < best - 1e-12:
                    best = vf
                    x_best = xf.copy()
            except Exception:
                pass
        it += 1
        if it > 500:
            break

    # final squeeze on best with remaining time
    try:
        if time.time() < DEADLINE:
            steps = 2500
            # scale steps to remaining time roughly
            rem = DEADLINE - time.time()
            if rem < 6:
                steps = 800
            xb, mb, xf = smooth_polish(x_best, steps=steps, T0=0.008, T1=0.0008, lr0=0.012, lr1=0.002, mu=0.85)
            if mb < best:
                best = mb
                x_best = xb.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
