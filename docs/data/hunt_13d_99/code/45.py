import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _polish_lse(X0, alpha, steps, lr, kappa=None, beta=0.9, time_lim=None, t0=None):
    cur = X0.copy()
    best = cur.copy()
    bs = _max_cos(cur)
    mom = np.zeros_like(cur)
    if kappa is None:
        kappa = alpha
    for s in range(steps):
        if time_lim is not None and t0 is not None and (s % 100 == 0):
            if time.time() - t0 > time_lim:
                break
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # weights
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / rs
        # worst-point emphasis: weight rows by softmax of row-max
        try:
            rm = np.max(G, axis=1, keepdims=True)
            rw = np.exp(kappa * (rm - m))
            rw = rw / (np.mean(rw) + 1e-300)
            # clip to avoid explosion
            rw = np.clip(rw, 0.2, 5.0)
            Gt = Gt * rw
        except Exception:
            pass
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        # decay lr slightly
        lrc = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - lrc * (mom / bc)
        cur = _norm_rows(cur)
        if (s % 60 == 59) or (s == steps - 1):
            v = _max_cos(cur)
            if v < bs:
                bs = v
                best = cur.copy()
    return best, bs

def _polish_top(X0, window, steps, lr, beta=0.9, time_lim=None, t0=None):
    cur = X0.copy()
    best = cur.copy()
    bs = _max_cos(cur)
    mom = np.zeros_like(cur)
    n = cur.shape[0]
    for s in range(steps):
        if time_lim is not None and t0 is not None and (s % 100 == 0):
            if time.time() - t0 > time_lim:
                break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - window).astype(float)
        np.fill_diagonal(Mask, 0.0)
        cnt = Mask.sum(axis=1, keepdims=True) + 1e-12
        Grad = Mask @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / cnt
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        lrc = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - lrc * (mom / bc)
        cur = _norm_rows(cur)
        if (s % 80 == 79) or (s == steps - 1):
            v = _max_cos(cur)
            if v < bs:
                bs = v
                best = cur.copy()
        # shrink window halfway to tighten
        if s == steps // 2:
            window = max(window * 0.3, 2e-07)
    return best, bs

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    BUDGET = 172.0
    rng = _np.random.default_rng(0)
    try:
        xb = _np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = _np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = _np.asarray(xb, float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        xb = tmp
    xb = _norm_rows(xb)
    n, d = 99, 13
    best_val = _max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val

    # initial direct squeeze of incumbent with large alpha (no noise)
    try:
        q, qv = _polish_lse(cur_x, 30000.0, 500, 0.008, time_lim=BUDGET - 8, t0=t0)
        if qv < best_val:
            best_val = qv; best_x = q.copy(); cur_x = q.copy(); cur_val = qv
        q2, qv2 = _polish_lse(cur_x, 120000.0, 500, 0.005, time_lim=BUDGET - 8, t0=t0)
        if qv2 < best_val:
            best_val = qv2; best_x = q2.copy(); cur_x = q2.copy(); cur_val = qv2
        q3, qv3 = _polish_top(cur_x, 3e-06, 400, 0.006, time_lim=BUDGET - 8, t0=t0)
        if qv3 < best_val:
            best_val = qv3; best_x = q3.copy(); cur_x = q3.copy(); cur_val = qv3
    except Exception:
        pass

    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]
    alphas = [20000.0, 50000.0, 120000.0, 250000.0, 400000.0]
    it = 0
    # explore until ~140s, reserve rest for final polish
    while _time.time() - t0 < 140.0 and it < 1000:
        it += 1
        progress = (_time.time() - t0) / 140.0
        Temp = 1.2e-06 * (1.0 - progress) + 1.5e-07
        mode = rng.random()
        try:
            Gc = cur_x @ cur_x.T
            _np.fill_diagonal(Gc, -2.0)
            rm = Gc.max(axis=1)
            order = _np.argsort(rm)
        except Exception:
            order = _np.arange(n)
        try:
            if mode < 0.45:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                # mostly perturb worst points more
                base = cur_x.copy()
                if rng.random() < 0.25:
                    base = best_x.copy()
                noise = rng.standard_normal(size=(n, d)) * sigma
                # amplify noise on worst 10 points
                kk = 10
                idx_w = order[-kk:]
                noise[idx_w] *= 2.0
                x0 = base + noise
                x0 = _norm_rows(x0)
                alpha = float(alphas[int(rng.integers(0, len(alphas)))])
                steps = int(rng.integers(350, 750))
                lr = float(rng.uniform(0.003, 0.009))
                x1, _ = _polish_lse(x0, alpha, steps, lr, time_lim=BUDGET - 8, t0=t0)
                # follow with short uniform equalization
                if rng.random() < 0.6:
                    w = float(rng.choice([5e-07, 1e-06, 2e-06, 4e-06]))
                    x1, _ = _polish_top(x1, w, 180, lr * 0.8, time_lim=BUDGET - 8, t0=t0)
                v1 = _max_cos(x1)
            elif mode < 0.75:
                # worst-point surgery: reinsert k worst into voids
                kk = int(rng.integers(1, 5))
                idx = order[-kk:]
                # find voids: random candidates far from all except removed
                keep = _np.delete(cur_x, idx, axis=0)
                n_cand = 1500
                cand = _norm_rows(rng.standard_normal(size=(n_cand, d)))
                # score by max inner prod to keep set
                scores = cand @ keep.T
                s_max = scores.max(axis=1)
                # pick kk most diverse (lowest smax), greedy
                pick = _np.argsort(s_max)[:kk]
                x0 = cur_x.copy()
                x0[idx] = cand[pick] + rng.standard_normal(size=(kk, d)) * 0.005
                x0 = _norm_rows(x0)
                alpha = float(rng.choice([30000.0, 80000.0, 200000.0]))
                steps = int(rng.integers(400, 800))
                lr = float(rng.uniform(0.003, 0.008))
                x1, _ = _polish_lse(x0, alpha, steps, lr, time_lim=BUDGET - 8, t0=t0)
                v1 = _max_cos(x1)
            else:
                # global small shake + top equalize
                sigma = float(rng.uniform(0.006, 0.02))
                x0 = cur_x + rng.standard_normal(size=(n, d)) * sigma
                x0 = _norm_rows(x0)
                w = float(rng.choice([8e-07, 1.5e-06, 3e-06]))
                x1, _ = _polish_top(x0, w, int(rng.integers(300, 600)), float(rng.uniform(0.003, 0.008)), time_lim=BUDGET - 8, t0=t0)
                # then LSE refine
                alpha = float(rng.choice([50000.0, 150000.0, 350000.0]))
                x1, _ = _polish_lse(x1, alpha, 300, 0.004, time_lim=BUDGET - 8, t0=t0)
                v1 = _max_cos(x1)

            # acceptance: best tracking + wandering current
            if v1 < best_val:
                best_val = v1; best_x = x1.copy()
                cur_x = x1.copy(); cur_val = v1
            elif v1 < cur_val or rng.random() < _np.exp(-(v1 - cur_val) / max(Temp, 1e-12)):
                cur_x = x1.copy(); cur_val = v1
            else:
                # occasionally restart current from best
                if rng.random() < 0.12:
                    cur_x = best_x.copy(); cur_val = best_val
        except Exception:
            continue
        if _time.time() - t0 > BUDGET - 25:
            break

    # final intensive polish of best
    try:
        remaining = BUDGET - (_time.time() - t0) - 2.0
        if remaining > 8:
            q = best_x.copy()
            # chain: medium -> large -> top equalize -> very large short
            q, qv = _polish_lse(q, 50000.0, 600, 0.005, time_lim=BUDGET - 3, t0=t0)
            if qv < best_val:
                best_val = qv; best_x = q.copy()
            q, qv = _polish_lse(q, 180000.0, 700, 0.0035, time_lim=BUDGET - 3, t0=t0)
            if qv < best_val:
                best_val = qv; best_x = q.copy()
            q, qv = _polish_top(q, 1.2e-06, 500, 0.003, time_lim=BUDGET - 3, t0=t0)
            if qv < best_val:
                best_val = qv; best_x = q.copy()
            q, qv = _polish_lse(q, 350000.0, 500, 0.002, time_lim=BUDGET - 2, t0=t0)
            if qv < best_val:
                best_val = qv; best_x = q.copy()
    except Exception:
        pass
    best_x = _norm_rows(_np.asarray(best_x, float))
    return best_x
