import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # load starts
    cands = []
    try:
        xb = np.load("best.npy")
        cands.append(np.asarray(xb, dtype=float))
    except Exception:
        pass
    try:
        xr = np.load("record.npy")
        cands.append(np.asarray(xr, dtype=float))
    except Exception:
        pass
    valid = []
    for c in cands:
        if c.shape != (453, 5):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        # guard nan
        if not np.all(np.isfinite(c)):
            continue
        valid.append((_max_cos(c), c))
    if len(valid) == 0:
        x0 = rng.standard_normal((453, 5))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        valid.append((_max_cos(x0), x0))
    valid.sort(key=lambda t: t[0])
    best_val = float(valid[0][0])
    best_x = valid[0][1].copy()
    n, d = best_x.shape

    def smooth_run(x, N, tau0, tau1, lr0, lr1, mu):
        nonlocal best_val, best_x
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if it % 50 == 0 and time.time() > DEAD - 8:
                break
            frac = it / max(N - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < best_val:
                best_val = m
                best_x = x.copy()
            # softmax weights near max
            # (G-m)<=0
            P = np.exp((G - m) / tau)
            # diag ~0 already
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if m > best_val + 0.02:
                x = best_x.copy()
                v *= 0.5
        return x

    def greedy_shave(x, steps):
        nonlocal best_val, best_x
        eps = 1e-3
        for it in range(steps):
            if it % 50 == 0 and time.time() > DEAD - 5:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < best_val:
                best_val = m
                best_x = x.copy()
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            cij = float(x[i] @ x[j])
            ti = x[j] - cij * x[i]
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d)
                ti -= float(ti @ x[i]) * x[i]
                ni = float(np.linalg.norm(ti)) + 1e-12
            ti /= ni
            tj = x[i] - cij * x[j]
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d)
                tj -= float(tj @ x[j]) * x[j]
                nj = float(np.linalg.norm(tj)) + 1e-12
            tj /= nj
            improved = False
            for scale in (eps, eps * 0.5, eps * 0.25):
                vi = x[i] - scale * ti
                vi /= np.linalg.norm(vi)
                vj = x[j] - scale * tj
                vj /= np.linalg.norm(vj)
                # quick eval: only need max involving i,j + old max?
                # full eval for safety (cheap enough at low steps)
                xp_i_old = x[i].copy()
                xp_j_old = x[j].copy()
                x[i] = vi
                x[j] = vj
                g2 = x @ x.T
                np.fill_diagonal(g2, -2.0)
                m2 = float(g2.max())
                if m2 < m - 1e-12:
                    if m2 < best_val:
                        best_val = m2
                        best_x = x.copy()
                    eps = min(scale * 1.08, 3e-3)
                    improved = True
                    break
                else:
                    x[i] = xp_i_old
                    x[j] = xp_j_old
            if not improved:
                eps *= 0.96
                if eps < 5e-07:
                    k = it % n
                    rnd = rng.standard_normal(d)
                    rnd -= float(rnd @ x[k]) * x[k]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        rnd /= rn
                        xn = x[k] + 1e-06 * rnd
                        xn /= np.linalg.norm(xn)
                        x[k] = xn
                    eps = 5e-07
        return x

    def row_max(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    def largest_hole_for(exclude_idx, Xref, n_cand=20000):
        # random candidates, pick one minimizing max cos to Xref (excluding point)
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        mask = np.ones(Xref.shape[0], dtype=bool)
        mask[exclude_idx] = False
        Xr = Xref[mask]
        # chunk to save memory
        best_c = None
        best_s = 1e9
        chunk = 4000
        # we need global best: compute scores per candidate
        scores = np.full(n_cand, 1e9)
        # Actually compute max in chunks of candidates
        for s in range(0, n_cand, chunk):
            e = min(s + chunk, n_cand)
            Cc = C[s:e]
            M = Cc @ Xr.T
            ms = M.max(axis=1)
            scores[s:e] = ms
        bi = int(np.argmin(scores))
        return C[bi]

    # initial tiny break
    x = best_x.copy()
    x = x + 2e-04 * rng.standard_normal(x.shape)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    # cycle plan: basin hopping
    # first quick anneal from start
    if time.time() < DEAD - 10:
        x = smooth_run(x, 2500, 2e-03, 2e-04, 3e-03, 8e-04, 0.9)

    cycle = 0
    sigmas = [0.012, 0.006, 0.003, 0.015, 0.004, 0.008, 0.002, 0.010, 0.005, 0.007]
    while time.time() < DEAD - 12 and cycle < 30:
        cycle += 1
        mode = cycle % 3
        base = best_x.copy()
        if mode == 0:
            # global shake
            sig = sigmas[(cycle - 1) % len(sigmas)]
            x = base + sig * rng.standard_normal(base.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            # worst-points shake
            rm, _ = row_max(base)
            order = np.argsort(-rm)
            k = 8 + (cycle % 8)  # 8..15 points
            worst = order[:k]
            sig_small = 0.002
            sig_big = sigmas[(cycle - 1) % len(sigmas)]
            noise = sig_small * rng.standard_normal(base.shape)
            noise[worst] = sig_big * rng.standard_normal((len(worst), d))
            x = base + noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # hole reinsertion of 1-2 worst points
            x = base.copy()
            rm, _ = row_max(x)
            order = np.argsort(-rm)
            n_move = 1 if (cycle % 2 == 1) else 2
            for t in range(n_move):
                idx = int(order[t])
                hole = largest_hole_for(idx, x, n_cand=15000)
                # small blend to keep some continuity? direct place
                x[idx] = hole
                # update rm order roughly (recompute after each)
                # recompute row max quickly for next t? skip
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # add tiny noise to neighbours to allow topology change
            x = x + 0.0015 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # anneal short-medium
        # alternate tau ranges to diversify
        if cycle % 2 == 1:
            x = smooth_run(x, 1800, 8e-04, 2e-05, 2e-03, 2e-04, 0.9)
        else:
            x = smooth_run(x, 1800, 3e-04, 5e-06, 8e-04, 8e-05, 0.92)
        if time.time() > DEAD - 10:
            break
        # light greedy polish each 3rd cycle
        if cycle % 3 == 0:
            x = greedy_shave(x, 400)

    # final intensive polish from global best
    if time.time() < DEAD - 5:
        x = best_x.copy()
        # fine anneal
        x = smooth_run(x, 3000, 1e-04, 3e-06, 4e-04, 5e-05, 0.92)
    if time.time() < DEAD - 3:
        x = best_x.copy()
        x = greedy_shave(x, 2500)

    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(best_x)):
        best_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return best_x
