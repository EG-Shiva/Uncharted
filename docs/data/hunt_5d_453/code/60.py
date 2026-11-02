import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 453, 5

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (n, d):
                continue
            nn = np.linalg.norm(cc, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    curX = bestX.copy()
    curV = float(bestV)
    # use lists for closure speed
    state = [curX, curV]

    iu_r, iu_c = np.triu_indices(n, 1)

    def note(x, m):
        if m < state[1] and np.isfinite(m):
            state[1] = m
            state[0] = x.copy()

    def smooth_minimax(x, N, tau0, tau1, lr0, lr1, mu):
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if (it & 63) == 0 and time.time() > DEAD - 8.0:
                break
            frac = it / max(N - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < state[1]:
                state[1] = m
                state[0] = x.copy()
            E = (G - m) / tau
            np.clip(E, -30.0, 0.0, out=E)
            np.exp(E, out=E)
            # E is P
            Grad = E @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Grad -= dot * x
            rms = float(np.sqrt(np.mean(Grad * Grad)))
            if not np.isfinite(rms) or rms < 1e-30:
                continue
            Grad /= rms
            v *= mu
            v += (1.0 - mu) * Grad
            x -= lr * v
            nn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            x /= nn
            if not np.all(np.isfinite(x)):
                x[:] = state[0]
                v *= 0.5
            elif m > state[1] + 0.03:
                # pull back gently
                x *= 0.995
                x += 0.005 * state[0]
                nn2 = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
                nn2[nn2 == 0] = 1.0
                x /= nn2
        return x

    def riesz_run(x, N, s0, s1, lr0, lr1, mu):
        v = np.zeros_like(x)
        ls = np.log(s1 / s0) if s0 > 0 and s1 > 0 else 0.0
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if (it & 63) == 0 and time.time() > DEAD - 8.0:
                break
            frac = it / max(N - 1, 1)
            s = s0 * np.exp(ls * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            # r2 = 2-2G, for diag G=-2 => r2=6 (will zero out)
            R2 = 2.0 - 2.0 * G
            # clip
            np.clip(R2, 1e-4, 4.0, out=R2)
            expn = -s / 2.0 - 1.0
            # W = s * R2^{expn}
            W = np.power(R2, expn)
            W *= s
            np.fill_diagonal(W, 0.0)
            # clip huge
            np.clip(W, 0.0, 1e6, out=W)
            Grad = W @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Grad -= dot * x
            rms = float(np.sqrt(np.mean(Grad * Grad)))
            if not np.isfinite(rms) or rms < 1e-30:
                continue
            Grad /= rms
            v *= mu
            v += (1.0 - mu) * Grad
            x -= lr * v
            nn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            x /= nn
            if not np.all(np.isfinite(x)):
                x[:] = state[0]
                v *= 0.5
        return x

    def greedy_shave(x, steps, eps0):
        eps = float(eps0)
        for it in range(steps):
            if (it & 31) == 0 and time.time() > DEAD - 5.0:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < state[1]:
                state[1] = m
                state[0] = x.copy()
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            if i == j:
                continue
            xi = x[i].copy()
            xj = x[j].copy()
            cij = float(xi @ xj)
            if cij > 1.0:
                cij = 1.0
            elif cij < -1.0:
                cij = -1.0
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12 or not np.isfinite(ni):
                rr = rng.standard_normal(d)
                rr -= float(rr @ xi) * xi
                ni2 = float(np.linalg.norm(rr))
                if ni2 < 1e-12:
                    continue
                ti = rr / ni2
            else:
                ti = ti / ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12 or not np.isfinite(nj):
                rr = rng.standard_normal(d)
                rr -= float(rr @ xj) * xj
                ni2 = float(np.linalg.norm(rr))
                if ni2 < 1e-12:
                    continue
                tj = rr / ni2
            else:
                tj = tj / nj
            G[i, :] = -2.0
            G[:, i] = -2.0
            G[j, :] = -2.0
            G[:, j] = -2.0
            rest_max = float(np.max(G))
            improved = False
            for scale in (eps, eps * 0.5):
                vi = xi - scale * ti
                nvi = float(np.linalg.norm(vi))
                if nvi == 0 or not np.isfinite(nvi):
                    continue
                vi = vi / nvi
                vj = xj - scale * tj
                nvj = float(np.linalg.norm(vj))
                if nvj == 0 or not np.isfinite(nvj):
                    continue
                vj = vj / nvj
                a1 = x @ vi
                a1[i] = -2.0
                a1[j] = -2.0
                mx1 = float(np.max(a1))
                a2 = x @ vj
                a2[i] = -2.0
                a2[j] = -2.0
                mx2 = float(np.max(a2))
                cij_new = float(vi @ vj)
                cand = rest_max
                if mx1 > cand:
                    cand = mx1
                if mx2 > cand:
                    cand = mx2
                if cij_new > cand:
                    cand = cij_new
                if cand < m - 1e-13:
                    x[i] = vi
                    x[j] = vj
                    if cand < state[1]:
                        state[1] = cand
                        state[0] = x.copy()
                    eps = min(eps * 1.05, 3e-4)
                    improved = True
                    break
            if not improved:
                eps *= 0.985
                if eps < 2e-7:
                    eps = 2e-7
        return x

    def find_holes(X, k, n_cand=6000, exclude=None):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        S = C @ X.T
        if exclude is not None and len(exclude) > 0:
            # ignore ejected points
            S[:, list(exclude)] = -2.0
        mc = np.max(S, axis=1)
        order = np.argsort(mc)
        picked = []
        picked_vecs = []
        for idx in order:
            if len(picked) >= k:
                break
            cv = C[idx]
            ok = True
            for pv in picked_vecs:
                if float(cv @ pv) > 0.90:
                    ok = False
                    break
            if ok:
                picked.append(int(idx))
                picked_vecs.append(cv)
            if len(picked) * 8 < len(order) and len(picked_vecs) > 0 and len(picked) < k:
                continue
        # if not enough due to separation, fill
        if len(picked) < k:
            for idx in order:
                if int(idx) not in picked:
                    picked.append(int(idx))
                    if len(picked) >= k:
                        break
        return C[picked[:k]]

    def row_worst(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = np.max(G, axis=1)
        # also degree within 1e-5 of global max as overload measure
        m = float(np.max(rm))
        # score = rm + 1e-4 * degree-ish; just use rm with tiny index jitter for diversity
        jitter = rng.standard_normal(n) * 1e-9
        score = rm + jitter
        idx = np.argsort(-score)[:k]
        return idx, rm, m

    try:
        # Phase 0: quick polish of incumbent to set baseline
        x = state[0].copy()
        if time.time() < DEAD - 30:
            x = smooth_minimax(x, 1800, 2.0e-3, 8e-5, 9e-4, 6e-6, 0.90)
        if time.time() < DEAD - 20:
            x = greedy_shave(x, 350, 6e-5)

        cycle = 0
        # main basin-hopping loop
        while time.time() < DEAD - 12.0:
            cycle += 1
            base = state[0].copy()
            mode = cycle % 5
            if mode == 0:
                # single hole surgery, worst point -> largest hole
                idx, rm, m = row_worst(base, 4)
                eject = idx[:1]
                holes = find_holes(base, 1, n_cand=7000, exclude=eject)
                y = base.copy()
                y[eject[0]] = holes[0]
                y = riesz_run(y, 350, 2.0, 6.0, 1.5e-3, 2e-4, 0.88)
                y = smooth_minimax(y, 1600, 1.5e-3, 5e-5, 8e-4, 5e-6, 0.89)
                y = greedy_shave(y, 250, 4e-5)
            elif mode == 1:
                # multi-hole surgery 3-4 points
                kk = 3 + (cycle % 2)  # 3 or 4
                idx, rm, m = row_worst(base, 12)
                # pick kk among top 12 randomly to diversify
                perm = rng.permutation(12)[:kk]
                eject = idx[perm]
                holes = find_holes(base, kk, n_cand=8000, exclude=eject)
                y = base.copy()
                y[eject] = holes
                y = riesz_run(y, 450, 3.0, 9.0, 1.8e-3, 2e-4, 0.88)
                y = smooth_minimax(y, 1700, 1.8e-3, 5e-5, 9e-4, 5e-6, 0.89)
                y = greedy_shave(y, 250, 4e-5)
            elif mode == 2:
                # global shake + Riesz anneal (different energy basin)
                sig = 0.008 if (cycle % 10) < 7 else 0.018
                y = base + sig * rng.standard_normal((n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
                s0 = 2.0 + float(rng.uniform(0, 2.0))
                s1 = s0 + 5.0
                y = riesz_run(y, 700, s0, s1, 2.0e-3, 2.5e-4, 0.90)
                y = smooth_minimax(y, 1500, 2.0e-3, 4e-5, 9e-4, 4e-6, 0.90)
                y = greedy_shave(y, 200, 3e-5)
            elif mode == 3:
                # tangent shake of worst 25 points + softmax only
                idx, rm, m = row_worst(base, 25)
                y = base.copy()
                sig = 0.006
                noise = sig * rng.standard_normal((25, d))
                # make tangent
                for t, gi in enumerate(idx):
                    nnv = y[gi]
                    noise[t] -= float(noise[t] @ nnv) * nnv
                y[idx] += noise
                y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True) + 1e-30
                y = smooth_minimax(y, 1900, 1.2e-3, 4e-5, 1.0e-3, 5e-6, 0.88)
                y = greedy_shave(y, 250, 4e-5)
            else:
                # 6-point surgery targeting high contact + random (topology break)
                idx, rm, m = row_worst(base, 30)
                eject = rng.choice(idx, size=5, replace=False)
                # add one random rattler/low point to move free volume
                G0 = base @ base.T
                np.fill_diagonal(G0, -2.0)
                rm0 = np.max(G0, axis=1)
                low = np.argsort(rm0)[:30]
                eject = np.append(eject, rng.choice(low, size=1, replace=False))
                holes = find_holes(base, 6, n_cand=9000, exclude=eject)
                y = base.copy()
                y[eject] = holes
                y = smooth_minimax(y, 1800, 2.2e-3, 5e-5, 1.0e-3, 5e-6, 0.88)
                y = greedy_shave(y, 250, 5e-5)
            # loop continues; elitist state already updated inside runs
            if time.time() > DEAD - 12.0:
                break

        # final micro polish from best
        if time.time() < DEAD - 5.0:
            z = state[0].copy()
            z = smooth_minimax(z, 900, 3.0e-4, 3.0e-5, 1.2e-4, 2.0e-6, 0.86)
            z = greedy_shave(z, 300, 1.2e-5)
    except Exception:
        pass

    Xout = state[0].copy()
    nn = np.linalg.norm(Xout, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xout = Xout / nn
    Xout = np.where(np.isfinite(Xout), Xout, 0.0)
    rn = np.linalg.norm(Xout, axis=1)
    bad = rn < 1e-12
    if np.any(bad):
        nb = int(np.sum(bad))
        tmp = rng.standard_normal((nb, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True) + 1e-30
        Xout[bad] = tmp
    return Xout
