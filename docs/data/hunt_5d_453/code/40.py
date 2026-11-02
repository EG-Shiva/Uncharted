import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
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
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc * cc, axis=1, keepdims=True))
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

    # state held in lists for closure
    cur_best_X = bestX.copy()
    cur_best_V = float(bestV)

    def smooth_run(x, N, tau0, tau1, lr0, lr1, mu):
        nonlocal cur_best_X, cur_best_V
        v = np.zeros_like(x)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(N):
            if it % 200 == 0 and time.time() > DEAD - 12.0:
                break
            frac = it / max(N - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < cur_best_V and np.isfinite(m):
                cur_best_V = m
                cur_best_X = x.copy()
            # softmax weights
            # (G-m)/tau in [-inf,0]
            E = (G - m) / tau
            # clip to avoid underflow cost, in place
            np.clip(E, -30.0, 0.0, out=E)
            P = np.exp(E, out=E)
            # P diag ~ exp((-2-m)/tau) ~ 0, fine
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            rms = float(np.sqrt(np.mean(Gt * Gt)))
            if not np.isfinite(rms) or rms < 1e-30:
                continue
            Gn = Gt / rms
            v *= mu
            v += (1.0 - mu) * Gn
            x = x - lr * v
            # renormalize
            nn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            x = x / nn
            if not np.all(np.isfinite(x)):
                x = cur_best_X.copy()
                v *= 0.5
            # rescue if diverged far
            if m > cur_best_V + 0.02:
                # pull back toward best
                x = 0.995 * x + 0.005 * cur_best_X
                nn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
                nn[nn == 0] = 1.0
                x = x / nn
        return x

    def greedy_shave(x, steps, eps0):
        nonlocal cur_best_X, cur_best_V
        eps = float(eps0)
        # preallocate mask helper
        for it in range(steps):
            if it % 200 == 0 and time.time() > DEAD - 6.0:
                break
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < cur_best_V:
                cur_best_V = m
                cur_best_X = x.copy()
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            if i == j:
                continue
            xi = x[i]
            xj = x[j]
            cij = float(xi @ xj)
            if cij > 1.0:
                cij = 1.0
            elif cij < -1.0:
                cij = -1.0
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12 or not np.isfinite(ni):
                # random tangent
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
            # rest max excluding i,j : mask rows/cols
            # compute rest_max by max of G with i,j rows/cols ignored
            # fast: set entries to -2 in a view copy? operate on G copy of two rows
            # we compute rest_max without full copy: max of upper part excluding
            # simplest: temporarily set
            # save rows/cols
            # use numpy max with masking via -inf assignment on copy of small slices
            # To avoid large copy, compute rest_max as max of G over mask
            # G is 453x453, copy is cheap (200k)
            # do inline:
            G[i, :] = -2.0
            G[:, i] = -2.0
            G[j, :] = -2.0
            G[:, j] = -2.0
            rest_max = float(np.max(G))
            improved = False
            # try scales
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
                    if cand < cur_best_V:
                        cur_best_V = cand
                        cur_best_X = x.copy()
                    eps = min(eps * 1.05, 3e-4)
                    improved = True
                    break
            if not improved:
                eps *= 0.985
                if eps < 2e-7:
                    eps = 2e-7
                    # tiny random kick to worst point occasionally to avoid lock
                    if (it % 250) == 249:
                        k = i
                        rr = rng.standard_normal(d)
                        rr -= float(rr @ x[k]) * x[k]
                        rn = float(np.linalg.norm(rr))
                        if rn > 1e-12:
                            rr /= rn
                            xn = x[k] + 1e-7 * rr
                            xn /= float(np.linalg.norm(xn))
                            x[k] = xn
        return x

    try:
        # Phase A : polish basin as-is
        x = cur_best_X.copy()
        if time.time() < DEAD - 20:
            x = smooth_run(x, 5500, 3.0e-3, 1.0e-4, 1.5e-3, 1.0e-5, 0.90)
        if time.time() < DEAD - 15:
            x = greedy_shave(x, 1800, 8e-5)
        # Phase B : perturbed re-anneals
        sigmas = [0.0012, 0.0025, 0.0045, 0.0018, 0.0032]
        for k, sig in enumerate(sigmas):
            if time.time() > DEAD - 10.0:
                break
            # start from best with tangent-ish noise
            y = cur_best_X + sig * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
            # vary schedule slightly per cycle
            tau0 = 1.2e-3 if k % 2 == 0 else 2.0e-3
            tau1 = 6e-5 if k < 3 else 4e-5
            lr0 = 8e-4 if k % 2 == 0 else 1.2e-3
            lr1 = 5e-6
            N = 3800 if time.time() < DEAD - 40 else 1800
            y = smooth_run(y, N, tau0, tau1, lr0, lr1, 0.88)
            if time.time() > DEAD - 8.0:
                break
            y = greedy_shave(y, 900, 5e-5)
            # keep x as best for next perturbation reference
            x = cur_best_X.copy()
        # final micro polish from best
        if time.time() < DEAD - 4.0:
            z = cur_best_X.copy()
            z = smooth_run(z, 1200, 3.0e-4, 4.0e-5, 1.2e-4, 3.0e-6, 0.85)
            z = greedy_shave(z, 600, 1.5e-5)
    except Exception:
        pass

    Xout = cur_best_X.copy()
    nn = np.sqrt(np.sum(Xout * Xout, axis=1, keepdims=True))
    nn[nn == 0] = 1.0
    Xout = Xout / nn
    # shrink by tiny margin guard (no-op, ensures finite)
    Xout = np.where(np.isfinite(Xout), Xout, 0.0)
    # if any zero row (should not), replace
    rn = np.linalg.norm(Xout, axis=1)
    bad = rn < 1e-12
    if np.any(bad):
        Xout[bad] = rng.standard_normal((int(np.sum(bad)), d))
        Xout[bad] /= np.linalg.norm(Xout[bad], axis=1, keepdims=True) + 1e-30
    return Xout
