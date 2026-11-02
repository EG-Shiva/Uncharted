import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    np.random.seed(0)
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
    Xcur = bestX.copy()
    Vcur = bestV

    def anneal(X, steps, tau0, tau1, lr0, lr1, mu, bal_delta, bal_w):
        nonlocal bestV, bestX
        v = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 > 0 and tau1 > 0 else 0.0
        llr = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
        for it in range(steps):
            if (it % 40 == 0) and time.time() > DEAD - 9.0:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = X.copy()
            # softmax weights near max
            with np.errstate(over='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            # normalize soft part
            s = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if np.isfinite(s) and s > 1e-14:
                Gt /= s
            else:
                Gt[:] = 0.0
            if bal_w > 0 and bal_delta > 0:
                thr = m - bal_delta
                # hinge weights for broad near-contact band -> equalize degrees
                H = G - thr
                H[H < 0] = 0.0
                H /= bal_delta
                np.fill_diagonal(H, 0.0)
                if float(np.max(H)) > 0:
                    Gb = H @ X
                    dotb = np.sum(Gb * X, axis=1, keepdims=True)
                    Gb -= dotb * X
                    sb = float(np.max(np.sqrt(np.sum(Gb * Gb, axis=1))))
                    if np.isfinite(sb) and sb > 1e-14:
                        Gb /= sb
                        Gt = (1.0 - bal_w) * Gt + bal_w * Gb
            v = mu * v + (1.0 - mu) * Gt
            X = X - lr * v
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
            # elitist reset if diverged
            if m > bestV + 0.02:
                X = bestX.copy()
                v *= 0.3
        return X

    def contact_degrees(X, m, thr=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        C = (G >= m - thr)
        deg = np.sum(C, axis=1)
        return G, deg

    def find_holes(Xkeep, k, n_cand=12000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        S = C @ Xkeep.T
        ms = np.max(S, axis=1)
        order = np.argsort(ms)
        picked = []
        for idx in order:
            if len(picked) >= k:
                break
            c = C[idx]
            ok = True
            for p in picked:
                if float(c @ p) > 0.92:
                    ok = False
                    break
            if ok:
                picked.append(c)
            if len(picked) < k and idx == order[-1]:
                break
        # fallback fill
        j = 0
        while len(picked) < k and j < len(order):
            picked.append(C[order[j]])
            j += 1
        return np.array(picked)

    def greedy_shave(X, steps, eps0):
        nonlocal bestV, bestX
        nloc = X.shape[0]
        eps = float(eps0)
        for it in range(steps):
            if (it % 60 == 0) and time.time() > DEAD - 6.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = X.copy()
            flat = int(np.argmax(G))
            i = flat // nloc
            j = flat % nloc
            xi = X[i].copy()
            xj = X[j].copy()
            cij = float(xi @ xj)
            cij = max(-1.0, min(1.0, cij))
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d)
                ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d)
                tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj
            improved = False
            for scale in (eps, eps * 0.5):
                vi = xi - scale * ti
                vi /= (float(np.linalg.norm(vi)) + 1e-30)
                vj = xj - scale * tj
                vj /= (float(np.linalg.norm(vj)) + 1e-30)
                oi = X[i].copy(); oj = X[j].copy()
                X[i] = vi; X[j] = vj
                G2 = X @ X.T
                np.fill_diagonal(G2, -2.0)
                m2 = float(np.max(G2))
                if m2 < m - 1e-13:
                    if m2 < bestV:
                        bestV = m2
                        bestX = X.copy()
                    eps = min(eps * 1.05, 3e-3)
                    improved = True
                    break
                else:
                    X[i] = oi; X[j] = oj
            if not improved:
                # try single-sided
                for scale, side in ((eps, 0), (eps, 1)):
                    if side == 0:
                        vi = xi - scale * ti
                        vi /= (float(np.linalg.norm(vi)) + 1e-30)
                        oi = X[i].copy(); X[i] = vi
                        G2 = X @ X.T
                        np.fill_diagonal(G2, -2.0)
                        m2 = float(np.max(G2))
                        if m2 < m - 1e-13:
                            if m2 < bestV:
                                bestV = m2; bestX = X.copy()
                            improved = True
                            break
                        X[i] = oi
                    else:
                        vj = xj - scale * tj
                        vj /= (float(np.linalg.norm(vj)) + 1e-30)
                        oj = X[j].copy(); X[j] = vj
                        G2 = X @ X.T
                        np.fill_diagonal(G2, -2.0)
                        m2 = float(np.max(G2))
                        if m2 < m - 1e-13:
                            if m2 < bestV:
                                bestV = m2; bestX = X.copy()
                            improved = True
                            break
                        X[j] = oj
            if not improved:
                eps *= 0.94
                if eps < 2e-9:
                    eps = 2e-9
                    # tiny random kick to worst of the pair
                    k = i if (it % 2 == 0) else j
                    rr = rng.standard_normal(d)
                    rr -= float(rr @ X[k]) * X[k]
                    rn = float(np.linalg.norm(rr))
                    if rn > 1e-12:
                        rr /= rn
                        X[k] = X[k] + 5e-7 * rr
                        X[k] /= float(np.linalg.norm(X[k])) + 1e-30
        return X

    # initial balanced anneal from incumbent
    if time.time() < DEAD - 20:
        Xcur = anneal(Xcur, 700, 0.004, 0.0004, 0.004, 0.0006, 0.85, 0.004, 0.35)
    if time.time() < DEAD - 15:
        Xcur = greedy_shave(Xcur, 1200, 8e-4)

    # basin-hopping with rattler/hub surgery
    cyc = 0
    while time.time() < DEAD - 12.0 and cyc < 25:
        cyc += 1
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        _, deg = contact_degrees(Xcur, mcur, thr=1e-9)
        # broad degree for hub detection
        _, degB = contact_degrees(Xcur, mcur, thr=5e-4)
        ratt = np.where(deg <= 1)[0]
        hubs = np.argsort(-degB)[:6]
        # worst pair members
        flat = int(np.argmax(G))
        wi = flat // n; wj = flat % n
        cand_replace = []
        for r in ratt[:4]:
            cand_replace.append(int(r))
        # add one worst-pair member with higher broad degree
        if int(degB[wi]) >= int(degB[wj]):
            cand_replace.append(int(wi))
        else:
            cand_replace.append(int(wj))
        # add top hub not already included
        for h in hubs:
            if int(h) not in cand_replace:
                cand_replace.append(int(h))
                break
        # unique, cap 2-4
        uniq = []
        for u in cand_replace:
            if u not in uniq:
                uniq.append(u)
        k = min(4, max(2, len(uniq)))
        repl = np.array(uniq[:k], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[repl] = False
        Xkeep = Xcur[keep_mask]
        holes = find_holes(Xkeep, k, n_cand=12000)
        # jitter holes slightly
        jit = rng.standard_normal(holes.shape) * 0.002
        jit -= np.sum(jit * holes, axis=1, keepdims=True) * holes
        holes = holes + jit
        holes /= np.linalg.norm(holes, axis=1, keepdims=True) + 1e-30
        Xt = Xcur.copy()
        Xt[repl] = holes
        # shake non-replaced a little along tangent (floppy exploration)
        amp = 0.004 if cyc % 2 == 0 else 0.0015
        Nz = rng.standard_normal(Xt.shape) * amp
        Nz -= np.sum(Nz * Xt, axis=1, keepdims=True) * Xt
        # freeze replaced rows from shake
        Nz[repl] = 0.0
        Xt = Xt + Nz
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-30
        # re-anneal: alternate tau schedules for diversity
        if cyc % 3 == 0:
            Xt = anneal(Xt, 380, 0.006, 0.0003, 0.005, 0.0005, 0.88, 0.006, 0.45)
        elif cyc % 3 == 1:
            Xt = anneal(Xt, 380, 0.002, 0.0002, 0.003, 0.0004, 0.80, 0.002, 0.25)
        else:
            Xt = anneal(Xt, 380, 0.004, 0.00025, 0.004, 0.0005, 0.85, 0.010, 0.5)
        Xt = greedy_shave(Xt, 500, 5e-4)
        vnew = _max_cos(Xt)
        # elitist + occasional sideways move to different basin
        if vnew < Vcur or (vnew < bestV + 0.001 and cyc % 4 == 0):
            Xcur = Xt.copy()
            Vcur = vnew
        else:
            # restart partly from best with shake
            if time.time() < DEAD - 20 and rng.random() < 0.5:
                Xcur = bestX.copy()
                Nz2 = rng.standard_normal(Xcur.shape) * 0.003
                Nz2 -= np.sum(Nz2 * Xcur, axis=1, keepdims=True) * Xcur
                Xcur = Xcur + Nz2
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True) + 1e-30
                Xcur = anneal(Xcur, 250, 0.003, 0.0004, 0.003, 0.0005, 0.85, 0.003, 0.3)
                Vcur = _max_cos(Xcur)

    # final fine polish of global best
    Xf = bestX.copy()
    if time.time() < DEAD - 6.0:
        # short balanced anneal at low temperature
        Xf = anneal(Xf, 250, 0.0008, 0.00015, 0.0012, 0.00025, 0.9, 0.0015, 0.2)
    if time.time() < DEAD - 3.0:
        Xf = greedy_shave(Xf, 800, 2e-4)
    # ensure best returned
    vf = _max_cos(Xf)
    if vf < bestV:
        bestV = vf
        bestX = Xf.copy()
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30
    return bestX
