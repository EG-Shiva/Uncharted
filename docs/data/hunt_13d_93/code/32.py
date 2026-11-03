import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = 170.0
    HOP_END = 152.0
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    M = len(iu0)

    best = None
    best_val = 1e9
    rec = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if rec is None and fn == "record.npy":
                rec = a.copy()
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)
    if rec is None:
        rec = best.copy()

    # ---------- shifted p-norm continuation ----------
    def p_run(X0, steps_per_stage, p_list, lr0, lr1, shift):
        X = _norm(X0.copy())
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        total = len(p_list) * steps_per_stage
        tglob = 0
        best_seen = _maxcos(X)
        best_X = X.copy()
        lr_ratio = lr1 / lr0 if lr0 != 0 else 1.0
        for p in p_list:
            # rescale step by resetting moments at each p increase
            m[:] = 0.0
            v[:] = 0.0
            tt = 0
            pm1 = float(p - 1.0)
            for s in range(steps_per_stage):
                if time.time() - t0 > DEADLINE - 4:
                    break
                frac = tglob / max(1, total - 1)
                lr = lr0 * (lr_ratio ** frac)
                # analytic tangent gradient of Fp = (sum max(c-shift,0)^p)^{1/p}
                G = X @ X.T
                c = G[iu0, iu1]
                sc = c - shift
                pos = sc > 0
                gt = None
                if np.any(pos):
                    idxp = np.where(pos)[0]
                    scp = sc[idxp]
                    # log-stable weights ~ s^{p-1}
                    logs = np.log(np.maximum(scp, 1e-300))
                    lmax = float(np.max(logs))
                    e = np.exp(pm1 * (logs - lmax))
                    esum = float(np.sum(e))
                    if esum > 0 and np.isfinite(esum):
                        wpos = e / esum
                        # effective active pairs (entropy of s^p) for monitoring
                        # q = exp(p*(logs-lmax))/sum ; Neff = exp(entropy)
                        # (computed cheaply, not used for step except diagnostics)
                        w = np.zeros(M, dtype=np.float64)
                        w[idxp] = wpos
                        W = np.zeros((n, n), dtype=np.float64)
                        W[iu0, iu1] = w
                        W[iu1, iu0] = w
                        grad = W @ X
                        dot = np.einsum('ij,ij->i', grad, X)[:, None]
                        gt = grad - dot * X
                tglob += 1
                if gt is None or not np.all(np.isfinite(gt)):
                    continue
                ng = float(np.linalg.norm(gt))
                if ng < 1e-16 or not np.isfinite(ng):
                    continue
                tt += 1
                m[:] = b1 * m + (1.0 - b1) * gt
                v[:] = b2 * v + (1.0 - b2) * (gt * gt)
                mh = m / (1.0 - b1 ** tt)
                vh = v / (1.0 - b2 ** tt)
                step = lr * mh / (np.sqrt(vh) + eps)
                X = X - step
                X = _norm(X)
                # track true maximum
                if (tglob % 25) == 0:
                    cur = _maxcos(X)
                    if cur < best_seen:
                        best_seen = cur
                        best_X = X.copy()
            cur = _maxcos(X)
            if cur < best_seen:
                best_seen = cur
                best_X = X.copy()
            if time.time() - t0 > DEADLINE - 4:
                break
        # return best seen
        return best_X, best_seen

    # ---------- softmax LSE Adam for basin hops ----------
    def smooth_adam(x, steps, tau0, tau1, lr0, lr1):
        x = _norm(x.copy())
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        for s in range(steps):
            if time.time() - t0 > DEADLINE - 3:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
            lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
            if tau < 1e-12:
                tau = 1e-12
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mg = float(G.max())
            W = np.exp((G - mg) / tau)
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-12
            grad = (W @ x) / rs
            dot = (grad * x).sum(axis=1, keepdims=True)
            gt = grad - dot * x
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m = b1 * m + (1.0 - b1) * gt
            vv = b2 * v + (1.0 - b2) * (gt * gt)
            v = vv
            mh = m / (1.0 - b1 ** tt)
            vh = vv / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            dot2 = (step * x).sum(axis=1, keepdims=True)
            step_t = step - dot2 * x
            x = x - step_t
            x = _norm(x)
        return x

    def hard_topk(X, iters, step0, K):
        X = _norm(X.copy())
        bv = _maxcos(X)
        bestX = X.copy()
        step = float(step0)
        K = int(min(K, M))
        grad = np.empty_like(X)
        for it in range(iters):
            if time.time() - t0 > DEADLINE - 2:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            if K < M:
                idx = np.argpartition(-c, K)[:K]
            else:
                idx = np.arange(M)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            rn = np.linalg.norm(gt, axis=1, keepdims=True)
            nz = (rn[:, 0] > 1e-14)
            direc = np.zeros_like(gt)
            direc[nz] = gt[nz] / rn[nz, None]
            if not np.all(np.isfinite(direc)):
                step *= 0.5
                continue
            Xt = _norm(X - step * direc)
            vv = _maxcos(Xt)
            if vv < bv:
                bv = vv
                bestX = Xt.copy()
                X = Xt
                step = min(step * 1.25, 0.006)
            else:
                step *= 0.55
                if step < 1e-12:
                    step = float(step0) * 0.25
                    # random tangent kick to escape
                    kick = rng.standard_normal(X.shape) * 1e-4
                    kick -= (kick * X).sum(axis=1, keepdims=True) * X
                    X = _norm(X + kick)
        return bestX, bv

    def find_void(others, n_cand=4000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-18
        D = C @ others.T
        worst = D.max(axis=1)
        vv = C[int(np.argmin(worst))].copy()
        lr = 0.05; tau = 0.008
        for _ in range(100):
            dots = others @ vv
            m0 = float(dots.max())
            w = np.exp((dots - m0) / tau)
            s = float(w.sum())
            if s <= 0 or not np.isfinite(s):
                break
            w /= s
            g = w @ others
            gt = g - float(g @ vv) * vv
            nr = float(np.linalg.norm(gt))
            if nr < 1e-12 or not np.isfinite(nr):
                break
            gt /= nr
            v2 = vv - lr * gt
            n2 = float(np.linalg.norm(v2))
            if n2 < 1e-12:
                lr *= 0.5
                continue
            v2 /= n2
            m2 = float((others @ v2).max())
            if m2 < m0:
                vv = v2
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return vv / (np.linalg.norm(vv) + 1e-18)

    p_list = [8, 16, 32, 64, 128, 256]
    # build starting configs: best, record, strongly perturbed, random
    starts = []
    starts.append(best.copy())
    starts.append(rec.copy())
    for sig in (0.15, 0.3, 0.6):
        starts.append(_norm(best + sig * rng.standard_normal(best.shape)))
    starts.append(_norm(rng.standard_normal((n, d))))
    starts.append(_norm(rec + 0.3 * rng.standard_normal(rec.shape)))
    starts.append(_norm(best + 0.15 * rng.standard_normal(best.shape)))

    contenders = []  # (val, X)
    for si, X0 in enumerate(starts):
        if time.time() - t0 > HOP_END - 25:
            break
        # fewer steps for many starts to fit budget; more for first two
        if si < 2:
            sps = 320
        else:
            sps = 220
        try:
            Xp, vp = p_run(X0, steps_per_stage=sps, p_list=p_list, lr0=3e-3, lr1=3e-4, shift=0.15)
        except Exception:
            continue
        # promote through hard-max polish
        try:
            Xh, vh = hard_topk(Xp, iters=220, step0=0.0012, K=700)
            Xh2, vh2 = hard_topk(Xh, iters=220, step0=0.0007, K=120)
            if vh2 < vh:
                Xh, vh = Xh2, vh2
        except Exception:
            Xh, vh = Xp, vp
        contenders.append((vh, Xh.copy()))
        if vh < best_val - 1e-12:
            best_val = vh
            best = Xh.copy()
        elif vp < best_val - 1e-12:
            best_val = vp
            best = Xp.copy()

    # keep best 3 continuations, polish further
    try:
        contenders.sort(key=lambda t: t[0])
        for k in range(min(3, len(contenders))):
            if time.time() - t0 > HOP_END:
                break
            Xc = contenders[k][1]
            Xp2, vp2 = hard_topk(Xc, iters=300, step0=0.0006, K=60)
            if vp2 < best_val:
                best_val = vp2
                best = Xp2.copy()
                contenders[k] = (vp2, Xp2.copy())
    except Exception:
        pass

    # ---------- topology-changing basin hops with bulk equalization ----------
    trial = 0
    while time.time() < HOP_END and trial < 60:
        if time.time() - t0 > HOP_END:
            break
        mode = trial % 5
        try:
            if mode == 0:
                sig = 0.02 if (trial % 2 == 0) else 0.035
                Xt = _norm(best + sig * rng.standard_normal(best.shape))
            elif mode == 1:
                # rattler / low-contact shake
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                low = np.argsort(rm)[:12]
                Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
                Xt[low] += 0.06 * rng.standard_normal((len(low), d))
                hi = np.argsort(-rm)[:2]
                Xt[hi] += 0.008 * rng.standard_normal((len(hi), d))
                Xt = _norm(Xt)
            elif mode == 2:
                # multi-void reinsertion of worst points
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm)
                chosen = []
                for cand in order:
                    ok = True
                    for cc in chosen:
                        if float(best[cand] @ best[cc]) > 0.7:
                            ok = False
                            break
                    if ok:
                        chosen.append(int(cand))
                    if len(chosen) >= 3:
                        break
                Xt = best.copy()
                mask = np.ones(n, dtype=bool)
                mask[np.array(chosen)] = False
                base = Xt[mask]
                for cc in chosen:
                    vv = find_void(base, n_cand=4000)
                    Xt[cc] = vv
                    base = np.vstack([base, vv[None, :]])
                Xt = _norm(Xt + 0.0015 * rng.standard_normal(Xt.shape))
            elif mode == 3:
                # single worst void + small noise
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                w = int(np.argmax(rm))
                mask = np.ones(n, dtype=bool); mask[w] = False
                vv = find_void(best[mask], n_cand=4000)
                Xt = best.copy(); Xt[w] = vv
                Xt = _norm(Xt + 0.002 * rng.standard_normal(Xt.shape))
            else:
                Xt = _norm(best + 0.012 * rng.standard_normal(best.shape))
        except Exception:
            trial += 1
            continue
        try:
            if mode == 2:
                Xt = smooth_adam(Xt, steps=800, tau0=0.012, tau1=0.0006, lr0=0.018, lr1=0.002)
            else:
                Xt = smooth_adam(Xt, steps=650, tau0=0.009, tau1=0.0006, lr0=0.014, lr1=0.0012)
        except Exception:
            trial += 1
            continue
        try:
            Xb, vb = hard_topk(Xt, iters=200, step0=0.0012, K=800)
            Xc, vc = hard_topk(Xb, iters=200, step0=0.0007, K=120)
            if vc < vb:
                Xb, vb = Xc, vc
        except Exception:
            trial += 1
            continue
        if vb < best_val - 1e-12:
            best_val = vb
            best = Xb.copy()
        trial += 1

    # ---------- final hard-max/active-set polish ----------
    try:
        for K, s0, it in [(800, 0.001, 400), (250, 0.0007, 400), (80, 0.0004, 400)]:
            if time.time() - t0 > DEADLINE - 5:
                break
            b, bv = hard_topk(best, iters=it, step0=s0, K=K)
            if bv < best_val:
                best_val = bv
                best = b.copy()
    except Exception:
        pass

    best = _norm(np.asarray(best, dtype=np.float64))
    return best
