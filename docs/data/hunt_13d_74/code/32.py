import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(12345)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    def adam_lse(X, schedule):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        for tau, steps, lr0 in schedule:
            for s in range(steps):
                if (s & 63) == 0 and time.time() - t0 > LIM - 5:
                    return X
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                gmax = float(G.max())
                E = np.exp((G - gmax) / tau)
                np.fill_diagonal(E, 0.0)
                S = float(E.sum())
                if not np.isfinite(S) or S <= 0:
                    continue
                Gr = (E @ X) / S
                dot = (Gr * X).sum(axis=1, keepdims=True)
                Gt = Gr - dot * X
                m[:] = 0.9 * m + 0.1 * Gt
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt += 1
                c1 = 1.0 - 0.9 ** cnt
                c2 = 1.0 - 0.999 ** cnt
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def minimax(X, steps, lr0, cur_init=None):
        cur = cur_init if cur_init is not None else _max_cos(X)
        lr = float(lr0)
        for it in range(steps):
            if (it & 15) == 0 and time.time() - t0 > LIM - 3:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def degrees(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        deg = (G > (gmax - thr)).sum(axis=1)
        return deg, gmax, G

    def refine_cand(c, S, steps=70, lr0=0.03):
        # S: (m,d) current set, minimize max_j c.Sj
        c = c.copy()
        lr = lr0
        for it in range(steps):
            dots = S @ c
            j = int(np.argmax(dots))
            cmax = float(dots[j])
            g = S[j] - cmax * c
            ng = float(np.linalg.norm(g))
            if ng < 1e-12 or not np.isfinite(ng):
                break
            g /= ng
            c = c - lr * g
            nr = float(np.linalg.norm(c))
            if nr == 0 or not np.isfinite(nr):
                break
            c /= nr
            lr *= (1.0 - 0.008)
        return c

    sched_full = [(0.008, 300, 0.008), (0.003, 400, 0.005), (0.0012, 300, 0.003)]
    sched_fast = [(0.008, 220, 0.008), (0.003, 280, 0.005), (0.0012, 220, 0.003)]

    # light initial polish to lock baseline
    X0 = best.copy()
    X0 = adam_lse(X0, sched_fast)
    X0, v0 = minimax(X0, 250, 0.002)
    if v0 < best_val:
        best, best_val = X0, v0
    if time.time() - t0 > LIM - 10:
        return _normalize(best)

    trial = 0
    targets = [6, 10, 14]
    while time.time() - t0 < LIM - 9:
        T = targets[trial % 3]
        H = 2 + (trial % 3)  # 2,3,4 hubs; for T=6 ->2 hubs, T=10->3, T=14->4
        # compute degrees on current best
        deg, gmax, G = degrees(best, thr=1e-7)
        rattlers = np.where(deg == 0)[0]
        # hubs: highest degree, must be >=9 else take top anyway
        order_deg = np.argsort(-deg)
        hubs = []
        for idx in order_deg:
            if len(hubs) >= H:
                break
            hubs.append(int(idx))
        # also require degree>=9 if possible; if top hubs have low degree still keep them
        hubs = np.array(hubs, dtype=int)
        # build deletion set of size T: hubs + rattlers fill + worst max fill
        delset = list(hubs)
        # add rattlers not already in delset (shuffled)
        rlist = [int(i) for i in rattlers if int(i) not in delset]
        # shuffle rattlers deterministically via rng
        if len(rlist) > 1:
            perm = rng.permutation(len(rlist))
            rlist = [rlist[k] for k in perm]
        need = T - len(delset)
        take_r = min(len(rlist), max(0, need))
        delset.extend(rlist[:take_r])
        need = T - len(delset)
        if need > 0:
            # add worst-max points
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)
            worder = np.argsort(-mx)
            for idx in worder:
                if int(idx) not in delset:
                    delset.append(int(idx))
                    if len(delset) >= T:
                        break
        if len(delset) > T:
            delset = delset[:T]
        delset = np.array(delset, dtype=int)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[delset] = False
        core = best[keep_mask].copy()  # (74-T, d), frozen

        # greedy reinsertion, frozen core
        cur_set = core.copy()
        # random order of insertion doesn't matter; do T insertions
        ok = True
        for si in range(T):
            if time.time() - t0 > LIM - 9:
                ok = False
                break
            m = cur_set.shape[0]
            NC = 20000
            cand = _normalize(rng.standard_normal((NC, d)))
            # score by max cosine to cur_set
            S = cand @ cur_set.T
            sc = S.max(axis=1)
            # best 20
            K = 20
            part = np.argpartition(sc, K)[:K]
            best_c = None
            best_s = 1e9
            for k in part:
                c0 = cand[k]
                cr = refine_cand(c0, cur_set, steps=60, lr0=0.03)
                # score refined
                s = float(np.max(cur_set @ cr))
                # also consider unrefined? keep refined
                if s < best_s:
                    best_s = s
                    best_c = cr
            if best_c is None:
                best_c = _normalize(rng.standard_normal((1, d)))[0]
            cur_set = np.vstack([cur_set, best_c.reshape(1, d)])
        if not ok:
            break
        Xcur = _normalize(cur_set)
        # unfreeze: annealed LogSumExp + active-set minimax
        sched = sched_full if (trial % 3 == 0) else sched_fast
        # small random nudge to break symmetry after insertion? no, keep frozen benefit
        Xcur = adam_lse(Xcur, sched)
        Xcur, vcur = minimax(Xcur, 280, 0.002)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        else:
            # opportunistic: even tiny gain counts; also try a second polish with tiny kick
            # if close, do extra minimax with smaller lr
            if vcur < best_val + 2e-9 and time.time() - t0 < LIM - 20:
                X2 = Xcur.copy()
                X2, v2 = minimax(X2, 150, 0.0006, cur_init=vcur)
                if v2 < best_val - 1e-12:
                    best = X2.copy()
                    best_val = float(v2)
        trial += 1
        if trial > 200:
            break

    # final squeeze
    if time.time() - t0 < LIM - 2:
        Xf = best.copy()
        Xf, vf = minimax(Xf, 250, 0.0008, cur_init=best_val)
        if vf < best_val:
            best, best_val = Xf, vf

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
