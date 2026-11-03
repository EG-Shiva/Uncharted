import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _adam_softmax(x, steps, beta0, beta1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        # tangent projection
        nd = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - nd * x
        m *= b1
        m += (1.0 - b1) * grad
        v *= b2
        v += (1.0 - b2) * (grad * grad)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mh = m / bc1
        vh = v / bc2
        upd = mh / (np.sqrt(vh) + eps)
        # re-project Adam step to tangent
        nd2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - nd2 * x
        # normalize update scale per point to avoid blowup
        sc = np.sqrt(np.sum(upd * upd, axis=1, keepdims=True)) + 1e-30
        # clip extreme
        upd = upd / np.maximum(sc / 3.0, 1.0)
        x = x - lr * upd
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        mv = c.max()
        thr = mv - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _find_voids(remaining, k, rng, n_sample=20000):
    # greedy farthest insertion using random candidates
    d = remaining.shape[1]
    cand = rng.normal(size=(n_sample, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    # max dot to remaining set
    G = cand @ remaining.T
    score = G.max(axis=1)
    chosen = []
    # iterative greedy to keep voids mutually separated
    for _ in range(k):
        if chosen:
            cc = cand[chosen]  # (c,d)
            gg = cand @ np.asarray(cc).T if False else None
        # pick best not too close to already chosen
        order = np.argsort(score)
        pick = None
        for o in order:
            if o in chosen:
                continue
            ok = True
            for ch in chosen:
                if float(cand[o] @ cand[ch]) > 0.75:
                    ok = False
                    break
            if ok:
                pick = int(o)
                break
        if pick is None:
            # fallback
            for o in order:
                if o not in chosen:
                    pick = int(o)
                    break
        chosen.append(pick)
        # penalize neighbours of pick to enforce spread
        dp = cand @ cand[pick]
        score = np.maximum(score, dp * 0.5 + 0.25)
        # also account for chosen point itself
        score[pick] = 2.0
    return cand[np.array(chosen)]


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    it = 0
    # temperature for basin hopping acceptance
    Temp = 0.002
    while time.time() < deadline - 7.0 and it < 500:
        it += 1
        now = time.time()
        time_left = deadline - 7.0 - now
        if time_left <= 0:
            break
        mode = it % 10
        y = None
        if mode == 0:
            # fresh random basin (rare)
            y = rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=900, beta0=8.0, beta1=250.0, lr0=0.03, lr1=0.002)
            y = _adam_softmax(y, steps=400, beta0=250.0, beta1=600.0, lr0=0.005, lr1=0.001)
            if time.time() < deadline - 7.0:
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 2.5))
        elif mode in (1, 4, 7):
            # kill worst k and rebirth into voids
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            k = 6 if (it % 20 == 1) else (4 if (it % 2 == 0) else 8)
            # vary k
            worst = np.argsort(-per)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[worst] = False
            remaining = cur[keep_mask]
            voids = _find_voids(remaining, k, rng, n_sample=25000)
            y = cur.copy()
            y[worst] = voids + 0.02 * rng.normal(size=(k, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=700, beta0=15.0, beta1=300.0, lr0=0.02, lr1=0.002)
            y = _adam_softmax(y, steps=250, beta0=300.0, beta1=600.0, lr0=0.004, lr1=0.001)
            # LP only if promising to save time
            gv = _max_cos(y)
            if gv < cur_val + 0.005 and time.time() < deadline - 7.0:
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.5))
        elif mode in (2, 5, 8):
            # overload shake + global kick
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            cmax = G.max()
            cnt = (G > cmax - 0.01).sum(axis=1)
            order = np.argsort(-cnt)
            y = cur.copy()
            # strong shake on top overloaded
            nover = 12
            idx_o = order[:nover]
            sigma_o = 0.20 if (it % 3 == 0) else 0.10
            y[idx_o] += sigma_o * rng.normal(size=(nover, d))
            # small global noise
            if it % 2 == 0:
                y += 0.02 * rng.normal(size=(n, d))
            else:
                y += 0.06 * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            # occasional random re-insert of 2 points
            if it % 4 == 0:
                kk = 3
                idx = rng.choice(n, size=kk, replace=False)
                rp = rng.normal(size=(kk, d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[idx] = rp
            y = _adam_softmax(y, steps=700, beta0=12.0, beta1=280.0, lr0=0.02, lr1=0.002)
            gv = _max_cos(y)
            if gv < cur_val + 0.005 and time.time() < deadline - 7.0:
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.5))
        else:
            # local fine shake
            sigma = 0.03 if (it % 2 == 0) else 0.008
            y = cur + sigma * rng.normal(size=(n, d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=600, beta0=20.0, beta1=350.0, lr0=0.015, lr1=0.0015)
            gv = _max_cos(y)
            if gv < cur_val + 0.003 and time.time() < deadline - 7.0:
                y = _lp_polish(y, min(deadline - 7.0, time.time() + 1.2))

        if y is None:
            continue
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-4)):
            cur = y.copy()
            cur_val = v
        # decay temperature slowly
        Temp *= 0.998
        Temp = max(Temp, 0.0003)

    try:
        best = _lp_polish(best, min(deadline, time.time() + 5.5))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
