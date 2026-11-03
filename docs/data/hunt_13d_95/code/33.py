import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, radius_init=1e-3, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if time.time() - start > seconds - 0.06:
            break
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = cfull > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.argpartition(cfull, -1100)[-1100:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        csel = cfull[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _gd(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize gradient scale
        gn = np.linalg.norm(Gt)
        if gn > 1e-12:
            Gt = Gt / (1.0 + gn)
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_END = t0 + 118.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (95, 13):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        c = rng.normal(size=(95, 13)); c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = (95, 13)
    # per-point max helper
    def per_pt(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G
    trial = 0
    # fast hole sampling
    while time.time() < EXPL_END and time.time() - t0 < BUDGET - 45:
        trial += 1
        remain = EXPL_END - time.time()
        if remain < 0.5:
            break
        mode = trial % 10
        # base
        if mode == 9 and trial > 5:
            # fresh random start for new topology (1 in 10)
            Xt = rng.normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 900, 0.02, 0.0008, 0.15, 0.008
        else:
            Xt = x_best.copy()
            ppm, _ = per_pt(Xt)
            order_hi = np.argsort(-ppm)
            order_lo = np.argsort(ppm)
            r = rng.random()
            if r < 0.45:
                # global / subset shake
                sig = float(rng.choice([0.01, 0.02, 0.04, 0.08, 0.15, 0.28]))
                if rng.random() < 0.4:
                    ksub = int(rng.integers(4, 16))
                    # bias to overloaded + rattler
                    idx = np.concatenate([order_hi[:3], order_lo[:2]])
                    if ksub > 5:
                        extra = rng.choice(n, size=ksub-5, replace=False)
                        idx = np.unique(np.concatenate([idx, extra]))[:ksub]
                        if len(idx) < ksub:
                            idx = rng.choice(n, size=ksub, replace=False)
                    Xt[idx] += sig * 2.0 * rng.normal(size=(len(idx), d))
                else:
                    Xt += sig * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 600 + int(rng.integers(0, 400))
                T0 = float(rng.choice([0.008, 0.012, 0.018]))
                T1 = 0.0008 + rng.random() * 0.0007
                lr0 = 0.06 + rng.random() * 0.08
                lr1 = 0.005 + rng.random() * 0.005
            else:
                # coordinated block reinsertion: hub(s) + rattler together
                if rng.random() < 0.5:
                    k = int(rng.choice([2, 3, 3, 4]))
                else:
                    k = 1
                # build index set: at least one hub and one rattler when k>=2
                idx = []
                if k >= 2:
                    idx.append(int(order_hi[0]))
                    # rattler = lowest contact point
                    idx.append(int(order_lo[0]))
                    while len(idx) < k:
                        c = int(rng.integers(0, n))
                        if c not in idx:
                            idx.append(c)
                else:
                    # k==1: mostly hub, sometimes rattler
                    if rng.random() < 0.7:
                        idx = [int(order_hi[int(rng.integers(0, 5))])]
                    else:
                        idx = [int(order_lo[0])]
                idx = np.array(idx, dtype=int)
                # sequential greedy hole fill
                for j in idx:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    others = Xt[mask]
                    nc = 3500 if k == 1 else 2500
                    C = rng.normal(size=(nc, d))
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    dots = C @ others.T
                    md = dots.max(axis=1)
                    b = int(np.argmin(md))
                    # try a couple of local jitters of best hole
                    if rng.random() < 0.6:
                        ref = C[b:b+1] + 0.02 * rng.normal(size=(1, d))
                        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
                        d0 = float((ref @ others.T).max())
                        if d0 < float(md[b]):
                            Xt[j] = ref[0]
                        else:
                            Xt[j] = C[b]
                    else:
                        Xt[j] = C[b]
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 700 + int(rng.integers(0, 300))
                T0 = 0.010 + rng.random() * 0.010
                T1 = 0.0006 + rng.random() * 0.0008
                lr0 = 0.08 + rng.random() * 0.06
                lr1 = 0.006 + rng.random() * 0.004
        # fast screen with soft-max repulsion
        try:
            Xt = _gd(Xt, steps, T0, T1, lr0, lr1)
        except Exception:
            continue
        v = _max_cos(Xt)
        improved_screen = v < best_val + 0.004
        # selective LP: polish if close or periodically
        do_lp = False
        if v < best_val - 2e-6:
            do_lp = True
        elif improved_screen and (trial % 4 == 0 or v < best_val + 0.0005):
            do_lp = True
        elif trial % 12 == 0:
            do_lp = True
        if do_lp:
            if time.time() - t0 > BUDGET - 42:
                break
            try:
                Xp = _lp_polish(Xt, seconds=0.9, radius_init=0.006, t_deadline=BUDGET + t0 - 40)
            except Exception:
                continue
            vp = _max_cos(Xp)
            if vp < best_val - 1e-12:
                x_best = Xp.copy(); best_val = vp
            elif vp < best_val + 0.002 and rng.random() < 0.10:
                # sideways basin change accepted rarely
                pass
            # also allow Xt itself if LP failed to improve but GD is better
            if v < best_val - 1e-12 and v < vp:
                # short second squeeze already is vp; keep better of two
                x_best = Xt.copy(); best_val = v
        else:
            if v < best_val - 1e-12:
                x_best = Xt.copy(); best_val = v
    # final long polish
    remain = BUDGET - (time.time() - t0)
    if remain > 4.0:
        try:
            fs = float(np.clip(remain - 2.0, 5.0, 40.0))
            x_best = _lp_polish(x_best, seconds=fs * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = _lp_polish(x_best, seconds=min(rest, 18.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
