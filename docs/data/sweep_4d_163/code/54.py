import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 400:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def _void_vec(x, rng, n_samp=1500):
    d = x.shape[1]
    cand = rng.normal(size=(n_samp, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = cand @ x.T
    mx = sc.max(axis=1)
    i = int(np.argmin(mx))
    v = cand[i].copy()
    # few projected repulsion steps to centre the void
    for _ in range(25):
        dots = x @ v
        w = np.exp(40.0 * (dots - dots.max()))
        g = w @ x
        gt = g - np.dot(g, v) * v
        nrm = np.linalg.norm(gt)
        if nrm < 1e-12:
            break
        gt /= nrm
        v = v - 0.06 * gt
        v /= np.linalg.norm(v)
    return v

def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(7)
    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            cc = np.asarray(np.load(fn), dtype=float)
            if cc.shape == (n, d):
                cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
                cc /= np.linalg.norm(cc, axis=1, keepdims=True)
                cands.append(cc)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for cc in cands:
        v = _max_cos(cc)
        if v < best_val:
            best_val = v
            best = cc.copy()
    cur = best.copy()
    cur_val = best_val
    # quick contact degrees of incumbent for hub targeting
    it = 0
    # hopping until reserve for final polish
    while time.time() - t0 < HARD - 30.0:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                # global moderate shake
                sig = 0.10 + 0.05 * float(rng.random())
                y = cur + rng.normal(scale=sig, size=cur.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=220, T0=0.014, T1=0.0012, lr0=0.012, lr1=0.0012)
            elif mode == 1:
                # worst patch heavy shake (topology change)
                rm, G = _row_max(cur)
                w = int(np.argmax(rm))
                order = np.argsort(-G[w])
                k = 5 + int(rng.integers(0, 4))  # 5-8 neighbours
                idx = order[:k + 1]
                y = cur.copy()
                y[idx] += rng.normal(scale=0.38, size=(len(idx), d))
                # tiny global jitter to unpin
                y += rng.normal(scale=0.005, size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=240, T0=0.016, T1=0.0010, lr0=0.014, lr1=0.0010)
            elif mode == 2:
                # hub resample to void: replace 1-2 worst points by voids
                rm, G = _row_max(cur)
                # worst 8 indices
                worst = np.argsort(-rm)[:8]
                y = cur.copy()
                nrep = 1 if (it % 2 == 0) else 2
                reps = rng.choice(worst, size=nrep, replace=False)
                for r in reps:
                    y[int(r)] = _void_vec(y, rng, n_samp=1200)
                y += rng.normal(scale=0.01, size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=240, T0=0.016, T1=0.0010, lr0=0.014, lr1=0.0010)
            elif mode == 3:
                # large global kick
                sig = 0.28 + 0.15 * float(rng.random())
                y = cur + rng.normal(scale=sig, size=cur.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=260, T0=0.018, T1=0.0012, lr0=0.016, lr1=0.0012)
            elif mode == 4:
                # fresh random with fast anneal (new basin)
                y = rng.normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=300, T0=0.022, T1=0.0015, lr0=0.020, lr1=0.0020)
            else:
                # degree-targeted: shake high-contact cluster
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                thr = m - 2e-3
                deg = (G > thr).sum(axis=1)
                hubs = np.argsort(-deg)[:6]
                y = cur.copy()
                y[hubs] += rng.normal(scale=0.22, size=(len(hubs), d))
                y += rng.normal(scale=0.008, size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, iters=220, T0=0.013, T1=0.0010, lr0=0.012, lr1=0.0010)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = y.copy()
            best_val = v
            cur = y.copy()
            cur_val = v
        elif v < cur_val:
            cur = y.copy()
            cur_val = v
        elif rng.random() < 0.04:
            # random walk to new basin
            cur = y.copy()
            cur_val = v
        # occasionally restart walk from best with shake to intensify
        if it % 25 == 0 and rng.random() < 0.5:
            try:
                yb = best + rng.normal(scale=0.06, size=best.shape)
                yb /= np.linalg.norm(yb, axis=1, keepdims=True)
                # don't smooth here, just set as cur if promising region
                cur = yb
                cur_val = _max_cos(cur)
            except Exception:
                pass
        if time.time() - t0 > HARD - 31.0:
            break
    # final polish: only best (keep fallback)
    try:
        left = HARD - (time.time() - t0) - 1.5
        if left > 3.0:
            best = _lp_polish(best, seconds=min(left, 26.0), r0=1.5e-3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best = best / nr
    return best
