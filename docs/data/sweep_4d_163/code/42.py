import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

RECORD = 0.856415430312

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contact_deg(x, thresh=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    B = G >= m - thresh
    deg = B.sum(axis=1)
    return deg, m

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

_IU_CACHE = {}
def _lp_polish(x, seconds, deadline=None, r0=2e-3, cap=2200, maxit=30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    key = n
    if key not in _IU_CACHE:
        _IU_CACHE[key] = np.triu_indices(n, 1)
    iu = _IU_CACHE[key]
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < maxit:
        if deadline is not None and time.time() >= deadline:
            break
        it += 1
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
            if cnt2 > cap + 800:
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
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_void(v0, Y, steps=120, lr0=0.09, beta=80.0):
    v = v0.copy()
    v /= np.linalg.norm(v)
    lr = lr0
    for _ in range(steps):
        dots = Y @ v
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        g = w @ Y
        d = float(g @ v)
        gt = g - d * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        gt /= nrm
        v = v - lr * gt
        n = float(np.linalg.norm(v))
        if n == 0:
            break
        v /= n
        lr *= 0.995
    return v / np.linalg.norm(v)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (163, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(163, 4))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n = 163
    # baseline degrees
    try:
        deg, _ = _contact_deg(best, thresh=1e-9)
    except Exception:
        deg = np.zeros(n)

    trial = 0
    # main delete-optimize-reinsert cycles
    while time.time() < DEADLINE - 12:
        trial += 1
        if time.time() >= DEADLINE - 12:
            break
        # recompute degrees periodically
        if trial == 1 or trial % 5 == 0:
            try:
                deg, _ = _contact_deg(best, thresh=1e-9)
            except Exception:
                pass
        # number of extra points beyond rattlers: 2..8 cycle
        k_extra = 2 + ((trial - 1) % 7)
        ratt = np.where(deg <= 0.5)[0]
        # keep exactly two rattlers if more, else all
        if len(ratt) >= 2:
            # pick two with smallest row-max (deepest voids) deterministically + rng jitter
            drop_r = list(ratt[:2])
        elif len(ratt) == 1:
            drop_r = list(ratt)
        else:
            drop_r = []
        need = k_extra
        order = np.argsort(-deg, kind="stable")
        # vary selection: every 3rd trial randomize among top 25
        candidates_top = [i for i in order if i not in drop_r]
        if trial % 3 == 0:
            pool = candidates_top[:25]
            # deterministic-ish random choice with rng
            perm = rng.permutation(len(pool))
            chosen = [pool[j] for j in perm[:need]]
        else:
            chosen = candidates_top[:need]
        drop = np.array(sorted(set(drop_r) | set(chosen)), dtype=int)
        k = len(drop)
        if k < 3 or k > 12:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop] = False
        Y = best[keep_mask].copy()
        m = Y.shape[0]
        # aggressive optimization of remainder: annealed log-sum-exp repulsion
        # vary annealing slightly per trial
        if trial % 2 == 0:
            Y = _smooth(Y, iters=350, T0=0.02, T1=0.0015, lr0=0.035, lr1=0.003)
        else:
            Y = _smooth(Y, iters=450, T0=0.03, T1=0.001, lr0=0.05, lr1=0.002)
        # short LP polish on remainder
        rem_budget = min(0.8, max(0.2, (DEADLINE - 12 - time.time()) / 4))
        if rem_budget > 0.15:
            Y = _lp_polish(Y, seconds=rem_budget, deadline=DEADLINE - 12, r0=2e-3)
        sub_max = _max_cos(Y)
        # explicit check vs published record: log but continue unless hopeless
        # if remainder already far above best, skip reinsertion (cannot win without moving Y later)
        if sub_max >= best_val + 0.004:
            continue
        # ---- reinsertion: sample >=50000 candidates ----
        if time.time() >= DEADLINE - 11:
            break
        n_samp = 50000
        # chunk generation to bound memory/time
        cand = rng.standard_normal(size=(n_samp, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        curY = Y.copy()
        # greedy insertion into largest voids
        for ri in range(k):
            if time.time() >= DEADLINE - 10:
                break
            # score pool against curY
            # chunked matmul to avoid huge temp
            chunk = 10000
            scores = np.empty(n_samp)
            for s0 in range(0, n_samp, chunk):
                s1 = min(n_samp, s0 + chunk)
                scores[s0:s1] = (cand[s0:s1] @ curY.T).max(axis=1)
            # promising candidates: smallest scores
            n_prom = 30
            part = np.argpartition(scores, n_prom)[:n_prom]
            # pick the absolute best first, refine it; also try a few and keep best refined
            best_v = None
            best_s = 1e9
            # try top 6 promising to limit cost
            order_p = part[np.argsort(scores[part])][:6]
            for pi in order_p:
                v0 = cand[pi]
                v = _refine_void(v0, curY, steps=130, lr0=0.09, beta=80.0)
                s = float(np.max(curY @ v))
                if s < best_s:
                    best_s = s
                    best_v = v
            if best_v is None:
                # fallback
                bi = int(np.argmin(scores))
                best_v = _refine_void(cand[bi], curY)
            curY = np.vstack([curY, best_v[None, :]])
            # remove the used region from pool by rescoring next iteration (implicit)
        if curY.shape[0] != n:
            continue
        # assemble full trial: curY already contains kept+new; polish full
        # small smooth to settle inserts
        curY = _smooth(curY, iters=80, T0=0.008, T1=0.001, lr0=0.015, lr1=0.002)
        full_budget = min(1.4, max(0.3, (DEADLINE - 12 - time.time()) / 3))
        if full_budget > 0.2:
            curY = _lp_polish(curY, seconds=full_budget, deadline=DEADLINE - 12, r0=2e-3)
        v = _max_cos(curY)
        if v < best_val:
            best = curY / np.linalg.norm(curY, axis=1, keepdims=True)
            best_val = v
            try:
                deg, _ = _contact_deg(best, thresh=1e-9)
            except Exception:
                pass
    # final LP polish on champion
    fbudget = max(0.0, DEADLINE - time.time() - 1.0)
    fbudget = min(fbudget, 11.0)
    if fbudget > 0.3:
        best = _lp_polish(best, seconds=fbudget, r0=1.5e-3)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
