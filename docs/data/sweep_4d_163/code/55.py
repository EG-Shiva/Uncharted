import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _descend(x, steps, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(steps):
        f = k / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** f if T0 != T1 else T0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # soft-max weights
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # clip huge steps
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(1)
    t0 = time.time()
    HARD = 168.0
    LP_RESERVE = 55.0

    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (163, 4):
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                c = _norm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.normal(size=(n, d))))
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    incumbent = best.copy(); incumbent_val = best_val

    # quick descent polish of incumbent to have clean baseline (cheap)
    # hopping phase
    cur = best.copy(); cur_val = best_val
    # precompute scales
    scales = [0.015, 0.03, 0.06, 0.12, 0.20]
    trial = 0
    # time limit for hopping
    hop_end = HARD - LP_RESERVE
    ntrials = 0
    while time.time() - t0 < hop_end - 1.0 and ntrials < 200:
        ntrials += 1
        trial += 1
        scale = scales[(ntrials - 1) % len(scales)]
        do_void = (ntrials % 3 == 0)
        xw = cur.copy()
        if do_void:
            # find worst point(s)
            G = xw @ xw.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            widx = int(np.argmax(rowmax))
            # find void: sample candidates, minimize max dot
            S = 1200
            cand = rng.normal(size=(S, d))
            cand = _norm(cand)
            # max dot to all points (excluding widx slightly downweighted)
            dots = cand @ xw.T
            # ignore self (widx) so void can be near old pos? keep it
            # score = max dot
            scores = dots.max(axis=1)
            # enforce diversity: penalize closeness to random subset? just take best
            order = np.argsort(scores)
            # pick among 5 best at random for diversity
            pick = order[int(rng.integers(0, min(5, S)))]
            void = cand[pick]
            # jitter void a bit
            void = _norm(void + 0.01 * rng.normal(size=4))
            xw[widx] = void
            # also shake neighbours of widx a little
            # neighbours = top 6 closest to widx
            nbrs = np.argsort(-G[widx])[:6]
            for nb in nbrs:
                nb = int(nb)
                if nb == widx:
                    continue
                nz = rng.normal(size=4)
                nz = nz - np.dot(nz, xw[nb]) * xw[nb]
                nn = np.linalg.norm(nz)
                if nn > 1e-12:
                    nz /= nn
                    xw[nb] = _norm(xw[nb] + 0.02 * nz)
            eff_scale = scale * 0.5
            if eff_scale > 1e-12:
                nz = rng.normal(size=(n, d))
                proj = np.sum(nz * xw, axis=1, keepdims=True)
                nz = nz - proj * xw
                nn = np.linalg.norm(nz, axis=1, keepdims=True)
                nz = nz / np.maximum(nn, 1e-12)
                # shake all a little, worst more
                amp = np.full((n, 1), eff_scale * 0.3)
                amp[widx] = 0.0
                xw = _norm(xw + amp * nz)
        else:
            # global / focused shake
            G = xw @ xw.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowmax = G.max(axis=1)
            # focus weight: points near max get larger shake
            w = np.clip((rowmax - (m - 0.02)) / 0.02, 0.2, 1.0)[:, None]
            nz = rng.normal(size=(n, d))
            proj = np.sum(nz * xw, axis=1, keepdims=True)
            nz = nz - proj * xw
            nn = np.linalg.norm(nz, axis=1, keepdims=True)
            nz = nz / np.maximum(nn, 1e-12)
            xw = _norm(xw + scale * w * nz)

        # annealed descent: hot then cold
        try:
            xw = _descend(xw, steps=160, T0=0.02, T1=0.004, lr0=0.06, lr1=0.02)
            xw = _descend(xw, steps=120, T0=0.004, T1=0.0009, lr0=0.02, lr1=0.006)
        except Exception:
            continue
        v = _max_cos(xw)
        if v < cur_val:
            cur = xw; cur_val = v
            if v < best_val - 1e-12:
                best = xw.copy(); best_val = v
        elif v < cur_val + 0.004 and rng.random() < 0.25:
            # Metropolis side-step to new basin
            cur = xw; cur_val = v
        # periodically restart walk from best
        if ntrials % 18 == 0:
            cur = best.copy(); cur_val = best_val
        # early exit if excellent improvement
        if best_val < 0.8545:
            break

    # LP polish phase: polish best and current runner
    remain = HARD - (time.time() - t0) - 1.0
    if remain < 3.0:
        return _norm(best)
    # allocate most to best, some to cur if different
    runners = [best]
    if cur_val < best_val + 0.002 and _max_cos(cur) < incumbent_val:
        # check distinct
        if np.linalg.norm(cur - best) > 1e-6:
            runners.append(cur)
    polished_best = best
    polished_val = best_val
    # first polish best with majority time
    t1 = min(remain * (0.75 if len(runners) > 1 else 1.0), 40.0)
    try:
        pb = _lp_polish(best, seconds=max(t1, 3.0))
        pv = _max_cos(pb)
        if pv < polished_val:
            polished_best = pb; polished_val = pv
    except Exception:
        pass
    remain2 = HARD - (time.time() - t0) - 1.0
    if len(runners) > 1 and remain2 > 5.0:
        try:
            pc = _lp_polish(runners[1], seconds=min(remain2 * 0.8, 20.0))
            pv2 = _max_cos(pc)
            if pv2 < polished_val:
                polished_best = pc; polished_val = pv2
        except Exception:
            pass
    # final short polish on winner with leftover
    remain3 = HARD - (time.time() - t0) - 1.0
    if remain3 > 3.0 and polished_val >= best_val - 1e-12:
        # try one more quick descend+polish hybrid on winner with tiny shake
        try:
            alt = _norm(polished_best + 0.004 * (rng.normal(size=(n, d))))
            # project to tangent
            alt = _descend(alt, steps=80, T0=0.003, T1=0.0008, lr0=0.015, lr1=0.005)
            av = _max_cos(alt)
            if av < polished_val:
                polished_best = alt; polished_val = av
            # final LP touch
            rleft = HARD - (time.time() - t0) - 0.5
            if rleft > 2.0:
                pf = _lp_polish(polished_best, seconds=min(rleft, 12.0), r0=1e-3)
                pvf = _max_cos(pf)
                if pvf < polished_val:
                    polished_best = pf; polished_val = pvf
        except Exception:
            pass
    if polished_val < best_val:
        best, best_val = polished_best, polished_val
    else:
        best, best_val = polished_best, polished_val
    if best_val > incumbent_val:
        best = incumbent
    return _norm(best)
