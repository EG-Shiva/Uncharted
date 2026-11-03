import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds, p_cap=1200):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            # keep tightest
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            remain = seconds - (time.time() - start)
            if remain <= 0.05:
                break
            opts = {"time_limit": max(0.2, min(remain, 5.0))}
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options=opts)
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = bx.copy()
    return bx

def _quick_smooth(v0, steps=180, beta=250.0, lr=0.004):
    v = v0.copy()
    n, d = v.shape
    for _ in range(steps):
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt = (grad - dot * v) / row
        v = v - lr * gt
        v = _normalize(v)
    return v

def _null_modes(x, I, J, k, rng):
    n, d = x.shape
    p = len(I)
    N = n * d
    q = p + n
    # dense S = [R; E]
    try:
        S = np.zeros((q, N), dtype=np.float64)
        # R rows
        for t in range(p):
            i = int(I[t]); j = int(J[t])
            S[t, i*d:(i+1)*d] = x[j]
            S[t, j*d:(j+1)*d] = x[i]
        for i in range(n):
            S[p+i, i*d:(i+1)*d] = x[i]
        Rmat = rng.randn(N, k)
        B = S @ Rmat  # q x k
        A = S @ S.T   # q x q
        A += 1e-12 * np.eye(q)
        try:
            Y = np.linalg.solve(A, B)
        except Exception:
            Y, *_ = np.linalg.lstsq(A, B, rcond=None)
        V = Rmat - S.T @ Y
        # normalize + re-tangentize
        modes = []
        for c in range(k):
            vc = V[:, c].reshape(n, d)
            # explicit tangent projection
            dot = np.sum(vc * x, axis=1, keepdims=True)
            vc = vc - dot * x
            nv = float(np.linalg.norm(vc))
            if nv < 1e-9 or not np.isfinite(nv):
                rv = rng.randn(n, d)
                dot2 = np.sum(rv * x, axis=1, keepdims=True)
                rv = rv - dot2 * x
                nv2 = float(np.linalg.norm(rv))
                if nv2 < 1e-12:
                    continue
                vc = rv / nv2
            else:
                vc = vc / nv
            modes.append(vc)
        if len(modes) == 0:
            return None
        # orthonormalize globally (stacked)
        M = np.stack([m.ravel() for m in modes], axis=1)
        Q, _ = np.linalg.qr(M)
        out = []
        for c in range(Q.shape[1]):
            vc = Q[:, c].reshape(n, d)
            dot = np.sum(vc * x, axis=1, keepdims=True)
            vc = vc - dot * x
            nv = float(np.linalg.norm(vc))
            if nv > 1e-12:
                vc = vc / nv
                out.append(vc)
        return out if out else None
    except Exception:
        return None

def solve():
    t0 = time.time()
    HARD = 168.0
    HOP_END = 138.0
    n, d = 123, 13
    rng = np.random.RandomState(7)
    np.random.seed(7)
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _normalize(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _normalize(rng.randn(n, d))
        best_val = _max_cos(xb)
    x_best = xb.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    srad_cycle = [0.04, 0.08, 0.12, 0.05, 0.10, 0.15, 0.03, 0.06]
    n_cycles = 0
    k_modes = 5
    while time.time() - t0 < HOP_END and n_cycles < 60:
        n_cycles += 1
        if time.time() - t0 > HOP_END - 2:
            break
        # contact graph within 1e-7
        try:
            G = x_best @ x_best.T
            c_all = G[iu0, iu1]
            m = float(c_all.max())
            sel = c_all > m - 1e-7
            cnt = int(np.count_nonzero(sel))
            if cnt < 10:
                sel = c_all > m - 1e-6
                cnt = int(np.count_nonzero(sel))
            I = iu0[sel]; J = iu1[sel]; cs = c_all[sel]
            # cap to tightest 1300 for rigidity
            if cnt > 1300:
                idx = np.argsort(-cs)[:1300]
                I = I[idx]; J = J[idx]; cs = cs[idx]
                cnt = 1300
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, I, 1); np.add.at(deg, J, 1)
            hubs = np.argsort(-deg)[:6]
        except Exception:
            hubs = np.argsort(-(x_best @ x_best.T).max(axis=1))[:6]
            I = None
        modes = None
        if I is not None and len(I) >= 10:
            try:
                modes = _null_modes(x_best, I, J, k_modes, rng)
            except Exception:
                modes = None
        if modes is None:
            # fallback: random tangent modes localized on hubs
            modes = []
            for kk in range(k_modes):
                rv = np.zeros((n, d))
                hh = hubs[kk % len(hubs)]
                rv += 0.3 * rng.randn(n, d)
                rv[hh] *= 3.0
                dot = np.sum(rv * x_best, axis=1, keepdims=True)
                rv = rv - dot * x_best
                nv = float(np.linalg.norm(rv))
                if nv < 1e-12:
                    continue
                modes.append(rv / nv)
        # build candidates: one per mode with cycling step size
        cands = []
        # precompute strains for scoring
        # need R for strain: use I,J
        for mi, vm in enumerate(modes):
            if time.time() - t0 > HOP_END:
                break
            # rigidity stress proxy: strain vector
            if I is not None:
                try:
                    # e_k = x_j.vm_i + x_i.vm_j
                    e = np.sum(x_best[J] * vm[I], axis=1) + np.sum(x_best[I] * vm[J], axis=1)
                    rms = float(np.sqrt(np.mean(e*e)) + 1e-12)
                except Exception:
                    e = None; rms = 1.0
            else:
                e = None; rms = 1.0
            # choose hub to break: cycle
            h = int(hubs[(n_cycles + mi) % len(hubs)])
            sgn = 1.0
            if e is not None:
                try:
                    mask = (I == h) | (J == h)
                    eh = e[mask]
                    if eh.size >= 3:
                        srt = np.sort(eh)
                        s_low = float(srt[:3].sum())
                        s_high = float(srt[-3:].sum())
                        # pick sign giving larger reduction
                        if abs(s_low) >= abs(s_high):
                            sgn = 1.0
                        else:
                            sgn = -1.0
                    elif eh.size > 0:
                        sgn = -float(np.sign(eh[np.argmax(np.abs(eh))]))
                        if sgn == 0:
                            sgn = 1.0
                except Exception:
                    sgn = 1.0
            else:
                sgn = 1.0 if rng.rand() > 0.5 else -1.0
            vm_s = vm * sgn
            rn = np.linalg.norm(vm_s, axis=1)
            mx = float(rn.max() + 1e-12)
            srad = srad_cycle[(n_cycles * k_modes + mi) % len(srad_cycle)]
            alpha = srad / mx
            # cap alpha to avoid huge jumps
            if alpha > 2.0:
                alpha = 2.0
            xc = _normalize(x_best + alpha * vm_s)
            # tiny jitter to avoid exact symmetry trap
            # deterministic small noise
            xc = _normalize(xc + 0.002 * rng.randn(n, d) * 0.0)  # keep deterministic, no extra
            v0 = _max_cos(xc)
            # score: prefer lower v0 but also modes preserving most (low rms) with big hub break
            cands.append((v0, rms, mi, xc, vm_s, alpha, h))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        # quick smooth + capped LP on top-2
        improved = False
        for ci in range(min(2, len(cands))):
            if time.time() - t0 > HOP_END:
                break
            v0, rms, mi, xc, vm_s, alpha, h = cands[ci]
            # skip hopeless far jumps to save time
            if v0 > best_val + 0.025 and ci == 1:
                continue
            # cheap repair
            try:
                xs = _quick_smooth(xc, steps=120, beta=200.0, lr=0.004)
                vs = _max_cos(xs)
                if vs < v0:
                    xc = xs; v0 = vs
            except Exception:
                pass
            remain = HOP_END - (time.time() - t0)
            if remain < 2.0:
                # no time for LP, just check raw
                if v0 < best_val:
                    best_val = v0; x_best = xc.copy(); improved = True
                continue
            lp_sec = 1.6 if v0 > best_val else 2.5
            lp_sec = min(lp_sec, remain - 0.5)
            if lp_sec < 0.6:
                if v0 < best_val:
                    best_val = v0; x_best = xc.copy(); improved = True
                continue
            try:
                xp = _lp_polish(xc, lp_sec)
                vp = _max_cos(xp)
            except Exception:
                continue
            if vp < best_val:
                best_val = vp
                x_best = xp.copy()
                improved = True
                break  # elitist: restart cycle from new basin
            # opportunistic keep raw if it beats best (shouldn't happen if LP failed)
            if v0 < best_val:
                best_val = v0
                x_best = xc.copy()
                improved = True
                break
        # if no improvement, occasionally try rattler reinsertion cheaply (no LP)
        # keep time bounded: don't add heavy ops
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0)
    if remain > 4:
        try:
            x_best = _lp_polish(x_best, min(remain - 1.0, 22.0))
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best
