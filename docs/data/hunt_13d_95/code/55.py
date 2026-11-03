import time
import numpy as np
from scipy.optimize import linprog, differential_evolution
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _fourier_basis():
    d = 13
    B = np.zeros((d, d))
    B[:, 0] = 1.0 / np.sqrt(d)
    t = np.arange(d)
    c = 1
    for f in range(1, 7):
        B[:, c] = np.sqrt(2.0 / d) * np.cos(2 * np.pi * f * t / d)
        c += 1
        B[:, c] = np.sqrt(2.0 / d) * np.sin(2 * np.pi * f * t / d)
        c += 1
    return B

_BASIS = _fourier_basis()

def _build_symmetric(amps, phases, signs, lens):
    # amps: (7,) [b0, r1..r6] unnormalized, r>=0
    # phases: (K,6), signs: (K,), lens: list length K sum 95
    b0_raw = amps[0]
    r = np.asarray(amps[1:], dtype=float)
    r = np.abs(r)
    nrm = np.sqrt(float(b0_raw)**2 + float(np.sum(r**2)))
    if nrm == 0:
        nrm = 1.0
    b0 = b0_raw / nrm
    rn = r / nrm
    K = len(lens)
    pts = []
    for k in range(K):
        c = np.zeros(13)
        c[0] = b0 * float(signs[k])
        for f in range(6):
            ph = phases[k, f]
            c[1 + 2 * f] = rn[f] * np.cos(ph)
            c[1 + 2 * f + 1] = rn[f] * np.sin(ph)
        base = _BASIS @ c
        # normalise (should be 1)
        nb = np.linalg.norm(base)
        if nb > 0:
            base = base / nb
        Lk = int(lens[k])
        for s in range(Lk):
            pts.append(np.roll(base, s))
    X = np.stack(pts, axis=0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _adam_softmax(X0, steps, b0=30.0, b1=500.0, lr0=0.02, lr1=0.005, seed=0):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1a, b2a = 0.9, 0.999
    eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        T = 1.0 / beta
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.finite(sw) or sw == 0:
            break
        W /= sw
        Gr = W @ X
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        mh = m / (1 - b1a ** (s + 1))
        vh = v / (1 - b2a ** (s + 1))
        X = X - lr * mh / (np.sqrt(vh) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.1:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    rng = np.random.default_rng(0)
    # load records
    cands_rec = []
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape == (95, 13):
            xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
            cands_rec.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (95, 13):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            cands_rec.append(xr)
    except Exception:
        pass
    if cands_rec:
        rec_vals = [_max_cos(c) for c in cands_rec]
        bi = int(np.argmin(rec_vals))
        x_rec_best = cands_rec[bi].copy()
        rec_best = float(rec_vals[bi])
    else:
        x_rec_best = rng.normal(size=(95, 13))
        x_rec_best /= np.linalg.norm(x_rec_best, axis=1, keepdims=True)
        rec_best = _max_cos(x_rec_best)

    # --- Phase 1: cyclic symmetric search, at most 60s ---
    T_PHASE1_END = t0 + 58.0
    K = 8
    lens = [13]*7 + [4]  # 91+4=95
    signs = np.array([1,1,1,1,-1,-1,-1,-1], dtype=float)
    # random search for amps+phases
    sym_cands = []  # (val, amps, phases)
    # seed phases set count
    t_rand_end = min(t0 + 28.0, T_PHASE1_END - 20.0)
    # quick random sampling
    it = 0
    while time.time() < t_rand_end and time.time() < DEADLINE - 100:
        it += 1
        b0r = float(rng.normal() * 0.5)
        rr = np.abs(rng.normal(size=6) * 1.0)
        amps = np.concatenate([[b0r], rr])
        phases = rng.uniform(0, 2*np.pi, size=(K, 6))
        try:
            X = _build_symmetric(amps, phases, signs, lens)
            vv = _max_cos(X)
        except Exception:
            continue
        sym_cands.append((vv, amps.copy(), phases.copy()))
        if len(sym_cands) > 400:
            # keep best 50
            sym_cands.sort(key=lambda z: z[0])
            sym_cands = sym_cands[:50]
    sym_cands.sort(key=lambda z: z[0])
    top = sym_cands[:3] if len(sym_cands) >= 3 else sym_cands
    # DE on amps for top phase sets (low-dim 7D)
    bounds = [(-1, 1)] + [(0, 1)]*6
    for ti, (vv0, a0, ph0) in enumerate(list(top)):
        if time.time() > T_PHASE1_END - 5.0:
            break
        if time.time() > DEADLINE - 110:
            break
        def obj(y, _ph=ph0):
            try:
                XX = _build_symmetric(np.asarray(y), _ph, signs, lens)
                return _max_cos(XX)
            except Exception:
                return 1.0
        try:
            res = differential_evolution(obj, bounds, maxiter=18, popsize=10,
                                         seed=ti, polish=False, updating='immediate',
                                         workers=1, disp=False)
            best_amps = np.asarray(res.x)
            Xb = _build_symmetric(best_amps, ph0, signs, lens)
            vb = _max_cos(Xb)
            sym_cands.append((vb, best_amps, ph0.copy()))
        except Exception:
            continue
    sym_cands.sort(key=lambda z: z[0])
    if len(sym_cands) == 0:
        # fallback random symmetric
        amps = np.array([0.2, 0.5, 0.5, 0.4, 0.3, 0.3, 0.2])
        phases = rng.uniform(0, 2*np.pi, size=(K, 6))
        Xf = _build_symmetric(amps, phases, signs, lens)
        sym_cands = [(_max_cos(Xf), amps, phases)]
    best_sym = sym_cands[:3]

    # --- Phase 2: symmetry breaking + annealed softmax Adam ---
    sigmas = [0.01, 0.03, 0.08]
    refined = []
    # budget until LP start: reserve 27s for LP + 3s margin
    LP_SEC = 25.0
    for (sv, samps, sph) in best_sym:
        try:
            Xs = _build_symmetric(samps, sph, signs, lens)
        except Exception:
            continue
        for sg in sigmas:
            if time.time() > DEADLINE - LP_SEC - 3.0:
                break
            # tangent Gaussian noise
            nz = rng.normal(size=Xs.shape)
            dot = np.sum(nz * Xs, axis=1, keepdims=True)
            nz_t = nz - dot * Xs
            Xt = Xs + float(sg) * nz_t
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # annealed softmax Adam, no LP
            # steps scaled by remaining time
            remain = DEADLINE - LP_SEC - 3.0 - time.time()
            # ~9 trials total; give ~1500-2200 steps
            steps = 2000
            if remain < 30:
                steps = 1200
            if remain < 12:
                steps = 600
            Xr = _adam_softmax(Xt, steps=steps, b0=30.0, b1=500.0, lr0=0.02, lr1=0.005)
            vr = _max_cos(Xr)
            refined.append((vr, Xr))
        if time.time() > DEADLINE - LP_SEC - 3.0:
            break
    if refined:
        refined.sort(key=lambda z: z[0])
        x_best_ref, v_best_ref = refined[0][1], refined[0][0]
    else:
        x_best_ref, v_best_ref = _build_symmetric(best_sym[0][1], best_sym[0][2], signs, lens), best_sym[0][0]

    # --- Phase 3: LP polish only best refined for final 25s ---
    remain = DEADLINE - time.time()
    lp_time = min(LP_SEC, max(2.0, remain - 2.0))
    if lp_time > 1.0:
        try:
            x_pol = _lp_polish(x_best_ref, seconds=lp_time, deadline=DEADLINE - 1.0)
            v_pol = _max_cos(x_pol)
        except Exception:
            x_pol, v_pol = x_best_ref, v_best_ref
    else:
        x_pol, v_pol = x_best_ref, v_best_ref

    # return better of polished and loaded record
    if v_pol < rec_best:
        out = x_pol
    else:
        out = x_rec_best
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out
