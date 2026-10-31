import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm < 1e-12] = 1.0
    x = x / nrm
    x[~np.isfinite(x)] = 0.0
    nrm2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (nrm2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _stats(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    pw = g.max(axis=1)
    thr = m - 1e-9
    cnt = (g >= thr).sum(axis=1)
    return m, pw, cnt, g


def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _adam_moving(x, mov, steps, tau0, tau1, lr0, lr1):
    # moving: 1D index array
    n, d = x.shape
    k = len(mov)
    m = np.zeros((k, d))
    v = np.zeros((k, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        # weights
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        W[W < 1e-12] = 0.0
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        g = gt[mov]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x[mov] = x[mov] - lr * step
        x[mov] = x[mov] / np.linalg.norm(x[mov], axis=1, keepdims=True)
    return x


def _adam_full(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        W[W < 1e-12] = 0.0
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x


def _void_for(others, rng, n_cand=4000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    sc = s.max(axis=1)
    bi = int(np.argmin(sc))
    return cand[bi], float(sc[bi])


def _patch_refill(x_base, patch, rng):
    # sequentially refill patch points greedily into voids
    n = x_base.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[patch] = False
    fixed = x_base[mask]
    placed = []
    y = x_base.copy()
    order = list(patch)
    rng.shuffle(order)
    for idx in order:
        others = np.vstack([fixed] + placed) if placed else fixed
        b, _ = _void_for(others, rng, n_cand=4000)
        # small local jitter hill-climb
        best_b = b
        others_t = others
        bs = float((best_b[None, :] @ others_t.T).max())
        step = 0.08
        for _ in range(120):
            prop = best_b + step * rng.standard_normal((32, 4))
            prop /= np.linalg.norm(prop, axis=1, keepdims=True)
            sc = (prop @ others_t.T).max(axis=1)
            j = int(np.argmin(sc))
            if sc[j] < bs:
                best_b = prop[j]
                bs = float(sc[j])
                step *= 1.05
            else:
                step *= 0.96
            if step < 1e-5:
                break
        y[idx] = best_b
        placed.append(best_b[None, :])
    return y


def _hopf_start(n_base_list, m_fib_list, rng):
    # build one Hopf-bundle start; caller picks factorisation
    # S2 base via fibonacci + fiber uniform
    outs = []
    for nb, mf in zip(n_base_list, m_fib_list):
        ga = np.pi * (3.0 - np.sqrt(5.0))
        base = []
        for i in range(nb):
            zz = 1.0 - 2.0 * (i + 0.5) / nb
            rr = np.sqrt(max(0.0, 1.0 - zz * zz))
            ph = ga * i
            base.append([rr * np.cos(ph), rr * np.sin(ph), zz])
        base = np.array(base)
        for bi in range(nb):
            bx, by, bz = base[bi]
            th = np.arccos(np.clip(bz, -1.0, 1.0))
            phi = np.arctan2(by, bx)
            shift = rng.uniform(0, 2 * np.pi)
            for k in range(mf):
                psi = 2 * np.pi * k / mf + shift
                # Hopf coords in C^2
                a = np.cos(th / 2.0) * np.exp(1j * (psi + phi) / 2.0)
                b = np.sin(th / 2.0) * np.exp(1j * (psi - phi) / 2.0)
                outs.append([a.real, a.imag, b.real, b.imag])
    outs = np.array(outs)
    return outs


def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _norm(tmp)
    x_best = x0.copy()
    best_val = _max_cos(x_best)

    patch_sizes = [48, 32, 24, 40, 16, 28]
    it = 0
    # reserve final polish ~30s
    while time.time() < t0 + 128.0 and time.time() < deadline - 38.0:
        it += 1
        mcur, pw, cnt, G = _stats(x_best)
        ratt = np.where(cnt == 0)[0]
        # choose centre
        if it % 5 == 0 and len(ratt) > 0:
            centre = int(ratt[rng.integers(0, len(ratt))])
        elif it % 3 == 1:
            centre = int(np.argmax(pw))
        elif it % 3 == 2:
            # overloaded hub
            centre = int(np.argmax(cnt + 0.1 * rng.standard_normal(272)))
        else:
            centre = int(rng.integers(0, 272))
        k = patch_sizes[it % len(patch_sizes)]
        # nearest neighbours by dot
        sims = x_best @ x_best[centre]
        order = np.argsort(-sims)
        patch = list(order[:k])
        # periodically force rattlers into patch
        if it % 4 == 0:
            for r in ratt:
                if int(r) not in patch:
                    patch[-1] = int(r)
                    break
        patch = np.array(sorted(set(int(p) for p in patch)), dtype=int)

        y = _patch_refill(x_best, patch, rng)
        # joint anneal of patch at two temperatures
        tau_pair = [(0.02, 0.002), (0.008, 0.0008), (0.035, 0.003)][it % 3]
        lr_pair = [(0.01, 0.001), (0.006, 0.0006), (0.014, 0.0012)][it % 3]
        y = _adam_moving(y, patch, steps=260,
                         tau0=tau_pair[0], tau1=tau_pair[1],
                         lr0=lr_pair[0], lr1=lr_pair[1])
        # short full relax to settle seams
        y = _adam_full(y, steps=120, tau0=0.006, tau1=0.0012, lr0=0.004, lr1=0.0006)
        val = _max_cos(y)
        # promising? snap with short LP
        thresh = best_val + (0.004 if it % 2 == 0 else 0.0015)
        if val < thresh and time.time() < deadline - 40.0:
            # quick LP 2.5-4s
            ys = _lp_polish(y, seconds=3.0, deadline=deadline - 32.0)
            vs = _max_cos(ys)
            if vs < best_val - 1e-12:
                x_best = ys
                best_val = vs
                continue
            # keep sideways basin if close and diverse? occasionally accept
            if vs < best_val + 1e-4 and it % 7 == 0:
                # accept as new base to walk along plateau
                x_best = ys
                best_val = vs
            continue
        # cheap acceptance if outright better (rare without LP)
        if val < best_val - 1e-9:
            ys = _lp_polish(y, seconds=3.0, deadline=deadline - 32.0)
            vs = _max_cos(ys)
            if vs < best_val - 1e-12:
                x_best = ys
                best_val = vs
                continue
        # occasional Hopf-bundle diversification injection
        if it % 11 == 0 and time.time() < t0 + 90:
            try:
                fac = [(68, 4), (34, 8), (17, 16)][(it // 11) % 3]
                nb, mf = fac
                h = _hopf_start([nb], [mf], rng)
                if h.shape[0] >= 272:
                    h = h[:272]
                else:
                    extra = rng.standard_normal((272 - h.shape[0], 4))
                    h = np.vstack([h, extra])
                h = _norm(h + 0.02 * rng.standard_normal(h.shape))
                h = _adam_full(h, steps=400, tau0=0.02, tau1=0.0015,
                               lr0=0.01, lr1=0.0008)
                hv = _max_cos(h)
                if hv < best_val + 0.01:
                    hs = _lp_polish(h, seconds=3.0, deadline=deadline - 32.0)
                    hvs = _max_cos(hs)
                    if hvs < best_val - 1e-12:
                        x_best = hs
                        best_val = hvs
            except Exception:
                pass

    # hemisphere twist escape attempts (coordinated global move)
    while time.time() < t0 + 138.0 and time.time() < deadline - 34.0:
        it += 1
        u = rng.standard_normal(4)
        u /= np.linalg.norm(u)
        proj = x_best @ u
        # rotate hemisphere points by small random SO(4) rotation
        # build random skew matrix
        A = rng.standard_normal((4, 4))
        A = A - A.T
        ang = float(rng.uniform(0.02, 0.12))
        # exp approx via I + sin*Ahat + (1-cos)*Ahat^2 ; normalize Ahat
        nA = float(np.linalg.norm(A))
        if nA < 1e-9:
            continue
        Ah = A / nA
        R = np.eye(4) + np.sin(ang) * Ah + (1 - np.cos(ang)) * (Ah @ Ah)
        # orthonormalize
        Q, _ = np.linalg.qr(R)
        y = x_best.copy()
        hemi = proj > 0.1
        if hemi.sum() < 10 or hemi.sum() > 262:
            continue
        y[hemi] = (y[hemi] @ Q.T)
        y = _norm(y)
        y = _adam_full(y, steps=250, tau0=0.012, tau1=0.001, lr0=0.008, lr1=0.0007)
        val = _max_cos(y)
        if val < best_val + 0.003:
            ys = _lp_polish(y, seconds=2.5, deadline=deadline - 32.0)
            vs = _max_cos(ys)
            if vs < best_val - 1e-12:
                x_best = ys
                best_val = vs
        if time.time() > t0 + 138.0:
            break

    # final long polish of best
    remain = deadline - time.time() - 1.0
    if remain > 3.0:
        x_best = _lp_polish(x_best, seconds=min(remain, 28.0), deadline=deadline - 0.5)
    x_best = _norm(x_best)
    return x_best
