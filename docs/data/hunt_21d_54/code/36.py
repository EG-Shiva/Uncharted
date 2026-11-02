import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_info(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - tol
    cnt = int(sel.sum())
    return m, cnt, G, iu, c, sel

def _adam_screen(y, steps=200, lr=0.01, T=0.006, mu=0.85):
    n, d = y.shape
    y = y.copy()
    vel = np.zeros_like(y)
    for s in range(steps):
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        # normalize global scale to keep step stable
        gn = np.linalg.norm(grad) + 1e-12
        grad = grad / gn * np.sqrt(n)
        vel = mu * vel + (1 - mu) * grad
        # bias correction-ish
        step = lr * (1.0 - 0.5 * s / steps)
        y = y - step * vel
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _lp_polish(x, seconds, cap=2500):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            idx = np.argpartition(-c[sel], cap)[:cap]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = cap
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x, rng):
    n, d = x.shape
    D = d - 1
    B = np.empty((n, d, D))
    for i in range(n):
        R = rng.normal(size=(d, D))
        # orthogonalize against x[i]
        R -= x[i][:, None] * (x[i] @ R)[None, :]
        Q, _ = np.linalg.qr(R)
        B[i] = Q
    return B

def _soft_modes(x, B, pairs_I, pairs_J, nmodes, rng):
    n, d = x.shape
    D = d - 1
    N = n * D
    p = len(pairs_I)
    if p == 0:
        Y = rng.normal(size=(nmodes, N))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
        return Y
    # Assemble Jacobian in tangent coords: A (p x N)
    A = np.zeros((p, N))
    # precompute projections: for each contact, need B_i^T x_j
    for k in range(p):
        i, j = int(pairs_I[k]), int(pairs_J[k])
        A[k, i * D:(i + 1) * D] = B[i].T @ x[j]
        A[k, j * D:(j + 1) * D] = B[j].T @ x[i]
    # economy SVD to characterize (truncated / sparse spirit); also used for stable projection
    try:
        _, S, _ = np.linalg.svd(A, full_matrices=False)
    except Exception:
        S = np.ones(1)
    _ = S  # diagnostics only; projection below gives nullspace (lowest singular) modes
    G = A @ A.T
    G += np.eye(p) * 1e-9
    # generate random tangent vectors and project onto nullspace: y0 = y - A^T (AA^T)^{-1} A y
    Y = rng.normal(size=(nmodes, N))
    # solve for all modes at once
    AY = A @ Y.T  # p x nmodes
    try:
        W = np.linalg.solve(G, AY)  # p x nmodes
    except Exception:
        W, *_ = np.linalg.lstsq(G, AY, rcond=None)
    Y0 = Y - (W.T @ A)
    nrm = np.linalg.norm(Y0, axis=1, keepdims=True) + 1e-12
    # if projection collapsed (row space filled), keep original
    collaps = (nrm.ravel() < 1e-8)
    if np.any(collaps):
        Y0[collaps] = Y[collaps]
        nrm[collaps] = np.linalg.norm(Y[collaps], axis=1, keepdims=True) + 1e-12
    Y0 /= nrm
    return Y0

def _gram_dist(a, b):
    Ga = a @ a.T
    Gb = b @ b.T
    return float(np.linalg.norm(Ga - Gb))

def solve():
    T0 = time.time()
    DEADLINE = 155.0
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _proj(np.nan_to_num(np.array(x0, dtype=float)))
    if x0.shape != (54, 21):
        x0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.array(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(x0):
                x0 = r0.copy()
    except Exception:
        pass
    n, d = 54, 21
    best = x0.copy()
    best_val = _max_cos(best)
    elite = [(best_val, best.copy())]
    elite_grams = [best @ best.T]

    def time_left():
        return DEADLINE - (time.time() - T0)

    # ensure starting point is locally polished a bit (cheap)
    if time_left() > 30:
        best = _lp_polish(best, min(4.0, time_left() - 25))
        best_val = _max_cos(best)
        elite = [(best_val, best.copy())]

    amps = [0.01, 0.03, 0.07, 0.13, 0.2]
    it = 0
    # main loop: one soft-mode move + short LP polish
    while time_left() > 22 and it < 120:
        it += 1
        # pick base: rotate through elite + best, to avoid same basin
        if len(elite) > 1 and (it % 3 == 0):
            bi = int(rng.integers(0, len(elite)))
            base_val, base = elite[bi][0], elite[bi][1].copy()
        else:
            base_val, base = best_val, best.copy()
        m0, cnt0, G0, iu0, c0, sel0 = _contact_info(base, tol=1e-6)
        # contact pairs within 1e-6
        iu = np.triu_indices(n, 1)
        I0, J0 = iu[0][sel0], iu[1][sel0]
        # cap pairs for Jacobian (tightest 800)
        if len(I0) > 800:
            order = np.argsort(-c0[sel0])[:800]
            I0, J0 = I0[order], J0[order]
        if len(I0) < 5:
            # not jammed: just jitter
            B = _tangent_bases(base, rng)
            Y0 = rng.normal(size=(2, n * (d - 1)))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True) + 1e-12
        else:
            B = _tangent_bases(base, rng)
            # lowest ~10-30 singular directions: get 4 random nullspace combos
            Y0 = _soft_modes(base, B, I0, J0, nmodes=4, rng=rng)
        # try each mode with a line-search amplitude
        improved = False
        for mi in range(Y0.shape[0]):
            if time_left() < 20:
                break
            amp = amps[int(rng.integers(0, len(amps)))]
            yv = Y0[mi]
            Dd = d - 1
            Ym = yv.reshape(n, Dd)
            dx = np.einsum('idk,ik->id', B, Ym)
            # scale so mean per-point norm == amp
            frob = np.linalg.norm(dx) + 1e-12
            dx = dx / frob * amp * np.sqrt(n)
            trial = _proj(base + dx)
            # cheap 200-step screening before any LP
            scr = _adam_screen(trial, steps=200, lr=0.02, T=0.006, mu=0.85)
            v_scr = _max_cos(scr)
            _, cnt1, _, _, _, _ = _contact_info(scr, tol=1e-6)
            # second-order predicted max proxy: use screened value
            # accept only if topology changed materially or predicted max decreased
            change = abs(cnt1 - cnt0) / max(cnt0, 1)
            if not (change > 0.05 or v_scr < base_val + 0.0015):
                # try a second amplitude quickly without LP (line search)
                amp2 = amp * 2.0 if amp < 0.1 else amp * 0.5
                amp2 = float(np.clip(amp2, 0.01, 0.2))
                dx2 = dx / (amp * np.sqrt(n) / frob + 1e-12) * 0 + dx * (amp2 / amp)
                trial2 = _proj(base + dx2)
                scr2 = _adam_screen(trial2, steps=120, lr=0.02, T=0.006, mu=0.85)
                v2 = _max_cos(scr2)
                _, cnt2, _, _, _, _ = _contact_info(scr2, tol=1e-6)
                change2 = abs(cnt2 - cnt0) / max(cnt0, 1)
                if change2 > change and (change2 > 0.05 or v2 < v_scr):
                    scr, v_scr, cnt1 = scr2, v2, cnt2
                    change = change2
                else:
                    continue
            if time_left() < 20:
                break
            # short 1-2s LP polish
            lp_sec = 1.0 if v_scr > best_val else 2.0
            lp_sec = min(lp_sec, time_left() - 18)
            if lp_sec <= 0.3:
                break
            pol = _lp_polish(scr, lp_sec)
            v_pol = _max_cos(pol)
            if v_pol < best_val - 1e-12:
                best, best_val = pol.copy(), v_pol
                # add to elite with separation
                gd = min([_gram_dist(pol, elite[k][1]) for k in range(len(elite))]) if elite else 1e9
                elite.append((v_pol, pol.copy()))
                # keep elite small, sorted, separated
                elite = sorted(elite, key=lambda e: e[0])
                # enforce separation: drop close worse ones
                kept = [elite[0]]
                for e in elite[1:]:
                    if all(_gram_dist(e[1], k[1]) > 0.15 for k in kept):
                        kept.append(e)
                    if len(kept) >= 5:
                        break
                elite = kept[:5]
                improved = True
                break
            elif v_pol < base_val and len(elite) < 5:
                # diverse decent point: add if separated
                if all(_gram_dist(pol, k[1]) > 0.15 for k in elite):
                    elite.append((v_pol, pol.copy()))
                    elite = sorted(elite, key=lambda e: e[0])[:5]
        if not improved:
            # occasional random restart from elite to avoid stagnation: perturb worst elite slightly
            if it % 10 == 0 and len(elite) > 1 and time_left() > 30:
                # drop worst if too close to best
                if _gram_dist(elite[-1][1], elite[0][1]) < 0.1:
                    elite.pop()
    # final squeeze on best with remaining budget (reserve ~5s buffer)
    tl = time_left()
    if tl > 6:
        best = _lp_polish(best, min(tl - 4.0, 10.0))
        best_val = _max_cos(best)
    best = _proj(best)
    # tiny shrink guard (no-op, ensures strict unit rows, no NaN)
    best = np.nan_to_num(best)
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    best = best / nrm
    return best
