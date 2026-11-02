import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def solve():
    t0 = time.time()
    DEAD = t0 + 168.0
    np.random.seed(0)
    rng = np.random.default_rng(0)

    try:
        best = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            best = np.load("record.npy").astype(np.float64)
        except Exception:
            best = rng.standard_normal((153, 4))
    best = np.asarray(best, dtype=np.float64)
    if best.shape != (153, 4):
        best = rng.standard_normal((153, 4))
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best_val = _max_cos(best)

    iu0 = np.triu_indices(153, 1)

    def lp_polish(x_init, deadline):
        x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
        n, d = x.shape
        cur = _max_cos(x)
        radius = 2e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            G = x @ x.T
            c = G[iu]
            m = float(np.max(c))
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                break
            if cnt > 3000:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:3000]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu[0][sel]
            J = iu[1][sel]
            p = len(I)
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
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel],
                              A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.6, 0.05)
            else:
                radius *= 0.5
        return x, cur

    def lse_adam(x_init, steps, tau0, tau1, lr0, lr1, noise0, deadline):
        x = x_init.copy()
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        n, d = x.shape
        m1 = np.zeros_like(x)
        m2 = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for it in range(steps):
            if (it & 127) == 0 and time.time() > deadline:
                break
            frac = it / max(1, steps - 1)
            tau = tau0 * ((tau1 / tau0) ** frac)
            lr = lr0 * ((lr1 / lr0) ** frac)
            G = x @ x.T
            np.fill_diagonal(G, -1e9)
            m = float(np.max(G))
            E = (G - m) / max(tau, 1e-12)
            np.clip(E, -30.0, 0.0, out=E)
            W = np.exp(E)
            # W symmetric, diag ~0
            Gr = W @ x
            dot = np.einsum('ij,ij->i', Gr, x)[:, None]
            Gt = Gr - dot * x
            m1 *= b1
            m1 += (1.0 - b1) * Gt
            m2 *= b2
            m2 += (1.0 - b2) * (Gt * Gt)
            bc1 = 1.0 - b1 ** (it + 1)
            bc2 = 1.0 - b2 ** (it + 1)
            step = (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
            sdot = np.einsum('ij,ij->i', step, x)[:, None]
            step -= sdot * x
            if noise0 > 0:
                amp = noise0 * (1.0 - frac)
                if amp > 0:
                    rnd = rng.standard_normal((n, d)) * amp
                    rdot = np.einsum('ij,ij->i', rnd, x)[:, None]
                    rnd -= rdot * x
                    x = x - lr * step + rnd
                else:
                    x = x - lr * step
            else:
                x = x - lr * step
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def refine_hole(c, others):
        # gradient descent to minimize max dot to others
        for it in range(220):
            dd = others @ c
            mx = float(np.max(dd))
            beta = 25.0 + it * 0.35
            w = np.exp(beta * (dd - mx))
            s = float(np.sum(w))
            if s <= 0:
                break
            w /= s
            gvec = w @ others
            g = gvec - float(gvec @ c) * c
            gn = float(np.linalg.norm(g))
            if gn < 1e-12:
                break
            g /= gn
            lr = 0.03 / (1.0 + it * 0.025)
            c = c - lr * g
            nrm = float(np.linalg.norm(c))
            if nrm < 1e-12:
                break
            c /= nrm
        return c

    def find_hole(others, n_sample=16000):
        cand = rng.standard_normal((n_sample, 4))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        dots = cand @ others.T
        m = dots.max(axis=1)
        order = np.argsort(m)
        best_c = None
        best_v = 1e9
        for k in order[:4]:
            c = cand[k].copy()
            c = refine_hole(c, others)
            vv = float(np.max(others @ c))
            if vv < best_v:
                best_v = vv
                best_c = c.copy()
        return best_c

    def greedy_reinsert(others, k):
        fixed = others.copy()
        outs = []
        for _ in range(k):
            ref = fixed if len(outs) == 0 else np.vstack([fixed] + outs)
            h = find_hole(ref)
            outs.append(h[None, :])
        return np.vstack([fixed] + outs)

    def contacts_info(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnt = np.sum(G > m - 1e-9, axis=1)
        return m, cnt, G

    it = 0
    # main basin-hopping loop, reserve ~28s for final polish
    while time.time() < DEAD - 28.0:
        it += 1
        mode = it % 8
        cand = None
        try:
            if mode == 0:
                # full random fresh basin
                x_init = rng.standard_normal((153, 4))
                x_init /= np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
                tmp = lse_adam(x_init, steps=1400, tau0=8e-3, tau1=4e-4,
                               lr0=0.012, lr1=0.0015, noise0=0.004, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=700, tau0=4e-4, tau1=1.2e-4,
                               lr0=0.004, lr1=0.0008, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
            elif mode in (1, 2):
                sig = 0.025 if mode == 1 else 0.06
                pert = best + rng.standard_normal(best.shape) * sig
                pert /= np.maximum(np.linalg.norm(pert, axis=1, keepdims=True), 1e-12)
                tmp = lse_adam(pert, steps=1100, tau0=5e-3, tau1=2e-4,
                               lr0=0.01, lr1=0.001, noise0=0.0025, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=500, tau0=2e-4, tau1=1e-4,
                               lr0=0.003, lr1=0.0006, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
            elif mode == 3:
                # large shake
                pert = best + rng.standard_normal(best.shape) * 0.22
                pert /= np.maximum(np.linalg.norm(pert, axis=1, keepdims=True), 1e-12)
                tmp = lse_adam(pert, steps=1300, tau0=8e-3, tau1=3e-4,
                               lr0=0.012, lr1=0.001, noise0=0.003, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=500, tau0=3e-4, tau1=1e-4,
                               lr0=0.003, lr1=0.0006, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
            elif mode == 4:
                # single-point surgery on overloaded point
                _, cnt, _ = contacts_info(best)
                top = np.argsort(-cnt)[:14]
                victim = int(rng.choice(top))
                others = np.delete(best, victim, axis=0)
                hole = find_hole(others)
                tmp0 = best.copy()
                tmp0[victim] = hole
                tmp = lse_adam(tmp0, steps=1000, tau0=4e-3, tau1=2e-4,
                               lr0=0.009, lr1=0.0009, noise0=0.002, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=500, tau0=2e-4, tau1=1e-4,
                               lr0=0.003, lr1=0.0006, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
            elif mode in (5, 6):
                # multi-point surgery k=4..7
                _, cnt, Gc = contacts_info(best)
                k = 5 if mode == 5 else int(rng.integers(3, 8))
                # victims: most contacts + one of worst pair
                order = np.argsort(-cnt)
                victims = list(order[:k])
                # ensure worst pair represented
                G2 = best @ best.T
                np.fill_diagonal(G2, -2.0)
                ij = np.unravel_index(int(np.argmax(G2)), G2.shape)
                if ij[0] not in victims and rng.random() < 0.7:
                    victims[-1] = int(ij[0])
                victims = sorted(set(int(v) for v in victims))
                k = len(victims)
                mask = np.ones(153, dtype=bool)
                mask[victims] = False
                others = best[mask]
                rebuilt = greedy_reinsert(others, k)
                # map rebuilt tail back to victim slots (order doesn't matter)
                tmp0 = best.copy()
                tail = rebuilt[-k:]
                # random assignment to victims
                perm = rng.permutation(k)
                for t, v in enumerate(victims):
                    tmp0[v] = tail[perm[t]]
                tmp = lse_adam(tmp0, steps=1200, tau0=6e-3, tau1=2.5e-4,
                               lr0=0.011, lr1=0.001, noise0=0.0025, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=600, tau0=2.5e-4, tau1=1e-4,
                               lr0=0.0035, lr1=0.0006, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
            else:
                # patch shuffle: nearest neighbours of random center get big kick
                cidx = int(rng.integers(0, 153))
                dots = best @ best[cidx]
                nn = np.argsort(-dots)[:7]
                tmp0 = best.copy()
                tmp0[nn] += rng.standard_normal((len(nn), 4)) * 0.35
                tmp0[nn] /= np.maximum(np.linalg.norm(tmp0[nn], axis=1, keepdims=True), 1e-12)
                tmp = lse_adam(tmp0, steps=1200, tau0=6e-3, tau1=2.5e-4,
                               lr0=0.011, lr1=0.001, noise0=0.0025, deadline=DEAD - 28.0)
                tmp = lse_adam(tmp, steps=500, tau0=2.5e-4, tau1=1e-4,
                               lr0=0.003, lr1=0.0006, noise0=0.0, deadline=DEAD - 28.0)
                cand = tmp
        except Exception:
            continue
        if cand is None:
            continue
        try:
            vv = _max_cos(cand)
        except Exception:
            continue
        # only LP-polish promising candidates
        if vv < best_val + 0.0012:
            try:
                cand2, vv2 = lp_polish(cand, deadline=min(time.time() + 2.2, DEAD - 28.0))
            except Exception:
                cand2, vv2 = cand, vv
            if vv2 < best_val:
                best, best_val = cand2, vv2
            elif vv < best_val:
                best, best_val = cand, vv
        # early exit if clearly record-beating and time low? continue to polish further
        if time.time() >= DEAD - 28.0:
            break

    try:
        best, best_val = lp_polish(best, deadline=DEAD - 1.0)
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (~np.isfinite(nr.squeeze())) | (nr.squeeze() < 1e-12)
    if np.any(bad):
        rnd = rng.standard_normal(best.shape)
        rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad] = rnd[bad]
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
