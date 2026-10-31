import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def quench_lse(x0, steps, tau0, tau1, lr0, lr1, beta, rng, noise0=0.0):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gmax = float(G.max())
        if Gmax < bv:
            bv = Gmax
            bx = x.copy()
        E = np.exp((G - Gmax) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-30)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        if noise0 > 0:
            ns = noise0 * (1.0 - frac)
            if ns > 1e-12:
                nz = rng.standard_normal((n, d))
                nz -= np.sum(nz * x, axis=1, keepdims=True) * x
                tang = tang + ns * nz
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def refine_hole(x, y0, steps=120, tau=0.01, lr=0.15):
    y = y0.copy()
    yn = np.linalg.norm(y)
    if yn == 0:
        return y
    y = y / yn
    for _ in range(steps):
        dots = x @ y
        m = float(dots.max())
        w = np.exp((dots - m) / tau)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            break
        grad = (x.T @ w) / s
        gd = float(grad @ y)
        gt = grad - gd * y
        y = y - lr * gt
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y


def hole_centers_refined(x, k, ncand, rng):
    n, d = x.shape
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    md = dots.max(axis=1)
    kk = min(k, ncand)
    idx = np.argpartition(md, kk - 1)[:kk]
    idx = idx[np.argsort(md[idx])]
    out = []
    for j in idx:
        y = refine_hole(x, cand[j])
        out.append(y)
    out = np.stack(out, axis=0)
    # rank refined by true hole quality
    q = (out @ x.T).max(axis=1)
    order = np.argsort(q)
    return out[order]


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        np.fill_diagonal(Gfull, -2.0)
        m = float(Gfull.max())
        c_full = Gfull[iu]
        sel = c_full > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        # cap constraints for speed
        if cnt > 3000:
            # keep largest 3000
            sidx = np.where(sel)[0]
            vals = c_full[sidx]
            keep = np.argpartition(vals, -3000)[-3000:]
            newsel = np.zeros_like(sel, dtype=bool)
            newsel[sidx[keep]] = True
            sel = newsel
            c_full_sel = c_full[sel]
        else:
            c_full_sel = c_full[sel]
        I, J = iu[0][sel], iu[1][sel]
        c = c_full_sel
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 118.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    vals = [max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best_x = cand[bi].copy()
    best_val = float(vals[bi])

    # quick initial polish to ensure local optimality
    if HARD_END - time.time() > 10:
        bx0 = lp_polish(best_x, 2.5, HARD_END - 8)
        bv0 = max_cos(bx0)
        if bv0 < best_val:
            best_x, best_val = bx0, bv0

    cur_x = best_x.copy()
    cur_v = best_val

    trial = 0
    iu0 = np.triu_indices(n, 1)
    while time.time() < EXPLORE_END and time.time() < HARD_END - 20:
        trial += 1
        elapsed = time.time() - t0
        prog = elapsed / max(1e-9, (EXPLORE_END - t0))
        Temp = 0.004 * (1.0 - prog) + 0.0003

        # choose base: mostly cur for wandering, sometimes best
        if rng.random() < 0.35:
            base = best_x
            base_v = best_val
        else:
            base = cur_x
            base_v = cur_v

        r = rng.random()
        if r < 0.30:
            ls = rng.uniform(np.log(0.003), np.log(0.28))
            sigma = float(np.exp(ls))
            x_try = base + sigma * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.50:
            k = int(rng.integers(3, 26))
            idx = rng.choice(n, size=k, replace=False)
            sb = float(np.exp(rng.uniform(np.log(0.07), np.log(0.6))))
            ss = float(np.exp(rng.uniform(np.log(0.0008), np.log(0.012))))
            noise = ss * rng.standard_normal((n, d))
            noise[idx] = sb * rng.standard_normal((k, d))
            x_try = base + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.68:
            # contact-targeted shake with repulsion push
            ppm = per_point_max(base)
            thr = base_v - float(rng.uniform(0.002, 0.025))
            involved = np.where(ppm > thr)[0]
            if len(involved) < 3:
                involved = np.argsort(-ppm)[:12]
            si = float(np.exp(rng.uniform(np.log(0.03), np.log(0.35))))
            so = 0.002
            noise = so * rng.standard_normal((n, d))
            noise[involved] += si * rng.standard_normal((len(involved), d))
            # push worst pairs apart along tangent
            try:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                # top few pairs
                flat = G[iu0]
                K = min(12, flat.size)
                top = np.argpartition(flat, -K)[-K:]
                II = iu0[0][top]
                JJ = iu0[1][top]
                push = float(rng.uniform(0.0, 0.05))
                if push > 0:
                    for a_, b_ in zip(II, JJ):
                        # move a away from b
                        dab = base[a_] - base[b_]
                        dab -= float(dab @ base[a_]) * base[a_]
                        dba = base[b_] - base[a_]
                        dba -= float(dba @ base[b_]) * base[b_]
                        noise[a_] += push * dab
                        noise[b_] += push * dba
            except Exception:
                pass
            x_try = base + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.86:
            # rattler to refined hole
            ppm = per_point_max(base)
            gap = float(rng.uniform(0.004, 0.03))
            ratt = np.where(ppm < base_v - gap)[0]
            x_try = base + 0.004 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            if len(ratt) > 0:
                nh = int(rng.integers(1, min(4, len(ratt) + 1)))
                pick = rng.choice(ratt, size=nh, replace=False)
                hc = hole_centers_refined(x_try, nh, 6000, rng)
                hc = hc + 0.008 * rng.standard_normal(hc.shape)
                hc /= np.linalg.norm(hc, axis=1, keepdims=True)
                x_try[pick] = hc[:nh]
            else:
                j = int(np.argmin(ppm))
                hc = hole_centers_refined(x_try, 1, 6000, rng)[0]
                x_try[j] = hc
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # moderate global + single-point hole insert
            sigma = float(np.exp(rng.uniform(np.log(0.008), np.log(0.09))))
            x_try = base + sigma * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            # relocate one worst rattler to hole occasionally
            if rng.random() < 0.5:
                ppm = per_point_max(x_try)
                j = int(np.argmin(ppm))
                hc = hole_centers_refined(x_try, 1, 4000, rng)[0]
                # blend
                w = float(rng.uniform(0.7, 1.0))
                y = w * hc + (1 - w) * x_try[j]
                y /= np.linalg.norm(y)
                x_try[j] = y

        # quench hyperparams
        if time.time() > EXPLORE_END - 25:
            steps = int(rng.choice([300, 400, 550]))
        else:
            steps = int(rng.choice([350, 500, 650, 850, 1050]))
        beta = float(rng.uniform(0.80, 0.96))
        tau0 = float(np.exp(rng.uniform(np.log(0.006), np.log(0.06))))
        tau1 = float(np.exp(rng.uniform(np.log(0.0006), np.log(0.0045))))
        if tau1 >= tau0:
            tau1 = tau0 * 0.12
        lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.14))))
        lr1 = float(np.exp(rng.uniform(np.log(0.0012), np.log(0.012))))
        nz = 0.0
        if rng.random() < 0.22:
            nz = float(np.exp(rng.uniform(np.log(0.0004), np.log(0.004))))

        bx, bv = quench_lse(x_try, steps, tau0, tau1, lr0, lr1, beta, rng, noise0=nz)

        # Metropolis update of cur
        if bv < cur_v or rng.random() < np.exp((cur_v - bv) / max(Temp, 1e-9)):
            cur_x, cur_v = bx.copy(), float(bv)
            # restart wandering if cur drifted too far worse than best
            if cur_v > best_val + 0.015 and rng.random() < 0.6:
                cur_x, cur_v = best_x.copy(), best_val

        if bv < best_val - 5e-13:
            rem = min(EXPLORE_END - time.time(), HARD_END - time.time() - 6)
            if rem > 4:
                bxp = lp_polish(bx, min(1.8, rem - 1.0))
                bvp = max_cos(bxp)
                if bvp < bv:
                    bx, bv = bxp, bvp
            if bv < best_val:
                best_x, best_val = bx.copy(), float(bv)
                cur_x, cur_v = bx.copy(), float(bv)

    remain = HARD_END - time.time() - 1.5
    if remain > 3:
        use = min(remain, 48.0)
        best_x = lp_polish(best_x, use, HARD_END - 1.0)
        # extra tiny shake + repolish cycles to squeeze last digits
        for _ in range(3):
            if time.time() >= HARD_END - 2.0:
                break
            best_val_now = max_cos(best_x)
            # micro jitter
            jitter = 1e-7
            y = best_x + jitter * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = lp_polish(y, min(3.0, HARD_END - time.time() - 1.0), HARD_END - 1.0)
            vy = max_cos(y)
            if vy < best_val_now:
                best_x = y
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
