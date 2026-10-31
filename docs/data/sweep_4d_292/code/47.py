import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, deadline, r_init=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    # cache c array reuse
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
        except Exception:
            break
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        # cap constraints to tightest 3500 for speed
        if cnt > 3500:
            # select top 3500
            idx = np.argpartition(-c, 3500)[:3500]
            mask = np.zeros_like(c, dtype=bool)
            mask[idx] = True
            # ensure threshold consistency: keep only those in sel? just use idx
            sel = mask
            cnt = 3500
        I = iu0[0][sel].astype(np.int64)
        J = iu0[1][sel].astype(np.int64)
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            ci = I[:, None] * d + np.arange(d)
            cj = J[:, None] * d + np.arange(d)
            cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            best = v
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # if very close to deadline, break
        if time.time() >= deadline:
            break
    return x

def _contact_degrees(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > (m - tol)
    deg = np.count_nonzero(mask, axis=1)
    return deg, m, G

def _refine_hole(X, y0, steps=120, beta=250.0):
    y = y0 / max(1e-12, float(np.linalg.norm(y0)))
    for s in range(steps):
        dots = X @ y
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = float(w.sum()) + 1e-300
        w /= sw
        grad = w @ X
        gt = grad - float(grad @ y) * y
        # decaying lr
        lr = 0.08 * (1.0 - 0.7 * s / max(1, steps))
        y = y - lr * gt
        nr = float(np.linalg.norm(y))
        if nr < 1e-12:
            break
        y = y / nr
        # anneal beta slightly
        if s == steps // 2:
            beta = beta * 1.5
    return y

def _find_holes(X, rng, n_holes=3, n_sample=22000, n_cand=8):
    # sample candidates, pick best diverse, refine
    C = rng.standard_normal((n_sample, 4))
    C = _norm(C)
    # scores = max dot to X
    # chunk to save memory
    best_idx = None
    best_scores = np.full(n_sample, 1e9)
    # compute in blocks
    bs = 4000
    scores = np.empty(n_sample)
    for a in range(0, n_sample, bs):
        b = min(n_sample, a + bs)
        scores[a:b] = np.max(C[a:b] @ X.T, axis=1)
    order = np.argsort(scores)  # ascending: smallest max-dot = largest hole
    # take spread candidates: greedily enforce angular separation
    picked = []
    for idx in order:
        if len(picked) >= n_cand:
            break
        c = C[idx]
        ok = True
        for q in picked:
            if float(c @ C[q]) > 0.96:
                ok = False
                break
        if ok:
            picked.append(int(idx))
    refined = []
    for idx in picked:
        y = _refine_hole(X, C[idx])
        s = float(np.max(X @ y))
        refined.append((s, y))
    refined.sort(key=lambda kv: kv[0])
    # return top n_holes (plus extras for branching)
    return refined

def solve():
    t0 = time.time()
    LIMIT = 172.0
    deadline_final = t0 + LIMIT
    rng = np.random.default_rng(292304)
    n_target = 292
    # load incumbent
    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                X0 = _norm(a)
                break
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((292, 4)))
    best_val = _max_cos(X0)
    Xbest = X0.copy()
    base = X0.copy()

    # quick signature helper
    def slack_sig(X, k=1500):
        G = X @ X.T
        iu = np.triu_indices(X.shape[0], 1)
        c = G[iu]
        m = float(c.max())
        s = m - c
        s.sort()
        return s[:k]

    seen = set()
    def sig_key(X):
        try:
            s = slack_sig(X, k=900)
            # quantize to 1e-7 to compare topology
            q = tuple(np.round(s[:40], 7).tolist())
            return q
        except Exception:
            return None

    # main ladder loop: alternate delete-rebuild and overshoot cycles
    # each trial ~4-6s; keep total < ~150s leaving 20s final polish
    trial = 0
    # precompute degrees of base
    while time.time() - t0 < 148.0:
        trial += 1
        rem_time = deadline_final - (time.time() - t0) * 0 - time.time()
        # rem absolute
        if time.time() > t0 + 148.0:
            break
        # refresh degrees of current base (or best)
        # occasionally restart from Xbest
        if trial % 7 == 0:
            base = Xbest.copy()
        try:
            deg, mcur, _ = _contact_degrees(base, tol=1e-9)
        except Exception:
            deg = np.zeros(292, dtype=int)
        order_desc = np.argsort(-deg)  # most over-contacted first
        order_asc = np.argsort(deg)   # rattlers first
        mode = trial % 4
        try:
            if mode in (0, 1):
                # DELETE ladder: remove 1 (or 2) high-degree points -> polish small -> refill
                if mode == 0:
                    # remove 1 point: branch over top list
                    k = (trial // 4) % 12
                    if k < 10:
                        ridx = int(order_desc[k % len(order_desc)])
                    else:
                        # sometimes remove rattler to test
                        ridx = int(order_asc[0])
                    Xm = np.delete(base, ridx, axis=0)  # 291
                    # small jitter to leave basin
                    sig = 0.004 if (trial % 2 == 0) else 0.008
                    Xm = _norm(Xm + sig * rng.standard_normal(Xm.shape))
                    d1 = min(2.6, max(1.2, (t0 + 148.0 - time.time()) / 12.0))
                    Xm_opt = _lp_polish_deadline(Xm, time.time() + d1, r_init=1e-3)
                    # find holes (branch over hole choices)
                    holes = _find_holes(Xm_opt, rng, n_holes=4, n_sample=18000, n_cand=8)
                    hole_choice = (trial // 2) % max(1, len(holes))
                    ystar = holes[hole_choice][1] if len(holes) > 0 else rng.standard_normal(4)
                    ystar = ystar / max(1e-12, float(np.linalg.norm(ystar)))
                    Xnew = np.vstack([Xm_opt, ystar[None, :]])
                    Xnew = _norm(Xnew + 0.0015 * rng.standard_normal(Xnew.shape))
                    d2 = min(2.8, max(1.2, (t0 + 148.0 - time.time()) / 10.0))
                    Xnew = _lp_polish_deadline(Xnew, time.time() + d2, r_init=8e-4)
                else:
                    # remove 2 points -> polish 290 -> refill 2 sequentially
                    k1 = (trial) % 10
                    k2 = (trial * 3 + 1) % 10
                    r1 = int(order_desc[k1]); r2 = int(order_desc[k2])
                    if r1 == r2:
                        r2 = int(order_desc[(k2 + 5) % 10])
                    keep = [i for i in range(292) if i != r1 and i != r2]
                    Xm = base[np.array(keep), :]
                    Xm = _norm(Xm + 0.005 * rng.standard_normal(Xm.shape))
                    d1 = min(2.2, max(1.0, (t0 + 148.0 - time.time()) / 12.0))
                    Xm_opt = _lp_polish_deadline(Xm, time.time() + d1, r_init=1e-3)
                    # first hole
                    h1 = _find_holes(Xm_opt, rng, n_holes=2, n_sample=14000, n_cand=6)
                    y1 = h1[0][1] if len(h1) > 0 else _norm(rng.standard_normal((1, 4)))[0]
                    Xt = _norm(np.vstack([Xm_opt, y1[None, :]]))
                    # light polish of 291 before second insert
                    Xt = _lp_polish_deadline(Xt, time.time() + 1.0, r_init=5e-4)
                    h2 = _find_holes(Xt, rng, n_holes=3, n_sample=14000, n_cand=6)
                    hc = (trial // 3) % max(1, len(h2))
                    y2 = h2[hc][1] if len(h2) > 0 else _norm(rng.standard_normal((1, 4)))[0]
                    Xnew = _norm(np.vstack([Xt, y2[None, :]]))
                    d2 = min(2.2, max(1.0, (t0 + 148.0 - time.time()) / 10.0))
                    Xnew = _lp_polish_deadline(Xnew, time.time() + d2, r_init=6e-4)
            elif mode == 2:
                # OVERSHOOT: insert into largest hole -> polish 293 -> delete different high-degree point
                holes = _find_holes(base, rng, n_holes=3, n_sample=18000, n_cand=8)
                hc = (trial // 2) % max(1, len(holes))
                yadd = holes[hc][1] if len(holes) > 0 else _norm(rng.standard_normal((1, 4)))[0]
                Xover = _norm(np.vstack([base, yadd[None, :]]))  # 293
                Xover = _lp_polish_deadline(Xover, time.time() + 1.8, r_init=8e-4)
                # delete a different point from the one just inserted (index 292 is inserted)
                try:
                    deg2, _, _ = _contact_degrees(Xover, tol=1e-9)
                except Exception:
                    deg2 = np.zeros(293, dtype=int)
                od = np.argsort(-deg2)
                # pick deletion that is not the inserted one; branch
                pick = None
                for cand in od[:14]:
                    if int(cand) != 292:
                        # cycle choice
                        pick = int(cand)
                        break
                if pick is None:
                    pick = 0
                # branch: shift choice by trial
                alt = od[(trial % 12)]
                if int(alt) != 292 and (trial % 3 == 0):
                    pick = int(alt)
                keep = [i for i in range(293) if i != pick]
                Xnew = Xover[np.array(keep), :]
                Xnew = _norm(Xnew + 0.001 * rng.standard_normal(Xnew.shape))
                d2 = min(2.6, max(1.0, (t0 + 148.0 - time.time()) / 10.0))
                Xnew = _lp_polish_deadline(Xnew, time.time() + d2, r_init=7e-4)
            else:
                # OVERSHOOT x2: insert 2 -> polish 294 -> delete 2 (different from inserted)
                h = _find_holes(base, rng, n_holes=2, n_sample=14000, n_cand=6)
                y1 = h[0][1] if len(h) > 0 else _norm(rng.standard_normal((1, 4)))[0]
                Xt = _norm(np.vstack([base, y1[None, :]]))
                h2 = _find_holes(Xt, rng, n_holes=2, n_sample=12000, n_cand=6)
                y2 = h2[(trial % max(1, len(h2)))][1] if len(h2) > 0 else _norm(rng.standard_normal((1, 4)))[0]
                Xover = _norm(np.vstack([Xt, y2[None, :]]))  # 294
                Xover = _lp_polish_deadline(Xover, time.time() + 1.6, r_init=7e-4)
                try:
                    deg2, _, _ = _contact_degrees(Xover, tol=1e-9)
                except Exception:
                    deg2 = np.zeros(294, dtype=int)
                od = np.argsort(-deg2)
                # choose 2 deletions not both inserted (292,293)
                dels = []
                for cand in od:
                    if int(cand) not in (292, 293):
                        dels.append(int(cand))
                    if len(dels) >= 2:
                        break
                # branch second deletion
                if len(dels) < 2:
                    dels = [0, 1]
                keep = [i for i in range(294) if i not in dels]
                Xnew = Xover[np.array(keep), :]
                Xnew = _norm(Xnew + 0.0012 * rng.standard_normal(Xnew.shape))
                d2 = min(2.4, max(1.0, (t0 + 148.0 - time.time()) / 10.0))
                Xnew = _lp_polish_deadline(Xnew, time.time() + d2, r_init=6e-4)
        except Exception:
            continue
        # score intermediate only after polish
        try:
            if Xnew.shape != (292, 4) or not np.all(np.isfinite(Xnew)):
                continue
            Xnew = _norm(Xnew)
            vnew = _max_cos(Xnew)
        except Exception:
            continue
        # diversity via slack signature (skip exact repeats)
        try:
            key = sig_key(Xnew)
            if key is not None:
                if key in seen and vnew >= best_val:
                    # same topology and not better: perturb base slightly for next trial
                    base = _norm(base + 0.002 * rng.standard_normal(base.shape))
                    continue
                seen.add(key)
        except Exception:
            pass
        if vnew < best_val - 1e-13:
            best_val = vnew
            Xbest = Xnew.copy()
            base = Xnew.copy()
        else:
            # occasional move to different basin even if slightly worse (if diverse)
            # accept if within 2e-4 and signature new, to keep ladder moving
            if vnew < best_val + 2e-4 and (trial % 5 == 0):
                base = Xnew.copy()
        if time.time() > t0 + 148.0:
            break

    # final tight polish of best with remaining time
    try:
        rem = (t0 + LIMIT - 1.0) - time.time()
        rem = max(4.0, min(22.0, rem))
        Xbest = _lp_polish_deadline(Xbest, time.time() + rem, r_init=1e-4)
        # extra ultra-tight pass
        rem2 = (t0 + LIMIT - 0.5) - time.time()
        if rem2 > 2.0:
            Xbest = _lp_polish_deadline(Xbest, time.time() + rem2, r_init=2e-5)
    except Exception:
        pass
    Xbest = _norm(np.asarray(Xbest, dtype=float))
    # strict feasibility: shrink margin not needed (unit rows already); ensure finite
    Xbest[~np.isfinite(Xbest)] = 0.25
    Xbest = _norm(Xbest)
    return Xbest
