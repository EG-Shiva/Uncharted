import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def build_group(gens, d=8, limit=128):
    ident_p = np.arange(d)
    ident_s = np.ones(d)
    ops = [(ident_p, ident_s)]
    seen = { (tuple(ident_p), tuple(ident_s)) }
    stack = ops[:]
    # BFS closure
    idx = 0
    all_ops = ops[:]
    # include generators
    for g in gens:
        t = (tuple(np.asarray(g[0])), tuple(np.asarray(g[1])))
        if t not in seen:
            seen.add(t)
            all_ops.append((np.asarray(g[0]).copy(), np.asarray(g[1]).copy()))
    # close under composition
    changed = True
    while changed:
        changed = False
        cur = all_ops[:]
        for a_p, a_s in cur:
            for b_p, b_s in cur:
                # apply a then b: y_i = b_s[i]* (a_s[b_p[i]] * x[a_p[b_p[i]]])
                n_p = a_p[b_p]
                n_s = b_s * a_s[b_p]
                key = (tuple(n_p), tuple(n_s))
                if key not in seen:
                    seen.add(key)
                    all_ops.append((n_p.copy(), n_s.copy()))
                    changed = True
                    if len(all_ops) > limit:
                        break
            if len(all_ops) > limit:
                break
        # also need compose with gens only? above closes fully
    return all_ops

def prune_to_n(X, target=138):
    X = X.copy()
    while X.shape[0] > target:
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i, j = divmod(flat, n)
        # row maxima excluding each other
        ri = G[i].copy(); ri[j] = -2.0
        rj = G[j].copy(); rj[i] = -2.0
        mi, mj = ri.max(), rj.max()
        rm = i if mi >= mj else j
        X = np.delete(X, rm, axis=0)
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    def tl():
        return BUDGET - (time.time() - t0)
    rng = np.random.default_rng(12345)
    n, d = 138, 8
    # load best
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    best = None; best_val = 1e9
    for c in [xb, xr]:
        if c is None: continue
        if c.shape != (n, d): continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    if best is None:
        best = rng.standard_normal((n, d)); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)

    # ---- groups ----
    shift = np.array([7,0,1,2,3,4,5,6])
    rev = np.array([0,7,6,5,4,3,2,1])
    eye = np.arange(d)
    ones = np.ones(d)
    neg = -np.ones(d)
    alt = np.array([1.,-1.,1.,-1.,1.,-1.,1.,-1.])
    G_shift = build_group([(shift, ones)])
    G_shift_glob = build_group([(shift, ones),(eye, neg)])
    G_dih = build_group([(shift, ones),(rev, ones)])
    G_dih_glob = build_group([(shift, ones),(rev, ones),(eye, neg)])
    G_shift_alt = build_group([(shift, ones),(eye, alt)])
    groups = [G_shift_glob, G_dih, G_dih_glob, G_shift, G_shift_alt]

    iu138 = np.triu_indices(n, 1)

    # precompute inverses for each group
    def prep_ops(ops):
        invs = []
        for p, s in ops:
            inv_p = np.empty_like(p); inv_p[p] = np.arange(d)
            inv_s = s[inv_p]  # since +-1, inv = same
            invs.append((inv_p, inv_s))
        perms = np.stack([p for p,_ in ops]).astype(int)
        signs = np.stack([s for _,s in ops])
        iperms = np.stack([p for p,_ in invs]).astype(int)
        isigns = np.stack([s for _,s in invs])
        return perms, signs, iperms, isigns

    # symmetric Adam optimiser (seeds on sphere + free on sphere)
    def sym_opt(perms, signs, iperms, isigns, n_seed, n_free, betas, steps_list, lrs, rng_init, t_deadline):
        mg = perms.shape[0]
        S = rng_init.standard_normal((max(n_seed,1), d))
        S /= np.linalg.norm(S, axis=1, keepdims=True)+1e-30
        if n_seed == 0:
            S = np.zeros((0,d))
        F = rng_init.standard_normal((max(n_free,1), d))
        F /= np.linalg.norm(F, axis=1, keepdims=True)+1e-30
        if n_free == 0:
            F = np.zeros((0,d))
        mS = np.zeros_like(S); vS = np.zeros_like(S)
        mF = np.zeros_like(F); vF = np.zeros_like(F)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        # total expanded size (before prune)
        for bi, (beta, steps, lr) in enumerate(zip(betas, steps_list, lrs)):
            for _ in range(int(steps)):
                if time.time() > t_deadline:
                    break
                t += 1
                # expand
                if n_seed > 0:
                    # Xorb: (n_seed*mg, d)
                    Xorb = np.empty((n_seed*mg, d))
                    for k in range(mg):
                        pk = perms[k]; sk = signs[k]
                        # S[:,pk]*sk
                        Xorb[k::mg] if False else None
                    # simpler loop over seeds blocks
                    for j in range(n_seed):
                        sj = S[j]
                        base = j*mg
                        for k in range(mg):
                            Xorb[base+k] = signs[k] * sj[perms[k]]
                else:
                    Xorb = np.zeros((0,d))
                if n_free > 0:
                    X = np.concatenate([Xorb, F], axis=0) if n_seed>0 else F.copy()
                else:
                    X = Xorb
                N = X.shape[0]
                iu = np.triu_indices(N, 1)
                G = X @ X.T
                c = G[iu]
                mv = c.max()
                E = np.exp(beta*(c-mv))
                Sm = E.sum()+1e-300
                w = E/Sm
                W = np.zeros((N,N))
                W[iu] = w
                W = W + W.T
                gradX = W @ X
                # split
                if n_seed > 0:
                    gOrb = gradX[:n_seed*mg]
                    # aggregate to seeds via inverse ops
                    gSraw = np.zeros_like(S)
                    for j in range(n_seed):
                        acc = np.zeros(d)
                        base = j*mg
                        for k in range(mg):
                            g = gOrb[base+k]
                            acc += isigns[k] * g[iperms[k]]
                        gSraw[j] = acc
                    # tangent project
                    dot = (gSraw*S).sum(axis=1, keepdims=True)
                    gS = gSraw - dot*S
                else:
                    gS = np.zeros_like(S)
                if n_free > 0:
                    gFraw = gradX[n_seed*mg:]
                    dotF = (gFraw*F).sum(axis=1, keepdims=True)
                    gF = gFraw - dotF*F
                else:
                    gF = np.zeros_like(F)
                # Adam
                if n_seed > 0:
                    mS = b1*mS + (1-b1)*gS
                    vS = b2*vS + (1-b2)*gS*gS
                    mhat = mS/(1-b1**t); vhat = vS/(1-b2**t)
                    S = S - lr*mhat/(np.sqrt(vhat)+eps)
                    S /= np.linalg.norm(S, axis=1, keepdims=True)+1e-30
                if n_free > 0:
                    mF = b1*mF + (1-b1)*gF
                    vF = b2*vF + (1-b2)*gF*gF
                    mhatF = mF/(1-b1**t); vhatF = vF/(1-b2**t)
                    F = F - lr*mhatF/(np.sqrt(vhatF)+eps)
                    F /= np.linalg.norm(F, axis=1, keepdims=True)+1e-30
            if time.time() > t_deadline:
                break
        # final expand
        if n_seed > 0:
            Xorb = np.empty((n_seed*mg, d))
            for j in range(n_seed):
                for k in range(mg):
                    Xorb[j*mg+k] = signs[k]*S[j][perms[k]]
        else:
            Xorb = np.zeros((0,d))
        if n_free > 0:
            X = np.concatenate([Xorb, F], axis=0) if n_seed>0 else F.copy()
        else:
            X = Xorb
        X /= np.linalg.norm(X, axis=1, keepdims=True)+1e-30
        return X

    def free_refine(X0, betas, steps_list, lrs, t_deadline):
        X = X0.copy()
        X /= np.linalg.norm(X, axis=1, keepdims=True)+1e-30
        N, D = X.shape
        iu = np.triu_indices(N, 1)
        m = np.zeros_like(X); v = np.zeros_like(X)
        b1,b2,eps = 0.9,0.999,1e-8
        t=0
        for beta, steps, lr in zip(betas, steps_list, lrs):
            for _ in range(int(steps)):
                if time.time() > t_deadline:
                    break
                t+=1
                G = X @ X.T
                c = G[iu]
                mv = c.max()
                E = np.exp(beta*(c-mv)); Sm = E.sum()+1e-300
                w = E/Sm
                W = np.zeros((N,N)); W[iu]=w; W=W+W.T
                graw = W @ X
                dot = (graw*X).sum(axis=1, keepdims=True)
                g = graw - dot*X
                m = b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
                mhat=m/(1-b1**t); vhat=v/(1-b2**t)
                X = X - lr*mhat/(np.sqrt(vhat)+eps)
                X /= np.linalg.norm(X, axis=1, keepdims=True)+1e-30
            if time.time() > t_deadline:
                break
        return X

    # ---- try symmetric configs ----
    # each: (ops, n_seed, n_free, prune_extra?)
    configs = []
    for ops in groups:
        mg = len(ops)
        if mg == 0 or mg > 64: continue
        ns = n // mg
        nf = n - ns*mg
        # keep seed count in 3..17
        if ns < 2 or ns > 18: continue
        configs.append((ops, ns, nf, False))
    # add one heavy prune config with 6 seeds of 32-group
    try:
        mg = len(G_dih_glob)
        if mg == 32:
            configs.append((G_dih_glob, 6, 0, True))
    except Exception:
        pass
    # limit to 6 configs, prefer mg 16/32
    # sort by preference: mg 16 first
    def pref(c):
        mg=len(c[0]); return (0 if mg in (16,32) else 1, -mg)
    configs = sorted(configs, key=pref)[:6]

    sym_best = None; sym_best_val = 1e9
    sym_cands = []
    # time allocation: use up to ~100s for sym search
    sym_deadline = time.time() + min(100.0, max(20.0, tl()-70.0))
    betas = [30.0, 80.0, 150.0, 250.0, 380.0]
    steps_list = [280, 280, 280, 280, 380]
    lrs = [0.03, 0.025, 0.02, 0.015, 0.01]
    for ci, (ops, ns, nf, do_prune_big) in enumerate(configs):
        if time.time() > sym_deadline: break
        if tl() < 60: break
        perms, signs, iperms, isigns = prep_ops(ops)
        mg = perms.shape[0]
        if do_prune_big:
            # optimize big set then prune
            Xbig = sym_opt(perms, signs, iperms, isigns, ns, 0, betas, steps_list, lrs, rng, sym_deadline)
            X = prune_to_n(Xbig, 138)
        else:
            X = sym_opt(perms, signs, iperms, isigns, ns, nf, betas, steps_list, lrs, rng, sym_deadline)
            if X.shape[0] > 138:
                X = prune_to_n(X, 138)
            elif X.shape[0] < 138:
                # pad random
                need = 138 - X.shape[0]
                R = rng.standard_normal((need, d)); R/=np.linalg.norm(R,axis=1,keepdims=True)+1e-30
                X = np.concatenate([X, R], axis=0)
        X /= np.linalg.norm(X, axis=1, keepdims=True)+1e-30
        val = max_cos(X)
        sym_cands.append((val, X.copy()))
        if val < sym_best_val:
            sym_best_val = val; sym_best = X.copy()
        # also quick free refine of symmetric point at high beta to see potential
        if tl() < 40: break

    # pick starting point for symmetry breaking: best sym cand (even if worse than record)
    # keep global best
    if sym_best is not None and sym_best_val < best_val + 0.01:
        cur = sym_best.copy(); cur_val = sym_best_val
    else:
        # fallback to best
        cur = best.copy(); cur_val = best_val
        if sym_best is not None:
            # still keep sym_best as alternative start if time allows
            pass

    # ---- gradual symmetry breaking ----
    # successive tangent noise 1e-3, 1e-2, 3e-2 with anneals
    amps = [1e-3, 1e-2, 3e-2]
    for ai, amp in enumerate(amps):
        if tl() < 25: break
        # noisy restart from cur
        Nn = rng.standard_normal(cur.shape)
        Nn -= (Nn*cur).sum(axis=1, keepdims=True)*cur
        Xn = cur + amp * Nn
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)+1e-30
        # refine
        deadline = time.time() + min(12.0, max(4.0, (tl()-18.0)/ (len(amps)-ai)))
        Xr = free_refine(Xn, [80.0, 200.0, 380.0], [300, 300, 400], [0.02, 0.015, 0.008], deadline)
        vr = max_cos(Xr)
        if vr < cur_val:
            cur, cur_val = Xr, vr
        if vr < best_val:
            best, best_val = Xr.copy(), vr
        # also try noisy restart from global best occasionally
        if ai == 1 and tl() > 40:
            Nn2 = rng.standard_normal(best.shape)
            Nn2 -= (Nn2*best).sum(axis=1, keepdims=True)*best
            Xb2 = best + 0.01 * Nn2
            Xb2 /= np.linalg.norm(Xb2, axis=1, keepdims=True)+1e-30
            deadline2 = time.time() + 8.0
            if tl() > 30:
                Xr2 = free_refine(Xb2, [150.0, 350.0], [300, 400], [0.015, 0.008], deadline2)
                vr2 = max_cos(Xr2)
                if vr2 < best_val:
                    best, best_val = Xr2.copy(), vr2
                if vr2 < cur_val:
                    cur, cur_val = Xr2.copy(), vr2

    # ensure best tracks cur
    if cur_val < best_val:
        best, best_val = cur.copy(), cur_val
    # final LP polish with remaining time
    if tl() > 4:
        ft = min(tl()-1.5, 16.0)
        if ft > 1:
            Bp = lp_polish(best, seconds=ft)
            vp = max_cos(Bp)
            if vp < best_val:
                best, best_val = Bp, vp
            # also polish cur if different
            if cur_val < best_val + 0.002 and tl() > 4:
                Cp = lp_polish(cur, seconds=min(4.0, tl()-1.0))
                vpc = max_cos(Cp)
                if vpc < best_val:
                    best, best_val = Cp, vpc

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    best = best / (np.linalg.norm(best, axis=1, keepdims=True)+1e-30)
    return best
