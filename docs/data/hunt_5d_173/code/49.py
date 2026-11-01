import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= m - tol
    cnt = mask.sum(axis=1)
    return m, cnt, G

def _frozen_step(x, r, target_other, radius, K=40):
    n, d = x.shape
    G = x @ x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c_all = G[iu0, iu1]
    sel_other = (c_all > target_other - max(20.0*radius, 1e-9)) & (iu0 != r) & (iu1 != r)
    I_o = iu0[sel_other]; J_o = iu1[sel_other]; c_o = c_all[sel_other]
    rmask_idx = np.where((iu0 == r) | (iu1 == r))[0]
    order = np.argsort(-c_all[rmask_idx])
    K2 = min(K, len(rmask_idx))
    chosen = rmask_idx[order[:K2]]
    I_r = iu0[chosen]; J_r = iu1[chosen]; c_r = c_all[chosen]
    p_o = len(I_o); p_r = len(I_r); p = p_o + p_r
    if p_r == 0:
        return None, None
    nv = n*d + 1
    # Build A_ub
    # rows: 0..p_o-1 other, p_o..p-1 r-pairs
    rows_list = []
    cols_list = []
    vals_list = []
    # other pairs: x[J] at I-block, x[I] at J-block, 0 at t
    for k in range(p_o):
        i = int(I_o[k]); j = int(J_o[k])
        row = k
        for a in range(d):
            rows_list.append(row); cols_list.append(i*d+a); vals_list.append(float(x[j, a]))
            rows_list.append(row); cols_list.append(j*d+a); vals_list.append(float(x[i, a]))
        # t coeff 0 -> no entry
    for k2 in range(p_r):
        i = int(I_r[k2]); j = int(J_r[k2])
        row = p_o + k2
        for a in range(d):
            rows_list.append(row); cols_list.append(i*d+a); vals_list.append(float(x[j, a]))
            rows_list.append(row); cols_list.append(j*d+a); vals_list.append(float(x[i, a]))
        rows_list.append(row); cols_list.append(n*d); vals_list.append(-1.0)
    A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(p, nv)).tocsr()
    b_ub = np.empty(p)
    b_ub[:p_o] = target_other - c_o if p_o>0 else np.empty(0)
    b_ub[p_o:] = -c_r
    # equality: x_i . dx_i = 0
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    bounds = [(-radius, radius)]*(n*d) + [(None, None)]
    # freeze r
    for a in range(d):
        bounds[r*d+a] = (0.0, 0.0)
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=bounds, method="highs")
    except Exception:
        return None, None
    if res.status != 0:
        return None, None
    dx = res.x[:-1].reshape(n, d)
    tr = float(res.x[-1])
    return dx, tr

def _standard_step(x, radius, r_focus=None):
    n, d = x.shape
    G = x @ x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c_all = G[iu0, iu1]
    m = float(c_all.max())
    if r_focus is None:
        sel = c_all > m - max(20.0*radius, 1e-9)
    else:
        sel_wide = c_all > m - 5e-3
        # nearest 15 of r_focus
        rmask = (iu0 == r_focus) | (iu1 == r_focus)
        ridx = np.where(rmask)[0]
        order = np.argsort(-c_all[ridx])
        take = min(15, len(ridx))
        chosen = ridx[order[:take]]
        sel = sel_wide.copy()
        sel[chosen] = True
        # cap size
        if int(sel.sum()) > 4000:
            idx = np.where(sel)[0]
            ord2 = np.argsort(-c_all[idx])[:4000]
            m2 = np.zeros_like(sel)
            m2[idx[ord2]] = True
            sel = m2
    I = iu0[sel]; J = iu1[sel]; c = c_all[sel]
    p = len(I)
    if p == 0:
        return None
    nv = n*d+1
    rows = np.repeat(np.arange(p), 2*d+1)
    cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    return res.x[:-1].reshape(n,d)

def _polish(x, seconds, r_focus=None, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    while time.time()-start < seconds:
        if deadline is not None and time.time() > deadline-1.0:
            break
        if radius < 1e-14:
            break
        dx = _standard_step(x, radius, r_focus=r_focus)
        if dx is None:
            radius *= 0.5
            continue
        y = x + dx
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (173,5):
                an = a/np.linalg.norm(a,axis=1,keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best_x = an.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.normal(size=(173,5))
        best_x /= np.linalg.norm(best_x,axis=1,keepdims=True)
        best_val = _max_cos(best_x)
    record0 = best_val
    cur_x = best_x.copy()
    cur_val = best_val
    n, d = 173, 5
    eps_cycle = [1e-4, 3e-4, 1e-3, 3e-3]
    outer = 0
    # quick identify rattler
    while time.time() < deadline - 12.0:
        outer += 1
        m, cnt, G = _contacts(cur_x, tol=1e-9)
        r = int(np.argmin(cnt))
        # if all have contacts, pick min max-cosine point (most isolated = smallest row max)
        # argmin cnt already does; if tie keep smallest row-max? fine
        eps = eps_cycle[(outer-1) % len(eps_cycle)]
        # alternate: sometimes re-pick rattler from best_x
        target_other = min(cur_val, best_val)
        # frozen phase: iterate LP steps
        x_f = cur_x.copy()
        # current r-max
        Gr = G[r].copy()
        Gr[r] = -2.0
        prev_rmax = float(Gr.max())
        radius = 2e-3
        # aim: target_r = record0 - eps (or target_other - eps)
        target_r = target_other - eps
        it = 0
        # merit: max(max_other-target_other, rmax-target_r)
        def merit_of(X):
            gg = X@X.T
            np.fill_diagonal(gg,-2.0)
            mo = float(gg.max())  # global, but need other-only
            # other-only max: mask out r row/col
            # compute quickly
            tmp = gg.copy()
            tmp[r,:] = -2.0; tmp[:,r] = -2.0
            mo_other = float(tmp.max())
            mr = float(gg[r].max())
            return mo_other, mr
        mo0, mr0 = merit_of(x_f)
        merit0 = max(mo0-target_other, mr0-target_r)
        for it in range(25):
            if time.time() > deadline-12.0:
                break
            if radius < 1e-7:
                radius = 1e-4
                # continue or break?
            dx, tr = _frozen_step(x_f, r, target_other, radius, K=40)
            if dx is None:
                radius *= 0.5
                if radius < 5e-7:
                    break
                continue
            y = x_f + dx
            y /= np.linalg.norm(y,axis=1,keepdims=True)
            mo1, mr1 = merit_of(y)
            merit1 = max(mo1-target_other, mr1-target_r)
            # accept if merit improves and other not exploding
            if merit1 < merit0 - 1e-12 and mo1 <= target_other + 5e-4:
                x_f = y
                mo0, mr0, merit0 = mo1, mr1, merit1
                radius = min(radius*1.3, 5e-3)
                if mr0 <= target_r + 2e-9:
                    break
                if mo0 > target_other + 1e-9:
                    # others violated, stop pushing
                    pass
            else:
                radius *= 0.5
                if radius < 5e-7:
                    break
        # unfreeze: polish with rattler-focused selection
        rem = deadline - time.time()
        if rem < 11.0:
            break
        pt = min(6.0, rem-6.0)
        if pt < 1.0:
            pt = max(1.0, rem-6.0)
        y2, v2 = _polish(x_f, seconds=pt, r_focus=r, deadline=deadline)
        # evaluate rattler contacts (original r index)
        m2, cnt2, G2 = _contacts(y2, tol=1e-9)
        cr = int(cnt2[r])
        # also check if any point now rattler? global min contacts
        # acceptance: if rattler acquired >=3 contacts, accept even if worse (within tolerance)
        if cr >= 3:
            # accept as new cur even if worse, but cap worsening
            if v2 < cur_val + 0.02:
                cur_x, cur_val = y2, v2
            if v2 < best_val:
                best_x, best_val = y2.copy(), v2
        else:
            # greedy only
            if v2 < cur_val:
                cur_x, cur_val = y2, v2
                if v2 < best_val:
                    best_x, best_val = y2.copy(), v2
            elif v2 < best_val:
                best_x, best_val = y2.copy(), v2
            else:
                # small random kick to cur to change basin, keep load balanced: perturb overloaded + r
                # perturb r and its nearest neighbours slightly
                if outer % 3 == 0:
                    yy = cur_x.copy()
                    # move r to void: try random direction far from others
                    # pick random candidate far from all except encourage new topology
                    cand = rng.normal(size=(30,5)); cand/=np.linalg.norm(cand,axis=1,keepdims=True)
                    # score by max cos to others (want small)
                    sm = (cand @ cur_x.T).max(axis=1)
                    yy[r] = cand[int(np.argmin(sm))]
                    # shake neighbours a bit
                    gr = (cur_x @ cur_x[r])
                    gr[r] = -2
                    nb = np.argsort(-gr)[:5]
                    yy[nb] += 0.01*rng.normal(size=(len(nb),5))
                    yy /= np.linalg.norm(yy,axis=1,keepdims=True)
                    cur_x = yy
                    cur_val = _max_cos(cur_x)
        # if we improved best, also set cur to best occasionally
        if best_val < cur_val - 1e-9 and outer % 4 == 0:
            # keep exploring from best every 4th
            cur_x = best_x.copy(); cur_val = best_val
        # early exit if beat record by margin
        if best_val < 0.758561688405 - 1e-7:
            # keep polishing best a bit
            pass
    # final squeeze of best
    rem = deadline - time.time()
    if rem > 3.0:
        bx, bv = _polish(best_x, seconds=min(rem-1.0, 8.0), r_focus=None, deadline=deadline)
        if bv < best_val:
            best_x, best_val = bx, bv
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
