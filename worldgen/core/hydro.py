"""Flow routing and landscape processes on the preview grid (Numba).

- route(): priority-flood from the sea (Barnes et al. 2014): depression-filled
  surface, one steepest receiver per cell, and the flood order (every receiver
  before its donors).
- mfd_area() / sfd_area(): drainage area with multiple flow directions
  (Freeman 1991: no straight D8 grooves) or a single one (clean river lines).
- incise(): implicit stream-power incision (Braun & Willett 2013, n = 1).
- transport(): sediment carried downstream and deposited where the river can no
  longer carry it (capacity model after Davy & Lague 2009 / Yuan et al. 2019):
  flat valley floors, alluvial fans, deltas on the shelf.
- creep(): nonlinear hillslope diffusion (Roering et al. 1999): convex crests,
  planar slopes near the critical gradient.
- landslide(): slopes above the talus angle collapse (Musgrave et al. 1989).
- breach(): priority-flood with carving (Soille 2004): cut an outlet through the
  rim of a depression instead of filling it.
- discharge(): runoff accumulated downstream, with lakes that pool their inflow
  and spill only what evaporation leaves.
- height_above_drainage(): HAND (Nobre et al. 2011), for floodplains.
Heights in metres; `land` marks cells that evolve; sea cells are base level.
"""
import numpy as np
import numba as nb

DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)
DD = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356])


@nb.njit(cache=True)
def _push(hk, hv, size, k, v):
    i = size
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return size + 1


@nb.njit(cache=True)
def _pop(hk, hv, size):
    v = hv[0]
    size -= 1
    hk[0] = hk[size]
    hv[0] = hv[size]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= size:
            break
        c = l
        if l + 1 < size and hk[l + 1] < hk[l]:
            c = l + 1
        if hk[i] <= hk[c]:
            break
        hk[c], hk[i] = hk[i], hk[c]
        hv[c], hv[i] = hv[i], hv[c]
        i = c
    return v, size


@nb.njit(cache=True)
def route(h, land):
    n = h.shape[0]
    n2 = n * n
    hf = h.ravel().copy()
    rec = np.arange(n2)
    dist = np.ones(n2)
    order = np.empty(n2, np.int64)
    seen = np.zeros(n2, np.bool_)
    hk = np.empty(n2 + 8)
    hv = np.empty(n2 + 8, np.int64)
    size = 0
    cnt = 0
    for i in range(n):
        for j in range(n):
            if land[i, j] and i > 0 and j > 0 and i < n - 1 and j < n - 1:
                continue
            idx = i * n + j
            seen[idx] = True
            order[cnt] = idx
            cnt += 1
            for d in range(8):
                a = i + DI[d]
                b = j + DJ[d]
                if 0 <= a < n and 0 <= b < n and land[a, b] and not seen[a * n + b]:
                    size = _push(hk, hv, size, hf[idx], idx)
                    break
    while size > 0:
        c, size = _pop(hk, hv, size)
        ci = c // n
        cj = c % n
        for d in range(8):
            a = ci + DI[d]
            b = cj + DJ[d]
            if a < 0 or b < 0 or a >= n or b >= n:
                continue
            k = a * n + b
            if seen[k]:
                continue
            seen[k] = True
            if hf[k] <= hf[c] + 1e-4:
                hf[k] = hf[c] + 1e-4
            order[cnt] = k
            cnt += 1
            size = _push(hk, hv, size, hf[k], k)
    for t in range(cnt):
        c = order[t]
        ci = c // n
        cj = c % n
        if not land[ci, cj]:
            continue
        best = 0.0
        br = c
        bd = 1.0
        for d in range(8):
            a = ci + DI[d]
            b = cj + DJ[d]
            if a < 0 or b < 0 or a >= n or b >= n:
                continue
            k = a * n + b
            s = (hf[c] - hf[k]) / DD[d]
            if s > best:
                best = s
                br = k
                bd = DD[d]
        rec[c] = br
        dist[c] = bd
    return hf, rec, dist, order[:cnt]


@nb.njit(cache=True)
def sfd_area(rec, order, cell_area):
    a = np.full(rec.shape[0], cell_area)
    for t in range(order.shape[0] - 1, -1, -1):
        c = order[t]
        r = rec[c]
        if r != c:
            a[r] += a[c]
    return a


@nb.njit(cache=True)
def mfd_area(hf, order, land, cell_area, p):
    n = land.shape[0]
    a = np.full(hf.shape[0], cell_area)
    w = np.zeros(8)
    for t in range(order.shape[0] - 1, -1, -1):
        c = order[t]
        ci = c // n
        cj = c % n
        if not land[ci, cj]:
            continue
        tot = 0.0
        for d in range(8):
            w[d] = 0.0
            x = ci + DI[d]
            y = cj + DJ[d]
            if x < 0 or y < 0 or x >= n or y >= n:
                continue
            s = (hf[c] - hf[x * n + y]) / DD[d]
            if s > 0.0:
                w[d] = s ** p
                tot += w[d]
        if tot <= 0.0:
            continue
        for d in range(8):
            if w[d] > 0.0:
                a[(ci + DI[d]) * n + cj + DJ[d]] += a[c] * w[d] / tot
    return a


@nb.njit(cache=True)
def incise(h, hf, rec, dist, order, area, K, land, dx, m):
    """One implicit stream-power step; returns the eroded depth per cell.
    K already includes the time step. Cells inside a filled depression wait."""
    hh = h.ravel()
    Kf = K.ravel()
    lf = land.ravel()
    eroded = np.zeros(hh.shape[0])
    for t in range(order.shape[0]):
        c = order[t]
        if not lf[c]:
            continue
        r = rec[c]
        if r == c or hf[c] > hh[c] + 1e-3:
            continue
        f = Kf[c] * (area[c] / 1e6) ** m / (dist[c] * dx)
        new = (hh[c] + f * hh[r]) / (1.0 + f)
        if new < hh[c]:
            eroded[c] = hh[c] - new
            hh[c] = new
    return eroded


@nb.njit(cache=True)
def transport(h, rec, dist, order, area, eroded, land, dx, m, capacity, fan_slope, deposited, max_dep):
    """Carry the eroded material downstream; drop the excess over the river's
    capacity (capacity * area^m * slope per cell), never above a donor (no new
    pits) nor steeper than `fan_slope` over the receiver. Sediment reaching the
    sea builds the shelf in front of the mouth (deltas), up to 1 m below sea level."""
    n = land.shape[0]
    hh = h.ravel()
    lf = land.ravel()
    cell = dx * dx
    qs = np.zeros(hh.shape[0])
    min_donor = np.full(hh.shape[0], np.inf)
    for t in range(order.shape[0] - 1, -1, -1):
        c = order[t]
        if not lf[c]:
            continue
        qs[c] += eroded[c] * cell
        r = rec[c]
        if r == c:
            continue
        run = dist[c] * dx
        s = max(hh[c] - hh[r], 0.0) / run
        cap = capacity * (area[c] / 1e6) ** m * s * cell
        if qs[c] > cap:
            room = min(hh[r] + fan_slope * run, min_donor[c] - 1e-3) - hh[c]
            if room > 0.0:
                dep = min((qs[c] - cap) * 0.3 / cell, room, max_dep)
                hh[c] += dep
                deposited[c] += dep
                qs[c] -= dep * cell
        if lf[r]:
            qs[r] += qs[c]
        else:
            # delta: raise the sea floor at the mouth, but keep it under water
            room = -1.0 - hh[r]
            if room > 0.0:
                dep = min(qs[c] / cell, room)
                hh[r] += dep
                deposited[r] += dep
        if hh[c] < min_donor[r]:
            min_donor[r] = hh[c]


@nb.njit(cache=True)
def creep(h, kd, land, dx, sc, steps):
    """Nonlinear hillslope diffusion: flux = kd * S / (1 - (S/Sc)^2) across each
    cell face (explicit, `steps` sub-steps). kd per cell in m^2 per iteration."""
    n = h.shape[0]
    for _ in range(steps):
        dh = np.zeros_like(h)
        for i in range(n - 1):
            for j in range(n - 1):
                for di, dj in ((0, 1), (1, 0)):
                    a, b = i + di, j + dj
                    if not (land[i, j] and land[a, b]):
                        continue                 # soil creeps over land only, never into the sea
                    s = (h[i, j] - h[a, b]) / dx
                    r = min(abs(s) / sc, 0.9)
                    k = 0.5 * (kd[i, j] + kd[a, b])
                    q = k * s / (1.0 - r * r) / steps / dx
                    if land[i, j]:
                        dh[i, j] -= q
                    if land[a, b]:
                        dh[a, b] += q
        h += dh


@nb.njit(cache=True)
def landslide(h, talus_tan, land, dx, rate):
    """Material above the talus slope slides to the lower neighbours."""
    n = h.shape[0]
    delta = np.zeros_like(h)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if not land[i, j]:
                continue
            tot = 0.0
            mx = 0.0
            for d in range(8):
                diff = h[i, j] - h[i + DI[d], j + DJ[d]] - talus_tan[i, j] * DD[d] * dx
                if diff > 0.0:
                    tot += diff
                    mx = max(mx, diff)
            if tot <= 0.0:
                continue
            move = rate * mx * 0.5
            delta[i, j] -= move
            for d in range(8):
                a = i + DI[d]
                b = j + DJ[d]
                diff = h[i, j] - h[a, b] - talus_tan[i, j] * DD[d] * dx
                if diff > 0.0:
                    delta[a, b] += move * diff / tot
    return h + delta


@nb.njit(cache=True)
def breach(h, land, keep):
    """Priority-flood with carving: every land cell drains to the sea except the
    cells marked `keep` (lakes kept on purpose, already filled to their level)."""
    n = h.shape[0]
    n2 = n * n
    hh = h.ravel()
    kf = keep.ravel()
    parent = np.full(n2, -1, np.int64)
    seen = np.zeros(n2, np.bool_)
    hk = np.empty(n2 + 8)
    hv = np.empty(n2 + 8, np.int64)
    size = 0
    lf = land.ravel()
    for i in range(n):
        for j in range(n):
            if land[i, j] and i > 0 and j > 0 and i < n - 1 and j < n - 1:
                continue
            idx = i * n + j
            seen[idx] = True
            size = _push(hk, hv, size, hh[idx], idx)
    while size > 0:
        c, size = _pop(hk, hv, size)
        ci = c // n
        cj = c % n
        for d in range(8):
            a = ci + DI[d]
            b = cj + DJ[d]
            if a < 0 or b < 0 or a >= n or b >= n:
                continue
            k = a * n + b
            if seen[k]:
                continue
            seen[k] = True
            parent[k] = c
            if hh[k] <= hh[c] and not kf[k]:
                lvl = hh[k] - 1e-3
                p = c
                while p >= 0 and lf[p] and not kf[p] and hh[p] > lvl:
                    hh[p] = lvl
                    lvl -= 1e-3
                    p = parent[p]
            size = _push(hk, hv, size, hh[k], k)


@nb.njit(cache=True)
def discharge(rec, order, local, lake, outlet, lake_net):
    """Water accumulated downstream (any unit per cell, e.g. m³/year). `lake` holds a
    lake index per cell (-1 = none) and `outlet[k]` the cell through which lake k spills
    (its first flooded cell, so every other cell of the lake comes before it in reverse
    flood order). Lake cells pool their inflow; the outlet releases the pool plus
    `lake_net[k]` (rain on the lake minus evaporation from it), never below zero: a lake
    that evaporates more than it receives has no outflow. Returns (flow, lake inflow)."""
    q = local.copy()
    pool = np.zeros(lake_net.shape[0])
    for t in range(order.shape[0] - 1, -1, -1):
        c = order[t]
        k = lake[c]
        if k >= 0:
            if c != outlet[k]:
                pool[k] += q[c]
                q[c] = 0.0
                continue
            inflow = pool[k] + q[c]
            pool[k] = inflow
            q[c] = max(inflow + lake_net[k], 0.0)
        r = rec[c]
        if r != c:
            q[r] += q[c]
    return q, pool


@nb.njit(cache=True)
def height_above_drainage(h, rec, order, channel):
    """HAND (Nobre et al. 2011): each cell's height above the channel cell its water
    reaches first. Water that reaches the sea without a channel gets a large value:
    the sea is not a river, and low coastal land is not a floodplain."""
    base = np.zeros(h.shape[0])
    for t in range(order.shape[0]):
        c = order[t]
        r = rec[c]
        if channel[c]:
            base[c] = h[c]
        elif r == c:
            base[c] = min(h[c], 0.0) - 1000.0
        else:
            base[c] = base[r]
    return h - base
