"""Hotspot island chains, modelled on the Hawaiian-Emperor chain.

A hotspot is fixed in the mantle; the plate slides over it, so every volcano
it builds is carried away and replaced by a new one. What that produces, and
what this module reproduces:
- a track that follows the plate's motion, curving where the plate rotates
  (the Hawaiian-Emperor bend);
- eruptions at irregular intervals, alternating between two parallel trends a
  little apart (Hawaii's Loa and Kea trends), so the chain is a band, not a
  single file;
- islands made of one to four overlapping shield volcanoes, separated from the
  next island by a channel of sea; each volcano has two or three rift
  zones (the arms that give Hawaiian islands their lobed, elongated shapes);
- ageing along the track: the youngest island, at the hotspot, is the biggest;
  older ones erode and subside, shrink, drown, and the oldest survive only as
  atolls (a ring of reef islets around a lagoon).
Returns a land-score contribution for stage 2.
"""
import numpy as np

from ..core.noise import fbm


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def hotspot_chains(n, km, hot, plate, vel, omega, cen_px, p, rng, seed):
    score = np.zeros((n, n), np.float32)
    rough = fbm(n, 4, 40, seed + 40)                      # irregular coasts on every volcano
    for y, x, pl in hot:
        pl = int(pl)
        pos = np.array([y, x], np.float64)
        travelled = 0.0
        length = p["chain_length_km"] * km
        trend = 1.0
        left_in_island = int(rng.integers(2, 5))            # volcanoes that build the current island
        while travelled <= length:
            age = travelled / max(length, 1e-6)                 # 0 at the hotspot, 1 at the far end
            # plate velocity here: translation + rotation about the plate centre
            r = (pos - cen_px[pl]) / n
            v = vel[pl] + omega[pl] * np.array([-r[1], r[0]])
            dirn = v / max(np.linalg.norm(v), 1e-9)
            side = np.array([-dirn[1], dirn[0]])
            R = p["young_island_km"] * km * (1.0 - 0.7 * age) * rng.uniform(0.7, 1.25)
            amp = 1.3 * (1.0 - p["aging"] * age)
            c = pos + side * trend * 0.35 * p["young_island_km"] * km
            trend = -trend
            _volcano(score, c, R, amp, age, dirn, p, rng, rough)
            left_in_island -= 1
            if left_in_island > 0:                          # next shield volcano of the same island
                step = p["eruption_spacing_km"] * km * rng.uniform(0.5, 1.0)
            else:                                           # a channel of sea, then a new island
                step = p["eruption_spacing_km"] * km * rng.uniform(2.2, 3.4)
                left_in_island = int(rng.integers(1, 5))
            pos = pos + dirn * step
            travelled += step
            if not (0 <= pos[0] < n and 0 <= pos[1] < n) or plate[int(pos[0]), int(pos[1])] != pl:
                break                                       # volcanoes ride their own plate only
    return score


def _volcano(score, c, R, amp, age, dirn, p, rng, rough):
    n = score.shape[0]
    reach = int(R * 2.2) + 2
    y0, y1 = max(int(c[0]) - reach, 0), min(int(c[0]) + reach + 1, n)
    x0, x1 = max(int(c[1]) - reach, 0), min(int(c[1]) + reach + 1, n)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dy, dx = yy - c[0], xx - c[1]
    r = np.hypot(dy, dx)
    th = np.arctan2(dy, dx)
    # rift zones: two or three arms, one roughly along the chain
    along = np.arctan2(dirn[0], dirn[1])
    arms = [along + rng.uniform(-0.4, 0.4), along + np.pi + rng.uniform(-0.4, 0.4)]
    if rng.random() < 0.6:
        arms.append(along + np.pi / 2 * rng.choice([-1, 1]) + rng.uniform(-0.3, 0.3))
    lobes = 1.0 + sum(p["rift_arms"] * np.exp(-_wrap(th - a) ** 2 / 0.12) for a in arms)
    rr = r / (R * lobes * (1.0 + 0.2 * rough[y0:y1, x0:x1]))
    if amp < 0.55 and p["atolls"] > 0:
        # drowned volcano: only its fringing reef survives, as a broken ring of islets
        ring = np.exp(-((rr - 0.9) / 0.12) ** 2) * (rough[y0:y1, x0:x1] > -0.3)
        score[y0:y1, x0:x1] = np.maximum(score[y0:y1, x0:x1], 0.75 * p["atolls"] * ring)
    else:
        score[y0:y1, x0:x1] = np.maximum(score[y0:y1, x0:x1], amp * np.exp(-2.0 * rr ** 2))
