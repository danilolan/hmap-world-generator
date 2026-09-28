"""Stage 6 — climate (WORLDGEN.md section 2).

The world is one temperate, slightly cool climate band (medieval Europe, New
Zealand): there is no latitude gradient. The climate is modelled as if the map
were a compressed piece of a much larger world, so that every continent holds
a bit of everything (wet and dry sides, milder and colder parts) instead of one
region owning each climate.

- Wind. The long-term wind comes from pressure systems (highs and lows) whose
  size is a knob: the wind turns around them nearly along the isobars, with a
  little inflow toward the lows (friction). Different regions therefore get
  different prevailing directions, so each range casts its rain shadow its own
  way. An optional westerly bias adds a background drift. This is the climate
  average only; the wind a sailor feels should come from a live weather system
  in the game, not from this map.
- Local exposure: the Winstral sheltering index (steepest upwind angle,
  Winstral & Marks 2002) along the local wind direction.
- Precipitation. The Linear Theory of orographic precipitation (Smith &
  Barstad 2004), solved with FFTs for eight wind directions and blended by the
  local direction: heavy rain on windward slopes, drying in the lee. The air's
  moisture is then carried along the wind field (semi-Lagrangian, on a coarse
  grid), recharged over sea and lakes and used up as it rains: rain shadows and
  dry interiors, deserts where deep.
- Temperature. Sea-level baseline minus the altitude lapse rate, regional
  variation, and continentality (the seasonal range grows away from the sea).
- Aridity index (precipitation / potential evapotranspiration, UNEP classes).
"""
import numpy as np
from scipy import ndimage

from ..core import colormaps, raster
from ..core.noise import fbm
from ..core.params import Float
from ..core.stage import Stage

DIRECTIONS = 8


def linear_orographic_rain(h, dx, u, v, n_bv, hw, tau, cw):
    """Smith & Barstad (2004) linear model for a uniform wind (u, v) in m/s along the
    grid's columns and rows. Returns precipitation in mm/h, signed (negative =
    descending, drying air)."""
    ny, nx = h.shape
    pad = max(ny, nx) // 2
    hp = np.pad(h, pad)
    H = np.fft.rfft2(hp)
    kx = np.fft.rfftfreq(hp.shape[1], d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(hp.shape[0], d=dx) * 2 * np.pi
    K, L = np.meshgrid(kx, ky)
    sigma = u * K + v * L
    zero = np.abs(sigma) < 1e-12
    sig = np.where(zero, 1e-12, sigma)
    msq = (n_bv ** 2 - sig ** 2) / sig ** 2 * (K ** 2 + L ** 2)
    m = np.where(msq >= 0, np.sign(sig) * np.sqrt(np.abs(msq)), 1j * np.sqrt(np.abs(msq)))
    P = cw * 1j * sig * H / ((1 - 1j * m * hw) * (1 + 1j * sig * tau) ** 2)
    P[zero] = 0
    p = np.fft.irfft2(P, s=hp.shape)[pad:pad + ny, pad:pad + nx]
    return p * 3600.0


def wind_field(n, cell_m, p, seed):
    """Long-term wind from pressure systems: unit direction (dy, dx) per cell and a
    relative speed. Rows grow southward."""
    cells = max(1.0, (n * cell_m / 1000.0) / p["system_km"] * p.get("_scale", 1.0))
    pressure = fbm(n, 3, cells, seed + 3, 0.35)
    gy, gx = np.gradient(pressure)
    # down-gradient direction turned ~70° (flow nearly along the isobars, a little inflow)
    a = np.radians(70.0)
    fy, fx = -gy, -gx
    vy = np.cos(a) * fy - np.sin(a) * fx
    vx = np.sin(a) * fy + np.cos(a) * fx
    mag = np.hypot(vy, vx)
    mag = mag / max(np.mean(mag), 1e-9)
    # optional background drift (compass direction it comes from)
    to = np.radians(p["drift_from_deg"] + 180.0)
    vy = vy / max(np.mean(np.hypot(vy, vx)), 1e-9) + p["drift"] * -np.cos(to)
    vx = vx / max(np.mean(np.hypot(vy, vx)), 1e-9) + p["drift"] * np.sin(to)
    norm = np.maximum(np.hypot(vy, vx), 1e-9)
    speed = np.clip(0.4 + 0.6 * norm / max(np.mean(norm), 1e-9), 0.3, 1.8)
    return vy / norm, vx / norm, speed


def advect_moisture(potential, water, dy, dx, cell_km, depletion, evap, recycle=0.0, coarse=256):
    """Air moisture carried along the wind field until steady: rain uses it up,
    water restores it, and over land a share `recycle` of the rain returns to the
    air by evapotranspiration (moisture recycling: roughly half of the rain over the
    Amazon), so large continents do not dry out after a few hundred km. Semi-
    Lagrangian on a coarse grid (moisture is smooth), then returned at full
    resolution as a 0..1 factor."""
    n = potential.shape[0]
    f = min(1.0, coarse / n)
    m = max(8, int(round(n * f)))
    z = m / n
    pot = ndimage.zoom(potential, z, order=1)
    wat = ndimage.zoom(water.astype(np.float32), z, order=1)
    vy = ndimage.zoom(dy, z, order=1)
    vx = ndimage.zoom(dx, z, order=1)
    step_cells = 1.0
    step_km = cell_km / z * step_cells
    yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
    up = [yy - vy * step_cells, xx - vx * step_cells]
    moist = np.ones((m, m))
    for _ in range(int(m * 1.6)):
        mu = ndimage.map_coordinates(moist, up, order=1, cval=1.0)
        kept = 1.0 - recycle * (1.0 - wat)
        new = np.clip(mu - depletion * kept * pot * mu / 1000.0 * step_km, 0.0, 1.0)
        new = new + evap * wat * (1.0 - new) * step_km
        if np.max(np.abs(new - moist)) < 1e-4:
            moist = new
            break
        moist = new
    return np.clip(ndimage.zoom(moist, n / m, order=1)[:n, :n], 0.0, 1.0)


def sheltering(h, dy, dx, cell_m, search_m):
    """Winstral Sx along the local wind: steepest upward angle (degrees) looking upwind."""
    n = h.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    best = np.full(h.shape, -90.0)
    for s in range(1, max(1, int(search_m / cell_m)) + 1):
        up = ndimage.map_coordinates(h, [yy - dy * s, xx - dx * s], order=1, mode="nearest")
        best = np.maximum(best, np.degrees(np.arctan((up - h) / (s * cell_m))))
    return best


class Climate(Stage):
    id = "climate"
    title = "Climate"
    description = ("One temperate band, no latitude gradient. Winds from pressure systems, so each region has its own "
                   "prevailing direction; orographic rain and rain shadows; temperature from altitude, region and distance to the sea.")
    params = [
        Float("system_km", "Weather system size (km)", 22, 6, 80, 0.5,
              "Size of the high and low pressure systems that set the long-term wind: smaller = the wind direction "
              "changes over shorter distances, so every continent gets wet and dry sides"),
        Float("drift", "Westerly drift", 0.3, 0.0, 2.0, 0.01,
              "A background wind added to the pressure systems (0 = none; high = one direction dominates everywhere)"),
        Float("sea_temp_c", "Sea-level mean temperature (°C)", 9.5, 2, 16, 0.1,
              "Yearly mean at sea level: ~9–11 °C is a cool temperate climate"),
        Float("regional_c", "Regional temperature variation (°C)", 1.6, 0, 4, 0.05,
              "Milder and colder regions within every continent"),
        Float("seasonal_c", "Seasonal range at the coast (°C)", 11, 4, 25, 0.5, "Summer minus winter monthly mean by the sea"),
        Float("continental_c", "Extra seasonal range inland (°C per 10 km)", 2.5, 0, 8, 0.1,
              "Winters colder and summers warmer away from the sea"),
        Float("rain_mm", "Background rainfall (mm/year)", 850, 200, 2500, 10, "Rain over flat land where the air arrives from the sea"),
        Float("orographic", "Mountain rain", 1.0, 0.0, 3.0, 0.01, "How much extra rain mountains pull out of the wind"),
        Float("rain_shadow", "Rain shadow", 1.0, 0.0, 3.0, 0.01,
              "How fast the air dries as it rains: higher = drier lee sides and interiors (deserts where it is deep)"),
        Float("drift_from_deg", "Drift comes from (°)", 250, 0, 359, 1, "Compass direction of the background drift (270 = west)", advanced=True),
        Float("lapse_c_km", "Cooling with altitude (°C per km)", 6.5, 4, 9, 0.1, advanced=True),
        Float("wind_ms", "Wind speed (m/s)", 10, 3, 25, 0.5, "Background wind speed for the rain model", advanced=True),
        Float("rain_hours", "Hours of rain per year", 1100, 300, 3000, 10,
              "Converts the rain model's rate (mm/h) into a yearly total", advanced=True),
        Float("stability", "Air stability N (1/s)", 0.005, 0.001, 0.02, 0.0005, "Brunt-Väisälä frequency", advanced=True),
        Float("moist_layer_m", "Moist layer depth (m)", 2500, 500, 5000, 50, advanced=True),
        Float("cloud_time_s", "Cloud delay (s)", 1000, 200, 3000, 10,
              "Time to turn condensed water into rain: longer = rain spills further past the crest", advanced=True),
        Float("evaporation", "Moisture recharge over water (per km)", 0.08, 0.0, 0.5, 0.005, advanced=True),
        Float("lee_drying", "Lee drying limit", 0.6, 0.0, 1.0, 0.01,
              "Largest share of the background rain the descending air behind a range can remove (1 = all)",
              advanced=True),
        Float("recycle", "Moisture recycling over land", 0.5, 0.0, 0.9, 0.01,
              "Share of the rain over land that evaporates back into the air and rains again further on: keeps the "
              "interiors of large continents from drying out", advanced=True),
        Float("shelter_km", "Shelter search distance (km)", 1.2, 0.2, 5, 0.1, advanced=True),
    ]
    views = {"temperature": "Mean temperature", "winter": "Winter", "rain": "Rainfall", "aridity": "Aridity",
             "wind": "Wind (direction and exposure)"}

    def run(self, ctx, inputs, p):
        seed = ctx.stage_seed(self.id)
        n, cell = ctx.res, ctx.cell_m
        h = inputs["height_eroded"] if "height_eroded" in inputs else inputs["height_m"]
        land = inputs["land"]
        coast = np.maximum(inputs["coast_dist_km"], 0)
        hl = np.where(land, np.maximum(h, 0), 0.0).astype(np.float64)
        # the climate is that of the uncompressed world: its heights and distances
        hv = hl / ctx.vscale
        cell_v = cell * ctx.scale
        coast = coast * ctx.scale

        # wind from pressure systems
        dy, dx, speed = wind_field(n, cell, dict(p, _scale=ctx.scale), seed)

        # temperature
        t_mean = p["sea_temp_c"] - p["lapse_c_km"] * hv / 1000.0 + p["regional_c"] * fbm(n, 3, 4, seed + 1) * 0.8
        amp = p["seasonal_c"] + p["continental_c"] * np.minimum(coast, 25.0) / 10.0
        t_winter, t_summer = t_mean - amp / 2, t_mean + amp / 2

        # orographic rain for eight directions, blended by the local wind direction
        ang = np.arctan2(dy, dx)                                      # grid angle the wind blows toward
        oro = np.zeros_like(hl)
        step = 2 * np.pi / DIRECTIONS
        for k in range(DIRECTIONS):
            a = k * step
            w = np.clip(1.0 - np.abs(np.angle(np.exp(1j * (ang - a)))) / step, 0, 1)
            if w.max() <= 0:
                continue
            u, v = p["wind_ms"] * np.cos(a), p["wind_ms"] * np.sin(a)
            oro += w * linear_orographic_rain(hv, cell_v, u, v, p["stability"], p["moist_layer_m"], p["cloud_time_s"], 0.004)
        # the lee's descending air dries out the mountains' own rain but only part of the
        # background rain of the weather systems passing over (Smith & Barstad's lee term
        # alone would wipe it out behind every ridge, in razor-sharp strips)
        oro_mm = p["orographic"] * oro * speed * p["rain_hours"]
        potential = p["rain_mm"] + np.maximum(oro_mm, -p["lee_drying"] * p["rain_mm"])
        water = ~land | (inputs.get("lake_id", np.zeros_like(land, int)) > 0)
        moist = advect_moisture(potential, water, dy, dx, cell_v / 1000.0, 0.02 * p["rain_shadow"], p["evaporation"],
                                p["recycle"])
        # rain fields are smooth at the kilometre scale (storms drift, clouds spread)
        rain = ndimage.gaussian_filter(np.maximum(potential * moist, 50.0), max(1.0, 700.0 / cell))

        pet = 350.0 + 40.0 * np.clip(t_mean, 0, None) + 1.5 * np.clip(t_summer, 0, None) ** 2
        aridity = rain / pet

        sx = sheltering(hl, dy, dx, cell, ctx.km(p["shelter_km"]) * 1000.0)
        exposure = np.clip(1.0 - sx / 25.0, 0.2, 1.6) * (1.0 + 0.12 * hv / 1000.0) * speed
        return {"temp_mean": t_mean.astype(np.float32), "temp_winter": t_winter.astype(np.float32),
                "temp_summer": t_summer.astype(np.float32), "rain_mm": rain.astype(np.float32),
                "aridity": aridity.astype(np.float32), "wind_exposure": exposure.astype(np.float32),
                "wind_dy": dy.astype(np.float32), "wind_dx": dx.astype(np.float32), "climate_params": dict(p)}

    def render(self, view, ctx, data):
        land = data["land"]
        sea = np.array([40, 75, 125], np.uint8)
        if view == "temperature":
            img = colormaps.heat(data["temp_mean"], -8, 14)
        elif view == "winter":
            img = colormaps.heat(data["temp_winter"], -15, 10)
        elif view == "rain":
            img = colormaps.moisture(data["rain_mm"], 150, 2200)
        elif view == "aridity":
            cls = np.digitize(data["aridity"], [0.05, 0.2, 0.5, 0.65, 1.0])
            img = colormaps.categorical(cls, [(200, 150, 90), (220, 190, 120), (225, 215, 150), (170, 200, 120),
                                              (100, 170, 90), (50, 130, 110)])
        else:
            img = colormaps.gray(data["wind_exposure"], 0.2, 2.0)
        img = np.where(land[..., None], img, sea).astype(np.uint8)
        # wind arrows on a coarse grid
        n = land.shape[0]
        k = max(12, n // 16)
        L = k * 0.38
        w = max(1, n // 400)
        for cy in range(k // 2, n, k):
            for cx in range(k // 2, n, k):
                d = np.array([data["wind_dy"][cy, cx], data["wind_dx"][cy, cx]])
                a, b = np.array([cy, cx]) - d * L, np.array([cy, cx]) + d * L
                raster.draw_line(img, a[0], a[1], b[0], b[1], (255, 255, 255), w)
                for side in (1, -1):
                    wing = b - d * L * 0.45 + np.array([d[1], -d[0]]) * side * L * 0.3
                    raster.draw_line(img, b[0], b[1], wing[0], wing[1], (255, 255, 255), w)
        return img

    def stats(self, ctx, data):
        land = data["land"]
        if not land.any():
            return {}
        t, r, ai = data["temp_mean"][land], data["rain_mm"][land], data["aridity"][land]
        ang = np.degrees(np.arctan2(-data["wind_dy"][land], data["wind_dx"][land]))      # 0 = toward east, 90 = toward north
        sectors = np.histogram((ang + 360 + 22.5) % 360, bins=8, range=(0, 360))[0] / max(land.sum(), 1) * 100
        names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        top = ", ".join(f"toward {names[i]} {sectors[i]:.0f}%" for i in np.argsort(sectors)[::-1][:3])
        return {"mean temperature on land (°C)": f"{t.min():.1f} to {t.max():.1f}",
                "rain on land (mm/yr) min / median / max": f"{int(r.min())} / {int(np.median(r))} / {int(r.max())}",
                "dry land (aridity < 0.5) %": round(float((ai < 0.5).mean() * 100), 1),
                "desert (aridity < 0.2) %": round(float((ai < 0.2).mean() * 100), 1),
                "commonest wind directions": top}
