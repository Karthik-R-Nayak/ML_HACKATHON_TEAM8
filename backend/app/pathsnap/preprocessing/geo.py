"""Projection (lon/lat -> metres) and planar geometry helpers."""
import numpy as np

EARTH_R = 6371008.8


class LocalProjector:
    """Spherical azimuthal-equidistant projection centred on (lon0, lat0).

    Distances/bearings from the centre are exact; error grows slowly with
    distance from the centre (well under 0.1% at city scale). No pyproj needed.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self._phi0 = np.radians(lat0)

    @classmethod
    def from_lonlat(cls, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        return cls((lon.min() + lon.max()) / 2, (lat.min() + lat.max()) / 2)

    def forward(self, lon, lat):
        lam = np.radians(np.asarray(lon, float) - self.lon0)
        phi = np.radians(np.asarray(lat, float))
        p0 = self._phi0
        cosc = np.clip(np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam), -1, 1)
        c = np.arccos(cosc)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(c < 1e-12, 1.0, c / np.sin(c))
        x = EARTH_R * k * np.cos(phi) * np.sin(lam)
        y = EARTH_R * k * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam))
        return x, y

    def inverse(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rho = np.hypot(x, y)
        c = rho / EARTH_R
        p0 = self._phi0
        with np.errstate(invalid="ignore", divide="ignore"):
            sinc, cosc = np.sin(c), np.cos(c)
            term = np.where(rho < 1e-9, 0.0, y * sinc * np.cos(p0) / np.where(rho == 0, 1, rho))
            phi = np.arcsin(np.clip(cosc * np.sin(p0) + term, -1, 1))
            lam = np.arctan2(x * sinc, rho * np.cos(p0) * cosc - y * np.sin(p0) * sinc)
        lam = np.where(rho < 1e-9, 0.0, lam)
        return self.lon0 + np.degrees(lam), np.degrees(phi)


def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass bearing (0=north, clockwise) of vector (dx, dy) in projected metres."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute angular difference in [0, 180]."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


def point_segment_projection(px, py, ax, ay, bx, by):
    """Distance from point(s) to segment(s) and clamped projection fraction t in [0,1]. Broadcasts."""
    px, py, ax, ay, bx, by = (np.asarray(v, float) for v in (px, py, ax, ay, bx, by))
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.clip(np.where(l2 > 0, t, 0.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t
