"""Parsers for .track / .nodes / .arcs / .route files.

ASSUMED formats (verify with scripts/inspect_samples.py on real files):
  .track : lon, lat, timestamp   (extra columns ignored)
  .nodes : lon, lat              (node id = row index + node_id_base)
  .arcs  : from_node_id, to_node_id (arc id = row index + arc_id_base)
  .route : arc ids, one or many per line, in chronological order
Delimiter may be comma, semicolon, tab or whitespace; one optional header line.
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IOConfig

log = logging.getLogger(__name__)


def _tokenise(path):
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"{path}: file is empty")
    first = lines[0]
    pat = next((d for d in (",", ";", "\t") if d in first), None)
    split = (lambda s: [t.strip() for t in s.split(pat)]) if pat else (lambda s: s.split())
    rows = [split(ln) for ln in lines]
    return rows


def _is_num(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _drop_header(rows, numeric_cols):
    if not all(_is_num(t) for t in rows[0][:numeric_cols]):
        return rows[1:]
    return rows


def _num_table(path, ncols, names):
    rows = _drop_header(_tokenise(path), ncols)
    bad = [i for i, r in enumerate(rows) if len(r) < ncols]
    if bad:
        raise ValueError(f"{path}: {len(bad)} rows have < {ncols} columns (first at data row {bad[0]})")
    if max(len(r) for r in rows) > ncols:
        log.warning("%s: extra columns ignored (expected %d)", path, ncols)
    arr = np.array([[float(t) for t in r[:ncols]] for r in rows])
    return pd.DataFrame(arr, columns=names)


def parse_nodes(path) -> pd.DataFrame:
    df = _num_table(path, 2, ["lon", "lat"])
    _check_lonlat(df, path)
    return df


def parse_arcs(path) -> pd.DataFrame:
    df = _num_table(path, 2, ["from_node", "to_node"])
    return df.astype("int64")


def parse_route(path) -> np.ndarray:
    rows = _tokenise(path)
    rows = _drop_header(rows, 1)
    return np.array([int(float(t)) for r in rows for t in r if t != ""], dtype=np.int64)


def parse_track(path, time_unit="auto") -> pd.DataFrame:
    rows = _tokenise(path)
    rows = _drop_header(rows, 2)
    if min(len(r) for r in rows) < 3:
        raise ValueError(f"{path}: need lon, lat, timestamp columns")
    lon = np.array([float(r[0]) for r in rows])
    lat = np.array([float(r[1]) for r in rows])
    tok = [r[2] for r in rows]
    if all(_is_num(t) for t in tok):
        t = np.array([float(x) for x in tok])
        unit = time_unit
        if unit == "auto":
            unit = "ms" if np.nanmedian(np.abs(t)) > 1e11 else "s"
        if unit == "ms":
            t = t / 1000.0
    else:
        dts = pd.to_datetime(pd.Series(tok), utc=True)  # unit-agnostic (pandas version differences)
        t = ((dts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    df = pd.DataFrame({"lon": lon, "lat": lat, "t": t, "orig_row": np.arange(len(rows))})
    _check_lonlat(df, path)
    if (np.diff(df["t"].to_numpy()) < 0).any():
        log.warning("%s: timestamps not monotone; stable-sorting by time", path)
        df = df.sort_values("t", kind="stable").reset_index(drop=True)
    return df


def _check_lonlat(df, path):
    if not (df["lat"].abs() <= 90).all() or not (df["lon"].abs() <= 180).all():
        raise ValueError(f"{path}: lon/lat out of range - columns swapped?")


def load_trip_files(track_path, route_path=None, io: IOConfig = IOConfig()):
    track = parse_track(track_path, io.time_unit)
    route = parse_route(route_path) if route_path and Path(route_path).exists() else None
    return track, route
