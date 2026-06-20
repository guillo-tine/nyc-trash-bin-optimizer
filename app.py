"""
NYC Trash Bin Optimizer
=======================

What this app does, in one sentence:
    It finds city blocks that are busy with people but far from any existing
    trash bin, and suggests those blocks as good spots for a new bin.

The whole program is just four steps:

    1. LOAD   - read three kinds of public NYC data:
                  • activity data  (where people are - a stand-in for foot traffic)
                  • litter baskets (where bins already exist)
                  • DOT counts     (114 places with REAL measured pedestrian counts)

    2. SCORE  - chop the city into 250m squares ("cells"). Give each cell an
                "activity score" = how much activity happened inside it. Then for
                every cell, measure the distance to the nearest existing bin.

    3. FILTER - keep a cell as a SUGGESTION only if it is:
                  • busy enough        (activity above the Sensitivity cutoff), AND
                  • far from any bin    (farther than the Minimum-gap setting).

    4. RANK   - give each suggestion a Priority score (0-100) so a city planner
                knows which ones to build first, and draw everything on a map.

Read the functions top-to-bottom and they follow those four steps.
"""

import os
import io
import requests
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree          # fast "nearest point" lookup
from pyproj import Transformer             # converts lat/lon to meters

import streamlit as st
import folium
from streamlit_folium import st_folium


# ===========================================================================
# Settings (constants you can tweak)
# ===========================================================================
ACTIVITY_DATASET = "d6zx-ckhd"   # NYPD incidents (the original, weaker activity source)
BINS_DATASET     = "8znf-7b2c"   # DSNY Litter Basket Inventory
DOT_DATASET      = "cqsj-cfgu"   # DOT Bi-Annual Pedestrian Counts (114 measured locations)

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
ACTIVITY_CSV   = os.path.join(DATA_DIR, "pedestrian_counts.csv")
PROXY_GRID_CSV = os.path.join(DATA_DIR, "activity_grid.csv")     # the better composite data (built offline)
BINS_CSV       = os.path.join(DATA_DIR, "litter_baskets.csv")
DOT_CSV        = os.path.join(DATA_DIR, "dot_counts.csv")        # saved copy of the 114 DOT points
SUBWAY_CSV     = os.path.join(DATA_DIR, "subway_entrances.csv")  # subway entrances (transit signal)
INTERSECTIONS_CSV = os.path.join(DATA_DIR, "intersections.csv")  # named street corners (for snapping)
BUSINESS_CSV      = os.path.join(DATA_DIR, "businesses.csv")     # active food businesses (overlay)
SOCRATA_BASE   = "https://data.cityofnewyork.us/resource"

GRID_SIZE_M  = 250        # each cell is 250m x 250m
DATA_LIMIT   = 100_000    # how many rows to download from NYC Open Data
MAP_BIN_CAP  = 3000       # most existing-bin dots we draw (just for context)
MAP_SUGG_CAP = 300        # most suggestion dots we draw (the top ones)

# Colorblind-safe palette (Okabe-Ito) - distinguishable for all common color-vision types.
COLORS = {
    "bin":        "#0072B2",   # blue   - existing baskets
    "suggestion": "#E69F00",   # orange - suggested corners
    "misuse":     "#D55E00",   # vermillion ring - household-misuse risk
    "business":   "#56B4E9",   # sky    - businesses
    "dot":        "#555555",   # grey   - DOT counts
    "subway":     "#CC79A7",   # purple - subway entrances
    "move":       "#009E73",   # green  - relocation lines
}

# Rough rectangle (lat_min, lat_max, lon_min, lon_max) for each borough.
# Used to label a point with the borough it falls inside.
BOROUGH_BOUNDS = {
    "Manhattan":     (40.685, 40.882, -74.020, -73.907),
    "Brooklyn":      (40.570, 40.740, -74.042, -73.833),
    "Queens":        (40.490, 40.800, -73.962, -73.700),
    "Bronx":         (40.785, 40.918, -73.934, -73.765),
    "Staten Island": (40.477, 40.651, -74.260, -74.034),
}

# Where to center/zoom the map for each borough choice (lat, lon, zoom).
BOROUGH_CENTERS = {
    "All Boroughs":  (40.7128, -74.0060, 11),
    "Manhattan":     (40.783,  -73.964,  13),
    "Brooklyn":      (40.651,  -73.949,  13),
    "Queens":        (40.654,  -73.830,  12),
    "Bronx":         (40.845,  -73.865,  13),
    "Staten Island": (40.579,  -74.152,  13),
}

# A converter from lat/lon (degrees) to meters. We need meters so that
# "distance" and "250m grid cells" actually mean meters on the ground.
_TO_METERS = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ===========================================================================
# STEP 1 - Load the data
# ===========================================================================
def download_csv(dataset_id: str, out_path: str) -> None:
    """Download one dataset from NYC Open Data and save it as a CSV file."""
    url = f"{SOCRATA_BASE}/{dataset_id}.csv?$limit={DATA_LIMIT}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)


def ensure_data() -> None:
    """Make sure the activity and bins CSVs exist on disk (download if missing)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ACTIVITY_CSV):
        download_csv(ACTIVITY_DATASET, ACTIVITY_CSV)
    if not os.path.exists(BINS_CSV):
        download_csv(BINS_DATASET, BINS_CSV)


@st.cache_data(show_spinner=False)
def load_sources():
    """Load every activity source we have, plus the bins.

    Returns three things:
        proxy  - the good composite grid (or None if it hasn't been built yet)
        nypd   - the original NYPD incident table (or None)
        bins   - the existing litter baskets
    """
    if not os.path.exists(BINS_CSV):
        ensure_data()
    bins = pd.read_csv(BINS_CSV)

    # The composite grid is optional - it only exists after running build_proxy.py.
    proxy = pd.read_csv(PROXY_GRID_CSV) if os.path.exists(PROXY_GRID_CSV) else None

    # Only force-download the NYPD data if we have no composite grid to fall back on.
    if not os.path.exists(ACTIVITY_CSV) and proxy is None:
        ensure_data()
    nypd = pd.read_csv(ACTIVITY_CSV) if os.path.exists(ACTIVITY_CSV) else None

    return proxy, nypd, bins


def build_source_options(proxy, nypd):
    """Decide which choices appear in the "Activity data source" dropdown.

    Returns a dictionary: { menu label : (which table, which column) }.
    Only sources we actually have data for are included.
    """
    options = {}
    if proxy is not None:
        # The composite grid carries several columns; each can be used on its own.
        options["Composite (311 + transit)"] = ("proxy", "activity_score")

        # Per-person version: only offered if the population denominator (LODES) was
        # actually built into the grid. It scores activity-per-person instead of raw
        # volume, which corrects for crowd-size bias.
        has_pop = "lodes_pop" in proxy.columns and pd.to_numeric(proxy["lodes_pop"], errors="coerce").fillna(0).sum() > 0
        if "composite_perperson" in proxy.columns and has_pop:
            options["Composite (per person)"] = ("proxy", "composite_perperson")

        for label, column in [("Basket need (311 requests + overflow)", "score_basket_need"),
                              ("311 street complaints", "score_311"),
                              ("Subway ridership", "score_mta"),
                              ("Citibike trips", "score_citibike")]:
            # Only offer a column if it actually has data (sum > 0).
            if column in proxy.columns:
                values = pd.to_numeric(proxy[column], errors="coerce").fillna(0)
                if values.sum() > 0:
                    options[label] = ("proxy", column)
    if nypd is not None:
        options["NYPD incidents (911 calls)"] = ("nypd", None)
    return options


@st.cache_data(show_spinner=False)
def load_subway_entrances() -> pd.DataFrame:
    """The subway entrance points (lat/lon), bundled so the eligibility layer can
    show the 'near transit' signal. Returns an empty frame if the file is missing."""
    if os.path.exists(SUBWAY_CSV):
        df = pd.read_csv(SUBWAY_CSV)
        df["borough"] = assign_borough(df["lat"], df["lon"])
        return df
    return pd.DataFrame(columns=["lat", "lon", "borough"])


@st.cache_data(show_spinner=False)
def load_intersections() -> pd.DataFrame:
    """Named street corners (lat/lon/name) used to snap a 250m cell to a real corner.
    Bundled offline; returns empty if missing (snapping then falls back to cell centers)."""
    if os.path.exists(INTERSECTIONS_CSV):
        return pd.read_csv(INTERSECTIONS_CSV).dropna(subset=["lat", "lon"])
    return pd.DataFrame(columns=["lat", "lon", "name"])


@st.cache_data(show_spinner=False)
def load_businesses() -> pd.DataFrame:
    """Active food businesses (lat/lon/name) for the optional 'show businesses' overlay."""
    if os.path.exists(BUSINESS_CSV):
        df = pd.read_csv(BUSINESS_CSV).dropna(subset=["lat", "lon"])
        df["borough"] = assign_borough(df["lat"], df["lon"])
        return df
    return pd.DataFrame(columns=["lat", "lon", "name", "borough"])


def activity_input(options, choice, proxy, nypd):
    """Given the dropdown choice, return the table to feed into the pipeline.

    For a proxy column, we copy the grid and rename the chosen column to
    'activity_score' so the rest of the code can always look for that one name.
    """
    table, column = options[choice]
    if table == "nypd":
        return nypd
    grid = proxy.copy()
    grid["activity_score"] = pd.to_numeric(grid[column], errors="coerce")
    return grid


@st.cache_data(show_spinner=False)
def load_dot_counts() -> pd.DataFrame:
    """Load the 114 DOT pedestrian-count locations - the only REAL measured
    foot-traffic numbers NYC publishes. We use them as a reality check.

    These 114 points basically never change, so we download them ONCE and save
    a local copy (dot_counts.csv). After that the app reads the local file and
    never needs the internet again. ped_count = average of the 6 most recent
    counting sessions. On any failure we return an empty table so the app still runs.
    """
    # Fast path: use the saved local copy if we already have it.
    if os.path.exists(DOT_CSV):
        return pd.read_csv(DOT_CSV)

    try:
        response = requests.get(f"{SOCRATA_BASE}/{DOT_DATASET}.csv",
                                params={"$limit": 500}, timeout=60)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))

        # Columns that are NOT pedestrian counts (everything else is a count column).
        non_count = {"the_geom", "objectid", "loc", "borough",
                     "street_nam", "from_stree", "to_street", "iex"}
        count_cols = [c for c in raw.columns if c not in non_count]

        # The location is stored as text like "POINT (-73.98 40.75)"; pull out lon & lat.
        point = raw["the_geom"].astype(str).str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)")
        counts = raw[count_cols].apply(pd.to_numeric, errors="coerce")

        dot = pd.DataFrame({
            "lon":    pd.to_numeric(point[0], errors="coerce"),
            "lat":    pd.to_numeric(point[1], errors="coerce"),
            "street": raw.get("street_nam", ""),
        })
        # ped_count = mean of the 6 newest count columns; if those are blank, use all columns.
        dot["ped_count"] = counts[count_cols[-6:]].mean(axis=1, skipna=True)
        dot["ped_count"] = dot["ped_count"].fillna(counts.mean(axis=1, skipna=True))

        dot = dot.dropna(subset=["lat", "lon", "ped_count"])
        dot["borough"] = assign_borough(dot["lat"], dot["lon"])
        dot = dot.reset_index(drop=True)

        dot.to_csv(DOT_CSV, index=False)   # save for next time → offline from now on
        return dot
    except Exception:
        return pd.DataFrame(columns=["lat", "lon", "street", "ped_count", "borough"])


# ===========================================================================
# Small geometry helpers
# ===========================================================================
def find_column(df: pd.DataFrame, candidate_names: set):
    """Return the first column whose name (lowercased) is one of candidate_names."""
    for col in df.columns:
        if col.lower() in candidate_names:
            return col
    return None


def pick_lat_lon_columns(df: pd.DataFrame):
    """Figure out which columns hold latitude and longitude.

    Some files have plain 'latitude'/'longitude' columns. Others store the
    location as text like "POINT (-73.98 40.75)" in a single column. This
    handles both and returns (df, lat_column_name, lon_column_name).
    """
    LAT_NAMES = {"latitude", "lat", "y", "y_coord", "ycoordinate"}
    LON_NAMES = {"longitude", "lon", "lng", "x", "x_coord", "xcoordinate"}

    lat_col = find_column(df, LAT_NAMES)
    lon_col = find_column(df, LON_NAMES)
    if lat_col and lon_col:
        return df, lat_col, lon_col

    # No plain lat/lon columns - try a geometry/point column instead.
    geom_col = find_column(df, {"the_geom", "geom", "geometry", "location", "point"})
    if geom_col:
        text = df[geom_col].astype(str)
        df = df.copy()
        if text.str.contains("POINT", case=False, na=False).any():
            # WKT format: "POINT (lon lat)"  - note longitude comes first.
            parts = text.str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", expand=True)
            df["_lon"] = pd.to_numeric(parts[0], errors="coerce")
            df["_lat"] = pd.to_numeric(parts[1], errors="coerce")
        else:
            # Plain "(lat, lon)" format.
            parts = text.str.extract(r"\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?", expand=True)
            df["_lat"] = pd.to_numeric(parts[0], errors="coerce")
            df["_lon"] = pd.to_numeric(parts[1], errors="coerce")
        if df["_lat"].notna().any():
            return df, "_lat", "_lon"

    return df, None, None


def to_meters(lat, lon) -> np.ndarray:
    """Convert arrays of lat/lon (degrees) into (x, y) meters, so distances work."""
    x, y = _TO_METERS.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.column_stack([x, y])


def assign_borough(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Label each point with the borough whose rectangle it falls inside."""
    borough = pd.Series("Other", index=lat.index)
    for name, (lat_lo, lat_hi, lon_lo, lon_hi) in BOROUGH_BOUNDS.items():
        # Only fill in points we haven't labeled yet (first match wins).
        not_yet_set = borough == "Other"
        inside_box = lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi)
        borough[not_yet_set & inside_box] = name
    return borough


def clean_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Return a tidy two-column table of numeric lat/lon, dropping bad rows."""
    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })
    return out.dropna()


# ===========================================================================
# STEP 2 - Score every cell, and measure distance to the nearest bin
# ===========================================================================
@st.cache_data(show_spinner=False)
def prepare_candidates(ped_df: pd.DataFrame, bins_df: pd.DataFrame):
    """Turn raw data into a table of candidate cells.

    Output columns: lat, lon, borough, activity_score, nearest_bin_m
    (one row per 250m cell). Also returns the cleaned bins table.
    This is the slow part, so Streamlit caches its result.
    """
    # --- Clean up the existing bins and label each with its borough ---
    bins_df, bins_lat, bins_lon = pick_lat_lon_columns(bins_df)
    if bins_lat is None:
        raise ValueError("Cannot find latitude/longitude in the litter basket dataset.")
    bins = clean_latlon(bins_df, bins_lat, bins_lon)
    bins["borough"] = assign_borough(bins["lat"], bins["lon"])
    # Carry a few readable attributes so an existing basket can show its info on click.
    for src, dst in [("basketid", "basket_id"), ("baskettype", "basket_type"),
                     ("streetname1", "street1"), ("streetname2", "street2"),
                     ("location_description", "basket_loc")]:
        if src in bins_df.columns:
            bins[dst] = bins_df.loc[bins.index, src].values

    # --- Build the activity cells ---
    if "activity_score" in ped_df.columns and "lat" in ped_df.columns:
        # The composite grid is ALREADY one row per cell with a score - just clean it.
        cand = ped_df.copy()
        cand["lat"] = pd.to_numeric(cand["lat"], errors="coerce")
        cand["lon"] = pd.to_numeric(cand["lon"], errors="coerce")
        cand["activity_score"] = pd.to_numeric(cand["activity_score"], errors="coerce")
        cand = cand.dropna(subset=["lat", "lon", "activity_score"])
        if "borough" not in cand.columns:
            cand["borough"] = assign_borough(cand["lat"], cand["lon"])
    else:
        # Raw NYPD points - count how many fall into each 250m cell.
        ped_df, ped_lat, ped_lon = pick_lat_lon_columns(ped_df)
        if ped_lat is None:
            raise ValueError("Cannot find latitude/longitude in the activity dataset.")
        points = clean_latlon(ped_df, ped_lat, ped_lon)

        # Convert to meters and snap each point to a grid cell (gx, gy).
        xy = to_meters(points["lat"].values, points["lon"].values)
        points["gx"] = np.floor(xy[:, 0] / GRID_SIZE_M).astype(int)
        points["gy"] = np.floor(xy[:, 1] / GRID_SIZE_M).astype(int)

        # One row per cell: score = number of points, position = their average.
        cand = (
            points.groupby(["gx", "gy"])
            .agg(activity_score=("lat", "size"),
                 lat=("lat", "mean"),
                 lon=("lon", "mean"))
            .reset_index(drop=True)
        )
        cand["borough"] = assign_borough(cand["lat"], cand["lon"])

    # --- For every cell, find the distance to the nearest existing bin ---
    # cKDTree builds a fast index of the bins, then answers "nearest" in one shot.
    if len(bins):
        bin_xy = to_meters(bins["lat"].values, bins["lon"].values)
        cand_xy = to_meters(cand["lat"].values, cand["lon"].values)
        distances, _ = cKDTree(bin_xy).query(cand_xy, k=1)
        cand["nearest_bin_m"] = distances
    else:
        cand["nearest_bin_m"] = np.inf

    # Walking distance (precomputed offline over the street network) is the operative
    # "nearest bin" distance when the grid carries it; straight-line is kept for reference.
    if "nearest_bin_walk_m" in cand.columns:
        cand["nearest_bin_straight_m"] = cand["nearest_bin_m"]
        cand["nearest_bin_m"] = pd.to_numeric(
            cand["nearest_bin_walk_m"], errors="coerce").fillna(cand["nearest_bin_m"])

    keep = ["lat", "lon", "borough", "activity_score", "nearest_bin_m"]
    # Carry optional grid columns through if the grid has them (composite sources):
    # eligibility signals, sanitation district / BID tags, businesses, and basket-need.
    for extra in ("eligible", "near_transit", "near_bus", "commercial_area",
                  "district", "in_bid", "business_count", "score_basket_need",
                  "proxy_divergence", "nearest_bin_straight_m"):
        if extra in cand.columns:
            keep.append(extra)
    return cand[keep].reset_index(drop=True), bins.reset_index(drop=True)


# ===========================================================================
# STEP 3 - Filter cells down to suggestions
# ===========================================================================
def suggest_new_bins(cand: pd.DataFrame, threshold_index: int,
                     min_distance_m: float, borough: str,
                     eligible_only: bool = False) -> pd.DataFrame:
    """Keep only the cells that are busy enough AND far enough from a bin.

    If eligible_only is True and the grid carries an 'eligible' column, the DSNY
    eligibility gate (commercial/transit land use) is applied FIRST, before scoring.
    """
    # If a borough is chosen, look only at that borough's cells.
    cells = cand if borough == "All Boroughs" else cand[cand["borough"] == borough]

    # DSNY eligibility gate: a hard filter applied before the busy/far tests.
    if eligible_only and "eligible" in cells.columns:
        cells = cells[cells["eligible"]]

    if cells.empty:
        return cells.assign(activity_index=pd.Series(dtype=int))

    cells = cells.copy()
    # Turn the raw score into a 0-100 "activity index" = how this cell ranks
    # against the others in view. (rank(pct=True) gives 0-1, so we x100.)
    # Ranking within the chosen borough is what keeps Queens judged fairly.
    cells["activity_index"] = (cells["activity_score"].rank(pct=True) * 100).round().astype(int)

    # The two rules: busy enough, and far enough from an existing bin.
    busy_enough = cells["activity_index"] >= threshold_index
    far_enough  = cells["nearest_bin_m"] >= min_distance_m
    suggestions = cells[busy_enough & far_enough]

    # Show the busiest, most-underserved ones first.
    return suggestions.sort_values(["activity_index", "nearest_bin_m"], ascending=[False, False])


# ===========================================================================
# STEP 4 - Rank suggestions by priority (which to build first)
# ===========================================================================
def compute_priority(sugg: pd.DataFrame, dot_df: pd.DataFrame | None = None,
                     use_commercial: bool = False) -> pd.DataFrame:
    """Give each suggestion a 0-100 Priority score and sort by it.

    Priority blends:
        • activity    - how busy the cell is          (the activity index, 0-100)
        • gap         - how far the nearest bin is     (ranked 0-100 among suggestions)
        • dot         - near a high REAL DOT count?    (a small confidence bonus)
        • commercial  - how much commercial floor area (DSNY's waste-generation logic)

    The weights shift depending on which signals are available:
        activity + gap + dot + commercial:  0.45 / 0.30 / 0.10 / 0.15
        activity + gap + dot:               0.55 / 0.35 / 0.10
        activity + gap + commercial:        0.50 / 0.35 / 0.15
        activity + gap:                     0.60 / 0.40
    Weights are a deliberate design choice, not a derived optimum.
    """
    if sugg.empty:
        return sugg.assign(priority=pd.Series(dtype=int), dot_verified=pd.Series(dtype=bool))

    df = sugg.copy()
    activity = df["activity_index"].astype(float)              # already 0-100
    gap = df["nearest_bin_m"].rank(pct=True) * 100             # 0-100 within the suggestions

    have_dot = dot_df is not None and len(dot_df)
    if have_dot:
        # Find the nearest DOT count location to each suggestion.
        dot_xy = to_meters(dot_df["lat"].values, dot_df["lon"].values)
        sugg_xy = to_meters(df["lat"].values, df["lon"].values)
        distance_to_dot, nearest_index = cKDTree(dot_xy).query(sugg_xy, k=1)
        # Rank the DOT counts 0-100; only counts if a DOT point is within 500m.
        dot_rank = (dot_df["ped_count"].rank(pct=True) * 100).values
        dot_bonus = np.where(distance_to_dot <= 500, dot_rank[nearest_index], 0.0)
        df["dot_verified"] = distance_to_dot <= 500
    else:
        df["dot_verified"] = False
        dot_bonus = 0.0

    # Commercial term = how this spot's commercial floor area ranks among the suggestions.
    # This is DSNY's "businesses generate disposable waste" logic, applied to ordering.
    have_comm = use_commercial and "commercial_area" in df.columns
    commercial = (df["commercial_area"].rank(pct=True) * 100) if have_comm else 0.0

    if have_dot and have_comm:
        score = 0.45 * activity + 0.30 * gap + 0.10 * dot_bonus + 0.15 * commercial
    elif have_dot:
        score = 0.55 * activity + 0.35 * gap + 0.10 * dot_bonus
    elif have_comm:
        score = 0.50 * activity + 0.35 * gap + 0.15 * commercial
    else:
        score = 0.60 * activity + 0.40 * gap

    df["priority"] = score.round().astype(int)
    return df.sort_values("priority", ascending=False).reset_index(drop=True)


# ===========================================================================
# STEP 4b - Make the output usable for DSNY (corners, baskets, risk, relocation)
# ===========================================================================
def snap_to_corners(sugg: pd.DataFrame, inter_df: pd.DataFrame,
                    max_m: float = 180.0) -> pd.DataFrame:
    """Snap each suggestion to the nearest real street corner within max_m.

    DSNY places baskets on a named corner, not in a 250m square. Adds corner_lat,
    corner_lon, corner_name. If no corner is close enough, it keeps the cell center.
    """
    out = sugg.copy()
    if out.empty:
        out["corner_lat"] = []; out["corner_lon"] = []; out["corner_name"] = []
        return out
    corner_lat = out["lat"].to_numpy(dtype=float).copy()
    corner_lon = out["lon"].to_numpy(dtype=float).copy()
    corner_name = np.array([""] * len(out), dtype=object)
    if inter_df is not None and not inter_df.empty:
        inter_xy = to_meters(inter_df["lat"].values, inter_df["lon"].values)
        sugg_xy = to_meters(out["lat"].values, out["lon"].values)
        dist, idx = cKDTree(inter_xy).query(sugg_xy, k=1)
        hit = dist <= max_m
        lat_a = inter_df["lat"].values; lon_a = inter_df["lon"].values
        name_a = inter_df["name"].astype(str).values
        corner_lat[hit] = lat_a[idx[hit]]
        corner_lon[hit] = lon_a[idx[hit]]
        corner_name[hit] = name_a[idx[hit]]
    out["corner_lat"] = corner_lat
    out["corner_lon"] = corner_lon
    out["corner_name"] = corner_name
    return out


def enrich_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Add the planner-facing fields: how many baskets, and household-misuse risk.

    • recommended_baskets - busier corners (and high basket-need) warrant more than one.
    • misuse_risk - residential / non-eligible / no commercial floor area. These are the
      blocks where baskets get filled with household trash and DSNY ends up removing them.
    """
    if df.empty:
        return df
    df = df.copy()
    ai = df["activity_index"].to_numpy(dtype=float)
    rec = np.ones(len(df), dtype=int)
    rec[ai >= 60] = 2
    rec[ai >= 80] = 3
    df["recommended_baskets"] = rec

    elig = df["eligible"].astype(bool).to_numpy() if "eligible" in df.columns else np.ones(len(df), bool)
    comm = df["commercial_area"].to_numpy(dtype=float) if "commercial_area" in df.columns else np.ones(len(df))
    df["misuse_risk"] = (~elig) | (comm <= 0)

    # Confidence: proxy_divergence is how much the activity sources DISAGREE for a cell.
    # The most divergent quarter of suggestions is flagged as a fragile estimate.
    if "proxy_divergence" in df.columns:
        pv = pd.to_numeric(df["proxy_divergence"], errors="coerce").fillna(0)
        thr = pv.quantile(0.75)
        df["low_confidence"] = (pv > thr) & (thr > 0)
    else:
        df["low_confidence"] = False
    return df


def removable_bins(bins_df: pd.DataFrame, cand_df: pd.DataFrame) -> pd.DataFrame:
    """Existing bins that are good candidates to MOVE (so a new corner costs nothing).

    A bin is 'removable' when its cell is NOT DSNY-eligible (residential / parks /
    industrial) AND it is either redundant (another bin within 150m) or in a very
    low-activity cell. This is conservative on purpose.
    """
    if bins_df.empty or "eligible" not in cand_df.columns:
        return bins_df.iloc[0:0]
    cell_xy = to_meters(cand_df["lat"].values, cand_df["lon"].values)
    bin_xy = to_meters(bins_df["lat"].values, bins_df["lon"].values)
    _, ci = cKDTree(cell_xy).query(bin_xy, k=1)
    elig = cand_df["eligible"].astype(bool).to_numpy()[ci]
    act = cand_df["activity_score"].to_numpy(dtype=float)[ci]
    act_rank = pd.Series(act).rank(pct=True).to_numpy() * 100
    nn = cKDTree(bin_xy).query(bin_xy, k=2)[0][:, 1]   # distance to the nearest OTHER bin
    b = bins_df.copy()
    b["near_other_bin_m"] = nn
    removable = (~elig) & ((nn <= 150) | (act_rank <= 25))
    return b[removable].reset_index(drop=True)


def pair_relocations(suggested: pd.DataFrame, removable: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Pair each of the top-N suggested corners with the nearest removable bin:
    a concrete 'move this bin to here, net cost zero' list."""
    if suggested.empty or removable.empty:
        return pd.DataFrame()
    top = suggested.head(n)
    rem_xy = to_meters(removable["lat"].values, removable["lon"].values)
    tgt_xy = to_meters(top["corner_lat"].values, top["corner_lon"].values)
    d, idx = cKDTree(rem_xy).query(tgt_xy, k=1)
    return pd.DataFrame({
        "Priority":          top["priority"].to_numpy(),
        "Move to":           top["corner_name"].to_numpy(),
        "To lat":            np.round(top["corner_lat"].to_numpy(), 5),
        "To lon":            np.round(top["corner_lon"].to_numpy(), 5),
        "From lat":          np.round(removable["lat"].to_numpy()[idx], 5),
        "From lon":          np.round(removable["lon"].to_numpy()[idx], 5),
        "Move distance (m)": np.round(d, 0).astype(int),
    })


@st.cache_data(show_spinner=False)
def calibrated_min_gap(proxy, bins_raw, fallback: int = 300) -> int:
    """Derive the default minimum bin spacing from the REAL median spacing between
    existing DSNY baskets that sit in commercial cells - instead of an invented number.
    Returns a value rounded to the nearest 25m, clamped to the slider's range.
    """
    try:
        if proxy is None or "commercial_area" not in proxy.columns:
            return fallback
        b, la, lo = pick_lat_lon_columns(bins_raw)
        bins = clean_latlon(b, la, lo)
        comm = proxy[pd.to_numeric(proxy["commercial_area"], errors="coerce").fillna(0) > 0]
        if bins.empty or comm.empty:
            return fallback
        comm_xy = to_meters(comm["lat"].values, comm["lon"].values)
        bin_xy = to_meters(bins["lat"].values, bins["lon"].values)
        dist_to_comm, _ = cKDTree(comm_xy).query(bin_xy, k=1)
        in_comm = bin_xy[dist_to_comm <= 180]          # baskets in/next to a commercial cell
        if len(in_comm) < 50:
            return fallback
        nn = cKDTree(in_comm).query(in_comm, k=2)[0][:, 1]
        med = float(np.median(nn))
        return int(min(800, max(25, round(med / 25) * 25)))
    except Exception:
        return fallback


def defend_html(row) -> str:
    """The 'why this corner' panel: a few plain-English facts a planner can paste
    into a memo. Works on a namedtuple row (uses getattr with safe defaults)."""
    pr = int(getattr(row, "priority", 50))
    ai = int(getattr(row, "activity_index", 0))
    gap_m = float(getattr(row, "nearest_bin_m", 0))
    corner = str(getattr(row, "corner_name", "") or "")
    head = f"<b>Suggested new bin</b>"
    if corner:
        head += f"<br/><b>{corner}</b>"
    straight = getattr(row, "nearest_bin_straight_m", None)
    walked = straight is not None and not pd.isna(straight)
    dist_label = "Walking distance to nearest bin" if walked else "Nearest existing bin"
    parts = [
        head,
        f"Priority: <b>{pr}</b>/100",
        f"Busier than <b>{ai}%</b> of corners in view",
        f"{dist_label}: <b>{gap_m:.0f} m</b> ({gap_m * 3.281:.0f} ft) away",
    ]
    bc = getattr(row, "business_count", None)
    if bc is not None and not pd.isna(bc):
        parts.append(f"Businesses on this block: <b>{int(bc)}</b>")
    bn = getattr(row, "score_basket_need", None)
    if bn is not None and not pd.isna(bn) and float(bn) > 0:
        parts.append(f"311 basket-service requests here: <b>{int(float(bn))}</b>")
    rb = getattr(row, "recommended_baskets", None)
    if rb is not None and not pd.isna(rb):
        parts.append(f"Recommended baskets: <b>{int(rb)}</b>")
    elig = getattr(row, "eligible", None)
    if elig is not None:
        risk = bool(getattr(row, "misuse_risk", False))
        parts.append("<hr style='margin:4px 0'>"
                     f"DSNY-eligible: <b>{'Yes' if bool(elig) else 'No'}</b>")
        parts.append(f"Household-misuse risk: <b>{'High' if risk else 'Low'}</b>")
    dist = str(getattr(row, "district", "") or "")
    if dist:
        in_bid = bool(getattr(row, "in_bid", False))
        parts.append(f"Sanitation district: <b>{dist}</b>"
                     + (" &middot; <b>inside a BID</b>" if in_bid else ""))
    if bool(getattr(row, "low_confidence", False)):
        parts.append("<i>Lower confidence: the activity sources disagree here.</i>")
    return "<br/>".join(parts)


def build_report_html(table: pd.DataFrame, district: str, borough: str,
                      source: str, gap_m: float) -> str:
    """A printable one-page shortlist (HTML). A planner opens it and uses the browser's
    Print -> Save as PDF, so we add zero PDF dependencies."""
    scope = district if district != "All districts" else borough
    head = "".join(f"<th>{c}</th>" for c in table.columns)
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in rec) + "</tr>"
        for rec in table.itertuples(index=False)
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>DSNY Litter Basket Shortlist - {scope}</title>
<style>
 body{{font-family:Arial,Helvetica,sans-serif;margin:32px;color:#111}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{color:#555;font-size:12px;margin-bottom:16px}}
 table{{border-collapse:collapse;width:100%;font-size:12px}}
 th,td{{border:1px solid #ccc;padding:4px 6px;text-align:left}} th{{background:#f2f2f2}}
</style></head><body>
<h1>NYC Litter Basket Shortlist - {scope}</h1>
<div class='sub'>Activity source: {source}. Minimum gap from an existing bin: {int(gap_m)} m.
Ranked by priority (busyness + coverage gap + commercial activity). {len(table)} corners.</div>
<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
<p style='margin-top:16px;color:#777;font-size:11px'>Generated by the NYC Trash Bin Optimizer.
Suggestions are candidates for a planner to confirm.</p>
</body></html>"""


def scope_to_borough(df: pd.DataFrame, borough: str) -> pd.DataFrame:
    """Filter a lat/lon table to one borough (or return it whole for All Boroughs)."""
    if borough == "All Boroughs" or "borough" not in df.columns:
        return df
    return df[df["borough"] == borough]


def to_geojson(table: pd.DataFrame) -> str:
    """Export the priority list as GeoJSON points so DSNY can drop it straight into GIS."""
    import json
    feats = []
    for d in table.to_dict("records"):            # preserves exact column names
        lon = float(d.pop("Longitude")); lat = float(d.pop("Latitude"))
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": {k: (None if pd.isna(v) else v) for k, v in d.items()}})
    return json.dumps({"type": "FeatureCollection", "features": feats})


@st.cache_data(show_spinner=False)
def district_scorecard(proxy, bins_raw) -> pd.DataFrame:
    """Per-district coverage stats DSNY can put in a budget case: baskets, population,
    commercial corners, baskets per 1,000 residents and per commercial corner, plus a
    citywide rank (1 = best covered per resident). Indexed by district code."""
    if proxy is None or "district" not in proxy.columns:
        return pd.DataFrame()
    try:
        g = proxy.copy()
        g["comm_corner"] = g["eligible"].astype(bool) & (g["commercial_area"] > 0)
        agg = g[g["district"] != ""].groupby("district").agg(
            population=("lodes_pop", "sum"),
            commercial_corners=("comm_corner", "sum")).reset_index()

        b, la, lo = pick_lat_lon_columns(bins_raw)
        bins = clean_latlon(b, la, lo)
        cells = g.dropna(subset=["lat", "lon"])
        _, ci = cKDTree(to_meters(cells["lat"].values, cells["lon"].values)).query(
            to_meters(bins["lat"].values, bins["lon"].values), k=1)
        bin_dist = pd.Series(cells["district"].to_numpy()[ci]).value_counts()
        agg["baskets"] = agg["district"].map(bin_dist).fillna(0).astype(int)

        agg["per_1k_residents"] = agg["baskets"] / (agg["population"] / 1000).replace(0, np.nan)
        agg["per_commercial_corner"] = agg["baskets"] / agg["commercial_corners"].replace(0, np.nan)
        # Rank 1 = fewest baskets per resident (most underserved) - where DSNY should look.
        agg["underserved_rank"] = agg["per_1k_residents"].rank(method="min").astype("Int64")
        return agg.set_index("district")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_validation(proxy, bins_raw, dot_df) -> dict:
    """Two honest, explainable checks of the placement logic:

      • recovery - of the city's existing DSNY baskets, what share sit in cells the model
                   would INDEPENDENTLY call basket-worthy (busy AND DSNY-eligible)? High =
                   the logic agrees with where DSNY already chose to put baskets.
      • dot_corr - how well does our activity rank track the 114 REAL DOT pedestrian
                   counts (Spearman correlation)?

    Returns a dict; values are None when a check can't be run.
    """
    res = {"recovery": None, "n_bins": 0, "dot_corr": None, "n_dot": 0,
           "holdout": None, "n_holdout": 0}
    try:
        if proxy is None or "eligible" not in proxy.columns:
            return res
        cells = proxy.dropna(subset=["lat", "lon", "activity_score"]).copy()
        cells["ai"] = cells["activity_score"].rank(pct=True) * 100
        cell_xy = to_meters(cells["lat"].values, cells["lon"].values)
        tree = cKDTree(cell_xy)

        b, la, lo = pick_lat_lon_columns(bins_raw)
        bins = clean_latlon(b, la, lo)
        if not bins.empty:
            _, bi = tree.query(to_meters(bins["lat"].values, bins["lon"].values), k=1)
            elig = cells["eligible"].astype(bool).to_numpy()
            busy = cells["ai"].to_numpy() >= 50
            res["recovery"] = float((elig[bi] & busy[bi]).mean())
            res["n_bins"] = int(len(bins))

            # Hold-out test: hide 20% of bins. Among the hidden ones whose removal actually
            # creates a gap (no retained bin within 200 m), what share sit where the model
            # would independently call for a basket (busy AND eligible)? That is the real
            # predictive question - and it isn't dragged down by how densely baskets cluster.
            if len(bins) > 50:
                rng = np.random.default_rng(0)
                test_mask = rng.random(len(bins)) < 0.2
                train, test = bins[~test_mask], bins[test_mask]
                if len(train) and len(test):
                    test_xy = to_meters(test["lat"].values, test["lon"].values)
                    tnn = cKDTree(to_meters(train["lat"].values, train["lon"].values)).query(test_xy, k=1)[0]
                    iso = test_xy[tnn >= 200.0]              # removing these makes a real gap
                    if len(iso):
                        ci = tree.query(iso, k=1)[1]         # their cells
                        res["holdout"] = float((busy[ci] & elig[ci]).mean())
                        res["n_holdout"] = int(len(iso))

        if dot_df is not None and len(dot_df):
            _, di = tree.query(to_meters(dot_df["lat"].values, dot_df["lon"].values), k=1)
            from scipy.stats import spearmanr
            r, _ = spearmanr(cells["ai"].to_numpy()[di], dot_df["ped_count"].to_numpy())
            res["dot_corr"] = float(r) if r == r else None   # r==r filters out NaN
            res["n_dot"] = int(len(dot_df))
    except Exception:
        pass
    return res


# ===========================================================================
# Drawing the map
# ===========================================================================
def thin_points(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Reduce a big set of dots to at most `cap`, spread evenly across the map.

    We first drop near-duplicates (rounded to ~100m), then keep every Nth row.
    This keeps the picture representative instead of randomly clumpy.
    """
    if len(df) <= cap:
        return df
    cell_key = df["lat"].round(3).astype(str) + "," + df["lon"].round(3).astype(str)
    df = df[~cell_key.duplicated()]
    if len(df) > cap:
        step = int(np.ceil(len(df) / cap))
        df = df.iloc[::step]
    return df


def make_map(bins_df: pd.DataFrame, suggested_df: pd.DataFrame, borough: str,
             dot_df: pd.DataFrame | None = None,
             cells_df: pd.DataFrame | None = None, show_all: bool = False,
             show_eligibility: bool = False, entrances_df: pd.DataFrame | None = None,
             businesses_df: pd.DataFrame | None = None,
             relocations_df: pd.DataFrame | None = None,
             center: tuple | None = None) -> folium.Map:
    """Build the Folium map: blue DOT circles (optional), red bins, green suggestions.

    When show_all is True ("Visualize everything"):
      • draw an activity HEATMAP of every cell (the input the whole tool is built on)
      • draw ALL existing bins and ALL suggestions (caps lifted)
    When show_eligibility is True:
      • draw a green heatmap of commercial floor area (the "businesses" signal)
      • draw subway entrances as purple dots (the "near transit" signal)
    """
    center_lat, center_lon, zoom = BOROUGH_CENTERS.get(borough, BOROUGH_CENTERS["All Boroughs"])
    if center is not None:                    # a search match overrides the borough center
        center_lat, center_lon, zoom = center
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="cartodbpositron")

    # Activity heatmap (drawn first, underneath everything) - shows the raw activity
    # surface that feeds the whole model. Each cell is weighted by its activity score.
    if show_all and cells_df is not None and len(cells_df):
        from folium.plugins import HeatMap
        scores = cells_df["activity_score"].astype(float)
        top = scores.max() or 1.0
        heat_points = [[row.lat, row.lon, float(w / top)]
                       for row, w in zip(cells_df.itertuples(), scores)]
        HeatMap(heat_points, radius=12, blur=15, min_opacity=0.25).add_to(fmap)

    # Eligibility layer: show WHY cells qualify under DSNY rules.
    #   green heat = commercial floor area (where the businesses are)
    #   purple dots = subway entrances (the "near transit" signal)
    if show_eligibility and cells_df is not None and "commercial_area" in cells_df.columns:
        commercial = cells_df[cells_df["commercial_area"] > 0]
        if len(commercial):
            from folium.plugins import HeatMap
            top = float(commercial["commercial_area"].max()) or 1.0
            pts = [[r.lat, r.lon, float(r.commercial_area / top)] for r in commercial.itertuples()]
            HeatMap(pts, radius=14, blur=18, min_opacity=0.2,
                    gradient={0.2: "#c8f7c5", 0.5: "#46c06a", 1.0: "#0a7d2c"}).add_to(fmap)
    if show_eligibility and entrances_df is not None and len(entrances_df):
        for e in entrances_df.itertuples():
            folium.CircleMarker(
                [e.lat, e.lon], radius=3, color=COLORS["subway"], weight=0, fill=True, fill_opacity=0.7,
                popup="Subway entrance (a 'near transit' signal for eligibility)",
            ).add_to(fmap)

    # Businesses overlay: a literal "where the businesses are" layer (orange dots),
    # so a planner can see what drives a commercial score. Thinned for speed.
    if businesses_df is not None and len(businesses_df):
        for biz in thin_points(businesses_df, 1500).itertuples():
            folium.CircleMarker(
                [biz.lat, biz.lon], radius=2, color=COLORS["business"], weight=0,
                fill=True, fill_opacity=0.5,
                popup=folium.Popup(f"<b>{getattr(biz, 'name', 'Business')}</b>", max_width=200),
            ).add_to(fmap)

    # Relocation lines: "move this bin to this corner" (net-zero). Orange line from the
    # removable bin to the target corner, with a hollow marker at the bin being moved.
    if relocations_df is not None and len(relocations_df):
        for _, row in relocations_df.iterrows():
            folium.PolyLine([[row["From lat"], row["From lon"]], [row["To lat"], row["To lon"]]],
                            color=COLORS["move"], weight=2, opacity=0.7).add_to(fmap)
            folium.CircleMarker(
                [row["From lat"], row["From lon"]], radius=5, color=COLORS["move"], weight=2,
                fill=True, fill_opacity=0.15,
                popup=folium.Popup(f"<b>Move this bin</b><br/>to {row['Move to']}", max_width=220),
            ).add_to(fmap)

    # In "everything" mode we lift the drawing caps so nothing is hidden.
    bin_cap  = 12000 if show_all else MAP_BIN_CAP
    sugg_cap = len(suggested_df) if show_all else MAP_SUGG_CAP

    # Blue circles: the 114 real DOT counts, drawn first so they sit underneath.
    # Circle size grows with the measured pedestrian count.
    if dot_df is not None and len(dot_df):
        max_count = float(dot_df["ped_count"].max()) or 1.0
        for d in dot_df.itertuples():
            radius = 5 + 22 * (np.sqrt(d.ped_count) / np.sqrt(max_count))
            folium.CircleMarker(
                [d.lat, d.lon], radius=float(radius), color=COLORS["dot"], weight=1,
                fill=True, fill_opacity=0.12,
                popup=folium.Popup(
                    f"<b>DOT verified pedestrian count</b><br/>{d.street}<br/>"
                    f"~{d.ped_count:,.0f} per count session (recent avg)", max_width=240),
            ).add_to(fmap)

    # Red dots: existing bins. Click one to see its ID, type, and cross-streets.
    has_attr = "basket_id" in bins_df.columns
    for b in thin_points(bins_df, bin_cap).itertuples():
        popup = None
        if has_attr:
            s1 = str(getattr(b, "street1", "") or "").strip()
            s2 = str(getattr(b, "street2", "") or "").strip()
            where = f"{s1} & {s2}" if s1 and s2 else str(getattr(b, "basket_loc", "") or "")
            popup = folium.Popup(
                f"<b>Existing basket</b><br/>{where}<br/>"
                f"ID {getattr(b, 'basket_id', '')} &middot; {getattr(b, 'basket_type', '')}",
                max_width=220)
        folium.CircleMarker(
            [b.lat, b.lon], radius=2, color=COLORS["bin"], weight=0, fill=True, fill_opacity=0.5,
            popup=popup,
        ).add_to(fmap)

    # Green dots: the suggestions, drawn at their snapped street corner when available.
    # Higher priority = bigger and brighter. The popup is the "defend this pick" panel.
    for s in suggested_df.head(sugg_cap).itertuples():
        plat = float(getattr(s, "corner_lat", s.lat))
        plon = float(getattr(s, "corner_lon", s.lon))
        priority = int(getattr(s, "priority", 50))
        radius = 4 + 5 * (priority / 100)
        opacity = 0.45 + 0.45 * (priority / 100)
        # Mark high household-misuse-risk picks with a hollow ring instead of a solid dot.
        risk = bool(getattr(s, "misuse_risk", False))
        corner = str(getattr(s, "corner_name", "") or "")
        tip = (corner + " - " if corner else "") + f"Priority {priority}/100"
        folium.CircleMarker(
            [plat, plon], radius=radius,
            color=COLORS["misuse"] if risk else COLORS["suggestion"],
            weight=2 if risk else 0, fill=True,
            fill_opacity=0.12 if risk else opacity,
            popup=folium.Popup(defend_html(s), max_width=300), tooltip=tip,
        ).add_to(fmap)

    # A small fixed color key (colorblind-safe palette) so the map reads on its own.
    legend_html = f"""<div style="position:fixed;bottom:18px;left:18px;z-index:9999;
      background:rgba(255,255,255,0.92);padding:8px 10px;border:1px solid #bbb;border-radius:6px;
      font:12px Arial,sans-serif;color:#222;line-height:1.6">
      <span style="color:{COLORS['bin']}">&#9679;</span> existing bin
      &nbsp;<span style="color:{COLORS['suggestion']}">&#9679;</span> suggested corner<br/>
      <span style="color:{COLORS['misuse']}">&#9711;</span> ring = household-misuse risk
      (residential; a basket here may collect household trash)<br/>
      <span style="color:{COLORS['business']}">&#9679;</span> business
      &nbsp;<span style="color:{COLORS['move']}">&#9679;</span> move
      &nbsp;<span style="color:{COLORS['dot']}">&#9679;</span> DOT
      &nbsp;<span style="color:{COLORS['subway']}">&#9679;</span> subway</div>"""
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


# ===========================================================================
# The page (this is what runs top-to-bottom when Streamlit loads)
# ===========================================================================
st.set_page_config(page_title="NYC Trash Bin Optimizer", layout="wide")

st.title("NYC Trash Bin Optimizer")
st.caption(
    "Finds street corners that are busy with people but far from any existing trash bin, "
    "and ranks them so the city knows where to add baskets first. Built only on NYC open data."
)

with st.expander("How to read this, and what the words mean"):
    st.markdown(
        "**The idea in one line:** a corner is suggested when it is *busy* **and** *far from "
        "an existing bin*. Busy alone isn't enough; far-from-a-bin alone isn't enough.\n\n"
        "**How to use it:** pick a goal under *What do you want to do?*, choose a borough, and "
        "read the ranked list under the map. Click any dot to see why it was chosen.\n\n"
        "**What the words mean**\n"
        "- **Activity index (0-100):** how busy a corner is, as a *percentile*: 80 means "
        "busier than 80% of corners in view. It's a stand-in for foot traffic, built from the "
        "data sources below (there is no citywide pedestrian count).\n"
        "- **Walking distance to nearest bin:** distance along real streets (not straight-line) "
        "to the closest existing basket, computed over the city street network.\n"
        "- **DSNY-eligible:** the kind of corner the city actually baskets: commercial / "
        "mixed-use, or near a subway entrance or bus stop, and *not* mid-residential, parks, "
        "industrial, or highway land. Found from city land-use (PLUTO) + transit locations.\n"
        "- **Priority (0-100):** the ranking score = mostly *activity* + *coverage gap*, plus a "
        "small commercial nudge. Higher = build sooner.\n"
        "- **Misuse risk:** a residential spot where a public basket tends to collect household "
        "trash (shown as a hollow ring on the map).\n"
        "- **Confidence:** lower when the data sources disagree about how busy a spot is.\n"
        "- **Recommended baskets:** 1-3, based on how busy the corner is.\n\n"
        "**Where the data comes from:** NYC Open Data: 311 complaints, PLUTO land use, MTA "
        "subway + bus stops, DOT pedestrian counts, DOHMH businesses, DSNY basket inventory + "
        "districts, and the city street centerline. Everything is bundled, so nothing downloads "
        "while you use it."
    )

# Load the data first so the dropdown can list whichever sources exist.
with st.spinner("Loading NYC data…"):
    proxy_df, nypd_df, bins_raw = load_sources()
source_options = build_source_options(proxy_df, nypd_df)

# Default minimum spacing, derived from the REAL spacing of existing commercial-area
# baskets (not an invented number). And the list of DSNY sanitation districts present.
default_gap = calibrated_min_gap(proxy_df, bins_raw)
district_list = (sorted(d for d in proxy_df["district"].dropna().unique() if d)
                 if proxy_df is not None and "district" in proxy_df.columns else [])

# Data vintage: when the bundled grid was assembled (the file's own timestamp).
try:
    import datetime
    data_date = datetime.date.fromtimestamp(os.path.getmtime(PROXY_GRID_CSV)).strftime("%b %d, %Y")
except Exception:
    data_date = None

# Shareable view: seed preset / borough / district from the URL on first load.
_qp = st.query_params
for _k, _ok in [("preset", None), ("borough", set(BOROUGH_CENTERS)), ("district", set(district_list))]:
    if _k in _qp and _k not in st.session_state and (_ok is None or _qp[_k] in _ok):
        st.session_state[_k] = _qp[_k]

# Figure out which layers are actually inside the composite grid, so the
# description we show the user is always honest (e.g. Citibike only appears
# if the build actually managed to include it).
composite_layers = []
if proxy_df is not None:
    for col, name in [("score_311", "311 street complaints"),
                      ("score_mta", "subway ridership"),
                      ("score_citibike", "Citibike trips")]:
        if col in proxy_df.columns and pd.to_numeric(proxy_df[col], errors="coerce").fillna(0).sum() > 0:
            composite_layers.append(name)

# ---- Sidebar controls ----
# Task presets set a few controls at once for a common job. "Custom" leaves them alone.
PRESETS = {
    "Find new corners":        {"gate": True,  "commercial": True,  "relocation": False},
    "Rebalance existing bins": {"gate": True,  "commercial": False, "relocation": True},
    "Plan a district":         {"gate": True,  "commercial": True,  "relocation": False},
    "Respond to complaints":   {"gate": True,  "commercial": False, "relocation": False,
                                "source": "Basket need (311 requests + overflow)"},
}
# Defaults for the preset-controlled widgets (set once, before the widgets are built).
for _k, _v in [("source", list(source_options.keys())[0]), ("gate", False),
               ("commercial", False), ("relocation", False), ("snap", True)]:
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("Settings")

    preset = st.selectbox(
        "What do you want to do?",
        ["Custom", "Find new corners", "Rebalance existing bins",
         "Plan a district", "Respond to complaints"], key="preset",
        help="A shortcut that sets the placement rules below for a common job. Pick Custom to "
             "set everything yourself.",
    )
    # When the preset changes, push its settings into the controlled widgets.
    if st.session_state.get("_prev_preset") != preset:
        st.session_state["_prev_preset"] = preset
        for k, v in PRESETS.get(preset, {}).items():
            if k == "source" and v not in source_options:
                continue
            st.session_state[k] = v

    borough = st.selectbox(
        "Borough",
        ["All Boroughs", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        key="borough",
        help="Focus on one borough. Activity is ranked within it, so Queens is judged against "
             "Queens, not crowded Manhattan.",
    )
    find_query = st.text_input(
        "Find a street or corner",
        help="Type a street or corner (e.g. 'Steinway' or 'Broadway & W 145'). The map "
             "recenters on the first match.",
    )
    district_choice = st.selectbox(
        "Sanitation district", ["All districts"] + district_list, key="district",
        help="Narrow the shortlist, export, and report to one DSNY district (e.g. BKN03).",
    ) if district_list else "All districts"

    st.caption("Pick a goal and a borough above. The options below are optional fine-tuning.")

    with st.expander("Advanced settings"):
        source_choice = st.selectbox(
            "Activity data source", list(source_options.keys()), key="source",
            help="Which data stands in for where people are. Composite (311 + transit) is "
                 "best; NYPD misses quiet residential areas.",
        )
        sensitivity = st.slider(
            "Sensitivity", 1, 10, 5,
            help="How busy an area must be to be flagged. Low (1-3) = fewer, stronger picks; "
                 "high (8-10) = more picks.",
        )
        min_distance_m = st.slider(
            "Minimum gap from existing bin (meters)", 25, 800, value=default_gap, step=25,
            help=f"A spot is only suggested if no bin is within this distance. The default "
                 f"({default_gap} m) is the measured median spacing of existing commercial-area "
                 "DSNY baskets.",
        )
        apply_gate = st.checkbox(
            "Apply DSNY eligibility rules", key="gate",
            help="Suggest only commercial / mixed-use, near-transit, or near-bus corners; not "
                 "residential, parks, industrial, or highway cells.",
        )
        prioritize_commercial = st.checkbox(
            "Prioritize commercial corners", key="commercial",
            help="Among equally busy, equally underserved spots, rank the one with more retail "
                 "/ office floor area higher.",
        )
        budget = st.number_input(
            "Budget: number of new baskets (0 = no limit)", 0, 2000, 0, 5,
            help="Fund a fixed number? The tool returns exactly that many, highest priority first.",
        )
        snap_corners = st.checkbox(
            "Snap suggestions to nearest street corner", key="snap",
            help="Move each suggestion to the nearest real intersection and label it, e.g. "
                 "'Broadway & W 145 St'.",
        )
        relocation_mode = st.checkbox(
            "Relocation mode (move low-value bins, net-zero)", key="relocation",
            help="Pair each top corner with the nearest low-value existing bin to move there, "
                 "so a new corner costs nothing.",
        )
        layers = st.multiselect(
            "Map layers", ["Eligibility signals", "Businesses", "DOT counts", "All-cell heatmap"],
            help="Optional overlays. Eligibility = commercial heat + subway entrances. "
                 "Businesses = DOHMH dots. DOT = the 114 real counts. All-cell heatmap is slower.",
        )
    show_eligibility = "Eligibility signals" in layers
    show_business    = "Businesses" in layers
    show_dot         = "DOT counts" in layers
    show_all         = "All-cell heatmap" in layers

    st.divider()
    if data_date:
        st.caption(f"Data assembled {data_date} from NYC Open Data (311, PLUTO, MTA, DOT, "
                   "DOHMH, DSNY). Bundled, so the app downloads nothing at runtime.")
    else:
        st.caption("Built with NYC Open Data. All data is bundled, so the app runs without "
                   "downloading anything.")

# Persist the shareable view in the URL (so a configured map can be emailed).
st.query_params.update({"preset": preset, "borough": borough, "district": district_choice})

# ---- Run the pipeline for the chosen settings ----
ped_df = activity_input(source_options, source_choice, proxy_df, nypd_df)
try:
    cand_df, bins_df = prepare_candidates(ped_df, bins_raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

dot_all = load_dot_counts()
entrances_all = load_subway_entrances()
intersections_all = load_intersections()
businesses_all = load_businesses()

# Sensitivity 1-10 becomes a 0-100 cutoff:
#   sensitivity 1  -> cutoff 90 (only the top 10% busiest cells)
#   sensitivity 5  -> cutoff 50
#   sensitivity 10 -> cutoff 0  (every cell passes the busyness test)
threshold_index = (10 - sensitivity) * 10

suggested = suggest_new_bins(cand_df, threshold_index, float(min_distance_m), borough,
                             eligible_only=apply_gate)
suggested = compute_priority(suggested, dot_all if len(dot_all) else None,
                             use_commercial=prioritize_commercial)

# Make the output usable: snap to a real corner, add basket count + misuse risk.
suggested = snap_to_corners(suggested, intersections_all if snap_corners else None)
suggested = enrich_suggestions(suggested)

# Narrow to one sanitation district if the planner picked one.
if district_choice != "All districts" and "district" in suggested.columns:
    suggested = suggested[suggested["district"] == district_choice].reset_index(drop=True)

# Budget cap: keep only the top-N highest-priority corners.
if budget and budget > 0:
    suggested = suggested.head(int(budget)).reset_index(drop=True)

# Relocation pairing (net-zero): only computed when the planner asks for it.
relocations = pd.DataFrame()
if relocation_mode:
    scope_for_reloc = cand_df if borough == "All Boroughs" else cand_df[cand_df["borough"] == borough]
    relocations = pair_relocations(suggested, removable_bins(bins_df, scope_for_reloc), n=50)

# Scope every map layer to the chosen borough in one place.
scope_bins = scope_to_borough(bins_df, borough)
if scope_bins.empty:
    scope_bins = bins_df
scope_cells = scope_to_borough(cand_df, borough)
dot_scope   = scope_to_borough(dot_all, borough)
ent_scope   = scope_to_borough(entrances_all, borough)
biz_scope   = scope_to_borough(businesses_all, borough)

# Search: recenter the map on the first matching named corner (no external geocoder).
search_center = None
if find_query and find_query.strip() and len(intersections_all):
    hits = intersections_all[intersections_all["name"].str.contains(
        find_query.strip(), case=False, na=False, regex=False)]
    if len(hits):
        search_center = (float(hits["lat"].mean()), float(hits["lon"].mean()), 15)

# ---- Main view (single column, mobile friendly) ----
st.subheader("Coverage map")
map_caption = "Blue = existing bins, orange = suggested corners (brighter = higher priority)."
if relocation_mode:
    map_caption += " Green lines = a bin to move and where to move it."
st.caption(map_caption + " Tap a dot for details; a color key sits on the map.")
if apply_gate and "eligible" not in cand_df.columns:
    st.caption("Eligibility rules need a Composite source, so they don't apply to NYPD data.")
if find_query and find_query.strip():
    st.caption(f"Centered on '{find_query.strip()}'." if search_center
               else f"No street or corner matched '{find_query.strip()}'.")
elif not show_all and len(suggested) > MAP_SUGG_CAP:
    st.caption(f"Showing the top {MAP_SUGG_CAP} of {len(suggested):,} suggestions by priority.")

# Height-only (no fixed width) keeps the map responsive on a phone.
st_folium(
    make_map(scope_bins, suggested, borough, dot_scope if show_dot else None,
             cells_df=scope_cells, show_all=show_all,
             show_eligibility=show_eligibility,
             entrances_df=ent_scope if show_eligibility else None,
             businesses_df=biz_scope if show_business else None,
             relocations_df=relocations if relocation_mode else None,
             center=search_center),
    height=520, returned_objects=[],
)

# One compact summary line + a plain description of the current data source.
st.markdown(
    f"**{len(suggested):,}** suggested corners &nbsp;&middot;&nbsp; "
    f"**{len(scope_bins):,}** existing bins &nbsp;&middot;&nbsp; "
    f"**{len(scope_cells):,}** blocks analyzed")

composite_desc = (
    "Mixes " + " and ".join(composite_layers) + " to estimate where people are walking."
    if composite_layers else
    "Mixes several NYC activity signals to estimate where people are walking."
)
SOURCE_BLURBS = {
    "Composite (311 + transit)": composite_desc,
    "Composite (per person)": "The same blend divided by how many people work and live nearby "
        "(Census LODES), so a packed transit hub doesn't automatically outrank a quieter but "
        "underserved neighborhood.",
    "Basket need (311 requests + overflow)": "311 complaints that directly ask for basket "
        "service or report overflow - the most direct public signal of where people want a basket.",
    "311 street complaints": "Outdoor 311 reports (street/sidewalk, litter, noise) - common in "
        "residential areas where NYPD data falls short.",
    "Subway ridership": "Riders per station - strong near stations, weak away from them.",
    "Citibike trips": "Trips per station - useful inside the bike network only.",
    "NYPD incidents (911 calls)": "911 records as a stand-in for foot traffic - leans commercial, "
        "misses quiet residential blocks.",
}
st.caption(f"**Source ({source_choice}):** {SOURCE_BLURBS.get(source_choice, '')}")

if len(suggested) == 0:
    st.warning("No recommendations with these settings. Open **Advanced settings** and raise "
               "**Sensitivity** or lower the **minimum gap**.")

# How accurate is this? (validation)
val = compute_validation(proxy_df, bins_raw, dot_all)
if val["recovery"] is not None or val["dot_corr"] is not None:
    with st.expander("How accurate is this? (validation)"):
        if val["recovery"] is not None:
            st.caption(
                f"**Recovery: {val['recovery'] * 100:.0f}%** of the city's {val['n_bins']:,} "
                "existing baskets sit where the model independently calls for one (busy and "
                "eligible) - it agrees with where DSNY already places baskets.")
        if val["dot_corr"] is not None:
            st.caption(
                f"**DOT agreement: r = {val['dot_corr']:.2f}** between our activity rank and the "
                f"{val['n_dot']} real DOT pedestrian counts (1 = perfect, 0 = none).")
        if val["holdout"] is not None:
            st.caption(
                f"**Hold-out test: {val['holdout'] * 100:.0f}%** - of {val['n_holdout']:,} hidden "
                "baskets whose removal leaves a real gap, this share sit where the model "
                "independently calls for a basket. Tests prediction, not just agreement.")
        st.caption("Transparent sanity checks, not a trained model's accuracy.")

# District scorecard: equity / coverage stats for the selected district.
if district_choice != "All districts":
    sc = district_scorecard(proxy_df, bins_raw)
    if not sc.empty and district_choice in sc.index:
        row = sc.loc[district_choice]
        with st.expander(f"District scorecard ({district_choice})"):
            st.caption(
                f"{int(row['baskets']):,} existing baskets &middot; "
                f"{int(row['population']):,} day+night people &middot; "
                f"{int(row['commercial_corners']):,} commercial corners.")
            if pd.notna(row["per_1k_residents"]):
                st.caption(
                    f"{row['per_1k_residents']:.2f} baskets per 1,000 residents "
                    f"(underserved rank {int(row['underserved_rank'])} of {len(sc)}; "
                    "rank 1 = most underserved).")

# ---- Priority list (full width, below the map) ----
st.divider()
if len(suggested) == 0:
    st.stop()

st.subheader("Priority list (build these first)")
priority_note = ("The priority score (0 to 100) combines how busy a spot is and how far it is "
                 "from the nearest bin")
if prioritize_commercial:
    priority_note += ", how much commercial activity is there"
if len(dot_all):
    priority_note += ", and whether a real DOT pedestrian count is nearby"
st.caption(priority_note + ".")

# Build a friendly, human-readable version of the table.
ranked = suggested.copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))
ranked["Priority"]       = ranked["priority"]
if "corner_name" in ranked.columns:
    ranked["Corner"]     = ranked["corner_name"].replace("", "(cell center)")
ranked["Borough"]        = ranked["borough"]
if "district" in ranked.columns:
    ranked["District"]   = ranked["district"]
ranked["Activity index"] = ranked["activity_index"]
gap_label = "Walk to bin (m)" if "nearest_bin_straight_m" in suggested.columns else "Gap to bin (m)"
ranked[gap_label] = ranked["nearest_bin_m"].round(0).astype(int)
if "recommended_baskets" in ranked.columns:
    ranked["Baskets"]    = ranked["recommended_baskets"]
if "misuse_risk" in ranked.columns:
    ranked["Misuse risk"] = np.where(ranked["misuse_risk"], "high", "low")
if "low_confidence" in ranked.columns:
    ranked["Confidence"] = np.where(ranked["low_confidence"], "low", "ok")
# Export the snapped corner coordinate when we have it, else the cell center.
lat_src = ranked["corner_lat"] if "corner_lat" in ranked.columns else ranked["lat"]
lon_src = ranked["corner_lon"] if "corner_lon" in ranked.columns else ranked["lon"]
ranked["Latitude"]       = lat_src.round(5)
ranked["Longitude"]      = lon_src.round(5)
if "dot_verified" in ranked.columns and ranked["dot_verified"].any():
    ranked["DOT-verified"] = np.where(ranked["dot_verified"], "yes", "")

show_cols = ["Rank", "Priority"]
if "Corner" in ranked.columns:      show_cols.append("Corner")
show_cols.append("Borough")
if "District" in ranked.columns:    show_cols.append("District")
show_cols += ["Activity index", gap_label]
if "Baskets" in ranked.columns:     show_cols.append("Baskets")
if "Misuse risk" in ranked.columns: show_cols.append("Misuse risk")
if "Confidence" in ranked.columns:  show_cols.append("Confidence")
show_cols += ["Latitude", "Longitude"]
if "DOT-verified" in ranked.columns:
    show_cols.insert(2, "DOT-verified")

st.dataframe(ranked[show_cols].head(50), use_container_width=True, hide_index=True)

scope_tag = (district_choice if district_choice != "All districts"
             else borough).replace(" ", "_").lower()
with st.expander("Download / export"):
    st.download_button(
        "CSV (spreadsheet)", ranked[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"bin_priority_{scope_tag}.csv", mime="text/csv")
    st.download_button(
        "GeoJSON (for GIS / ArcGIS)", to_geojson(ranked[show_cols]).encode("utf-8"),
        file_name=f"bin_priority_{scope_tag}.geojson", mime="application/geo+json")
    st.download_button(
        "Printable report (HTML → print to PDF)",
        build_report_html(ranked[show_cols], district_choice, borough,
                          source_choice, min_distance_m).encode("utf-8"),
        file_name=f"bin_report_{scope_tag}.html", mime="text/html")

if relocation_mode and len(relocations):
    st.divider()
    st.subheader(f"Relocation plan (net-zero): {len(relocations)} bins to move")
    st.caption("Each top corner is paired with the nearest low-value existing bin to move "
               "there, so adding the corner costs nothing.")
    st.dataframe(relocations.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download relocation plan (CSV)", relocations.to_csv(index=False).encode("utf-8"),
        file_name=f"bin_relocations_{scope_tag}.csv", mime="text/csv")
