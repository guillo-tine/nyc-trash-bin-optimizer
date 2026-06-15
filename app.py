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
SOCRATA_BASE   = "https://data.cityofnewyork.us/resource"

GRID_SIZE_M  = 250        # each cell is 250m x 250m
DATA_LIMIT   = 100_000    # how many rows to download from NYC Open Data
MAP_BIN_CAP  = 3000       # most existing-bin dots we draw (just for context)
MAP_SUGG_CAP = 300        # most suggestion dots we draw (the top ones)

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

    keep = ["lat", "lon", "borough", "activity_score", "nearest_bin_m"]
    # Carry DSNY-eligibility columns through if the grid has them (composite sources).
    for extra in ("eligible", "near_transit", "commercial_area"):
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
             show_eligibility: bool = False, entrances_df: pd.DataFrame | None = None) -> folium.Map:
    """Build the Folium map: blue DOT circles (optional), red bins, green suggestions.

    When show_all is True ("Visualize everything"):
      • draw an activity HEATMAP of every cell (the input the whole tool is built on)
      • draw ALL existing bins and ALL suggestions (caps lifted)
    When show_eligibility is True:
      • draw a green heatmap of commercial floor area (the "businesses" signal)
      • draw subway entrances as purple dots (the "near transit" signal)
    """
    center_lat, center_lon, zoom = BOROUGH_CENTERS.get(borough, BOROUGH_CENTERS["All Boroughs"])
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
                [e.lat, e.lon], radius=3, color="#6f42c1", weight=0, fill=True, fill_opacity=0.7,
                popup="Subway entrance (a 'near transit' signal for eligibility)",
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
                [d.lat, d.lon], radius=float(radius), color="#1f6feb", weight=1,
                fill=True, fill_opacity=0.12,
                popup=folium.Popup(
                    f"<b>DOT verified pedestrian count</b><br/>{d.street}<br/>"
                    f"~{d.ped_count:,.0f} per count session (recent avg)", max_width=240),
            ).add_to(fmap)

    # Red dots: existing bins (just context, so we thin them down).
    for b in thin_points(bins_df, bin_cap).itertuples():
        folium.CircleMarker(
            [b.lat, b.lon], radius=2, color="#cc2200", weight=0, fill=True, fill_opacity=0.5,
        ).add_to(fmap)

    # Green dots: the suggestions. Higher priority = bigger and brighter.
    for s in suggested_df.head(sugg_cap).itertuples():
        feet = s.nearest_bin_m * 3.281
        priority = int(getattr(s, "priority", 50))
        radius = 4 + 5 * (priority / 100)
        opacity = 0.45 + 0.45 * (priority / 100)

        popup_html = (
            f"<b>Suggested new bin</b><br/>"
            f"Priority: <b>{priority}</b>/100<br/>"
            f"Activity index: <b>{s.activity_index}</b>/100<br/>"
            f"Nearest existing bin: <b>{s.nearest_bin_m:.0f} m</b> ({feet:.0f} ft)"
        )
        # If the grid carries DSNY signals, show WHY this spot qualifies.
        eligible = getattr(s, "eligible", None)
        if eligible is not None:
            near = bool(getattr(s, "near_transit", False))
            comm = float(getattr(s, "commercial_area", 0) or 0)
            popup_html += (
                "<hr style='margin:4px 0'>"
                f"<b>Why it's DSNY-eligible</b><br/>"
                f"Commercial / mixed-use here: <b>{'Yes' if comm > 0 else 'No'}</b><br/>"
                f"Near a subway entrance: <b>{'Yes' if near else 'No'}</b><br/>"
                f"Commercial floor area: <b>{comm:,.0f} sq ft</b>"
            )

        folium.CircleMarker(
            [s.lat, s.lon], radius=radius, color="#00aa44", weight=0, fill=True, fill_opacity=opacity,
            popup=folium.Popup(popup_html, max_width=300), tooltip=f"Priority {priority}/100",
        ).add_to(fmap)

    return fmap


# ===========================================================================
# The page (this is what runs top-to-bottom when Streamlit loads)
# ===========================================================================
st.set_page_config(page_title="NYC Trash Bin Optimizer", layout="wide")

st.title("NYC Trash Bin Optimizer")
st.caption(
    "This map points out blocks that might need more public trash bins. It looks for areas "
    "that are busy but don't have many bins nearby, all using the city's public data."
)

# Load the data first so the dropdown can list whichever sources exist.
with st.spinner("Loading NYC data…"):
    proxy_df, nypd_df, bins_raw = load_sources()
source_options = build_source_options(proxy_df, nypd_df)

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
with st.sidebar:
    st.header("Settings")

    source_choice = st.selectbox(
        "Activity data source",
        list(source_options.keys()),
        help=(
            "Pick which data stands in for where people are.\n\n"
            "**Composite** mixes 311 and transit, and is the best option.\n\n"
            "**311, Subway, or Citibike** each use a single signal.\n\n"
            "**NYPD incidents** is the original, and it misses quiet residential areas.\n\n"
            "To add more sources, run `python data/build_proxy.py`."
        ),
    )

    borough = st.selectbox(
        "Borough",
        ["All Boroughs", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        help=(
            "Focus on one borough. Activity is ranked inside the borough you pick, so "
            "quieter boroughs like Queens get compared to themselves instead of to "
            "crowded Manhattan."
        ),
    )

    st.divider()

    sensitivity = st.slider(
        "Sensitivity", min_value=1, max_value=10, value=5,
        help=(
            "How busy an area has to be before it gets flagged.\n\n"
            "**Low (1 to 3):** only the busiest spots, so you get fewer but stronger picks.\n\n"
            "**Medium (5):** a balance.\n\n"
            "**High (8 to 10):** also includes moderately busy areas, so you get more picks."
        ),
    )

    min_distance_m = st.slider(
        "Minimum gap from existing bin (meters)", min_value=25, max_value=800, value=300, step=25,
        help=(
            "A spot is only suggested if there's no bin within this distance.\n\n"
            "**300 m is about one city block.** Raise it to find only the most isolated gaps."
        ),
    )

    apply_gate = st.checkbox(
        "Apply DSNY eligibility rules", value=False,
        help=(
            "DSNY only places baskets on commercial or mixed-use corners and near transit, "
            "not on residential or industrial streets, in parks, or mid-block. With this on, "
            "a cell is suggested only if it has commercial/mixed-use land use or sits near a "
            "subway entrance, and is not dominated by parks, industrial, or highway land. "
            "(Eligibility uses NYC PLUTO land use and MTA subway entrances.) "
            "This is off by default so you can show the before/after."
        ),
    )

    prioritize_commercial = st.checkbox(
        "Prioritize commercial corners", value=False,
        help=(
            "DSNY places baskets where commercial activity generates disposable waste. "
            "With this on, among equally busy and equally underserved spots, the one with "
            "more retail/office floor area ranks higher. Adds a commercial term to the "
            "priority score (weight 0.15). Off by default."
        ),
    )

    show_eligibility = st.checkbox(
        "Show eligibility signals", value=False,
        help=(
            "Reveals the data behind the DSNY rule: a green heatmap of commercial floor "
            "area (where the businesses are) and purple dots for subway entrances (the "
            "'near transit' signal). Each green suggestion's popup also explains why it "
            "qualifies."
        ),
    )

    show_dot = st.checkbox(
        "Show DOT verified pedestrian counts", value=False,
        help=(
            "Shows the 114 NYC DOT count locations as blue circles, sized by how many "
            "pedestrians were measured. These are the only real foot-traffic counts the city "
            "publishes, so they make a good reality check."
        ),
    )

    show_all = st.checkbox(
        "Visualize everything", value=False,
        help=(
            "Adds a heatmap of activity across every cell (the raw signal behind the whole "
            "tool) and draws every bin and every suggestion, not just the top ones.\n\n"
            "Good for seeing the full picture. Tip: pick one borough, since drawing the whole "
            "city at full detail can get slow."
        ),
    )

    st.divider()
    st.caption("Built with NYC Open Data. All data is bundled, so the app runs without "
               "downloading anything.")

# ---- Run the pipeline for the chosen settings ----
ped_df = activity_input(source_options, source_choice, proxy_df, nypd_df)
try:
    cand_df, bins_df = prepare_candidates(ped_df, bins_raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

dot_all = load_dot_counts()
entrances_all = load_subway_entrances()

# Sensitivity 1-10 becomes a 0-100 cutoff:
#   sensitivity 1  -> cutoff 90 (only the top 10% busiest cells)
#   sensitivity 5  -> cutoff 50
#   sensitivity 10 -> cutoff 0  (every cell passes the busyness test)
threshold_index = (10 - sensitivity) * 10

suggested = suggest_new_bins(cand_df, threshold_index, float(min_distance_m), borough,
                             eligible_only=apply_gate)
suggested = compute_priority(suggested, dot_all if len(dot_all) else None,
                             use_commercial=prioritize_commercial)

# What to show on the map for the chosen borough (existing bins + DOT points).
scope_bins = bins_df if borough == "All Boroughs" else bins_df[bins_df["borough"] == borough]
if scope_bins.empty:
    scope_bins = bins_df
scope_cells = cand_df if borough == "All Boroughs" else cand_df[cand_df["borough"] == borough]
dot_scope = dot_all if borough == "All Boroughs" else dot_all[dot_all["borough"] == borough]
ent_scope = entrances_all if borough == "All Boroughs" else entrances_all[entrances_all["borough"] == borough]

# ---- Layout: map on the left, summary on the right ----
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Coverage Map")
    legend = "Red dots are existing bins. Green dots are suggested new spots (brighter means higher priority)."
    if show_dot and len(dot_scope):
        legend += " Blue circles are DOT pedestrian counts."
    if show_all:
        legend += " The heat shows activity level."
    if show_eligibility:
        legend += " Green heat shows commercial floor area (businesses); purple dots are subway entrances."
    st.caption(legend + " Click any dot for details.")
    if apply_gate:
        if "eligible" in cand_df.columns:
            st.caption("DSNY eligibility rules are ON: showing only commercial or transit-eligible "
                       "corners (residential, parks, industrial, and highway cells are removed).")
        else:
            st.caption("DSNY eligibility rules need a Composite source (PLUTO land use); they don't "
                       "apply to the NYPD source.")
    if show_all:
        st.caption("Showing everything: the activity heatmap, all bins, and all suggestions.")
        if borough == "All Boroughs":
            st.caption("Tip: pick a single borough if the full-city view feels slow.")
    elif len(suggested) > MAP_SUGG_CAP:
        st.caption(f"Showing the top {MAP_SUGG_CAP} suggestions by priority (of {len(suggested):,} total).")
    # returned_objects=[] tells st_folium not to send map state back to Python,
    # so panning/zooming the map doesn't trigger a full app rerun.
    st_folium(
        make_map(scope_bins, suggested, borough, dot_scope if show_dot else None,
                 cells_df=scope_cells, show_all=show_all,
                 show_eligibility=show_eligibility,
                 entrances_df=ent_scope if show_eligibility else None),
        width=900, height=650, returned_objects=[],
    )

with right:
    st.subheader("Summary")
    st.metric("Existing bins (this view)", f"{len(scope_bins):,}")
    st.metric("Activity cells analyzed",   f"{len(scope_cells):,}")
    st.metric("Suggested new bins",        f"{len(suggested):,}")

    if len(suggested) == 0:
        st.warning(
            "No recommendations with these settings. Try raising **Sensitivity** "
            "or lowering the **minimum gap**."
        )

    st.divider()
    st.caption("**How it works**")
    st.caption(
        f"The city is split into **250 m by 250 m squares**. Each square gets an **activity "
        f"score from 0 to 100** based on how it ranks against the others in view. A square is "
        f"suggested when its score is at least **{threshold_index}** and the nearest bin is more "
        f"than **{min_distance_m} m** away."
    )

    st.divider()
    st.caption(f"**About the data: {source_choice}**")
    composite_desc = (
        "This mixes " + " and ".join(composite_layers) + " to estimate where people are walking."
        if composite_layers else
        "This mixes several NYC activity signals to estimate where people are walking."
    )
    SOURCE_BLURBS = {
        "Composite (311 + transit)": composite_desc,
        "Composite (per person)": "The same blend, but divided by how many people work and "
            "live nearby (from Census LODES). This scores activity per person instead of raw "
            "crowd size, so a packed transit hub doesn't automatically outrank a quieter "
            "but underserved neighborhood.",
        "Basket need (311 requests + overflow)": "Counts of 311 complaints that directly ask "
            "for basket service: 'Litter Basket Request', 'Litter Basket Complaint', and "
            "'Overflowing Litter Baskets'. This is the most direct public signal of where "
            "people actually want or need a basket.",
        "311 street complaints": "Counts of outdoor 311 reports like street and sidewalk "
            "conditions, litter, and noise. These are common in residential areas, where the "
            "NYPD data falls short.",
        "Subway ridership": "How many people ride the subway at each station. It's strong near "
            "stations and weaker once you move away from them.",
        "Citibike trips": "How many Citibike trips start or end at each station. It's useful "
            "inside the bike network but doesn't cover areas without stations.",
        "NYPD incidents (911 calls)": "NYPD incident records, used as a stand-in for foot "
            "traffic. It leans toward commercial streets and tends to miss quiet residential blocks.",
    }
    st.caption(SOURCE_BLURBS.get(source_choice, ""))
    if proxy_df is None:
        st.caption("Run `python data/build_proxy.py` to add the 311 and transit sources.")

    if len(suggested) > 0:
        st.divider()
        st.caption("**Priority list (build these first)**")
        priority_note = "The priority score (0 to 100) combines how busy a spot is and how far it is from the nearest bin"
        if prioritize_commercial:
            priority_note += ", how much commercial activity is there"
        if len(dot_all):
            priority_note += ", and whether a real DOT pedestrian count is nearby"
        st.caption(priority_note + ".")

        # Build a friendly, human-readable version of the table.
        ranked = suggested.copy()
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        ranked["Priority"]       = ranked["priority"]
        ranked["Borough"]        = ranked["borough"]
        ranked["Activity index"] = ranked["activity_index"]
        ranked["Gap to bin (m)"] = ranked["nearest_bin_m"].round(0).astype(int)
        ranked["Latitude"]       = ranked["lat"].round(5)
        ranked["Longitude"]      = ranked["lon"].round(5)
        if "dot_verified" in ranked.columns and ranked["dot_verified"].any():
            ranked["DOT-verified"] = np.where(ranked["dot_verified"], "yes", "")

        show_cols = ["Rank", "Priority", "Borough", "Activity index", "Gap to bin (m)", "Latitude", "Longitude"]
        if "DOT-verified" in ranked.columns:
            show_cols.insert(2, "DOT-verified")

        st.dataframe(ranked[show_cols].head(20), use_container_width=True, hide_index=True)
        st.download_button(
            "Download full priority list (CSV)",
            ranked[show_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"bin_priority_{borough.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
