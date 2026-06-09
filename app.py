"""
NYC Trash Bin Optimizer
=======================

What this app does, in one sentence:
    It finds city blocks that are busy with people but far from any existing
    trash bin, and suggests those blocks as good spots for a new bin.

The whole program is just four steps:

    1. LOAD   – read three kinds of public NYC data:
                  • activity data  (where people are — a stand-in for foot traffic)
                  • litter baskets (where bins already exist)
                  • DOT counts     (114 places with REAL measured pedestrian counts)

    2. SCORE  – chop the city into 250m squares ("cells"). Give each cell an
                "activity score" = how much activity happened inside it. Then for
                every cell, measure the distance to the nearest existing bin.

    3. FILTER – keep a cell as a SUGGESTION only if it is:
                  • busy enough        (activity above the Sensitivity cutoff), AND
                  • far from any bin    (farther than the Minimum-gap setting).

    4. RANK   – give each suggestion a Priority score (0-100) so a city planner
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
# STEP 1 — Load the data
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
        proxy  – the good composite grid (or None if it hasn't been built yet)
        nypd   – the original NYPD incident table (or None)
        bins   – the existing litter baskets
    """
    if not os.path.exists(BINS_CSV):
        ensure_data()
    bins = pd.read_csv(BINS_CSV)

    # The composite grid is optional — it only exists after running build_proxy.py.
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
        for label, column in [("311 street complaints", "score_311"),
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
    """Load the 114 DOT pedestrian-count locations — the only REAL measured
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

    # No plain lat/lon columns — try a geometry/point column instead.
    geom_col = find_column(df, {"the_geom", "geom", "geometry", "location", "point"})
    if geom_col:
        text = df[geom_col].astype(str)
        df = df.copy()
        if text.str.contains("POINT", case=False, na=False).any():
            # WKT format: "POINT (lon lat)"  — note longitude comes first.
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
# STEP 2 — Score every cell, and measure distance to the nearest bin
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
        # The composite grid is ALREADY one row per cell with a score — just clean it.
        cand = ped_df.copy()
        cand["lat"] = pd.to_numeric(cand["lat"], errors="coerce")
        cand["lon"] = pd.to_numeric(cand["lon"], errors="coerce")
        cand["activity_score"] = pd.to_numeric(cand["activity_score"], errors="coerce")
        cand = cand.dropna(subset=["lat", "lon", "activity_score"])
        if "borough" not in cand.columns:
            cand["borough"] = assign_borough(cand["lat"], cand["lon"])
    else:
        # Raw NYPD points — count how many fall into each 250m cell.
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
    return cand[keep].reset_index(drop=True), bins.reset_index(drop=True)


# ===========================================================================
# STEP 3 — Filter cells down to suggestions
# ===========================================================================
def suggest_new_bins(cand: pd.DataFrame, threshold_index: int,
                     min_distance_m: float, borough: str) -> pd.DataFrame:
    """Keep only the cells that are busy enough AND far enough from a bin."""
    # If a borough is chosen, look only at that borough's cells.
    cells = cand if borough == "All Boroughs" else cand[cand["borough"] == borough]
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
# STEP 4 — Rank suggestions by priority (which to build first)
# ===========================================================================
def compute_priority(sugg: pd.DataFrame, dot_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Give each suggestion a 0-100 Priority score and sort by it.

    Priority blends three ideas:
        • activity   – how busy the cell is        (the activity index, 0-100)
        • gap        – how far the nearest bin is   (ranked 0-100 among suggestions)
        • dot        – is it near a high REAL DOT count?  (a small confidence bonus)

    With DOT data:    priority = 0.55*activity + 0.35*gap + 0.10*dot
    Without DOT data: priority = 0.60*activity + 0.40*gap
    """
    if sugg.empty:
        return sugg.assign(priority=pd.Series(dtype=int), dot_verified=pd.Series(dtype=bool))

    df = sugg.copy()
    activity = df["activity_index"].astype(float)              # already 0-100
    gap = df["nearest_bin_m"].rank(pct=True) * 100             # 0-100 within the suggestions

    if dot_df is not None and len(dot_df):
        # Find the nearest DOT count location to each suggestion.
        dot_xy = to_meters(dot_df["lat"].values, dot_df["lon"].values)
        sugg_xy = to_meters(df["lat"].values, df["lon"].values)
        distance_to_dot, nearest_index = cKDTree(dot_xy).query(sugg_xy, k=1)

        # Rank the DOT counts 0-100; a suggestion gets that score only if a DOT
        # point is within 500m, otherwise it gets 0 (no nearby ground truth).
        dot_rank = (dot_df["ped_count"].rank(pct=True) * 100).values
        dot_bonus = np.where(distance_to_dot <= 500, dot_rank[nearest_index], 0.0)

        df["dot_verified"] = distance_to_dot <= 500
        df["priority"] = (0.55 * activity + 0.35 * gap + 0.10 * dot_bonus).round().astype(int)
    else:
        df["dot_verified"] = False
        df["priority"] = (0.60 * activity + 0.40 * gap).round().astype(int)

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
             cells_df: pd.DataFrame | None = None, show_all: bool = False) -> folium.Map:
    """Build the Folium map: blue DOT circles (optional), red bins, green suggestions.

    When show_all is True ("Visualize everything"):
      • draw an activity HEATMAP of every cell (the input the whole tool is built on)
      • draw ALL existing bins and ALL suggestions (caps lifted)
    """
    center_lat, center_lon, zoom = BOROUGH_CENTERS.get(borough, BOROUGH_CENTERS["All Boroughs"])
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="cartodbpositron")

    # Activity heatmap (drawn first, underneath everything) — shows the raw activity
    # surface that feeds the whole model. Each cell is weighted by its activity score.
    if show_all and cells_df is not None and len(cells_df):
        from folium.plugins import HeatMap
        scores = cells_df["activity_score"].astype(float)
        top = scores.max() or 1.0
        heat_points = [[row.lat, row.lon, float(w / top)]
                       for row, w in zip(cells_df.itertuples(), scores)]
        HeatMap(heat_points, radius=12, blur=15, min_opacity=0.25).add_to(fmap)

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
        popup = folium.Popup(
            f"<b>Suggested new bin</b><br/>"
            f"Priority: <b>{priority}</b>/100<br/>"
            f"Activity index: <b>{s.activity_index}</b>/100<br/>"
            f"Nearest existing bin: <b>{s.nearest_bin_m:.0f} m</b> ({feet:.0f} ft)",
            max_width=280,
        )
        folium.CircleMarker(
            [s.lat, s.lon], radius=radius, color="#00aa44", weight=0, fill=True, fill_opacity=opacity,
            popup=popup, tooltip=f"Priority {priority}/100",
        ).add_to(fmap)

    return fmap


# ===========================================================================
# The page (this is what runs top-to-bottom when Streamlit loads)
# ===========================================================================
st.set_page_config(page_title="NYC Trash Bin Optimizer", layout="wide")

st.title("NYC Trash Bin Optimizer")
st.caption(
    "Flags city blocks where public trash bins are most likely needed — places with high "
    "local activity but poor existing bin coverage, using NYC's own public data."
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
            "Which signal to use as a stand-in for where people are.\n\n"
            "**Composite** blends 311 + transit (best).\n\n"
            "**311 / Subway / Citibike** isolate one signal.\n\n"
            "**NYPD incidents** is the original (weak in quiet residential areas).\n\n"
            "Build more sources by running `python data/build_proxy.py`."
        ),
    )

    borough = st.selectbox(
        "Borough",
        ["All Boroughs", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        help=(
            "Focus recommendations on one borough. Activity is ranked within the "
            "selected borough, so quieter boroughs like Queens are judged against "
            "themselves — not against dense Manhattan."
        ),
    )

    st.divider()

    sensitivity = st.slider(
        "Sensitivity", min_value=1, max_value=10, value=5,
        help=(
            "How active an area must be before it's flagged.\n\n"
            "**Low (1–3):** only the very busiest spots — fewest, highest-confidence picks.\n\n"
            "**Medium (5):** balanced.\n\n"
            "**High (8–10):** includes moderately active areas — more, broader picks."
        ),
    )

    min_distance_m = st.slider(
        "Minimum gap from existing bin (meters)", min_value=25, max_value=800, value=300, step=25,
        help=(
            "A spot is only suggested if no existing bin is within this distance.\n\n"
            "**300 m ≈ 1 city block.** Larger = only the most isolated gaps."
        ),
    )

    show_dot = st.checkbox(
        "Show DOT verified pedestrian counts", value=False,
        help=(
            "Overlays the 114 NYC DOT count locations as blue circles, sized by measured "
            "pedestrian volume. These are the only verified foot-traffic numbers the city "
            "publishes — useful as a reality check on the activity estimate."
        ),
    )

    show_all = st.checkbox(
        "Visualize everything", value=False,
        help=(
            "Adds an **activity heatmap** of every cell (the raw signal the whole tool is "
            "built on) and draws ALL existing bins and ALL suggestions, not just the top ones.\n\n"
            "Great for seeing the full picture. Tip: pick a single borough — drawing the whole "
            "city at full detail can make the map slow."
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

# Sensitivity 1-10 becomes a 0-100 cutoff:
#   sensitivity 1  -> cutoff 90 (only the top 10% busiest cells)
#   sensitivity 5  -> cutoff 50
#   sensitivity 10 -> cutoff 0  (every cell passes the busyness test)
threshold_index = (10 - sensitivity) * 10

suggested = suggest_new_bins(cand_df, threshold_index, float(min_distance_m), borough)
suggested = compute_priority(suggested, dot_all if len(dot_all) else None)

# What to show on the map for the chosen borough (existing bins + DOT points).
scope_bins = bins_df if borough == "All Boroughs" else bins_df[bins_df["borough"] == borough]
if scope_bins.empty:
    scope_bins = bins_df
scope_cells = cand_df if borough == "All Boroughs" else cand_df[cand_df["borough"] == borough]
dot_scope = dot_all if borough == "All Boroughs" else dot_all[dot_all["borough"] == borough]

# ---- Layout: map on the left, summary on the right ----
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Coverage Map")
    legend = "Red = existing bins  |  Green = suggested new locations (brightest = highest priority)"
    if show_dot and len(dot_scope):
        legend += "  |  Blue = DOT verified pedestrian counts"
    if show_all:
        legend += "  |  Heat = activity level"
    st.caption(legend + "  — click any marker for details")
    if show_all:
        st.caption("Visualize-everything mode: activity heatmap + all bins + all suggestions shown.")
        if borough == "All Boroughs":
            st.caption("Tip: pick a single borough if the full-city view feels slow.")
    elif len(suggested) > MAP_SUGG_CAP:
        st.caption(f"Showing the top {MAP_SUGG_CAP} suggestions by priority (of {len(suggested):,} total).")
    # returned_objects=[] tells st_folium not to send map state back to Python,
    # so panning/zooming the map doesn't trigger a full app rerun.
    st_folium(
        make_map(scope_bins, suggested, borough, dot_scope if show_dot else None,
                 cells_df=scope_cells, show_all=show_all),
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
        f"The city is split into **250 m × 250 m cells**. Each cell gets an **activity index "
        f"(0–100)** ranking it against others in the current view. A cell is suggested when its "
        f"index is **≥ {threshold_index}** *and* the nearest existing bin is more than "
        f"**{min_distance_m} m** away."
    )

    st.divider()
    st.caption(f"**About the data — {source_choice}**")
    composite_desc = (
        "A blend of " + " and ".join(composite_layers) + " — a proxy for where people are on foot."
        if composite_layers else
        "A blend of NYC activity signals — a proxy for where people are on foot."
    )
    SOURCE_BLURBS = {
        "Composite (311 + transit)": composite_desc,
        "311 street complaints": "Counts of outdoor 311 complaints (street/sidewalk conditions, "
            "litter, noise). Dense in residential areas where NYPD data is blind.",
        "Subway ridership": "MTA hourly ridership totals per station. Strong near transit, "
            "weaker away from stations.",
        "Citibike trips": "Citibike trip start/end counts per station. Good within the system "
            "footprint, absent outside it.",
        "NYPD incidents (911 calls)": "NYPD incident records used as a foot-traffic proxy. "
            "Over-weights commercial corridors and is weak in quiet residential blocks.",
    }
    st.caption(SOURCE_BLURBS.get(source_choice, ""))
    if proxy_df is None:
        st.caption("Run `python data/build_proxy.py` to add the richer 311/transit sources.")

    if len(suggested) > 0:
        st.divider()
        st.caption("**Priority list — build these first**")
        st.caption(
            "Priority (0–100) blends how busy the area is, how far the nearest bin is, "
            + ("and how close it sits to a verified DOT count." if len(dot_all) else "and coverage gap.")
        )

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
