"""
build_proxy.py - builds the "composite" activity data for the Trash Bin Optimizer.
====================================================================================

You run this ONCE, offline:

    python data/build_proxy.py

It writes  data/activity_grid.csv , and the app automatically uses that file if it
exists (otherwise the app falls back to the basic NYPD data).

WHY THIS FILE EXISTS
--------------------
The app needs a number for "how busy is each 250m square of the city?" No single
dataset measures that, so this script blends several public datasets that each hint
at where people are, into one score per square.

WHAT IT DOES (same four-ish steps for every data source)
--------------------------------------------------------
    1. DOWNLOAD a dataset of points (e.g. every 311 street complaint).
    2. SNAP each point to a 250m grid square.
    3. COUNT (or sum) the points in each square  -> a "score" for that square.
    4. COMBINE all the sources into one weighted "composite" score per square,
       then save lat/lon + score for every square to a CSV.

THE DATA SOURCES
----------------
    1. 311 street complaints  (NYC)   - dense everywhere people live   -> score_311
    2. Subway ridership       (MTA)   - strong near stations           -> score_mta
    3. Citibike trips         (S3)    - only inside the bike network    -> score_citibike
    4. DOT pedestrian counts  (NYC)   - 114 REAL counts, used as a check (not a score)
    5. LODES population       (Census)- optional, to divide out density (off by default)

Every download is cached in  _proxy_cache/  so re-running is fast. If a source can't
be reached, that layer is simply skipped instead of crashing the whole build.
"""

import io
import os
import zipfile
import warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")


# ===========================================================================
# Settings
# ===========================================================================
DATA_DIR  = Path(__file__).parent
OUT_CSV   = DATA_DIR / "activity_grid.csv"      # the file this script produces
CACHE_DIR = DATA_DIR / "_proxy_cache"           # downloaded data is cached here
CACHE_DIR.mkdir(exist_ok=True)

GRID_SIZE_M      = 250        # square size in meters - must match app.py
EQUITY_FLOOR_PCT = 0.05       # populated squares never score below the 5th percentile

# How much data to pull / which optional steps to run.
LIMIT_311    = 150_000                                  # number of 311 complaints to download
ENABLE_LODES = os.environ.get("ENABLE_LODES", "1") == "1"   # population denominator; on by default
DOT_DATASET  = "cqsj-cfgu"                              # the real DOT counts table

# NYC Open Data lives on two portals (city + state).
SOCRATA_CITY  = "https://data.cityofnewyork.us/resource"
SOCRATA_STATE = "https://data.ny.gov/resource"

# Optional free API token (speeds up downloads). Empty = works but slower/rate-limited.
SOCRATA_APP_TOKEN = os.environ.get("SOCRATA_APP_TOKEN", "")

# The first 5 digits of a census block ID tell you the borough.
BOROUGH_FIPS = {
    "36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
    "36081": "Queens", "36085": "Staten Island",
}

# Rough lat/lon box for each borough (lat_min, lat_max, lon_min, lon_max).
BOROUGH_BOUNDS = {
    "Manhattan":     (40.685, 40.882, -74.020, -73.907),
    "Brooklyn":      (40.570, 40.740, -74.042, -73.833),
    "Queens":        (40.490, 40.800, -73.962, -73.700),
    "Bronx":         (40.785, 40.918, -73.934, -73.765),
    "Staten Island": (40.477, 40.651, -74.260, -74.034),
}


# ===========================================================================
# Small shared helpers
# ===========================================================================
def log(msg: str) -> None:
    """Print a progress line."""
    print(f"  {msg}", flush=True)


def socrata_headers() -> dict:
    """Add the API token to a request if we have one."""
    headers = {"Accept": "application/json"}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN
    return headers


def fetch_socrata_csv(domain, dataset_id, where, limit, cache_name, select=None) -> pd.DataFrame:
    """Download a dataset from NYC Open Data as a table (and cache it to disk)."""
    cache_path = CACHE_DIR / f"{cache_name}.parquet"
    if cache_path.exists():
        log(f"Cache hit: {cache_name}")
        return pd.read_parquet(cache_path)

    params = {"$limit": limit, "$where": where}
    if select:
        params["$select"] = select         # only download the columns we need = much faster
    if SOCRATA_APP_TOKEN:
        params["$$app_token"] = SOCRATA_APP_TOKEN

    log(f"Downloading {cache_name} (~{limit:,} rows)...")
    response = requests.get(f"{domain}/{dataset_id}.csv", params=params, timeout=300)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df.to_parquet(cache_path, index=False)
    log(f"  {len(df):,} rows cached")
    return df


def assign_borough(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Label each point with the borough whose box it falls inside (first match wins)."""
    result = pd.Series("Other", index=lat.index)
    for name, (lat_lo, lat_hi, lon_lo, lon_hi) in BOROUGH_BOUNDS.items():
        not_yet_set = result == "Other"
        inside_box = lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi)
        result[not_yet_set & inside_box] = name
    return result


def to_grid_cells(lat, lon, weight=None) -> pd.DataFrame:
    """Snap each lat/lon point to a 250m square.

    Returns one row per input point: which square it's in (grid_x, grid_y) and its
    weight (1 for a plain point, or e.g. ridership when we want to sum a value).
    """
    from pyproj import Transformer
    to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = to_meters.transform(lon.values, lat.values)
    grid_x = np.floor(x / GRID_SIZE_M).astype(int)
    grid_y = np.floor(y / GRID_SIZE_M).astype(int)
    w = weight.values if weight is not None else np.ones(len(lat))
    return pd.DataFrame({"grid_x": grid_x, "grid_y": grid_y, "w": w})


def aggregate_to_grid(cells_df: pd.DataFrame) -> pd.Series:
    """Add up the weights in each square. Result is a score per (grid_x, grid_y)."""
    return cells_df.groupby(["grid_x", "grid_y"])["w"].sum()


# ===========================================================================
# Source 1 - NYC 311 street complaints
# ===========================================================================
# We only keep OUTDOOR complaint types (these track street activity, not indoor issues).
STREET_311_TYPES = (
    "'Street Condition','Sidewalk Condition','Dirty Conditions',"
    "'Litter Basket / Request','Derelict Vehicles',"
    "'Noise - Street/Sidewalk','Blocked Driveway'"
)

def fetch_311() -> pd.Series:
    """Score per square = number of 311 street complaints in it.

    The dataset is huge, so we download it in 50k-row pages and retry a page up to
    3 times - that way one slow request can't kill the whole download.
    """
    cache_path = CACHE_DIR / "311_street.parquet"
    try:
        if cache_path.exists():
            log("Cache hit: 311_street")
            df = pd.read_parquet(cache_path)
        else:
            where = f"complaint_type in({STREET_311_TYPES}) AND latitude IS NOT NULL"
            page_size, pages, total = 50_000, [], 0
            for offset in range(0, LIMIT_311, page_size):
                params = {"$limit": page_size, "$offset": offset, "$where": where,
                          "$select": "latitude,longitude", "$order": ":id"}
                if SOCRATA_APP_TOKEN:
                    params["$$app_token"] = SOCRATA_APP_TOKEN

                page = None
                for attempt in range(3):       # retry a page up to 3 times
                    try:
                        response = requests.get(f"{SOCRATA_CITY}/fhrw-4uyv.csv", params=params, timeout=120)
                        response.raise_for_status()
                        page = pd.read_csv(io.StringIO(response.text))
                        break
                    except Exception as e:
                        log(f"  311 page @{offset} attempt {attempt+1} failed ({str(e)[:60]})")

                if page is None or page.empty:
                    break
                pages.append(page)
                total += len(page)
                log(f"  311: {total:,} rows so far")
                if len(page) < page_size:      # last page reached
                    break

            if not pages:
                raise RuntimeError("no 311 pages returned")
            df = pd.concat(pages, ignore_index=True)
            df.to_parquet(cache_path, index=False)
            log(f"  311 total: {len(df):,} rows cached")

        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        cells = to_grid_cells(df["latitude"], df["longitude"])
        return aggregate_to_grid(cells).rename("score_311")
    except Exception as e:
        log(f"WARNING: 311 fetch failed ({e}). Skipping 311 layer.")
        return pd.Series(dtype=float, name="score_311")


# ===========================================================================
# Source 2 - MTA subway ridership
# ===========================================================================
def fetch_mta() -> pd.Series:
    """Score per square = total subway riders at stations in it.

    Each row is one station for one hour, so we SUM the ridership (the weight) per square.
    """
    try:
        cache_path = CACHE_DIR / "mta_ridership.parquet"
        if cache_path.exists():
            log("Cache hit: mta_ridership")
            df = pd.read_parquet(cache_path)
        else:
            log("Downloading MTA subway ridership...")
            params = {"$limit": 300_000, "$select": "latitude,longitude,ridership"}
            response = requests.get(f"{SOCRATA_STATE}/wujg-7c2s.csv", params=params,
                                    headers=socrata_headers(), timeout=300)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.to_parquet(cache_path, index=False)
            log(f"  {len(df):,} rows cached")

        for col in ["latitude", "longitude", "ridership"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "ridership"])
        cells = to_grid_cells(df["latitude"], df["longitude"], weight=df["ridership"])
        return aggregate_to_grid(cells).rename("score_mta")
    except Exception as e:
        log(f"WARNING: MTA fetch failed ({e}). Skipping MTA layer.")
        return pd.Series(dtype=float, name="score_mta")


# ===========================================================================
# Source 3 - Citibike trips (most recent month)
# ===========================================================================
def fetch_citibike() -> pd.Series:
    """Score per square = number of Citibike trip starts/ends in it.

    Citibike publishes monthly zip files; we try the last few months until one
    downloads. If none work (common), we just skip this layer.
    """
    cache_path = CACHE_DIR / "citibike_stations.parquet"
    if cache_path.exists():
        log("Cache hit: citibike_stations")
        df = pd.read_parquet(cache_path)
    else:
        log("Fetching Citibike trip data from S3...")
        import datetime
        found = None
        for months_back in range(1, 6):           # try the last 5 months
            month = datetime.date.today().replace(day=1)
            for _ in range(months_back):
                month = (month - datetime.timedelta(days=1)).replace(day=1)
            slug = month.strftime("%Y%m")
            url = f"https://s3.amazonaws.com/tripdata/{slug}-citibike-tripdata.csv.zip"
            try:
                response = requests.get(url, timeout=120)
                if response.status_code == 200:
                    found = (slug, response.content)
                    break
            except Exception:
                continue

        if found is None:
            log("  WARNING: Could not fetch Citibike data. Skipping.")
            return pd.Series(dtype=float, name="score_citibike")

        slug, content = found
        log(f"  Using {slug} Citibike data")
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            raw = pd.read_csv(zf.open(csv_name))

        # Column names have changed over the years - find the start/end lat/lon columns.
        raw.columns = raw.columns.str.lower().str.replace(" ", "_")
        lat_start = next((c for c in raw.columns if "start" in c and "lat" in c), None)
        lon_start = next((c for c in raw.columns if "start" in c and ("lon" in c or "lng" in c)), None)
        lat_end   = next((c for c in raw.columns if "end"   in c and "lat" in c), None)
        lon_end   = next((c for c in raw.columns if "end"   in c and ("lon" in c or "lng" in c)), None)

        # Treat both the start point and the end point of each trip as activity.
        endpoints = []
        for lat_c, lon_c in [(lat_start, lon_start), (lat_end, lon_end)]:
            if lat_c and lon_c:
                sub = raw[[lat_c, lon_c]].copy()
                sub.columns = ["latitude", "longitude"]
                endpoints.append(sub)

        if not endpoints:
            log("  WARNING: Could not parse Citibike columns. Skipping.")
            return pd.Series(dtype=float, name="score_citibike")

        df = pd.concat(endpoints, ignore_index=True)
        for col in ["latitude", "longitude"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        df.to_parquet(cache_path, index=False)
        log(f"  {len(df):,} trip endpoints cached")

    cells = to_grid_cells(df["latitude"], df["longitude"])
    return aggregate_to_grid(cells).rename("score_citibike")


# ===========================================================================
# Source 4 - DOT pedestrian counts (a REALITY CHECK, not a score)
# ===========================================================================
def fetch_dot_counts() -> pd.DataFrame:
    """The 114 locations where DOT actually measured pedestrians.

    Returns latitude/longitude/ped_count. The location is stored as text like
    "POINT (-73.98 40.75)", and ped_count is the average of the 6 most recent counts.
    """
    cache_path = CACHE_DIR / "dot_counts.parquet"
    if cache_path.exists():
        log("Cache hit: dot_counts")
        return pd.read_parquet(cache_path)

    try:
        log("Downloading DOT pedestrian counts...")
        response = requests.get(f"{SOCRATA_CITY}/{DOT_DATASET}.csv", params={"$limit": 500}, timeout=60)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))

        # Everything that isn't a label column is a pedestrian-count column.
        non_count = {"the_geom", "objectid", "loc", "borough",
                     "street_nam", "from_stree", "to_street", "iex"}
        count_cols = [c for c in raw.columns if c not in non_count]

        point = raw["the_geom"].astype(str).str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)")
        counts = raw[count_cols].apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame({
            "longitude": pd.to_numeric(point[0], errors="coerce"),
            "latitude":  pd.to_numeric(point[1], errors="coerce"),
            # average the 6 newest counts; if blank, fall back to all counts
            "ped_count": counts[count_cols[-6:]].mean(axis=1, skipna=True).fillna(
                counts.mean(axis=1, skipna=True)),
        }).dropna(subset=["latitude", "longitude", "ped_count"])
        log(f"  {len(out)} DOT calibration locations")
    except Exception as e:
        log(f"  WARNING: DOT fetch failed ({e}). Calibration skipped.")
        out = pd.DataFrame(columns=["latitude", "longitude", "ped_count"])

    out.to_parquet(cache_path, index=False)
    return out


# ===========================================================================
# Source 5 - LODES population (the denominator: how many people are present)
# ===========================================================================
# Census LEHD LODES gives jobs-per-block (daytime) and residents-per-block
# (nighttime). We use jobs + residents as a behavior-independent population count,
# so the proxies can be measured as activity PER PERSON instead of raw volume.
# LODES counts are privacy-fuzzed at the block level, so we aggregate up to block
# group (the first 12 digits of the 15-digit census GEOID) to smooth them, then
# join to TIGER 2020 block-group center points to get lat/lon.
LODES_BASE   = "https://lehd.ces.census.gov/data/lodes/LODES8/ny"
TIGER_BG_URL = "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_36_bg.zip"  # all of NY state
LODES_YEAR   = 2022

def fetch_lodes_denominator() -> pd.Series:
    """Population (daytime jobs + nighttime residents) summed per 250m square.

    Returns a Series indexed by (grid_x, grid_y), named lodes_pop. Empty on failure.
    """
    cache_path = CACHE_DIR / "lodes_grid.parquet"
    if cache_path.exists():
        log("Cache hit: lodes_grid")
        return pd.read_parquet(cache_path).set_index(["grid_x", "grid_y"])["lodes_pop"]

    boroughs = set(BOROUGH_FIPS)   # the 5 borough county prefixes

    def lodes_by_block_group(kind, value_name):
        # kind "wac" = jobs at workplace (daytime); "rac" = workers by home (nighttime)
        fname = f"ny_{kind}_S000_JT00_{LODES_YEAR}.csv.gz"
        cache = CACHE_DIR / fname
        if cache.exists():
            log(f"  Cache hit: {fname}")
            raw = cache.read_bytes()
        else:
            log(f"  Downloading {fname}...")
            r = requests.get(f"{LODES_BASE}/{kind}/{fname}", timeout=300)
            r.raise_for_status()
            cache.write_bytes(r.content)
            raw = r.content
        geo = "w_geocode" if kind == "wac" else "h_geocode"
        df = pd.read_csv(io.BytesIO(raw), compression="gzip", dtype={geo: str})
        df[geo] = df[geo].str.zfill(15)
        df = df[df[geo].str[:5].isin(boroughs)]   # keep the 5 boroughs
        df["bg"] = df[geo].str[:12]               # first 12 digits = block group
        return df.groupby("bg")["C000"].sum().rename(value_name).reset_index()

    try:
        jobs = lodes_by_block_group("wac", "jobs")
        residents = lodes_by_block_group("rac", "residents")

        # TIGER block-group center points for NY, then keep the 5 boroughs.
        zcache = CACHE_DIR / "tiger_bg_36.zip"
        if not zcache.exists():
            log("  Downloading TIGER 2020 block groups (NY state)...")
            r = requests.get(TIGER_BG_URL, timeout=300)
            r.raise_for_status()
            zcache.write_bytes(r.content)
        zdir = CACHE_DIR / "tiger_bg_36"
        if not zdir.exists():
            with zipfile.ZipFile(zcache) as zf:
                zf.extractall(zdir)
        shp = next(zdir.glob("*.shp"))
        bg = gpd.read_file(shp)
        geoid = "GEOID" if "GEOID" in bg.columns else next(c for c in bg.columns if c.upper().startswith("GEOID"))
        bg = bg[[geoid, "geometry"]].rename(columns={geoid: "bg"})
        bg = bg[bg["bg"].str[:5].isin(boroughs)].to_crs("EPSG:4326")
        centers = bg.geometry.centroid
        bg = pd.DataFrame({"bg": bg["bg"].values, "lat": centers.y.values, "lon": centers.x.values})

        # Attach jobs + residents to each block-group center, then sum into the grid.
        merged = bg.merge(jobs, on="bg", how="left").merge(residents, on="bg", how="left").fillna(0)
        merged["lodes_pop"] = merged["jobs"] + merged["residents"]
        cells = to_grid_cells(merged["lat"], merged["lon"], weight=merged["lodes_pop"])
        grid_pop = aggregate_to_grid(cells).rename("lodes_pop")

        grid_pop.reset_index().to_parquet(cache_path, index=False)
        log(f"  LODES grid: {len(grid_pop):,} cells, {merged['lodes_pop'].sum():,.0f} day+night people")
        return grid_pop
    except Exception as e:
        log(f"WARNING: LODES denominator failed ({e}). Skipping.")
        return pd.Series(dtype=float, name="lodes_pop")


# ===========================================================================
# DOT reality-check column (diagnostic only - not used to place bins)
# ===========================================================================
def dot_calibration_flag(grid_df: pd.DataFrame, dot_df: pd.DataFrame) -> pd.Series:
    """For each square, the real DOT count of the nearest DOT location within 500m
    (or blank if none is that close). Lets us later compare our estimate to ground truth.
    """
    if dot_df.empty or "latitude" not in dot_df.columns:
        return pd.Series(np.nan, index=grid_df.index, name="dot_calibration")

    from scipy.spatial import cKDTree
    from pyproj import Transformer
    to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    dot_valid = dot_df.dropna(subset=["latitude", "longitude"])
    if dot_valid.empty:
        return pd.Series(np.nan, index=grid_df.index, name="dot_calibration")

    # Build a fast nearest-point index of the DOT locations (in meters).
    dx, dy = to_meters.transform(dot_valid["longitude"].values, dot_valid["latitude"].values)
    tree = cKDTree(np.column_stack([dx, dy]))

    # Each square's center, in meters, then find its nearest DOT point.
    cx = grid_df["grid_x"].values * GRID_SIZE_M + GRID_SIZE_M / 2
    cy = grid_df["grid_y"].values * GRID_SIZE_M + GRID_SIZE_M / 2
    distances, nearest = tree.query(np.column_stack([cx, cy]), k=1)

    nearby_count = np.where(distances <= 500, dot_valid["ped_count"].values[nearest], np.nan)
    return pd.Series(nearby_count, index=grid_df.index, name="dot_calibration")


# ===========================================================================
# Combine everything into one grid (this is the heart of the script)
# ===========================================================================
def build_grid() -> pd.DataFrame:
    # --- Get a score-per-square from each source ---
    print("\n[1/5] Fetching 311 street complaints...")
    score_311 = fetch_311()
    print("\n[2/5] Fetching MTA subway ridership...")
    score_mta = fetch_mta()
    print("\n[3/5] Fetching Citibike trip endpoints...")
    score_citibike = fetch_citibike()
    print("\n[4/5] Fetching DOT pedestrian counts (calibration)...")
    dot_df = fetch_dot_counts()

    print("\n[5/5] Fetching LODES population denominator...")
    if ENABLE_LODES:
        try:
            lodes = fetch_lodes_denominator()
        except Exception as e:
            log(f"WARNING: LODES/TIGER step failed ({e}). Population normalization skipped.")
            lodes = pd.Series(dtype=float, name="lodes_pop")
    else:
        log("LODES disabled (set ENABLE_LODES=1 to add population normalization). Skipping.")
        lodes = pd.Series(dtype=float, name="lodes_pop")

    print("\nBuilding composite grid...")

    # --- Put every square that ANY source touched onto one shared list ---
    all_squares = set(score_311.index) | set(score_mta.index) | set(score_citibike.index)
    index = pd.MultiIndex.from_tuples(sorted(all_squares), names=["grid_x", "grid_y"])

    # Line up each source against that shared list (missing squares = 0).
    grid = pd.DataFrame(index=index)
    grid["score_311"]      = score_311.reindex(index, fill_value=0)
    grid["score_mta"]      = score_mta.reindex(index, fill_value=0)
    grid["score_citibike"] = score_citibike.reindex(index, fill_value=0) if not score_citibike.empty else 0.0
    grid["lodes_pop"]      = lodes.reindex(index, fill_value=0)

    # --- Raw composite: the weighted blend of raw scores. This is the app's default
    #     activity score (unchanged from before). MTA weighted highest (real measured
    #     riders), then Citibike, then 311 as the dense base. ---
    grid["composite_raw"] = (
        1.0 * grid["score_311"] +
        2.0 * grid["score_mta"] +
        1.5 * grid["score_citibike"]
    )

    # --- Per-person composite: divide each source by population FIRST, so a square is
    #     scored on activity-per-person, not raw crowd. This is the bias correction;
    #     the app offers it as a separate source so a demo can compare raw vs per-person.
    #     (+1 avoids dividing by zero. If LODES is unavailable, lodes_pop is 0 and this
    #     equals the raw composite.) ---
    denominator = grid["lodes_pop"] + 1
    grid["norm_311"]      = grid["score_311"]      / denominator
    grid["norm_mta"]      = grid["score_mta"]      / denominator
    grid["norm_citibike"] = grid["score_citibike"] / denominator
    grid["composite_perperson"] = (
        1.0 * grid["norm_311"] +
        2.0 * grid["norm_mta"] +
        1.5 * grid["norm_citibike"]
    )

    # --- Fairness floor (applied to the per-person score): a populated square never
    #     drops below the 5th percentile, so inhabited areas are not starved by gaps
    #     in correlated proxies. ---
    inhabited = grid["lodes_pop"] > 0
    if inhabited.any():
        floor_value = float(grid.loc[inhabited, "composite_perperson"].quantile(EQUITY_FLOOR_PCT))
        grid.loc[inhabited & (grid["composite_perperson"] < floor_value), "composite_perperson"] = floor_value
        grid["equity_floored"] = inhabited & (grid["composite_perperson"] <= floor_value)
    else:
        grid["equity_floored"] = False

    # --- How much the sources disagree per square (high = fragile estimate) ---
    grid["proxy_divergence"] = grid[["norm_311", "norm_mta", "norm_citibike"]].std(axis=1)

    # --- Turn each square's (grid_x, grid_y) back into a real lat/lon center point ---
    from pyproj import Transformer
    to_latlon = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    grid_x = grid.index.get_level_values("grid_x").values
    grid_y = grid.index.get_level_values("grid_y").values
    center_x = grid_x * GRID_SIZE_M + GRID_SIZE_M / 2
    center_y = grid_y * GRID_SIZE_M + GRID_SIZE_M / 2
    lons, lats = to_latlon.transform(center_x, center_y)

    grid = grid.reset_index()
    grid["lat"] = lats
    grid["lon"] = lons
    grid["borough"] = assign_borough(pd.Series(lats), pd.Series(lons))

    # Keep only squares that fall inside one of the 5 boroughs.
    grid = grid[grid["borough"] != "Other"].copy()

    # Attach the DOT reality-check column.
    grid["dot_calibration"] = dot_calibration_flag(grid, dot_df).values

    # The app reads "activity_score" by default - that's the raw composite (unchanged).
    grid["activity_score"] = grid["composite_raw"]

    log(f"Grid complete: {len(grid):,} cells across NYC")
    return grid


# ===========================================================================
# Run it
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NYC Trash Bin Optimizer - Pedestrian Proxy Grid Builder")
    print("=" * 60)

    if not SOCRATA_APP_TOKEN:
        print("\nTIP: Set SOCRATA_APP_TOKEN env var for faster 311 downloads.")
        print("     Free at https://data.cityofnewyork.us/profile/app_tokens\n")

    grid = build_grid()

    # Save only the columns the app + diagnostics need (this column order is fixed).
    keep_cols = [
        "lat", "lon", "borough", "activity_score", "composite_perperson",
        "score_311", "score_mta", "score_citibike",
        "lodes_pop", "proxy_divergence", "equity_floored", "dot_calibration",
    ]
    grid[keep_cols].to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(f"Rows:  {len(grid):,}")
    print(f"Boroughs: {grid['borough'].value_counts().to_dict()}")
    print("\nProxy divergence summary (high = fragile estimate):")
    print(grid["proxy_divergence"].describe().round(4).to_string())
    print("\nDone. Run `streamlit run app.py` - it will pick up activity_grid.csv automatically.")
