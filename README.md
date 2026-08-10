# NYC Trash Bin Optimizer

An interactive map that flags city blocks where NYC should consider adding public
trash bins - places with **high local activity** but **poor existing bin coverage** -
using the city's own open data.

## The idea in one sentence

> Find blocks that are busy with people but far from any existing trash bin.

## How it works (four steps)

1. **Load** three kinds of public NYC data:
   - **Activity data** - a stand-in for where people are on foot (see "Data sources" below)
   - **Litter baskets** - where bins already exist (DSNY inventory)
   - **DOT counts** - 114 locations with *real, measured* pedestrian counts (used as a reality check)

2. **Score** - split the city into **250 m × 250 m cells**. Each cell gets an
   **activity score** = how much activity happened inside it. Then, for every cell,
   measure the **walking distance along real streets** to the nearest existing basket
   (precomputed offline; straight-line is kept only as a fallback).

3. **Filter** - keep a cell as a suggestion only if it is **both**:
   - busy enough (activity above the *Sensitivity* cutoff), **and**
   - far enough from a basket (farther than the *Minimum gap* setting).

4. **Rank** - give each suggestion a **Priority score (0-100)** so a planner knows
   which to build first, and draw everything on a map.

## Built for DSNY (what it does beyond the four steps)

- **Task presets** - pick a job (find new corners, rebalance existing bins, plan a
  district, respond to complaints) and the rules set themselves.
- **Corner-level output** - each suggestion snaps to a real named street intersection
  (e.g. "Broadway & W 145 St"), from 59k corners built off the city street centerline.
- **DSNY eligibility gate** - keep only commercial / mixed-use or near-transit corners
  (PLUTO land use + subway entrances); drop residential, parks, industrial, highway.
- **Calibrated spacing** - the default minimum gap is the *measured* median spacing of
  existing commercial-area baskets, not an invented number.
- **Relocation mode (net-zero)** - moves a basket that is already *redundant* (another
  within 120 m, in a quiet ineligible spot) onto a busy corner. Each basket is used at
  most once, and moves stay **inside the same sanitation district** and under 800 m,
  because baskets are serviced on district collection routes.
- **Budget cap, per-district shortlist, district scorecard** (baskets per 1,000
  residents + underserved rank), **household-misuse risk flag**, **confidence flag**.
- **Servicing load** - baskets added per district, since the real constraint is crew
  time on a collection route, not the price of the basket.
- **Community board labels** - "BKS06 (Brooklyn CB 6)", the unit planners actually use.
- **Validation panel** - recovery vs. a chance baseline, a hold-out AUC, and an explicit
  statement of where the model fails.
- **Community reports** - residents can flag a corner with a note and photo.
- **Search box, clickable existing baskets, business overlay (DOHMH), BID tagging.**
- **Exports** - CSV, GeoJSON (for GIS), and a printable HTML report that carries a
  **field-verification checklist** (curb ramps, hydrant setback, bus pads, cellar doors).

All layers are bundled, so the app downloads nothing at runtime.

## Data sources (the activity layer is a *proxy*, not a true count)

No dataset measures citywide pedestrian foot traffic, so the app uses proxies you can
switch between in the **"Activity data source"** dropdown:

| Source | What it is | Strength / weakness |
|---|---|---|
| **Composite** (default) | A blend of the proxies below | Broadest signal |
| **311 street complaints** | Counts of outdoor 311 reports (street/sidewalk, litter, noise) | Dense in residential areas |
| **Subway ridership** | MTA hourly ridership per station | Strong near transit only |
| **NYPD incidents** | 911/incident records | The original; weak in quiet residential blocks |

The **activity index** shown in the app is a **0-100 percentile rank** - it means
"busier than X% of other cells in view," **not** a literal count of people. Ranking is
done **within the selected borough**, so quieter boroughs (e.g. Queens) are judged
against themselves, not against dense Manhattan.

## Priority score (exact formula)

For each suggestion, with all parts on a 0-100 scale:

- **activity** = the cell's activity index
- **gap** = how far the nearest bin is, ranked 0-100 among the suggestions
- **dot** = nearby verified DOT count (0-100 rank), but only if a DOT point is within 500 m, else 0
- **commercial** = commercial floor area, ranked 0-100 (only when "Prioritize commercial" is on)

```
DOT + commercial:  priority = 0.45 * activity + 0.30 * gap + 0.10 * dot + 0.15 * commercial
with DOT data:     priority = 0.55 * activity + 0.35 * gap + 0.10 * dot
with commercial:   priority = 0.50 * activity + 0.35 * gap + 0.15 * commercial
neither:           priority = 0.60 * activity + 0.40 * gap
```

The weights are a deliberate design choice, not a fitted optimum.

Suggestions are sorted highest-priority first, and the map draws higher-priority dots
larger and brighter.

## Controls

- **Activity data source** - which proxy to use
- **Borough** - focus on one borough (activity is re-ranked within it)
- **Sensitivity (1-10)** - how busy a cell must be. Internally this becomes a cutoff:
  `cutoff = (10 − sensitivity) × 10`. So sensitivity 1 → top 10% only; 5 → top 50%; 10 → all.
- **Minimum gap (m)** - a cell is only suggested if no bin is within this distance (300 m ≈ 1 block)
- **Show DOT verified pedestrian counts** - overlay the 114 real count locations as blue circles

You can also **download the full priority list as a CSV**.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

The app reads local CSVs in `data/`. On first run it downloads anything missing from
NYC Open Data and saves it, so later runs work offline.

## Community reports: pointing them at Supabase (do this before sharing the link)

Reports work with no setup, but they are then written to the server's own disk, which
**Streamlit Cloud wipes on every restart and redeploy**. To make them durable, give the
app a Postgres database. The app creates its own table on first use.

1. Create a free project at [supabase.com](https://supabase.com).
2. In the Supabase dashboard: **Project Settings, Database, Connection string, URI**.
   Copy the **connection pooler** string (port `6543`), not the direct one.
3. In Streamlit Cloud: **your app, Settings, Secrets**, and paste:

   ```toml
   [connections.reports]
   url = "postgresql://postgres.PROJECT:PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres"
   ```

   For local development, put the same block in `.streamlit/secrets.toml`
   (already git-ignored, so the password is never committed).
4. Redeploy. The banner saying reports are not durable disappears once it connects.

Photos are stored in the database, capped at 3 MB each to stay inside Supabase's free
500 MB. With no `[connections.reports]` secret the app silently falls back to the CSV,
so nothing breaks when you run it locally.

## Building the better composite data (optional)

The composite source comes from `data/activity_grid.csv`, produced offline by:

```bash
python data/build_proxy.py
```

This downloads 311 + subway data, aggregates them into the 250 m grid, and writes
`activity_grid.csv`. Optional layers (Citibike, and LODES population normalization) can be
enabled but may need a good connection - Citibike depends on Citibike's S3 files being
reachable, and LODES is turned on with `ENABLE_LODES=1`. If a source can't be reached,
the build simply skips that layer instead of failing.

## Honest limitations (good to know for questions)

- The activity layer is a **proxy**, not measured foot traffic. The DOT counts (114 points)
  are the only ground truth, used as a reality check - not to place baskets.
- **How well does it actually work?** On a hold-out test (hide 20% of baskets, then ask how
  often the model scores a real basket location above a random street cell) it reaches
  **AUC 0.66**, where 0.50 is chance. That is a real but moderate signal: good for
  narrowing ~15,000 blocks to a shortlist, not for trusting one corner over the next.
- **Where it fails:** on baskets that sit alone (200 m from any other) it scores **0.50 -
  pure chance**. The model ranks by how busy a place is, so it cannot find the quiet,
  isolated spots DSNY still baskets. Use it for busy corridors, not for those.
- Suggestions are **desk-research candidates, not approved sites.** Nothing is inspected;
  siting still has to clear curb ramps, hydrant setbacks, bus stops, cellar doors, and the
  district superintendent. The printable report carries that checklist.
- Boroughs come from the **DSNY district code**, not lat/lon rectangles. (The rectangles
  overlap and had mislabelled 18% of cells, including over half of "Manhattan".)
- **Community reports are not durable on Streamlit Cloud.** They are written to the
  server's own disk, which is wiped on every restart and redeploy. Use the in-app
  download to back them up, or move the store to an external database.
- Data is pulled as a capped sample, not every historical record.
