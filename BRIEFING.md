# NYC Trash Bin Optimizer — Full Briefing

A complete explanation of how the app works, written so you can answer any question
about the core app and its logic. Read top-to-bottom once and you'll be ready.

---

## 1. The one-sentence pitch

> The app finds city blocks that are **busy with people** but **far from any existing
> trash bin**, and suggests those blocks as good spots for a new bin — ranked so a
> planner knows which to build first.

It is a **transparent, rule-based tool**, not a black-box AI. Every recommendation can
be explained with two facts: "this area is busy" and "the nearest bin is far away."

---

## 2. The whole process in four steps

**STEP 1 — LOAD** three public NYC datasets:
- **Activity data** — a stand-in ("proxy") for where people are on foot
- **Litter baskets** — where bins already exist (DSNY inventory)
- **DOT counts** — 114 locations with *real measured* pedestrian counts (a reality check)

**STEP 2 — SCORE** — chop the city into **250 m × 250 m squares** ("cells"). For each cell:
- **activity score** = how much activity happened inside it
- **nearest_bin distance** = how far the closest existing bin is

**STEP 3 — FILTER** — a cell becomes a **suggestion** only if BOTH are true:
- it's **busy enough** (activity above the Sensitivity cutoff), AND
- it's **far enough** from a bin (farther than the Minimum-gap setting)

**STEP 4 — RANK** — give each suggestion a **Priority score (0–100)** and sort the list,
then draw everything on the map.

That's the entire app. Everything below is detail on those four steps.

---

## 3. The data sources (this is a PROXY for foot traffic — be honest about it)

There is **no dataset that measures citywide pedestrian foot traffic.** So the app uses
proxies — public datasets that *correlate* with where people are. You pick one in the
**"Activity data source"** dropdown:

| Source | What each row is | Why it's a proxy | Weakness |
|---|---|---|---|
| **Composite** (default) | a pre-built grid blending the sources below | broadest signal | only as good as its parts |
| **311 street complaints** | one outdoor complaint (street/sidewalk, litter, noise) | more people → more complaints | biased toward who complains |
| **Subway ridership** | riders per station per hour | real people, measured | only meaningful near stations |
| **NYPD incidents** | one 911/incident record | more people → more incidents | skews to commercial/crime areas |

**Key honesty point:** the app does **not** count pedestrians or trash. It counts *records*
(complaints, rides, incidents) as a stand-in for "how many people are around." The 114
**DOT counts** are the only true measured foot-traffic numbers, and they're used as a
reality-check overlay, not as the main data.

---

## 4. EXACTLY how a trash-can location is found (trace one suggestion)

Follow a single block from raw data to a green dot:

1. **Every activity record is a point** with a latitude/longitude.
2. **Snap each point to a 250 m cell.** We convert lat/lon to meters (so distances are
   real), then divide by 250 and round down to get the cell's (x, y) id.
3. **Count points per cell.** A cell's **activity score** = how many records landed in it.
   (For the composite grid this is already done; the score is a blended number.)
4. **Rank cells into an "activity index" (0–100).** We take the percentile rank of each
   cell's score *within the current view*. So index 80 = "busier than 80% of cells here."
   - If a **borough** is selected, ranking happens **within that borough** — this is what
     keeps Queens judged against Queens, not crushed by Manhattan.
5. **Measure the gap to the nearest bin.** Using a fast nearest-neighbor index (a "KD-tree")
   built from all existing bins, we get each cell's distance to its closest bin, in meters.
6. **Apply the two rules:**
   - **busy enough:** activity index ≥ the Sensitivity cutoff
   - **far enough:** nearest bin distance ≥ the Minimum-gap setting
   - A cell that passes **both** becomes a suggestion. Its location is the cell's center.
7. **Rank the survivors by Priority** (Section 5) and draw them; the highest-priority dots
   are drawn larger and brighter.

**Say this out loud to the inspector:** *"A spot is recommended only when it's both busy
and underserved. Busy alone isn't enough; far-from-a-bin alone isn't enough. It needs both."*

---

## 5. The Priority ("relevance") score — exact formula

Once we have the list of suggestions, each gets a **Priority from 0–100**. All three
inputs are on a 0–100 scale:

```
With DOT data:     priority = 0.55 × activity + 0.35 × gap + 0.10 × dot
Without DOT data:  priority = 0.60 × activity + 0.40 × gap
```
The result is rounded to a whole number, and the list is sorted highest-first.

**What each term means:**
- **activity** = the cell's activity index (0–100) — how busy it is.
- **gap** = the distance to the nearest bin, percentile-ranked (0–100) **among the
  suggestions**. The most isolated suggestion ≈ 100; the least ≈ 0.
- **dot** = a corroboration bonus. For each suggestion we find the nearest of the 114
  DOT count locations:
  - if it's **within 500 m** → dot = that DOT point's pedestrian count, ranked 0–100
  - if the nearest DOT point is **farther than 500 m** → dot = 0

**Worked example (with DOT):** activity 90, gap 70, nearby DOT count ranks 80:
```
priority = 0.55×90 + 0.35×70 + 0.10×80 = 49.5 + 24.5 + 8.0 = 82
```

**IMPORTANT two-stage distinction (an inspector may probe this):**
- The DOT bonus is part of **ranking only** (Step 4). It affects the **order** suggestions
  are listed in.
- It does **NOT** affect **which** cells get suggested (Step 3 uses only activity + distance).
- So: DOT never changes *whether* a spot is recommended — it only nudges *which comes first*.

**The weights (0.55 / 0.35 / 0.10) are a deliberate design choice, not a derived optimum.**
They say "busyness matters most, coverage gap second, real-count corroboration is a small
tiebreaker." They can be adjusted.

---

## 6. The Sensitivity setting (how the slider becomes a cutoff)

Sensitivity is 1–10. Internally:
```
cutoff = (10 − sensitivity) × 10
```
- Sensitivity **1** → cutoff 90 → only the **top 10%** busiest cells qualify (strictest)
- Sensitivity **5** → cutoff 50 → the **top 50%**
- Sensitivity **10** → cutoff 0 → **every** cell passes the busyness test (broadest)

A cell qualifies if its **activity index ≥ cutoff**. Higher sensitivity = more suggestions.

---

## 7. Every control, and exactly what it does

- **Activity data source** — which proxy dataset to use (Section 3).
- **Borough** — limits suggestions to one borough AND ranks activity within it (fairness).
- **Sensitivity (1–10)** — the busyness cutoff (Section 6).
- **Minimum gap from existing bin (m)** — a cell is only suggested if no bin is within this
  distance. 300 m ≈ one city block. Larger = only the most isolated gaps.
- **Show DOT verified pedestrian counts** — overlays the 114 real count locations as blue
  circles, sized by measured volume. Pure reality-check overlay.
- **Visualize everything** — adds an **activity heatmap** of every cell (the raw input the
  whole tool runs on) and draws ALL bins and ALL suggestions instead of capped subsets.
  Display only — changes nothing about the recommendations.
- **Refresh data from NYC Open Data** — deletes local CSVs and re-downloads. (Don't click
  during a live demo on shaky wifi.)

---

## 8. The map — what every color means

- 🔴 **Red dots** — existing litter baskets (DSNY). Thinned for speed unless "Visualize
  everything" is on.
- 🟢 **Green dots** — suggested new bin locations. Bigger/brighter = higher priority.
  Click one to see its priority, activity index, and distance to nearest bin.
- 🔵 **Blue circles** — the 114 DOT measured-count locations (only if that toggle is on),
  sized by pedestrian volume.
- 🌡️ **Heat layer** — activity level per cell (only in "Visualize everything"). Hotter =
  busier. This is the input that produces the green dots.

---

## 9. Honest limitations (say these BEFORE they ask — it builds trust)

- **The activity layer is a proxy, not measured foot traffic.** Only the 114 DOT points are
  real counts; everything else is a stand-in.
- **Distance is straight-line, not walking distance.** A bin across a highway counts as close.
- **Suggestions are candidates, not final placements.** The tool doesn't yet exclude parks,
  cemeteries, highways, or water — a human filters those out.
- **One suggestion = one cell center.** It doesn't yet decide *how many* bins an area needs.
- **Data is a capped sample**, not every historical record, so very quiet areas can look
  emptier than reality.
- **311 bias:** complaint data over-represents neighborhoods that report more.

---

## 10. Likely DSNY questions + crisp answers

**"Is this measuring trash or litter?"**
No. It measures *where people are* (a proxy) and *where bins aren't*. The assumption: more
people + no nearby bin = more likely to need one.

**"Why are you using crime/911 data for foot traffic?"**
That's the original, weakest source and just a fallback. The default is the **composite**,
built mainly from **311 street complaints**, which track street activity in residential
areas far better. You can switch sources live in the dropdown.

**"How accurate is it?"**
It's a transparent heuristic, not a prediction model, so "accuracy" means "does the activity
proxy match reality?" We can check that against the 114 DOT measured counts — that's exactly
why they're built in as a reality check.

**"Does it tell me the exact spot to install a bin?"**
It points to a 250 m cell and its center. It's a **candidate** for a human to confirm —
it narrows thousands of blocks down to a ranked short list.

**"Why might my neighborhood show nothing?"**
Three reasons: it's genuinely well-covered by bins; it's parkland/highway with no activity;
or the activity data is sparse there (worse with the NYPD source, much better with 311).

**"Why 250 m cells and 300 m spacing?"**
250 m ≈ 1–2 blocks — fine enough to be local, coarse enough to smooth out noise. 300 m ≈ one
block — a reasonable minimum spacing between bins. Both are adjustable.

**"Can it recommend how many bins, or optimize a budget?"**
Not yet. It ranks *where*, one candidate per cell. Counting/optimization is a clear next step.

**"Is the data current?"**
It's pulled from NYC Open Data (the city's official open-data portal). The composite grid is
built from recent 311 and subway data; it can be rebuilt anytime by re-running the builder.

**"What stops it from suggesting a bin in the middle of a park or highway?"**
Right now, nothing automatic — that's a known limitation and a human filters those out. A
land-use mask is a planned improvement.

---

## 11. The 16 functions (one line each, in case they ask about code)

- `download_csv` / `ensure_data` — download a dataset from NYC Open Data if it's missing.
- `load_sources` — load the composite grid, the NYPD table, and the bins.
- `build_source_options` — decide which choices appear in the data-source dropdown.
- `activity_input` — hand the pipeline the right table for the chosen source.
- `load_dot_counts` — load the 114 DOT points (downloads once, then reads a local copy).
- `find_column` — find a column by name from a list of possibilities.
- `pick_lat_lon_columns` — figure out which columns hold lat/lon (or parse a "POINT(...)" text).
- `to_meters` — convert lat/lon to meters so distances are real.
- `assign_borough` — label a point with the borough whose box it falls in.
- `clean_latlon` — make a tidy numeric lat/lon table.
- `prepare_candidates` — **the core**: build the cells, score them, and find each cell's nearest bin.
- `suggest_new_bins` — **the filter**: keep cells that are busy enough AND far enough from a bin.
- `compute_priority` — **the ranking**: give each suggestion a 0–100 priority and sort.
- `thin_points` — reduce a huge set of map dots to a representative sample (for speed).
- `make_map` — draw the Folium map (bins, suggestions, optional DOT circles + heatmap).

---

## 12. The 5 "advanced" lines and your one-sentence answer for each

- **`cKDTree`** → "a fast index that finds the closest bin to each cell."
- **`pyproj Transformer`** → "converts lat/lon degrees into meters so distance math works."
- **`rank(pct=True)`** → "turns a raw number into a 0–100 percentile."
- **`groupby(...).agg(...)`** → "counts how many points fall into each grid square."
- **the regex `POINT\(...\)`** → "pulls the two coordinate numbers out of a text string."

That's the complete list of non-obvious code. Everything else is plain arithmetic and `if`s.
