"""
Download NYC OpenData CSVs used by the app (no API key needed for basic access).

Usage:
  python data/download_data.py --ped 100000 --bins 100000
"""
import os
import argparse
import requests

PEDESTRIAN_DATASET = "d6zx-ckhd"
BINS_DATASET = "yg8p-rkbm"
SOCRATA_BASE = "https://data.cityofnewyork.us/resource"

def socrata_csv_url(dataset_id: str, limit: int) -> str:
    return f"{SOCRATA_BASE}/{dataset_id}.csv?$limit={int(limit)}"

def download(url: str, out_path: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ped", type=int, default=50000, help="Pedestrian rows to download")
    ap.add_argument("--bins", type=int, default=50000, help="Bin rows to download")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    out_dir = here
    ped_path = os.path.join(out_dir, "pedestrian_counts.csv")
    bins_path = os.path.join(out_dir, "litter_baskets.csv")

    os.makedirs(out_dir, exist_ok=True)
    print("Downloading pedestrian counts...")
    download(socrata_csv_url(PEDESTRIAN_DATASET, args.ped), ped_path)
    print(f"Saved: {ped_path}")

    print("Downloading litter basket locations...")
    download(socrata_csv_url(BINS_DATASET, args.bins), bins_path)
    print(f"Saved: {bins_path}")

    print("Done.")

if __name__ == "__main__":
    main()
