"""
HerbalSalt-AI | Step 1: Download public NHANES data (2007 - March 2020)
Saves everything under:  D:\\My research papers\\Harbal Salt\\01_raw_data
Run:  python 01_download_nhanes.py
"""
import os
import time
import requests

ROOT = r"D:\My research papers\Harbal Salt"
RAW = os.path.join(ROOT, "01_raw_data")
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"

# cycle label -> (folder year, file suffix/prefix rule)
CYCLES = {
    "2007-2008": ("2007", "_E"),
    "2009-2010": ("2009", "_F"),
    "2011-2012": ("2011", "_G"),
    "2013-2014": ("2013", "_H"),
    "2015-2016": ("2015", "_I"),
    "2017-2020": ("2017", "P_"),   # pre-pandemic combined cycle uses a prefix
}

# Files needed in every cycle
COMMON = ["DEMO", "BMX", "GHB", "DIQ", "BPQ", "DPQ", "MCQ", "SMQ", "DSQIDS"]
# Cycle-specific files
EXTRA = {
    "2007-2008": ["BPX", "CRP"],
    "2009-2010": ["BPX", "CRP"],
    "2011-2012": ["BPX"],
    "2013-2014": ["BPX"],
    "2015-2016": ["BPX", "HSCRP"],
    "2017-2020": ["BPXO", "HSCRP"],
}
# NHANES Dietary Supplement Database (product, ingredient, blend) 1999-2020
DSD = ["DSPI", "DSII", "DSBI"]

HEADERS = {"User-Agent": "Mozilla/5.0 (research download; HerbalSalt-AI)"}


def fname(base, suffix):
    return f"P_{base}" if suffix == "P_" else f"{base}{suffix}"


def download(year, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.xpt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  [skip] {name} already downloaded")
        return True
    for ext in (".xpt", ".XPT"):
        url = BASE.format(year=year, name=name + ext)
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200 and len(r.content) > 1000 and not r.content[:15].lower().startswith(b"<!doctype"):
                with open(out_path, "wb") as f:
                    f.write(r.content)
                print(f"  [ok]   {name}  ({len(r.content)/1e6:.1f} MB)")
                return True
        except requests.RequestException as e:
            print(f"  [err]  {name}: {e}")
        time.sleep(1)
    print(f"  [MISSING] {name}  -> download manually from {BASE.format(year=year, name=name + '.xpt')}")
    return False


def main():
    missing = []
    for cycle, (year, sfx) in CYCLES.items():
        print(f"\n== NHANES {cycle} ==")
        out_dir = os.path.join(RAW, "nhanes", cycle)
        for base in COMMON + EXTRA[cycle]:
            name = fname(base, sfx)
            if not download(year, name, out_dir):
                missing.append(f"{cycle}/{name}")

    print("\n== NHANES Dietary Supplement Database ==")
    for name in DSD:
        if not download("1999", name, os.path.join(RAW, "dsd")):
            missing.append(f"dsd/{name}")

    print("\nDONE.")
    if missing:
        print("Files that need manual download:")
        for m in missing:
            print("  -", m)
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
