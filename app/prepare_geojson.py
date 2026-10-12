"""One-off: turn the geoBoundaries ADM1 file into app/static/oblasts.geojson with dataset region names.

Source: https://www.geoboundaries.org (gbOpen UKR ADM1, simplified), ODbL 1.0.
Usage: uv run python -m app.prepare_geojson
"""
import json
from pathlib import Path
from urllib.request import urlretrieve

from app.regions import PERMANENT, REGIONS

URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/UKR/ADM1/"
       "geoBoundaries-UKR-ADM1_simplified.geojson")
STATIC = Path(__file__).resolve().parent / "static"


def round_coords(c, nd=3):
    return [round(x, nd) for x in c] if isinstance(c[0], (int, float)) else [round_coords(x, nd) for x in c]


def main() -> None:
    raw = STATIC / "ukr_adm1_raw.geojson"
    if not raw.exists():
        urlretrieve(URL, raw)
    shape_to_region = {shape: r for r, (shape, _) in REGIONS.items()}
    feats = []
    for f in json.loads(raw.read_text())["features"]:
        name = f["properties"]["shapeName"]
        region = shape_to_region.get(name) or PERMANENT[name]
        geom = {"type": f["geometry"]["type"], "coordinates": round_coords(f["geometry"]["coordinates"])}
        feats.append({"type": "Feature", "properties": {"region": region, "permanent": name in PERMANENT},
                      "geometry": geom})
    missing = set(REGIONS) - {f["properties"]["region"] for f in feats}
    assert not missing, f"regions without a shape: {missing}"
    out = STATIC / "oblasts.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    raw.unlink()
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(feats)} features)")


if __name__ == "__main__":
    main()
