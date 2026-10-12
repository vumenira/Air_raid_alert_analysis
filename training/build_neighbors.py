"""One-off: derive oblast adjacency from the map polygons -> training/neighbors.json.

Adjacent oblasts share boundary vertices in the geoBoundaries file (distance 0), so the threshold is
only a tolerance; the result is identical for 0.005-0.05 degrees.
Usage: uv run python -m training.build_neighbors
"""
import json
from pathlib import Path

import numpy as np

from training.data import ROOT

GEOJSON = ROOT / "app" / "static" / "oblasts.geojson"
OUT = Path(__file__).resolve().parent / "neighbors.json"
THRESHOLD_DEG = 0.01


def vertices(geom: dict) -> np.ndarray:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return np.array([p for poly in polys for ring in poly for p in ring])


def min_dist(a: np.ndarray, b: np.ndarray) -> float:
    return min(np.sqrt(((a[i:i + 500, None] - b[None]) ** 2).sum(-1)).min() for i in range(0, len(a), 500))


def main() -> None:
    feats = json.loads(GEOJSON.read_text())["features"]
    # permanent-alert areas are kept: e.g. Luhansk borders Kharkiv and Donetsk, but it never has alert starts
    V = {f["properties"]["region"]: vertices(f["geometry"]) for f in feats}
    names = sorted(V)
    nb = {n: [] for n in names}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if min_dist(V[x], V[y]) < THRESHOLD_DEG:
                nb[x].append(y)
                nb[y].append(x)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}: {sum(map(len, nb.values())) // 2} adjacent pairs")


if __name__ == "__main__":
    main()
