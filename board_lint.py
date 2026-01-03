#!python
"""
board_lint.py — validate Scythe board.yaml internal consistency

Checks:
1) Every referenced hex id exists (neighbors / river_neighbors / lake_neighbors)
2) Reciprocity:
   - neighbors are symmetric
   - river_neighbors are symmetric
   - lake_neighbors are symmetric
3) No orphan hexes (a hex with no neighbors/river/lake connections)
4) Optional sanity:
   - terrain strings are valid Terrain members
   - board_position is present and is [int, int]
   - board_position coordinates are unique (warn)

Usage:
  python board_lint.py path/to/board.yaml

Requires:
  pip install pyyaml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml


class Terrain(Enum):
    FARM = "FARM"
    FOREST = "FOREST"
    MOUNTAIN = "MOUNTAIN"
    TUNDRA = "TUNDRA"
    VILLAGE = "VILLAGE"
    FACTORY = "FACTORY"
    LAKE = "LAKE"
    HOME_BASE = "HOME_BASE"


EDGE_FIELDS = ("neighbors", "river_neighbors", "lake_neighbors")


@dataclass
class Issue:
    level: str  # "ERROR" or "WARN"
    message: str


def _as_list(spec: Dict[str, Any], key: str) -> List[str]:
    v = spec.get(key, [])
    if v is None:
        return []
    if isinstance(v, list):
        return v
    raise TypeError(f'Field "{key}" must be a list (or omitted), got {type(v).__name__}')


def lint_board_yaml(path: Path) -> List[Issue]:
    issues: List[Issue] = []

    # --- Load YAML ---
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [Issue("ERROR", f"Failed to read/parse YAML: {e}")]

    if not isinstance(data, dict) or "hexes" not in data:
        return [Issue("ERROR", 'YAML root must be a mapping containing key "hexes".')]

    hexes = data["hexes"]
    if not isinstance(hexes, dict) or not hexes:
        return [Issue("ERROR", '"hexes" must be a non-empty mapping of hex_id -> spec.')]

    all_ids: Set[str] = set(hexes.keys())

    # --- Terrain validation + board_position validation ---
    positions: Dict[Tuple[int, int], str] = {}
    for hid, spec in hexes.items():
        if not isinstance(spec, dict):
            issues.append(Issue("ERROR", f"{hid}: spec must be a mapping/dict."))
            continue

        # terrain
        terr = spec.get("terrain")
        if terr is None:
            issues.append(Issue("ERROR", f"{hid}: missing required field terrain."))
        else:
            if not isinstance(terr, str):
                issues.append(Issue("ERROR", f"{hid}: terrain must be a string, got {type(terr).__name__}."))
            else:
                if terr not in {t.value for t in Terrain}:
                    issues.append(Issue("ERROR", f"{hid}: invalid terrain '{terr}'. Must be one of {[t.value for t in Terrain]}."))
        # board_position
        bp = spec.get("board_position")
        if bp is None:
            issues.append(Issue("WARN", f"{hid}: missing board_position (recommended for sanity/visualization)."))
        else:
            if not (isinstance(bp, list) and len(bp) == 2 and all(isinstance(x, int) for x in bp)):
                issues.append(Issue("ERROR", f"{hid}: board_position must be a 2-item list of ints, got {bp!r}."))
            else:
                pos = (bp[0], bp[1])
                if pos in positions:
                    issues.append(Issue("WARN", f"board_position {pos} used by both {positions[pos]} and {hid}."))
                else:
                    positions[pos] = hid

    # --- Reference validation (every edge points to an existing hex id) ---
    for hid, spec in hexes.items():
        if not isinstance(spec, dict):
            continue
        for field in EDGE_FIELDS:
            try:
                refs = _as_list(spec, field)
            except TypeError as e:
                issues.append(Issue("ERROR", f"{hid}: {e}"))
                continue

            for ref in refs:
                if not isinstance(ref, str):
                    issues.append(Issue("ERROR", f"{hid}: {field} contains non-string entry {ref!r}."))
                    continue
                if ref not in all_ids:
                    issues.append(Issue("ERROR", f"{hid}: {field} references unknown hex id '{ref}'."))

    # --- Reciprocity checks ---
    def check_reciprocal(field: str) -> None:
        for hid, spec in hexes.items():
            if not isinstance(spec, dict):
                continue
            refs = _as_list(spec, field)
            for ref in refs:
                if ref not in all_ids:
                    continue  # already reported
                ref_spec = hexes[ref]
                if not isinstance(ref_spec, dict):
                    issues.append(Issue("ERROR", f"{ref}: spec must be a mapping/dict (referenced by {hid}.{field})."))
                    continue
                back = _as_list(ref_spec, field)
                if hid not in back:
                    issues.append(Issue(
                        "ERROR",
                        f"{hid}: {field} includes '{ref}' but '{ref}' does not include '{hid}' in its {field} (non-reciprocal)."
                    ))

    for field in EDGE_FIELDS:
        check_reciprocal(field)

    # --- Orphan detection ---
    # orphan = no links in any adjacency list
    for hid, spec in hexes.items():
        if not isinstance(spec, dict):
            continue
        total_deg = sum(len(_as_list(spec, f)) for f in EDGE_FIELDS)
        if total_deg == 0:
            issues.append(Issue("WARN", f"{hid}: appears to be an orphan (no neighbors/river_neighbors/lake_neighbors)."))

    # --- (Optional) sanity: if terrain == LAKE, it should probably have lake_neighbors (warn only) ---
    for hid, spec in hexes.items():
        if not isinstance(spec, dict):
            continue
        terr = spec.get("terrain")
        if terr == "LAKE":
            ln = _as_list(spec, "lake_neighbors")
            if not ln:
                issues.append(Issue("WARN", f"{hid}: terrain is LAKE but lake_neighbors is empty."))

    return issues


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python board_lint.py path/to/board.yaml", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2

    issues = lint_board_yaml(path)

    errors = [i for i in issues if i.level == "ERROR"]
    warns = [i for i in issues if i.level == "WARN"]

    for i in issues:
        print(f"{i.level}: {i.message}")

    print(f"\nSummary: {len(errors)} error(s), {len(warns)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
