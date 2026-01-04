from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
import yaml
from units import Resource

# -----------------------------
# Board model (graph)
# -----------------------------

HexId = str


@dataclass(frozen=True)
class Hex:
    hid: HexId
    terrain: Terrain
    neighbors: Tuple[HexId, ...] = ()
    river_neighbors: Tuple[HexId, ...] = ()
    lake_neighbors: Tuple[HexId, ...] = ()
    has_encounter: bool = False
    is_tunnel: bool = False
    board_position: Tuple[int, int] = ()


@dataclass(frozen=True)
class Board:
    """Graph-like board representation. Expand later with rivers, lakes, tunnels, etc."""
    hexes: Dict[HexId, Hex]

    def neighbors(self, hid: HexId) -> Tuple[HexId, ...]:
        return self.hexes[hid].neighbors

    def river_neighbors(self, hid: HexId) -> Tuple[HexId, ...]:
        return self.hexes[hid].river_neighbors

    def terrain(self, hid: HexId) -> Terrain:
        return self.hexes[hid].terrain

    @classmethod
    def load_board_from_yaml(cls, path: str | Path) -> Board:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        hexes: dict[str, Hex] = {}

        for hid, spec in data["hexes"].items():
            hexes[hid] = Hex(
                hid=hid,
                terrain=Terrain[spec["terrain"]],
                neighbors=tuple(spec.get("neighbors", [])),
                river_neighbors=tuple(spec.get("river_neighbors", [])),
                lake_neighbors=tuple(spec.get("lake_neighbors", [])),
                has_encounter=spec.get("has_encounter", False),
                is_tunnel=spec.get("is_tunnel", False),
                board_position=tuple(spec.get("board_position", (0, 0))),
            )

        return Board(hexes=hexes)


class Terrain(IntEnum):
    FARM = auto()
    FOREST = auto()
    MOUNTAIN = auto()
    TUNDRA = auto()
    VILLAGE = auto()
    FACTORY = auto()
    LAKE = auto()
    HOME_BASE = auto()

    @classmethod
    def produces(cls, terrainType):
        match terrainType:
            case Terrain.HOME_BASE:
                return None
            case Terrain.FARM:
                return Resource.FOOD
            case Terrain.FOREST:
                return Resource.WOOD
            case Terrain.MOUNTAIN:
                return Resource.METAL
            case Terrain.TUNDRA:
                return Resource.OIL
            case Terrain.VILLAGE:
                return Resource.WORKER
