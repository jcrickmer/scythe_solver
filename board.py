from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
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
