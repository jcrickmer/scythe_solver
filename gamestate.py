#!python

from __future__ import annotations

from dataclasses import dataclass, field, replace
from units import BottomActionType, Resource
from board import Board
from factions import Faction

# -----------------------------
# Game state
# -----------------------------


@dataclass(frozen=True)
class GameState:
    faction: Faction
    mat: PlayerMat
    board: Board

    units: Units
    econ: Economy
    prog: Progress

    # Turn bookkeeping
    turn: int = 0
    last_top_action: Optional[TopActionType] = None  # Scythe bans repeating same top action back-to-back

    # Cache-friendly signature pieces (optional): you can compute on demand.
    # But keeping state fully immutable + hashable is already great.

    def bottom_action_cost(self, bottom_action: BottomActionType) -> Dict[BottomActionType: int]:
        ''' For each resource cost in the bottom action coss from the mat, modify it by the current bottom modifiers in Progress. '''
        cost = self.mat.bottom_cost[bottom_action].copy()

        cost_modifier = 0
        for (ba, mod) in self.prog.bottom_modifiers:
            if ba == bottom_action:
                cost_modifier = mod
        result = {k: v + cost_modifier for k, v in cost.items()}
        # print("bottom_action_cost: {}".format(result))
        return result


@dataclass(frozen=True)
class Units:
    """Positions of units. Keep simple for openings."""
    character: HexId
    mechs: Tuple[Tuple[HexId, Mech], ...] = ()
    workers: Tuple[Tuple[HexId, int], ...] = ()
    structures: Tuple[Tuple[Structure, HexId], ...] = ()

    def worker_count(self):
        return sum(value for _, value in self.workers)

    def territories_controlled(self):
        return set(key for key, _ in self.workers) | set([self.character]) | set(hexid for (hexid, _) in self.mechs)


@dataclass(frozen=True)
class Economy:
    coins: int = 0
    power: int = 0
    popularity: int = 0
    resources: Tuple[Tuple[Resource, int], ...] = ()
    combat_cards: int = 0

    def __str__(self) -> str:
        res = dict(self.resources)
        return (f"Coins={self.coins}  "
                f"Res(F/W/M/O)=({res.get(Resource.FOOD, 0)}/{res.get(Resource.WOOD, 0)}/"
                f"{res.get(Resource.METAL, 0)}/{res.get(Resource.OIL, 0)})  "
                f"Popularity={self.popularity}  Power={self.power}  Cards={self.combat_cards} "
                )

    def res_dict(self) -> Dict[Resource, int]:
        return dict(self.resources)

    def with_res(self, new_res: Dict[Resource, int]) -> "Economy":
        return replace(self, resources=tuple(sorted(new_res.items(), key=lambda x: x[0].value)))


@dataclass(frozen=True)
class Progress:
    upgrades_done: int = 0
    mechs_deployed: int = 0
    structures_built: int = 0
    enlists: int = 0
    top_upgrade_opportunities: Tuple[TopUpgradeChoice] = ()
    bottom_upgrade_opportunities: Tuple = ()
    encounters: Tuple[HexId] = ()

    # top actions that can be modified by upgrades
    bolster_modifier: int = 0
    combat_cards_modifier: int = 0
    produce_modifier: int = 0
    move_modifier: int = 0
    gain_modifier: int = 0
    popularity_modifier: int = 0

    # bottom actions that can be modified by upgrades
    bottom_modifiers: Tuple[Tuple[BottomActionType, int], ...] = ([BottomActionType.BUILD, 0],
                                                                  [BottomActionType.DEPLOY, 0],
                                                                  [BottomActionType.UPGRADE, 0],
                                                                  [BottomActionType.ENLIST, 0])

    # On the Faction Mat, which of the 4 enlist bonuses have been taken
    enlist_power_bonus: bool = False
    enlist_coin_bonus: bool = False
    enlist_popularity_bonus: bool = False
    enlist_card_bonus: bool = False

    # On the Player Mat, each BottomAction can have a bonus becuase of past Enlist actions
    bottom_bonuses: Tuple[BottomActionType] = ()

    def __str__(self) -> str:
        return (f"Upgrades={self.upgrades_done} "
                f"Enlists={self.enlists} "
                f"Stars={self.stars_earned()}"
                )

    # Track “which bottom actions have been upgraded” etc. later.
    def stars_earned(self) -> int:
        stars = 0
        if self.upgrades_done == 6:
            stars = stars + 1
        if self.mechs_deployed == 4:
            stars = stars + 1
        if self.structures_built == 4:
            stars = stars + 1

        return stars
