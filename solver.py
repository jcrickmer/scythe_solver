#!python
"""
Scythe opening explorer
- Focus: first N turns optimization (single-player, no combat)
- Starting scenario: Nordic Kingdoms + Industrial player mat

Python 3.12+ recommended.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum, Enum, auto
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Callable
import heapq
import math
import statistics
from itertools import combinations
import psutil
process = psutil.Process()
#print(process.memory_info().rss)

#from collections import Counter


# -----------------------------
# Basic identifiers / enums
# -----------------------------
class Scythe():
    GOAL_UPGRADES = 6
    GOAL_MECHS = 4
    GOAL_STRUCTURES = 4
    GOAL_ENLISTS = 4
    GOAL_WORKERS = 8
    GOAL_OBJECTIVE = 1
    GOAL_COMBAT = 2
    GOAL_POPULARITY = 18
    GOAL_POWER = 16

    POPULARITY_TIER_0 = {'ceiling': 6,
                         'stars': 3,
                         'territory': 2,
                         'resources': 1,
                         }
    POPULARITY_TIER_1 = {'ceiling': 12,
                         'stars': 4,
                         'territory': 3,
                         'resources': 2,
                         }
    POPULARITY_TIER_2 = {'ceiling': 18,
                         'stars': 5,
                         'territory': 4,
                         'resources': 3,
                         }
    
class Resource(IntEnum):
    FOOD = auto()
    WOOD = auto()
    METAL = auto()
    OIL = auto()
    WORKER = auto()

class TopActionType(Enum):
    MOVE = auto()
    TRADE = auto()
    PRODUCE = auto()
    BOLSTER = auto()

    # for MOVE parameters (unit_type, source_hid, dest_hid, unit_count)
    MOVE_PARAM_UNIT_TYPE = 0
    MOVE_PARAM_SOURCE_HID = 1
    MOVE_PARAM_DEST_HID = 2
    MOVE_PARAM_UNIT_COUNT = 3
    MOVE_UNIT_CHARACTER = 'CHAR'
    MOVE_UNIT_MECH = 'MECH'
    MOVE_UNIT_WORKER = 'WORKER'

class BottomActionType(Enum):
    UPGRADE = auto()
    DEPLOY = auto()
    BUILD = auto()
    ENLIST = auto()

class BottomActionBonus(Enum):
    POWER = auto()
    COIN = auto()
    POPULARITY = auto()
    CARD = auto()

class TopUpgradeChoice(Enum):
    BOLSTER_MILITARY = auto()
    BOLSTER_CARD = auto()
    PRODUCE_RESOURCE = auto()
    TRADE_POPULARITY = auto()
    MOVE_UNIT = auto()
    MOVE_COIN = auto()

class BottomUpgradeChoice(Enum):
    UPGRADE_COST = auto()
    DEPLOY_COST = auto()
    BUILD_COST = auto()
    ENLIST_COST = auto()
    @classmethod
    def action_type(cls, choice) -> BottomActionType:
        match choice:
            case BottomUpgradeChoice.UPGRADE_COST:
                return BottomActionType.UPGRADE
            case BottomUpgradeChoice.DEPLOY_COST:
                return BottomActionType.DEPLOY
            case BottomUpgradeChoice.BUILD_COST:
                return BottomActionType.BUILD
            case BottomUpgradeChoice.ENLIST_COST:
                return BottomActionType.ENLIST

class Structure(IntEnum):
    MONUMENT = auto()
    MILL = auto()
    MINE = auto()
    ARMORY = auto()
    
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

class Popularity(Enum):
    POPULARITY = auto()

COMBAT_CARDS_AVG = (16*2 + 12*3 + 8*4 + 6*5)/42

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


def make_minimal_opening_board() -> Board:
    """
    Placeholder board for early-turn modeling.
    Replace with full Scythe map later.
    """
    hx = {
        "N_HOME":     Hex("N_HOME",     Terrain.HOME_BASE, neighbors=("N_FOREST", "N_TUNDRA",), board_position=(0,1)),
        "A_VILLAGE":  Hex("A_VILLAGE",  Terrain.VILLAGE, neighbors=(), river_neighbors=("TUNDRA_2_3", "N_FOREST"), has_encounter=True, board_position=(1,3)),
        "N_FOREST":   Hex("N_FOREST",   Terrain.FOREST, neighbors=("N_HOME", "N_TUNDRA", "N_MOUNTAIN",), river_neighbors=('A_VILLAGE', 'TUNDRA_2_3'), board_position=(1,4)),
        "N_TUNDRA":   Hex("N_TUNDRA",   Terrain.TUNDRA, neighbors=("N_HOME", "N_FOREST", "N_MOUNTAIN",), river_neighbors=('N_FARM','N_VILLAGE'), board_position=(1,5)),
        "N_VILLAGE":  Hex("N_VILLAGE",  Terrain.VILLAGE, neighbors=("N_FARM",), river_neighbors=('N_TUNDRA',), board_position=(1,6)),
        "TUNDRA_2_3": Hex("TUNDRA_2_3", Terrain.TUNDRA, neighbors=(), river_neighbors=('A_VILLAGE','N_FOREST','N_MOUNTAIN',), is_tunnel=True, board_position=(2,3)),
        "N_MOUNTAIN": Hex("N_MOUNTAIN", Terrain.MOUNTAIN, neighbors=("N_FOREST", "N_TUNDRA",), has_encounter=True, board_position=(2,4)),
        "N_FARM":     Hex("N_FARM",     Terrain.FARM, neighbors=("N_VILLAGE",), river_neighbors=('N_TUNDRA', 'N_MOUNTAIN',), board_position=(2, 5)),

        "C_HOME":     Hex("C_HOME",     Terrain.HOME_BASE, neighbors=("C_FARM", "C_VILLAGE"), board_position=(9,1)), #REVISIT board position
        "C_FARM":     Hex("C_FARM",     Terrain.FARM, neighbors=("C_HOME", "C_VILLAGE", "C_MOUNTAIN"), river_neighbors=("C_TUNDRA"), board_position=(9,2)), #REVISIT board position
        "C_VILLAGE":  Hex("C_VILLAGE",  Terrain.VILLAGE, neighbors=("C_HOME", "C_FARM", "C_MOUNTAIN"), river_neighbors=(), board_position=(9,3)), #REVISIT board position
        "C_MOUNTAIN": Hex("C_MOUNTAIN", Terrain.MOUNTAIN, neighbors=("C_FARM", "C_VILLAGE"), river_neighbors=("FOREST"), board_position=(9,4)), #REVISIT board position
                          
    }
    return Board(hexes=hx)


# -----------------------------
# Faction + Mat configuration
# -----------------------------

@dataclass(frozen=True)
class Faction:
    name: str
    # Nordic specifics you might later encode:
    # - riverwalk rules
    # - "Swim" (workers may move into/through lakes?) etc.
    # Keep placeholders so the engine can reference them.
    special_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PlayerMat:
    name: str

    # How the 4 top actions pair with bottom actions on this mat
    # Example structure: {TopActionType.MOVE: BottomActionType.UPGRADE, ...}
    pairings: Dict[TopActionType, BottomActionType]

    # Base costs and coin rewards for bottom actions (before upgrades)
    # These are placeholders; you’ll fill in the real Industrial mat values.
    bottom_cost: Dict[BottomActionType, Dict[Resource, int]]
    bottom_coin_reward: Dict[BottomActionType, int]
    bottom_bonus: Dict[BottomActionType, BottomActionBonus]
    
    # Upgrades affect either top costs/effects or bottom costs/rewards
    # Keep it abstract for now.
    # Later: implement a mapping of "upgrade slots" -> (what it improves).
    #upgrade_slots: int = 6

    def init_progress(self):
        return Progress()


def nordic_config() -> Faction:
    return Faction(
        name="Nordic",
        special_rules=("TODO: Nordic riverwalk / swim rules",),
    )
def crimea_config() -> Faction:
    return Faction(
        name="Crimea",
        special_rules=("TODO: Nordic riverwalk / swim rules",),
    )


@dataclass(frozen=True)
class IndustrialMat(PlayerMat):
    name = "Industrial"
    pairings = {
        TopActionType.BOLSTER: BottomActionType.UPGRADE,
        TopActionType.PRODUCE: BottomActionType.DEPLOY,
        TopActionType.MOVE: BottomActionType.BUILD,
        TopActionType.TRADE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 3},
        BottomActionType.DEPLOY: {Resource.METAL: 3},
        BottomActionType.BUILD: {Resource.WOOD: 3},
        BottomActionType.ENLIST: {Resource.FOOD: 4},
    }
    # REVISIT - need to add how upgrades impact costs. Maybe just set a "minimum" cost that represents the floor upgrades can move toward.
    bottom_coin_reward = {
        BottomActionType.UPGRADE: 3,
        BottomActionType.DEPLOY: 2,
        BottomActionType.BUILD: 1,
        BottomActionType.ENLIST: 0,
    }
    bottom_bonus = {
        BottomActionType.UPGRADE: BottomActionBonus.POWER,
        BottomActionType.DEPLOY: BottomActionBonus.COIN,
        BottomActionType.BUILD: BottomActionBonus.POPULARITY,
        BottomActionType.ENLIST: BottomActionBonus.CARD,
    }
    
    def __init__(self):
        pass
    
    def init_progress(self):
        prog = Progress(top_upgrade_opportunities = (TopUpgradeChoice.BOLSTER_MILITARY,
                                                     TopUpgradeChoice.BOLSTER_CARD,
                                                     TopUpgradeChoice.PRODUCE_RESOURCE,
                                                     TopUpgradeChoice.MOVE_UNIT,
                                                     TopUpgradeChoice.MOVE_COIN,
                                                     TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities = (BottomUpgradeChoice.UPGRADE_COST,
                                                        BottomUpgradeChoice.DEPLOY_COST,
                                                        BottomUpgradeChoice.DEPLOY_COST,
                                                        BottomUpgradeChoice.BUILD_COST,
                                                        BottomUpgradeChoice.ENLIST_COST,
                                                        BottomUpgradeChoice.ENLIST_COST,
                                                        ),)
        return prog


# -----------------------------
# Game state
# -----------------------------

@dataclass(frozen=True)
class Units:
    """Positions of units. Keep simple for openings."""
    character: HexId
    mechs: Tuple[Tuple[HexId, int], ...] = ()
    # store workers as a multiset (hid repeated) or a count map for compactness
    workers: Tuple[Tuple[HexId, int], ...] = ()
    structures: Tuple[Tuple[Structure, HexId], ...] = ()
    
    def worker_count(self):
        return sum(value for _, value in self.workers)
    
    def territories_controlled(self):
        return set(key for key, _ in self.workers) | set([self.character]) | set(self.mechs)

@dataclass(frozen=True)
class Economy:
    coins: int = 0
    power: int = 0
    popularity: int = 0
    resources: Tuple[Tuple[Resource, int], ...] = ()
    combat_cards: int = 0
    
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

    # Track “which bottom actions have been upgraded” etc. later.
    def stars_earned(self):
        stars = 0
        if self.upgrades_done == 6:
            stars = stars +1
        if self.mechs_deployed == 4:
            stars = stars +1
        if self.structures_built == 4:
            stars = stars +1
            
        return stars

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
        #print("bottom_action_cost: {}".format(result))
        return result

# -----------------------------
# Action abstraction
# -----------------------------

@dataclass(frozen=True)
class TurnChoice:
    """A concrete top action choice (with parameters)."""
    action: TopActionType
    params: Tuple = ()


@dataclass(frozen=True)
class BottomChoice:
    action: BottomActionType
    params: Tuple = ()


class RulesError(Exception):
    pass


class Engine:
    """
    Owns rules and generates/apply actions.
    Keep this as the single authority so later you can add opponent modeling.
    """

    def __init__(self) -> None:
        # You can attach helpers here (pathfinding caches, etc.)
        pass

    # ---------
    # Utilities
    # ---------

    def _assert(self, cond: bool, msg: str) -> None:
        if not cond:
            raise RulesError(msg)

    def _can_pay(self, econ: Economy, cost: Dict[Resource, int]) -> bool:
        res = econ.res_dict()
        return all(res.get(r, 0) >= amt for r, amt in cost.items())

    def _pay(self, econ: Economy, cost: Dict[Resource, int]) -> Economy:
        res = econ.res_dict()
        for r, amt in cost.items():
            res[r] = res.get(r, 0) - amt
            if res[r] < 0:
                raise RulesError("Tried to pay with insufficient resources")
        return econ.with_res(res)

    def _gain(self, econ: Economy, gains: Dict[Resource, int] | None = None, coins: int = 0,
              power: int = 0, popularity: int = 0, combat_cards: int = 0) -> Economy:
        res = econ.res_dict()
        if gains:
            for r, amt in gains.items():
                if r is not None:
                    res[r] = res.get(r, 0) + amt
        return Economy(
            coins=econ.coins + coins,
            power=econ.power + power,
            popularity=econ.popularity + popularity,
            combat_cards=econ.combat_cards + combat_cards,
            resources=tuple(sorted(res.items(), key=lambda x: x[0].value)),
        )

    # ------------------------
    # Generate legal top turn actions
    # ------------------------

    def legal_top_choices(self, s: GameState) -> List[TurnChoice]:
        choices: List[TurnChoice] = []

        for a in TopActionType:
            if s.last_top_action == a:
                if s.faction.name == 'Rusviet':
                    pass
                else:
                    continue  # cannot repeat top action

            # For now: treat each top action as a single generic choice.
            # Later: move has parameters; produce may choose which worker hexes to produce; trade selects resource, etc.
            if a == TopActionType.MOVE:
                choices.extend(self._legal_move_choices(s))
            elif a == TopActionType.PRODUCE:
                # REVISIT - produce may require paying based on Faction Mat and workers
                choices.extend(self._legal_produce_choices(s))
            elif a == TopActionType.TRADE:
                # must have money to trade.
                if s.econ.coins > 0:
                    choices.extend(self._legal_trade_choices(s))
            elif a == TopActionType.BOLSTER:
                # must have money to bolster.
                if s.econ.coins > 0:
                    choices.extend(self._legal_bolster_choices(s))

        return choices

    def _legal_move_choices(self, s: GameState) -> List[TurnChoice]:
        # Skeleton: allow character to move to any neighbor.
        here = s.units.character
        result = [TurnChoice(TopActionType.MOVE, params=(TopActionType.MOVE_UNIT_CHARACTER, here, n, 1)) for n in s.board.neighbors(here)]

        # and now allow workers to move
        for (territory, worker_count) in s.units.workers:
            #print("L363 considering moving {} workers from {}".format(worker_count, territory))
            # territory_tuple is a [hid, worker_count]
            neighbors = s.board.neighbors(territory)
            if s.faction.name == 'Nordic':
                # swim rule for workers
                neighbors = neighbors + s.board.river_neighbors(territory)
            for workers in range(1, worker_count + 1):
                mmm = [TurnChoice(TopActionType.MOVE, params=(TopActionType.MOVE_UNIT_WORKER, territory, n, workers)) for n in neighbors]
                result = result + mmm
                #print("    L369 and choices are {}".format(result))
        # REVISIT - mech movement

        # REVISIT - speed mech movement

        # REVISIT - upgrade allows more moves
        
        return result

    def _legal_produce_choices(self, s: GameState) -> List[TurnChoice]:
        result: List[TurnChoice] = []

        # --- Production limit: base 2 + upgrades (per your Progress model) ---
        production_limit = 2 + s.prog.produce_modifier  # or s.prog.production_limit if you renamed it

        # --- Mill hex (if any) ---
        mill_hex: Optional[HexId] = None
        for struct, hid in s.units.structures:
            if struct == Structure.MILL:
                mill_hex = hid
                break

        # --- Current worker slots remaining (cap 8) ---
        current_workers = s.units.worker_count()
        remaining_slots = max(0, 8 - current_workers)

        # --- Eligible producer hexes: (hid, base_count, produced_resource, mill_bonus_on_this_hex?) ---
        eligible: List[Tuple[HexId, int, Resource, bool]] = []
        for hid, worker_count in s.units.workers:
            terr = s.board.terrain(hid)
            produced = Terrain.produces(terr)
            if produced is None:
                continue
            eligible.append((hid, worker_count, produced, (mill_hex == hid)))

        if not eligible or production_limit <= 0:
            return result

        kmax = min(production_limit, len(eligible))

        # We'll generate "up to kmax" territories. If you want to allow producing on 0 territories,
        # add an explicit TurnChoice for that, but most engines skip it.
        for k in range(1, kmax + 1):
            for combo in combinations(eligible, k):
                fixed_outputs: List[Tuple[HexId, int, Resource]] = []
                village_hexes: List[HexId] = []
                village_caps: List[int] = []

                # 1) Compute per-hex production amounts (including mill bonus if selected)
                for hid, base_cnt, res, has_mill in combo:
                    cnt = base_cnt + (1 if has_mill else 0)

                    if res == Resource.WORKER:
                        # This hex produces workers. We cannot finalize the count yet if we might exceed cap.
                        village_hexes.append(hid)
                        village_caps.append(cnt)
                    else:
                        fixed_outputs.append((hid, cnt, res))

                # 2) Handle cases with no villages -> single deterministic choice
                if not village_hexes:
                    # Canonical ordering to avoid duplicates in params
                    params = tuple(sorted(fixed_outputs, key=lambda x: (x[2], x[0])))
                    result.append(TurnChoice(TopActionType.PRODUCE, params=params))
                    continue

                # 3) Villages exist. Determine how many workers we can actually add.
                # Total workers requested by this selection:
                requested_workers = sum(village_caps)

                # If you want to enforce "must produce as many workers as possible", use:
                # produced_total = min(requested_workers, remaining_slots)
                # If you want to allow "produce fewer than possible" (usually not needed), you'd enumerate totals 0..produced_total.
                produced_total = min(requested_workers, remaining_slots)

                # 4) Generate allocations of produced_total across villages with per-hex caps
                allocs = bounded_allocations(village_caps, produced_total)

                # If produced_total is 0, bounded_allocations returns [ (0,0,...0) ] -> fine.
                for alloc in allocs:
                    outputs = fixed_outputs[:]
                    for hid, w in zip(village_hexes, alloc):
                        if w > 0:
                            outputs.append((hid, w, Resource.WORKER))

                    # Canonical ordering for params (helps pruning/caching)
                    params = tuple(sorted(outputs, key=lambda x: (x[2], x[0])))
                    result.append(TurnChoice(TopActionType.PRODUCE, params=params))

        return result

    
    def _legal_trade_choices(self, s: GameState) -> List[TurnChoice]:
        # Skeleton: trade chooses 2 resources or 1 resource + popularity, etc depending on mat upgrades.
        # Start by choosing a single resource to gain (placeholder).
        result = [TurnChoice(TopActionType.TRADE, params=(r,)) for r in Resource]
        already_added_choices = list() # since resources are essentially unordered, let's not add the same pair twice.
        for r in Resource:
            for r2 in Resource:
                param = tuple(sorted((r,r2)))
                if param not in already_added_choices:
                    already_added_choices.append(param)
                    tc = TurnChoice(TopActionType.TRADE, params=param)
                    result = result + [tc, ]
        # REVISIT - curently hard-coded for just 1 popularity. This can be improved with upgrades
        result.append(TurnChoice(TopActionType.TRADE, params=(Popularity.POPULARITY, )))
        return result

    def _legal_bolster_choices(self, s: GameState) -> List[TurnChoice]:
        # Skeleton: choose power or combat cards, etc.
        return [TurnChoice(TopActionType.BOLSTER, params=("CARDS",)), TurnChoice(TopActionType.BOLSTER, params=("POWER",))]

    # ----------------------
    # Apply top + bottom
    # ----------------------

    def apply_top(self, s: GameState, c: TurnChoice) -> GameState:
        a = c.action
        if a == TopActionType.MOVE:
            return self._apply_move(s, c)
        if a == TopActionType.PRODUCE:
            return self._apply_produce(s, c)
        if a == TopActionType.TRADE:
            return self._apply_trade(s, c)
        if a == TopActionType.BOLSTER:
            return self._apply_bolster(s, c)
        raise RulesError(f"Unhandled top action {a}")

    def _apply_move(self, s: GameState, c: TurnChoice) -> GameState:
        (unit_type, source_hid, dest_hid, unit_count) = c.params
        #self._assert(dest_hid in s.board.neighbors(source_hid), "Illegal move destination")
        # REVISIT - works for Character only right now
        if unit_type == TopActionType.MOVE_UNIT_CHARACTER:
            new_units = replace(s.units, character=dest_hid)
        elif unit_type == TopActionType.MOVE_UNIT_WORKER:
            #print("Thinking about moving workers {}".format(c.params))
            #going to have to manipulate a lot of immutable data, like s.unit.workers
            new_worker_list = list(list(pair) for pair in s.units.workers)
            if dest_hid not in [hidwc[0] for hidwc in new_worker_list]:
                new_worker_list.append([dest_hid, 0]) # make sure that we have a destination to go to
            for hid_worker_pair in new_worker_list:
                if hid_worker_pair[0] == source_hid:
                    hid_worker_pair[1] = hid_worker_pair[1] - unit_count
                if hid_worker_pair[0] == dest_hid:
                    hid_worker_pair[1] = hid_worker_pair[1] + unit_count
            new_worker_tuple = tuple(tuple(hidwc) for hidwc in new_worker_list if hidwc[1] > 0)
            new_units = replace(s.units, workers=new_worker_tuple)
        elif unit_type == TopActionType.MOVE_UNIT_MECH:
            # REVISIT - To do
            pass
        
        return replace(s, units=new_units, last_top_action=TopActionType.MOVE)

    def _apply_produce(self, s: GameState, c: TurnChoice) -> GameState:
        # c.params: Tuple[Tuple[HexId, int, Resource], ...]
        new_econ = s.econ
        new_units = s.units

        # Pay produce "costs" based on current worker count (per your model)
        wc = new_units.worker_count()
        if wc >= 4:
            new_econ = self._gain(new_econ, power=-1)
        if wc >= 6:
            new_econ = self._gain(new_econ, popularity=-1)
        if wc >= 8:
            new_econ = self._gain(new_econ, coins=-1)

        # Apply production outputs
        for hid, count, res in c.params:
            if count <= 0:
                continue

            if res == Resource.WORKER:
                # Enforce cap defensively (choices should already respect it)
                remaining = max(0, 8 - new_units.worker_count())
                add = min(count, remaining)
                if add <= 0:
                    continue

                # Add workers to that hex in units
                updated_workers = add_to_tuple_map(new_units.workers, (hid, add))
                new_units = replace(new_units, workers=updated_workers)

                # Track worker gain in econ if you want (proxy)
                new_econ = self._gain(new_econ, gains={Resource.WORKER: add})
            else:
                new_econ = self._gain(new_econ, gains={res: count})

        return replace(s, units=new_units, econ=new_econ, last_top_action=TopActionType.PRODUCE)

    
    def _apply_trade(self, s: GameState, c: TurnChoice) -> GameState:
        # Trading always costs 1 coin
        new_econ = self._gain(s.econ, coins=-1)

        # If Armory is acquired, gain 1 power
        if Structure.ARMORY in (struct for (struct, _) in s.units.structures):
            new_econ = self._gain(s.econ, power=1)
        
        r2 = None
        if len(c.params) == 2:
            (r,r2) = c.params
        else:
            (r,) = c.params
        if r == Popularity.POPULARITY:
            # REVISIT - read the GameState to understand how many popularities are gained. Hardcoded to 1 right now.
            new_econ = self._gain(new_econ, popularity=1)
        else:
            new_econ = self._gain(new_econ, gains={r: 1})
            if r2:
                new_econ = self._gain(new_econ, gains={r2: 1})
        return replace(s, econ=new_econ, last_top_action=TopActionType.TRADE)

    def _apply_bolster(self, s: GameState, c: TurnChoice) -> GameState:
        # Bolster always costs 1 coin
        new_econ = self._gain(s.econ, coins=-1)

        # If Monument is acquired, gain 1 popularity
        if Structure.MONUMENT in (struct for (struct, _) in s.units.structures):
            new_econ = self._gain(s.econ, popularity=1)
        
        # Placeholder: +2 power REVISIT
        if 'POWER' in c.params:
            new_econ = self._gain(new_econ, power=2 + s.prog.bolster_modifier)
        elif 'CARDS' in c.params:
            new_econ = self._gain(new_econ, combat_cards=1 + s.prog.combat_cards_modifier)
        return replace(s, econ=new_econ, last_top_action=TopActionType.BOLSTER)

    # ----------------------
    # Bottom action handling
    # ----------------------

    def legal_bottom_choice(self, s_after_top: GameState, top_action: TopActionType) -> List[BottomChoice]:
        """
        In Scythe, after top action you may take the paired bottom action if you can pay.
        Some upgrades change costs/rewards; handle later.

        Return [] or [BottomChoice(...)] or multiple if parameterized (build which structure, deploy which mech, etc.)
        """
        bottom = s_after_top.mat.pairings[top_action]
        cost = s_after_top.bottom_action_cost(bottom)

        if not self._can_pay(s_after_top.econ, cost):
            return []

        # Parameterization stubs:
        if bottom == BottomActionType.BUILD:
            return self._legal_build_choices(s_after_top)
        if bottom == BottomActionType.DEPLOY:
            return self._legal_deploy_choices(s_after_top)
        if bottom == BottomActionType.UPGRADE:
            return self._legal_upgrade_choices(s_after_top)
        if bottom == BottomActionType.ENLIST:
            return [BottomChoice(bottom, params=("TODO: which enlist",))]

        return [BottomChoice(bottom, params=())]

    def _legal_deploy_choices(self, s: GameState) -> List[BottomChoice]:
        # To Do:
        #   * which occupied territory does the mech go on
        #   * mech ability
        #   * mech abilities by faction
        #   * mech abilities altering things like move by picking up workers
        #   * mech ability River Walk
        #   * mech bonus of Speed to all moves
        return [BottomChoice(BottomActionType.DEPLOY, params=("MECH needs stuff"))]
    
    def _legal_build_choices(self, s: GameState) -> List[BottomChoice]:
        # for any hex that has a worker on it we can put a structure on it.
        param_extension = list()
        # iterate through the list of hexes with workers
        for (hid, worker_count) in s.units.workers:
            # territories can only have one structure on them.
            if hid in (used_hid for (_, used_hid) in s.units.structures):
                pass
            # structures cannot be on home bases
            if s.board.terrain(hid) == Terrain.HOME_BASE:
                pass
            for struct in Structure:
                if struct in (used_struct for (used_struct, _) in s.units.structures):
                    pass
                param_extension.append([struct, hid])
        return [BottomChoice(BottomActionType.BUILD, params=tuple(x)) for x in param_extension]
    
    def _legal_upgrade_choices(self, s: GameState) -> List[BottomChoice]:
        param_extension = list()
        # REVISIT - need to remove already-selected choices! DONE?? Needs to be verified
        for tuc in s.prog.top_upgrade_opportunities:
            for buc in s.prog.bottom_upgrade_opportunities:
                param_extension.append([tuc, buc])
        return [BottomChoice(BottomActionType.UPGRADE, params=tuple(x)) for x in param_extension]
    
    def apply_bottom(self, s: GameState, b: BottomChoice) -> GameState:
        #cost = s.mat.bottom_cost[b.action]
        cost = s.bottom_action_cost(b.action)
        reward_coins = s.mat.bottom_coin_reward[b.action]

        # pay the cost for this action
        s2 = replace(s, econ=self._pay(s.econ, cost))

        # get the coin benefit
        s3 = replace(s2, econ=self._gain(s2.econ, coins=reward_coins))

        # get the bonus benefit
        #print(f"Looking for {b.action} in {s3.prog.bottom_bonuses}")
        if b.action in s3.prog.bottom_bonuses:
            bonus = s.mat.bottom_bonus[b.action]
            #print(f"applying {bonus} bottom bonus for {b.action}")
            # REVISIT - when we work on Enlist, we need to come back and add these bonuses!!!
        
        # Update progress counters (placeholder).
        prog = s3.prog
        if b.action == BottomActionType.BUILD:
            return self._apply_build(s3, params=b.params)
        elif b.action == BottomActionType.DEPLOY:
            return self._apply_deploy(s3, params=b.params)
        elif b.action == BottomActionType.UPGRADE:
            return self._apply_upgrade(s3, params=b.params)
        elif b.action == BottomActionType.ENLIST:
            prog = replace(prog, enlists=prog.enlists + 1)

        return replace(s3, prog=prog)

    def _apply_deploy(self, s: GameState, params) -> GameState:
        prog = s.prog
        prog = replace(prog, mechs_deployed=prog.mechs_deployed + 1)
        hid: HexId = None
        for (whid, wcount) in s.units.workers:
            hid = whid
            break
        fff = replace(s.units, mechs=add_to_tuple_map(s.units.mechs, (hid, 1)))
        return replace(s, prog=prog, units=fff)
            
    def _apply_build(self, s: GameState, params: Tuple[Structure, HexId]) -> GameState:
        prog = s.prog
        structs = list(s.units.structures)
        #print("params is {}".format(params))
        structs.append(params)
           
        prog = replace(prog, structures_built=len(structs))
        #print("    structs as a list is {}".format(structs))
        #print("    setting structures to be {}".format(tuple(structs)))
        units = replace(s.units, structures=tuple(structs))
        return replace(s, prog=prog, units=units)

    def _apply_upgrade(self, s: GameState, params: Tuple[TopUpgradeChoice, BottomUpgradeChoice]) -> GameState:
        prog = s.prog
        #print("applying bottom upgrade with params {}".format(params))
        # we need to look at prog.top_upgrade_choices and then take this one out
        tou = list(prog.top_upgrade_opportunities)
        tou.remove(params[0]) # take this choice out of the list
        prog = replace(prog, top_upgrade_opportunities=tuple(tou))
        # now we need to change the cost/value of the top action
        match params[0]:
            case TopUpgradeChoice.BOLSTER_MILITARY:
                prog = replace(prog, bolster_modifier=1)
            case TopUpgradeChoice.BOLSTER_CARD:
                prog = replace(prog, combat_cards_modifier=1)
            case TopUpgradeChoice.PRODUCE_RESOURCE:
                prog = replace(prog, produce_modifier=1)
            case TopUpgradeChoice.MOVE_UNIT:
                prog = replace(prog, move_modifier=1)
            case TopUpgradeChoice.MOVE_COIN:
                prog = replace(prog, gain_modifier=1)
            case TopUpgradeChoice.TRADE_POPULARITY:
                prog = replace(prog, popularity_modifier=1)

        # Now for the BOTTOM upgrade benefit
        # we need to look at prog.top_upgrade_choices and then take this one out
        bou = list()
        to_remove = params[1]
        for val in prog.bottom_upgrade_opportunities:
            if val == to_remove:
                # make to_remove None so that we only remove one of these bad boys
                to_remove = None
            else:
                bou.append(val)

        new_bm_list = list()
        for (ba, mod) in s.prog.bottom_modifiers:
            #print("BM eval {} vs {}".format(BottomUpgradeChoice.action_type(params[1]), ba))
            if ba == BottomUpgradeChoice.action_type(params[1]):
                mod = mod - 1
            new_bm_list.append((ba,mod))
    
        prog = replace(prog,
                       upgrades_done=prog.upgrades_done + 1,
                       bottom_modifiers=tuple(new_bm_list),
                       bottom_upgrade_opportunities=tuple(bou))
        
        return replace(s, prog=prog)
    

    # ----------------------
    # Turn advance
    # ----------------------

    def end_turn(self, s: GameState) -> GameState:
        return replace(s, turn=s.turn + 1)

def power_bell(power: int, n_mechs: int, sigma: float = 3.0) -> float:
    """
    Returns a value in (0, 1] that peaks when power ~= (n_mechs + 1) * 7,
    capped at 18 (Scythe’s max power).
    """
    n_units = n_mechs + 1  # character + mechs
    ideal = min(18, 7 * n_units)

    # Standard Gaussian centered at `ideal`, max = 1.0 at the peak
    return math.exp(-0.5 * ((power - ideal) / sigma) ** 2)

def add_to_tuple_map(data: tuple[tuple[HexId, int], ...],
                     addition: tuple[HexId, int],
                     ) -> tuple[tuple[HexId, int], ...]:
    if len(addition) == 0:
        return data
    k, delta = addition
    d = dict(data)
    d[k] = d.get(k, 0) + delta
    return tuple(sorted(d.items()))

def bounded_allocations(bounds: List[int], total: int) -> List[Tuple[int, ...]]:
    """
    Return all tuples a where:
      - len(a) == len(bounds)
      - 0 <= a[i] <= bounds[i]
      - sum(a) == total

    Example:
      bounds=[3,2], total=2 -> [(0,2),(1,1),(2,0)]
    """
    result: List[Tuple[int, ...]] = []

    def rec(i: int, remaining: int, acc: List[int]) -> None:
        if i == len(bounds):
            if remaining == 0:
                result.append(tuple(acc))
            return

        max_take = min(bounds[i], remaining)
        for take in range(0, max_take + 1):
            acc.append(take)
            rec(i + 1, remaining - take, acc)
            acc.pop()

    rec(0, total, [])
    return result

# -----------------------------
# Scoring (tune as you like)
# -----------------------------

def line_scoring(s: GameState) -> float:
    opening_weight = max(0.1, (10.0 - s.turn)/9.0)
    sigma = 4.0
    midgame_weight = math.exp(-0.5 * ((s.turn - 14) / sigma) ** 2)
    endgame_weight = max(0.1, (s.turn - 12)/10)

    score = opening_weight * opening_score(s)
    score += midgame_weight * midgame_score(s)
    score += endgame_weight * endgame_score(s)
    return score

def midgame_score(s: GameState) -> float:
    workers = sum(value for _, value in s.units.workers)
    mechs = sum(value for _, value in s.units.mechs)
    territory_count = len(s.units.territories_controlled())
    score = 0.0
    score += 6.0 * mechs
    score += 2.5 * territory_count
    score += 1.1 * s.econ.coins
    score += 1.9 * s.econ.popularity
    score += 1.75 * power_bell(s.econ.power, mechs)
    return score

def endgame_score(s: GameState) -> float:
    territory_count = len(s.units.territories_controlled())
    workers = sum(value for _, value in s.units.workers)
    mechs = sum(value for _, value in s.units.mechs)
    score = 0.0
    score += 1.0
    score += 3.0 * s.econ.popularity
    score += 3.5 * territory_count
    return score

def opening_score(s: GameState) -> float:
    """
    Heuristic score for openings.
    Adjust weights as you learn what “good” looks like for Nordic + Industrial.
    """
    workers = sum(value for _, value in s.units.workers)
    mechs = sum(value for _, value in s.units.mechs)

    # Weighted progress
    score = 0.0
    score += 3.0 * workers
    score += 5.0 * mechs
    score += 4.0 * s.prog.upgrades_done * ((25.0 - min(24, s.turn)) / 25.0)
    score += 4.1 * s.prog.enlists * ((25.0 - min(24, s.turn)) / 25.0)
    score += 4.0 * s.prog.structures_built * ((25.0 - min(24, s.turn)) / 25.0)
    score += 0.76 * COMBAT_CARDS_AVG * max(1, (6 - s.econ.combat_cards))
    score += 1.1 * s.econ.coins
    score += 1.1 * s.econ.popularity
    score += 0.75 * power_bell(s.econ.power, mechs)
    # Small bonus for having some resources banked
    score += 0.2 * sum(dict(s.econ.resources).values())

    # REVISIT - also, score needs to be resource weighted... the resources that fund low-cost advancements should have a lower weight than those that fund high-cost advancements.

    # REVISIT - lastly, use some scoring that reflects the actual scoreboard of the game.

    # REVISIT - think about how goals change over time.
    #   Opening game = resources and production possiblity
    #   Mid game = tempo, board control, power potential
    #   End game = focus on Scythe scoring system
    #
    # Maybe as the game progresses, the weight of each of these scoring systems shifts. Turns 1-6 are opening, 7-18 or mid game, 19+ is end game.
    
    return score


# -----------------------------
# Search (beam search)
# -----------------------------

@dataclass(frozen=True)
class Line:
    """A candidate line of play."""
    state: GameState
    turn_actions: Tuple[str, ...] = ()

    def with_step(self, text: str, new_state: GameState) -> "Line":
        return Line(state=new_state, turn_actions=self.turn_actions + (text,))


def beam_search_openings(
    engine: Engine,
    start: GameState,
    turns: int = 5,
    beam_width: int = 200,
    scorer: Callable[[GameState], float] = line_scoring,
) -> List[Line]:
    """
    Expand turn by turn. Keep best K lines each turn.
    """
    frontier: List[Line] = [Line(state=start, turn_actions=())]

    for _ in range(turns):
        candidates: List[Tuple[float, int, Line]] = []
        uid = 0
        print("Turn {}, frontier size is {}, memory is {} MB".format(_, len(frontier), process.memory_info().rss / 1024.0 / 1024.0))
        for line in frontier:
            s = line.state
            top_choices = engine.legal_top_choices(s)

            for top in top_choices:
                s_top = engine.apply_top(s, top)

                # optional bottom action
                bottoms = engine.legal_bottom_choice(s_top, top.action)
                if bottoms:
                    for b in bottoms:
                        s_bot = engine.apply_bottom(s_top, b)
                        s_next = engine.end_turn(s_bot)
                        desc = f"T{s.turn+1}: {top.action.name} {top.params} + {b.action.name} {b.params}"
                        new_line = line.with_step(desc, s_next)
                        candidates.append((scorer(s_next), uid, new_line))
                        uid += 1
                else:
                    s_next = engine.end_turn(s_top)
                    desc = f"T{s.turn+1}: {top.action.name} {top.params} + (no bottom)"
                    new_line = line.with_step(desc, s_next)
                    candidates.append((scorer(s_next), uid, new_line))
                    uid += 1

        # keep best beam_width
        print("   candidate size is {}".format(len(candidates)))
        all_scrs = [scr for scr, _, _ in candidates]
        print("      max={}, min={}, mean={}, median={}".format(max(all_scrs), min(all_scrs), statistics.mean(all_scrs), statistics.median(all_scrs)))
        best = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        frontier = [ln for _, _, ln in best]
        scrs = [scr for scr, _, _ in best]
        print("   candidates post beam {}".format(len(best)))
        print("      max={}, min={}, mean={}, median={}".format(max(scrs), min(scrs), statistics.mean(scrs), statistics.median(scrs)))

    # return best lines overall
    return sorted(frontier, key=lambda ln: scorer(ln.state), reverse=True)


# -----------------------------
# Starting state (Nordic + Industrial)
# -----------------------------

def make_start_state_nordic_industrial() -> GameState:
    board = make_minimal_opening_board()
    faction = nordic_config()
    #mat = industrial_mat_config()
    mat = IndustrialMat()

    # Placeholder start:
    # - Character starts at home
    # - 2 workers on home (tuple of locations)
    # - 0 mechs
    # - starting resources/coins/power/popularity: fill in true values later
    units = Units(
        character="N_HOME",
        mechs=(),
        workers=(("N_FOREST", 1), ("N_TUNDRA", 1)),
        structures=()
    )
    econ = Economy(
        coins=4,
        power=4,
        popularity=2,
        resources=tuple(sorted({Resource.FOOD: 0, Resource.WOOD: 0, Resource.METAL: 0, Resource.OIL: 0, Resource.WORKER: 0}.items(),
                              key=lambda x: x[0].value)),
        combat_cards=1,
    )
    prog = mat.init_progress()

    return GameState(
        faction=faction,
        mat=mat,
        board=board,
        units=units,
        econ=econ,
        prog=prog,
        turn=0,
        last_top_action=None,
    )

def make_start_state_crimea_industrial() -> GameState:
    board = make_minimal_opening_board()
    faction = crimea_config()
    #mat = industrial_mat_config()
    mat = IndustrialMat()

    # Placeholder start:
    # - Character starts at home
    # - 2 workers on home (tuple of locations)
    # - 0 mechs
    # - starting resources/coins/power/popularity: fill in true values later
    units = Units(
        character="C_HOME",
        mechs=(),
        workers=(("C_VILLAGE", 1), ("C_FARM", 1)),
        structures=()
    )
    econ = Economy(
        coins=4,
        power=5,
        popularity=2,
        resources=tuple(sorted({Resource.FOOD: 0, Resource.WOOD: 0, Resource.METAL: 0, Resource.OIL: 0, Resource.WORKER: 0}.items(),
                              key=lambda x: x[0].value)),
        combat_cards=0,
    )
    prog = mat.init_progress()

    return GameState(
        faction=faction,
        mat=mat,
        board=board,
        units=units,
        econ=econ,
        prog=prog,
        turn=0,
        last_top_action=None,
    )


# -----------------------------
# Demo runner
# -----------------------------

def summarize_state(s: GameState) -> str:
    res = dict(s.econ.resources)
    territories = s.units.territories_controlled()
    return (
        f"Turn={s.turn}  Coins={s.econ.coins}  Pow={s.econ.power}  Pop={s.econ.popularity}  "
        f"Res(F/W/M/O)=({res.get(Resource.FOOD,0)}/{res.get(Resource.WOOD,0)}/"
        f"{res.get(Resource.METAL,0)}/{res.get(Resource.OIL,0)})  "
        f"Territories={territories}  "
        f"Workers={sum(x for _, x in s.units.workers)}  Mechs={sum(x for _, x in s.units.mechs)}  "
        f"Upg={s.prog.upgrades_done}  Dep={s.prog.mechs_deployed}  "
        f"Build={s.prog.structures_built}  Enlist={s.prog.enlists}  "
        f"Combat Cards={s.econ.combat_cards}  "
        f"Char@{s.units.character}"
        f"\n    prog={s.prog}"
        f"\n    units={s.units}"
        f"\n    econ={s.econ}"
    )


if __name__ == "__main__":
    engine = Engine()
    start = make_start_state_nordic_industrial()
    #start = make_start_state_crimea_industrial()

    lines = beam_search_openings(engine, start, turns=8, beam_width=1000)

    print("Top 10 opening lines:")
    for i, ln in enumerate(lines[:10], start=1):
        print(f"\n#{i}  Score={opening_score(ln.state):.2f}")
        for step in ln.turn_actions:
            print(" ", step)
        print(" ", summarize_state(ln.state))
