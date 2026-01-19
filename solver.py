#!python
"""
Scythe opening explorer
- Focus: first N turns optimization (single-player, no combat)
- Starting scenario: Nordic Kingdoms + Industrial player mat

Python 3.12+ recommended.
"""

from __future__ import annotations
import sys
import argparse

from dataclasses import dataclass, field, replace
from enum import IntEnum, Enum, auto
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Callable
import heapq
import math
import statistics
from itertools import combinations
from pathlib import Path
import psutil
from board import HexId, Hex, Board, Terrain
from units import Resource
import scorer
from gamestate import GameState, Units, Progress, Economy
from factions import Faction, albion_config, togawa_config, nordic_config, crimea_config, polania_config, rusviet_config, saxony_config
from units import TopActionType, BottomActionType, BottomActionBonus, TopUpgradeChoice, BottomUpgradeChoice, Popularity, Structure, MoveableUnit, Mech
from util import add_to_tuple_map, bounded_allocations

process = psutil.Process()
# print(process.memory_info().rss)

# from collections import Counter


# -----------------------------
# Faction + Mat configuration
# -----------------------------

@dataclass(frozen=True)
class PlayerMat:
    name: str

    start_coins: int
    start_pop: int

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
    # upgrade_slots: int = 6

    def init_progress(self):
        return Progress()


@dataclass(frozen=True)
class IndustrialMat(PlayerMat):  # Mat 1
    name = "Industrial"
    start_coins = 4
    start_pop = 2
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.UPGRADE_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class EngineeringMat(PlayerMat):  # Mat 2
    name = "Engineering"
    start_coins = 5
    start_pop = 2
    pairings = {
        TopActionType.PRODUCE: BottomActionType.UPGRADE,
        TopActionType.TRADE: BottomActionType.DEPLOY,
        TopActionType.BOLSTER: BottomActionType.BUILD,
        TopActionType.MOVE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 3},
        BottomActionType.DEPLOY: {Resource.METAL: 3},
        BottomActionType.BUILD: {Resource.WOOD: 4},
        BottomActionType.ENLIST: {Resource.FOOD: 3},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 2,
        BottomActionType.DEPLOY: 0,
        BottomActionType.BUILD: 3,
        BottomActionType.ENLIST: 1,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.UPGRADE_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class MilitantMat(PlayerMat):  # Mat 2A
    name = "Militant"
    start_coins = 4
    start_pop = 3
    pairings = {
        TopActionType.BOLSTER: BottomActionType.UPGRADE,
        TopActionType.MOVE: BottomActionType.DEPLOY,
        TopActionType.PRODUCE: BottomActionType.BUILD,
        TopActionType.TRADE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 3},
        BottomActionType.DEPLOY: {Resource.METAL: 3},
        BottomActionType.BUILD: {Resource.WOOD: 4},
        BottomActionType.ENLIST: {Resource.FOOD: 3},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 0,
        BottomActionType.DEPLOY: 3,
        BottomActionType.BUILD: 1,
        BottomActionType.ENLIST: 2,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.UPGRADE_COST,
                                                      BottomUpgradeChoice.UPGRADE_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class PatrioticMat(PlayerMat):  # Mat 3
    name = "Patriotic"
    start_coins = 6
    start_pop = 2
    pairings = {
        TopActionType.MOVE: BottomActionType.UPGRADE,
        TopActionType.BOLSTER: BottomActionType.DEPLOY,
        TopActionType.TRADE: BottomActionType.BUILD,
        TopActionType.PRODUCE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 2},
        BottomActionType.DEPLOY: {Resource.METAL: 4},
        BottomActionType.BUILD: {Resource.WOOD: 4},
        BottomActionType.ENLIST: {Resource.FOOD: 3},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 1,
        BottomActionType.DEPLOY: 3,
        BottomActionType.BUILD: 0,
        BottomActionType.ENLIST: 2,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class InnovativeMat(PlayerMat):  # Mat 3A
    name = "Innovative"
    start_coins = 5
    start_pop = 3
    pairings = {
        TopActionType.TRADE: BottomActionType.UPGRADE,
        TopActionType.PRODUCE: BottomActionType.DEPLOY,
        TopActionType.BOLSTER: BottomActionType.BUILD,
        TopActionType.MOVE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 3},
        BottomActionType.DEPLOY: {Resource.METAL: 3},
        BottomActionType.BUILD: {Resource.WOOD: 4},
        BottomActionType.ENLIST: {Resource.FOOD: 3},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 3,
        BottomActionType.DEPLOY: 1,
        BottomActionType.BUILD: 2,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class MechanicalMat(PlayerMat):  # Mat 4
    name = "Mechanical"
    start_coins = 6
    start_pop = 3
    pairings = {
        TopActionType.TRADE: BottomActionType.UPGRADE,
        TopActionType.BOLSTER: BottomActionType.DEPLOY,
        TopActionType.MOVE: BottomActionType.BUILD,
        TopActionType.PRODUCE: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 3},
        BottomActionType.DEPLOY: {Resource.METAL: 3},
        BottomActionType.BUILD: {Resource.WOOD: 3},
        BottomActionType.ENLIST: {Resource.FOOD: 4},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 0,
        BottomActionType.DEPLOY: 2,
        BottomActionType.BUILD: 2,
        BottomActionType.ENLIST: 2,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.UPGRADE_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


@dataclass(frozen=True)
class AgriculturalMat(PlayerMat):  # Mat 5
    name = "Agricultural"
    start_coins = 7
    start_pop = 4
    pairings = {
        TopActionType.MOVE: BottomActionType.UPGRADE,
        TopActionType.TRADE: BottomActionType.DEPLOY,
        TopActionType.PRODUCE: BottomActionType.BUILD,
        TopActionType.BOLSTER: BottomActionType.ENLIST,
    }
    bottom_cost = {
        BottomActionType.UPGRADE: {Resource.OIL: 2},
        BottomActionType.DEPLOY: {Resource.METAL: 4},
        BottomActionType.BUILD: {Resource.WOOD: 4},
        BottomActionType.ENLIST: {Resource.FOOD: 3},
    }

    bottom_coin_reward = {
        BottomActionType.UPGRADE: 1,
        BottomActionType.DEPLOY: 0,
        BottomActionType.BUILD: 2,
        BottomActionType.ENLIST: 3,
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
        prog = Progress(top_upgrade_opportunities=(TopUpgradeChoice.BOLSTER_MILITARY,
                                                   TopUpgradeChoice.BOLSTER_CARD,
                                                   TopUpgradeChoice.PRODUCE_RESOURCE,
                                                   TopUpgradeChoice.MOVE_UNIT,
                                                   TopUpgradeChoice.MOVE_COIN,
                                                   TopUpgradeChoice.TRADE_POPULARITY),
                        bottom_upgrade_opportunities=(BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.DEPLOY_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.BUILD_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      BottomUpgradeChoice.ENLIST_COST,
                                                      ),)
        return prog


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
        result = list()

        if 1:  # character movement
            # Skeleton: allow character to move to any neighbor.
            here = s.units.character
            all_neighbors = s.board.neighbors(here) + s.board.river_neighbors(here)
            for to_hexid in all_neighbors:
                if self._can_move(s, MoveableUnit.CHARACTER, here, to_hexid):
                    result.append(TurnChoice(TopActionType.MOVE, params=(TopActionType.MOVE_UNIT_CHARACTER, here, to_hexid, 1)))

        # and now allow workers to move
        for (territory, worker_count) in s.units.workers:
            # print("L363 considering moving {} workers from {}".format(worker_count, territory))
            # territory_tuple is a [hid, worker_count]
            all_neighbors = s.board.neighbors(territory) + s.board.river_neighbors(territory)
            available_hexids = list()
            for to_hexid in all_neighbors:
                if self._can_move(s, MoveableUnit.WORKER, territory, to_hexid):
                    available_hexids.append(to_hexid)
            for workers in range(1, worker_count + 1):
                for n in all_neighbors:
                    result.append(TurnChoice(TopActionType.MOVE, params=(TopActionType.MOVE_UNIT_WORKER, territory, n, workers)))

        # REVISIT - mech movement

        # REVISIT - speed mech movement

        # REVISIT - upgrade allows more moves

        return tuple(result)

    def _can_move(self, s: GameState, unit_type: MoveableUnit, from_hexid: HexId, to_hexid: HexId):
        result = True
        neighbors = s.board.neighbors(from_hexid)
        river_neighbors = s.board.river_neighbors(from_hexid)
        match unit_type:
            case MoveableUnit.WORKER:
                if to_hexid in neighbors:
                    result = True
                elif s.faction.name == 'Nordic' and to_hexid in river_neighbors:
                    # swim rule for workers
                    result = True
                else:
                    result = False
            case MoveableUnit.CHARACTER:
                if to_hexid in neighbors:
                    result = True
                else:
                    result = False
            case MoveableUnit.MECH:
                if to_hexid in neighbors:
                    result = True
                else:
                    result = False

        return result

    def _legal_produce_choices(self, s: GameState) -> List[TurnChoice]:
        result: List[TurnChoice] = []

        # first, can we even produce? We must be able to pay produce "costs" based on current worker count (per your model)
        wc = s.units.worker_count()
        if wc >= 4 and s.econ.power < 1:
            # you cannot pay the cost
            return result
        if wc >= 6 and s.econ.popularity < 1:
            # you cannot pay the cost
            return result
        if wc >= 8 and s.econ.coins < 1:
            # you cannot pay the cost
            return result

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
        already_added_choices = list()  # since resources are essentially unordered, let's not add the same pair twice.
        for r in Resource:
            for r2 in Resource:
                param = tuple(sorted((r, r2)))
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
        prog = s.prog
        (unit_type, source_hid, dest_hid, unit_count) = c.params
        # self._assert(dest_hid in s.board.neighbors(source_hid), "Illegal move destination")
        # REVISIT - works for Character only right now
        if unit_type == TopActionType.MOVE_UNIT_CHARACTER:
            new_units = replace(s.units, character=dest_hid)
            # did we trigger an encounter?
            if s.board.hexes[dest_hid].has_encounter and dest_hid not in prog.encounters:
                # print(f"moving a character to {dest_hid} with encounter!!")
                prog = replace(prog, encounters=prog.encounters + (dest_hid, ))
                # BOOKMARK!! Test that this is working.
        elif unit_type == TopActionType.MOVE_UNIT_WORKER:
            # print("Thinking about moving workers {}".format(c.params))
            # going to have to manipulate a lot of immutable data, like s.unit.workers
            new_worker_list = list(list(pair) for pair in s.units.workers)
            if dest_hid not in [hidwc[0] for hidwc in new_worker_list]:
                new_worker_list.append([dest_hid, 0])  # make sure that we have a destination to go to
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

        return replace(s, units=new_units, last_top_action=TopActionType.MOVE, prog=prog)

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
            (r, r2) = c.params
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
        choices = list()
        available = tuple(item for item in s.faction.mech_bench if item not in set(val for (_, val) in s.units.mechs))
        for avail_mech in available:
            for (terr_hex, wcount) in set(tht for tht in s.units.workers):
                choices.append(BottomChoice(BottomActionType.DEPLOY, params=(terr_hex, avail_mech)))
        return choices

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
        # cost = s.mat.bottom_cost[b.action]
        cost = s.bottom_action_cost(b.action)
        reward_coins = s.mat.bottom_coin_reward[b.action]

        # pay the cost for this action
        s2 = replace(s, econ=self._pay(s.econ, cost))

        # get the coin benefit
        s3 = replace(s2, econ=self._gain(s2.econ, coins=reward_coins))

        # get the bonus benefit
        # print(f"Looking for {b.action} in {s3.prog.bottom_bonuses}")
        if b.action in s3.prog.bottom_bonuses:
            bonus = s.mat.bottom_bonus[b.action]
            # print(f"applying {bonus} bottom bonus for {b.action}")
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
        # params are going to be (HexId, Mech)
        (hid, mech) = params
        prog = s.prog
        prog = replace(prog, mechs_deployed=prog.mechs_deployed + 1)
        # REVISIT - mechs tuple is HexId,Mech
        hex_mech_t = s.units.mechs + (params,)
        fff = replace(s.units, mechs=hex_mech_t)
        return replace(s, prog=prog, units=fff)

    def _apply_build(self, s: GameState, params: Tuple[Structure, HexId]) -> GameState:
        prog = s.prog
        structs = list(s.units.structures)
        # print("params is {}".format(params))
        structs.append(params)

        prog = replace(prog, structures_built=len(structs))
        # print("    structs as a list is {}".format(structs))
        # print("    setting structures to be {}".format(tuple(structs)))
        units = replace(s.units, structures=tuple(structs))
        return replace(s, prog=prog, units=units)

    def _apply_upgrade(self, s: GameState, params: Tuple[TopUpgradeChoice, BottomUpgradeChoice]) -> GameState:
        prog = s.prog
        # print("applying bottom upgrade with params {}".format(params))
        # we need to look at prog.top_upgrade_choices and then take this one out
        tou = list(prog.top_upgrade_opportunities)
        tou.remove(params[0])  # take this choice out of the list
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
            # print("BM eval {} vs {}".format(BottomUpgradeChoice.action_type(params[1]), ba))
            if ba == BottomUpgradeChoice.action_type(params[1]):
                mod = mod - 1
            new_bm_list.append((ba, mod))

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
    line_scorer: Callable[[GameState], float] = scorer.line_score,
) -> List[Line]:
    """
    Expand turn by turn. Keep best K lines each turn.
    """
    frontier: List[Line] = [Line(state=start, turn_actions=())]
    verbose = False
    for _ in range(turns):
        candidates: List[Tuple[float, int, Line]] = []
        uid = 0
        if verbose:
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
                        desc = f"T{s.turn + 1}: {top.action.name} {top.params} + {b.action.name} {b.params}"
                        new_line = line.with_step(desc, s_next)
                        candidates.append((line_scorer(s_next), uid, new_line))
                        uid += 1
                else:
                    s_next = engine.end_turn(s_top)
                    desc = f"T{s.turn + 1}: {top.action.name} {top.params} + (no bottom)"
                    new_line = line.with_step(desc, s_next)
                    candidates.append((line_scorer(s_next), uid, new_line))
                    uid += 1

        # keep best beam_width
        if verbose:
            print("   candidate size is {}".format(len(candidates)))
        all_scrs = [scr for scr, _, _ in candidates]
        if verbose:
            print(
                "      max={}, min={}, mean={}, median={}".format(
                    max(all_scrs),
                    min(all_scrs),
                    statistics.mean(all_scrs),
                    statistics.median(all_scrs)))
        best = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        frontier = [ln for _, _, ln in best]
        scrs = [scr for scr, _, _ in best]
        if verbose:
            print("   candidates post beam {}".format(len(best)))
            print("      max={}, min={}, mean={}, median={}".format(max(scrs), min(scrs), statistics.mean(scrs), statistics.median(scrs)))

    # return best lines overall
    return sorted(frontier, key=lambda ln: line_scorer(ln.state), reverse=True)


# -----------------------------
# Starting state (Nordic + Industrial)
# -----------------------------

def make_start_state_general(faction_, mat_) -> GameState:
    board = Board.load_board_from_yaml("board.yaml")

    faction = faction_()
    mat = mat_()

    # Placeholder start:
    # - Character starts at home
    # - 2 workers on home (tuple of locations)
    # - 0 mechs
    # - starting resources/coins/power/popularity: fill in true values later
    units = faction.unit_start
    econ = Economy(
        coins=mat.start_coins,
        power=faction.start_power,
        popularity=mat.start_pop,
        resources=tuple(sorted({Resource.FOOD: 0, Resource.WOOD: 0, Resource.METAL: 0, Resource.OIL: 0, Resource.WORKER: 0}.items(),
                               key=lambda x: x[0].value)),
        combat_cards=faction.start_cards,
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
    territories = s.units.territories_controlled()
    return (
        f"Turn={s.turn}  {s.econ}  "
        f"Territories={territories}  "
        f"Workers={sum(x for _, x in s.units.workers)}  Mechs={len(set(x for _, x in s.units.mechs))}  "
        f"Upg={s.prog.upgrades_done}  Dep={s.prog.mechs_deployed}  "
        f"Build={s.prog.structures_built}  Enlist={s.prog.enlists}  "
        f"Combat Cards={s.econ.combat_cards}  "
        f"Char@{s.units.character}"
        f"\n    prog={s.prog}"
        f"\n    units={s.units}"
    )


# --- Registries ---
FACTIONS = {
    "albion": albion_config,
    "crimea": crimea_config,
    "nordic": nordic_config,
    "polania": polania_config,
    "togawa": togawa_config,
    "saxony": saxony_config,
    "polania": polania_config,
    "rusviet": rusviet_config
}

MATS = {
    "agricultural": AgriculturalMat,
    "industrial": IndustrialMat,
    "engineering": EngineeringMat,
    "militant": MilitantMat,
    "patriotic": PatrioticMat,
    "innovative": InnovativeMat,
    "mechanical": MechanicalMat,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scythe opening solver (beam search)"
    )

    parser.add_argument(
        "--faction",
        required=True,
        choices=sorted(FACTIONS.keys()),
        help="Faction name",
    )

    parser.add_argument(
        "--mat",
        required=True,
        choices=sorted(MATS.keys()),
        help="Player mat name",
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=8,
        help="Number of turns to search (default: 8)",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=1000,
        help="Beam width (default: 1000)",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top lines to print (default: 10)",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # Resolve faction + mat from registries
    faction_fn = FACTIONS[args.faction]
    mat_cls = MATS[args.mat]

    engine = Engine()

    start = make_start_state_general(
        faction_fn,
        mat_cls,
    )

    lines = beam_search_openings(
        engine,
        start,
        turns=args.turns,
        beam_width=args.beam_width,
    )

    print(f"\nTop {args.top} opening lines:")
    for i, ln in enumerate(lines[:args.top], start=1):
        print(f"\n#{i}  Score={scorer.line_score(ln.state):.2f}")
        for step in ln.turn_actions:
            print(" ", step)
        print(" ", summarize_state(ln.state))
