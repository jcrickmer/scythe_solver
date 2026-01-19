#!python
from __future__ import annotations

from dataclasses import dataclass, field, replace
from units import Mech
# from gamestate import Units


@dataclass(frozen=True)
class Faction:
    name: str
    start_power: int
    start_cards: int
    special_rules: Tuple[str, ...] = ()
    unit_start: Tuple[str, ...] = ()
    mech_bench: Tuple[Mech, Mech, Mech, Mech] = ()


def albion_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Albion",
        start_power=3,
        start_cards=0,
        special_rules=("TODO: ",),
        unit_start=Units(character="A_HOME", mechs=(), workers=(("A_MOUNTAIN", 1), ("A_FARM", 1)), structures=()),
        mech_bench=(Mech.BURROW, Mech.SWORD, Mech.SHIELD, Mech.RALLY),
    )


def nordic_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Nordic",
        start_power=4,
        start_cards=1,
        special_rules=("TODO: Nordic riverwalk / swim rules",),
        unit_start=Units(character="N_HOME", mechs=(), workers=(("N_FOREST", 1), ("N_TUNDRA", 1)), structures=()),
        mech_bench=(Mech.RIVERWALK_FORESTS_MOUNTAINS, Mech.SEAWORTHY, Mech.ARTILLERY, Mech.SPEED),
    )


def polania_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Polania",
        start_power=2,
        start_cards=3,
        special_rules=("TODO: ",),
        unit_start=Units(character="P_HOME", mechs=(), workers=(("P_FOREST", 1), ("P_FARM", 1)), structures=()),
        mech_bench=(Mech.RIVERWALK_VILLAGES_MOUNTAINS, Mech.SUBMERGE, Mech.CAMARADERIE, Mech.SPEED),
    )


def rusviet_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Rusviet",
        start_power=3,
        start_cards=2,
        special_rules=("TODO: ",),
        unit_start=Units(character="R_HOME", mechs=(), workers=(("R_MOUNTAIN", 1), ("R_VILLAGE", 1)), structures=()),
        mech_bench=(Mech.RIVERWALK_FARMS_VILLAGES, Mech.TOWNSHIP, Mech.PEOPLES_ARMY, Mech.SPEED),
    )


def saxony_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Saxony",
        start_power=1,
        start_cards=4,
        special_rules=("TODO: ",),
        unit_start=Units(character="S_HOME", mechs=(), workers=(("S_MOUNTAIN", 1), ("S_TUNDRA", 1)), structures=()),
        mech_bench=(Mech.RIVERWALK_FORESTS_MOUNTAINS, Mech.UNDERPASS, Mech.DISARM, Mech.SPEED),
    )


def crimea_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Crimea",
        start_power=5,
        start_cards=0,
        special_rules=("TODO: ",),
        unit_start=Units(character="C_HOME", mechs=(), workers=(("C_FARM", 1), ("C_VILLAGE", 1)), structures=()),
        mech_bench=(Mech.RIVERWALK_FARMS_TUNDRA, Mech.WAYFARE, Mech.SCOUT, Mech.SPEED),
    )


def togawa_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Togawa",
        start_power=0,
        start_cards=2,
        special_rules=("TODO: ",),
        unit_start=Units(character="T_HOME", mechs=(), workers=(("T_FARM", 1), ("T_TUNDRA", 1)), structures=()),
        mech_bench=(Mech.TOKA, Mech.SUITON, Mech.RONIN, Mech.SHINOBI),
    )
