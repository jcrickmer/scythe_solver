#!python
from __future__ import annotations

from dataclasses import dataclass, field, replace
# from gamestate import Units


@dataclass(frozen=True)
class Faction:
    name: str
    start_power: int
    start_cards: int
    # Nordic specifics you might later encode:
    # - riverwalk rules
    # - "Swim" (workers may move into/through lakes?) etc.
    # Keep placeholders so the engine can reference them.
    special_rules: Tuple[str, ...] = ()
    unit_start: Tuple[str, ...] = ()


def nordic_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Nordic",
        start_power=4,
        start_cards=1,
        special_rules=("TODO: Nordic riverwalk / swim rules",),
        unit_start=Units(character="N_HOME", mechs=(), workers=(("N_FOREST", 1), ("N_TUNDRA", 1)), structures=()),
    )


def crimea_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Crimea",
        start_power=5,
        start_cards=0,
        special_rules=("TODO: ",),
        unit_start=Units(character="C_HOME", mechs=(), workers=(("C_FARM", 1), ("C_VILLAGE", 1)), structures=()),
    )


def togawa_config() -> Faction:
    from gamestate import Units
    return Faction(
        name="Togawa",
        start_power=0,
        start_cards=2,
        special_rules=("TODO: ",),
        unit_start=Units(character="T_HOME", mechs=(), workers=(("T_FARM", 1), ("T_TUNDRA", 1)), structures=()),
    )
