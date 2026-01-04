from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum, auto

# -----------------------------
# Basic identifiers / enums
# -----------------------------


class Resource(IntEnum):
    FOOD = auto()
    WOOD = auto()
    METAL = auto()
    OIL = auto()
    WORKER = auto()


class Structure(IntEnum):
    MONUMENT = auto()
    MILL = auto()
    MINE = auto()
    ARMORY = auto()


class Popularity(Enum):
    POPULARITY = auto()


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
