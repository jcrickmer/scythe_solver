from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto


class Resource(IntEnum):
    FOOD = auto()
    WOOD = auto()
    METAL = auto()
    OIL = auto()
    WORKER = auto()
