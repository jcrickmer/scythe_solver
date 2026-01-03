#!python

from __future__ import annotations

from dataclasses import dataclass, field, replace
from board import Board


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
