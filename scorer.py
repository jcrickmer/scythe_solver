#!python
from gamestate import GameState
import math


# -----------------------------
# Scoring (tune as you like)
# -----------------------------

COMBAT_CARDS_AVG = (16 * 2 + 12 * 3 + 8 * 4 + 6 * 5) / 42


def line_score(s: GameState) -> float:
    opening_weight = max(0.1, (10.0 - s.turn) / 9.0)
    sigma = 4.0
    midgame_weight = math.exp(-0.5 * ((s.turn - 14) / sigma) ** 2)
    endgame_weight = max(0.1, (s.turn - 12) / 10)

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
    score += 3.3 * len(s.prog.encounters)
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
    score += 0.5 * len(s.prog.encounters)
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
    score += 2.0 * workers
    score += 5.0 * mechs
    score += 3.3 * len(s.prog.encounters)
    score += 4.0 * s.prog.upgrades_done * ((25.0 - min(24, s.turn)) / 25.0)
    score += 4.1 * s.prog.enlists * ((25.0 - min(24, s.turn)) / 25.0)
    score += 4.0 * s.prog.structures_built * ((25.0 - min(24, s.turn)) / 25.0)
    score += 0.76 * COMBAT_CARDS_AVG * max(1, (6 - s.econ.combat_cards))
    score += 1.1 * s.econ.coins
    score += 1.1 * s.econ.popularity
    score += 0.75 * power_bell(s.econ.power, mechs)
    # Small bonus for having some resources banked
    score += 0.2 * sum(dict(s.econ.resources).values())

    # REVISIT - also, score needs to be resource weighted... the resources
    # that fund low-cost advancements should have a lower weight than those
    # that fund high-cost advancements.

    # REVISIT - lastly, use some scoring that reflects the actual scoreboard of the game.

    # REVISIT - think about how goals change over time.
    #   Opening game = resources and production possiblity
    #   Mid game = tempo, board control, power potential
    #   End game = focus on Scythe scoring system
    #
    # Maybe as the game progresses, the weight of each of these scoring
    # systems shifts. Turns 1-6 are opening, 7-18 or mid game, 19+ is end
    # game.

    return score


def power_bell(power: int, n_mechs: int, sigma: float = 3.0) -> float:
    """
    Returns a value in (0, 1] that peaks when power ~= (n_mechs + 1) * 7,
    capped at 18 (Scythe’s max power).
    """
    n_units = n_mechs + 1  # character + mechs
    ideal = min(18, 7 * n_units)

    # Standard Gaussian centered at `ideal`, max = 1.0 at the peak
    return math.exp(-0.5 * ((power - ideal) / sigma) ** 2)


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
