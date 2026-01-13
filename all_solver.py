#!python

from __future__ import annotations
import sys
import argparse

import solver


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scythe opening solver for all mats and factions (beam search)"
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

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    for mat_name, mat_cls in solver.MATS.items():
        for faction_name, faction_fn in solver.FACTIONS.items():
            engine = solver.Engine()

            start = solver.make_start_state_general(
                faction_fn,
                mat_cls,
            )

            lines = solver.beam_search_openings(
                engine,
                start,
                turns=args.turns,
                beam_width=args.beam_width,
            )

            print(f"\n===============\nFaction {faction_name}, mat {mat_name}:")
            for i, ln in enumerate(lines[:1], start=1):
                print(f"Score={solver.scorer.line_score(ln.state):.2f}")
                for step in ln.turn_actions:
                    print(" ", step)
                print(" ", solver.summarize_state(ln.state))
