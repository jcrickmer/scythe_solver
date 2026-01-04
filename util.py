#!python

from board import HexId
from typing import List, Tuple

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
