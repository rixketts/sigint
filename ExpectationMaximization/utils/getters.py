from collections.abc import Iterable
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def get_final_two_values(
        items: Iterable[T] | dict[K, V]
) -> tuple[T | V, T | V]:
    """
    Return (final item, second to last item)
    """
    if type(items) == dict:
        items = list(items.values())

    return (items[-1], items[-2])
