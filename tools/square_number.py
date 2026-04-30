"""Square a number tool."""

from tools import tool


@tool
def square_number(x: float) -> float:
    """Square a number.

    Args:
        x: The number to square
    """
    return x * x
