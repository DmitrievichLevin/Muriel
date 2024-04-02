"""ECS Utility Functions"""

from typing import Generator


def four_point_float(
    *values,
) -> Generator:
    """Four Point Float Formatter

    Yields:
        Generator: formated floats.
    """
    _format = lambda a: float("{:10.4f}".format(a))
    yield from [_format(v) for v in values]
