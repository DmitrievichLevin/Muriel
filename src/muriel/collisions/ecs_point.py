from __future__ import annotations
from collections.abc import Iterator
import enum
from functools import reduce
from math import floor
from typing import Any, TypeVar
from .ecs_bsp_utils import (
    point_bounded_by_triangle,
    point_bounded_by_polygon,
    Polygon,
    Normal,
)
import collections


class MatrixIterator(Iterator):
    def __init__(self, mt: Matrix, length=None):
        self.mt = mt
        self.pos = -1
        self.length = length or mt.rows

    def __next__(self):
        self.pos += 1
        if self.pos < self.length:
            if self.pos >= len(self.mt):
                return Point()
            return self.mt[self.pos]
        raise StopIteration


_S = TypeVar("_S", float, int)


class Point(list[float | int]):
    """3-Dimensional Coordinate

    Attributes:
        is_float(bool): Whether vector includes floats
    Args:
         values (tuple[int | float] * 3): list of int(s) and/or float(s).
    """

    is_float: bool
    empty: bool

    def __init__(self, *values, **_kwargs):

        self.__type_check(values)
        super().__init__(values)

    def __type_check(self, values):
        length = (
            0
            if not isinstance(values, (list, tuple))
            else len(values)
        )
        print(values)
        match length:
            case 3:
                self.is_float = False
                for v in values:
                    if not isinstance(v, (float, int)):
                        raise TypeError(
                            "Expected elements of type float | int."
                        )
                    elif v - int(v) != 0:
                        self.is_float = True

                    else:
                        v = int(v)
                self.empty = False
            case 0:
                self.empty = True
            case _:
                raise TypeError(
                    "Expected three elements of type float | int."
                )

    def __add__(self, vector: list[Any], /) -> Point:  # type: ignore[override]
        _addition = [self[idx] + v for idx, v in enumerate(vector)]
        return Point(*_addition)

    def __sub__(self, vector: list[_S], /) -> Point:

        _subtraction = [self[idx] - v for idx, v in enumerate(vector)]
        return Point(*_subtraction)

    def __mul__(self, value) -> Point:
        if isinstance(value, (float, int)):
            _scalar = [v * value for v in self]
            return Point(*_scalar)

        return self.cross(self, value)

    def __rmul__(self, value) -> Point:

        if isinstance(value, (float, int)):
            _scalar = [v * value for v in self]
            return Point(*_scalar)

        return self.cross(value, self)

    def __imul__(self, value) -> Point:

        if isinstance(value, (float, int)):
            _scalar = [v * value for v in self]
            return Point(*_scalar)

        return self.cross(self, value)

    def __pow__(self, value: int) -> Point:
        power = [v**value for v in self]
        return Point(*power)

    def __floor__(self):
        """Reduce Vector by GCD

        Returns:
            point(Point): Vector reduced by GCD.
        """

        # includes: Float = No reduction
        if self.is_float:
            return self

        gcd = 1

        gcd = lambda a, b: (  # pylint: disable=unnecessary-lambda-assignment
            int(a)
            if b == 0
            else gcd(  # pylint: disable=not-callable
                # type: ignore[reportCallIssue]
                abs(b),
                abs(a) % abs(b),
            )
        )

        gcd = reduce(gcd, self)

        return Point(*[0 if v == 0 else int(v / gcd) for v in self])

    @classmethod
    def cross(cls, right, left) -> Point:
        """Matrix Multiplication

        - Scalar: if left is scalar
        - Matrix: if left is matrix

        Args:
            right (Point): 3D Vector
            left (Point | int | float): 3D Vector or Scalar

        Returns:
            result(Point | int | float): result of matrix/scalar multiplication
        """
        u1, u2, u3 = right
        v1, v2, v3 = left
        t1, t2, t3 = u1 - u2, v2 + v3, u1 * v3
        t4 = (t1 * t2) - t3

        cross = Point(
            (v2 * (t1 - u3)) - t4,
            (u3 * v1) - t3,
            t4 - (u2 * (v1 - t2)),
        )

        return cross

    @classmethod
    def sqrt(cls, obj: Point):
        """Square Root Point Function"""
        return obj.__sqrt__()

    def __sqrt__(self) -> Point:
        res = []
        for v in self:

            res.append(v**0.5)

        return Point(*res)

    def __abs__(self) -> Point:
        """Absolute Value of Point

        - Point as a unit

        Returns:
            Point: point as a unit -1,0,1
        """
        return Point(
            *[0 if v == 0 else int(v / abs(v)) for v in self]
        )


class Matrix(tuple[Point, ...]):
    cols: int
    rows: int

    def __new__(cls, *points: Point):
        instance = super(Matrix, cls).__new__(cls, points)

        try:
            cols = len(instance[0])
            rows = len(instance)
            instance.cols = cols
            instance.rows = rows
            return instance
        except Exception as e:
            raise TypeError("Error initializing matrix.") from e

    def __eq__(self, other: Any) -> Any:
        """Comparison Override"""

        if not isinstance(other, Matrix) and not issubclass(
            other.__class__, Matrix
        ):
            return False

        if self.rows != other.rows or self.cols != other.cols:
            return False

        close = []

        for i in range(self.rows):
            for j in range(self.cols):
                close.append(
                    abs(self[i][j] - other[i][j])
                    < getattr(self, "__thickness", 1e-08)
                )

        return all(close)

    def __iter__(self):
        try:
            return super().__iter__()
        except ValueError:
            return MatrixIterator(self)


def sort_points(*_points):

    left = 0
    right = 1
    points = list(_points)
    origin = Point(0, 0, 0)
    while left != len(points) - 1 and right != len(points):
        if points[left] == points[right]:
            points[right] = points[left + 1]
            points[left + 1] = points[left]
            continue
        _u, _v, w = Normal(points[left], points[right], origin).unit

        if w < 0:
            right += 1

        else:
            if right + 1 == left:
                right += 1
                left += 1
                continue
            points[left + 1] = points[right]
            points[right] = points[left + 1]
            left += 1
            right = left + 1

    return points


def hulls_algo(*_points: Point):
    import numpy as np

    points = np.unique(_points, axis=0)
    poly_avg = np.average(points, axis=0)

    print("unsorted", points)
    a, b, c, d, *rest = [
        Point(*points[i].tolist())
        for i in np.argsort(
            [(lambda x: sum(np.abs(x - poly_avg)))(p) for p in points]
        )
    ]

    poly = Polygon(*sort_points(a, b, c, d))
    print(
        "sorted poly",
        *poly,
        "points",
    )
    filtered = filter(
        lambda x: not point_bounded_by_polygon(x, poly),
        rest,
    )

    # TODO - Recurively add points furthest from each edge to polygon until none left
    print("filtered", [v for v in filtered])


def test_point_matrix():
    v1, v2, v3 = (
        Point(3, 5, 12),
        Point(6, 5, 9),
        Point(6, 8, 15),
    )

    vector = (v2 - v1) * (v3 - v1)

    assert all(v == [9, -18, 9][idx] for idx, v in enumerate(vector))

    simplified = floor(vector)

    assert all(
        v == [1, -2, 1][idx] for idx, v in enumerate(simplified)
    )

    not_simplified = floor(Point(1 / 2, 8, 4))

    assert all(
        v == [0.5, 8, 4][idx] for idx, v in enumerate(not_simplified)
    )

    unit_denom = abs(vector)

    assert all(
        v == [1, -1, 1][idx] for idx, v in enumerate(unit_denom)
    )

    matrix = Matrix(v1, v2, v3)

    matrix_1 = Matrix(v1, v2, Point(1 / 2, 8, 4))

    assert matrix == Matrix(v1, v2, v3)

    assert matrix != matrix_1

    # Unpacking Matrix
    _a, _b, _c, d = MatrixIterator(matrix, length=4)

    a, b, c = MatrixIterator(matrix)

    assert d.empty

    assert all(not v.empty for v in [a, b, c])

    CBD, BED, FCH = (
        Matrix(Point(0, 5, -3), Point(5, 0, -1), Point(5, 5, -4)),
        Matrix(
            Point(5, 0, -1),
            Point(10, 5, 4),
            Point(5, 5, -4),
        ),
        Matrix(Point(-1, 5, -1), Point(0, 4, -2), Point(0, 5, -3)),
    )

    hulls_algo(*CBD, *BED, *FCH)


test_point_matrix()
