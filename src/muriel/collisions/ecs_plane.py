"""ECS Plane Module"""

# pylint: disable=line-too-long
from __future__ import annotations
from collections.abc import Iterator
from math import floor
import os
import types
from typing import Any, TypeVar
from functools import reduce
import numpy as np
from .ecs_constants import Shape, Orientation


def sqrt(obj):
    return obj.__sqrt__()


# TODO - move to top level
float_formatter = "{:.5f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})


def dot_product(x, y, z, i, j, k):
    return (i * x) + (j * y) + (k * z)


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
    """3D Matrix Base Class

    Args:
        tuple (Point): Point args

    Raises:
        TypeError: Error initializing Matrix

    Returns:
        Matrix: instance of 3D Matrix
    """

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


class MatrixIterator(Iterator):
    """Matrix Iterator

    - Handles ValueError Exception with length arguement.
    """

    def __init__(self, mt, length=None):
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


class Line(Matrix):
    """Two 3D Vectors
    - represents a line segment in 3D space

    Args:
         input_array (list[int | float]): list of int(s) and/or float(s) of length 2.
    """

    _a: Point
    _b: Point
    cross_product: Point
    difference: Point

    def __new__(cls, *points: Point):
        instance = super().__new__(cls, *points)
        a, b = instance
        instance.cross_product = a * b
        instance.difference = b - a

        return instance


class Normal(object):
    """Plane Normal
    - vector perpendicular to plane.

    Attributes:
        normal (Vector3d): simplified raw_normal(gcd).
        unit (Vector3d): normal as a unit.
        raw_normal (Vector3d): vector perpendicular to plane.

    Args:
        input_array (list[Vector3d]): list of 3d vectors 2>vector_list>infinity.
    """

    def __init__(
        self,
        v1: Point,
        v2: Point,
        v3: Point,
        *_rest: Point,
    ):
        """Calculate normals
        - Normal of v1-3
        - Normal as a unit
        - Simplified Normal (gcd)

        Args:
            v1(Point): point v1
            v2(Point): point v2
            v3(Point): point v3

        Returns:
            Normal(list[float | int]): Normal of three points
            Unit normal(list[int]): Normal as a unit
            Simplified Normal(list[float | int]): Normal simplified by GCD
        """
        normal: Point = (v2 - v1) * (v3 - v1)

        simplified_normal = floor(normal)
        unit_normal = abs(normal)

        self.raw_normal = normal
        self.normal = simplified_normal
        self.orientation = Orientation(str(unit_normal))
        self.unit = unit_normal


class Plane(Matrix):
    """Plane

    Attributes:
        normal (Vector3d): vector perpendicular to plane.
        unit_normal (Vector3d): normal as a unit.
        dot_product (float): point on plane.
        vertices (int): 0 Plane is not closed.
        orientation (Orientation): Unit Normal Enum.

    Args:
        input_array(list[Vector3d]): list of 3d vectors 2>vector_list>infinity
        name(str): optional naming for debugging.
    """

    dot_product: float
    normal: Point
    unit_normal: Point
    raw_normal: Point
    ab: Point
    bc: Point
    ca: Point
    vertices: int
    name: str
    orientation: Orientation
    capacity: int = 3

    def __new__(cls, *points: Point, name="Unknown:Plane"):
        instance = super().__new__(cls, *points)

        instance.name = name

        if instance.rows < cls.capacity:
            raise TypeError(
                f"Expected {cls.capacity} vertices, but found {instance.rows}."
            )

        a, b, c, *_rest = instance

        _normal = Normal(a, b, c)

        instance.__dict__.update(
            {
                "normal": _normal.normal,
                "unit_normal": _normal.unit,
                "dot_product": dot_product(*_normal.normal, *a),
                "vertices": instance.rows,
                "raw_normal": _normal.raw_normal,
                "orientation": _normal.orientation,
                "__thickness": float(
                    os.environ["PLANE_THICKNESS_EPSILON"]
                ),
                "ab": b - a,
                "bc": c - b,
                "ca": a - c,
            }
        )

        return instance


class Polygon(Plane):
    """Closed Plane Subclass

    - Polygons are closed plane figures that have three or more sides.
    - concavity boolean

    Attributes:
        normal (Vector3d): vector perpendicular to polygon.
        unit_normal (Vector3d): normal as a unit.
        dot_product (float): point on polygon.
        vertices (int): Number of enclosing points.
        convex(bool): convex-polygon or concave-polygon.
        solid(bool): Polygon is solid.
        supporting(bool): Flag to prevent reselection in auto-partitioning bsp.

    Args:
        input_array(list[Vector3d]): list of 3d vectors 2>vector_list>infinity.
        name(str): optional naming for debugging.
    """

    convex: bool
    solid: bool = True
    supporting: bool = False

    def __new__(cls, *points: Point, name="Unknown:Polygon"):
        instance = super().__new__(cls, *points, name=name)
        instance.convex = is_convex(instance)
        return instance

    def __str__(self):

        match self.rows * 2 - self.cols:
            case Shape.QUADRILATERAL:
                _type = "Quadrilateral"

            case Shape.TRIANGLE:
                _type = "Triangle"
            case _:
                return super().__str__()
        _normal = self.raw_normal
        _dot = self.dot_product
        _convex = self.convex
        return f"{_type}: \n Concavity: {_convex} \n Vertices: {np.array(self)} \n Normal: {_normal} \n Dot Product: {_dot}"


class Quadrilateral(Polygon):
    """Polygon w/ 4 verticies"""

    capacity = 4


class Triangle(Polygon):
    """Polygon w/ 3 verticies"""

    capacity = 3


def is_convex(
    matrix: Polygon,
):
    """Matrix concavity."""
    match (matrix.rows):
        # Triangles are always convex
        case 3:
            return True
        case 4:
            _a, _b, _c, _d = matrix
            d_b = Line(_d, _b)
            a_b = Line(_a, _b)
            c_b = Line(_c, _b)

            bda = d_b.difference * a_b.difference
            bdc = d_b.difference * c_b.difference

            # Non-Convex
            if dot_product(*bda, *bdc) >= 0.0:
                return False
            # Convex
            else:
                c_a = Line(_c, _a)
                d_a = Line(_d, _a)
                b_a = Line(_b, _a)

                acd = c_a.difference * d_a.difference
                acb = c_a.difference * b_a.difference

                return dot_product(*acd, *acb) < 0.0

        case _:
            # Polygons with vertices < 3 = convex
            return True


def scalar_triple(*uwv: Point) -> int | float:
    """Scalar Triple Product

    - formula: u · (v · w)
    """
    u, w, v = uwv
    idx = [0, 1, 2]

    product = 0
    for i in range(3):
        i, j, k = np.roll(idx, shift=-i)
        _i, _j, _k = np.roll(idx[::-1], shift=i)

        product += u[i] * v[j] * w[k] - u[_i] * v[_j] * w[_k]

    return product


def test_scalar_triple():
    """Test Scalar Triple Product"""

    u, v, w = Point(3, 7, 10), Point(6, 10, 13), Point(6, 7, 7)

    assert scalar_triple(u, v, w) == 9


class Intersection(Point):
    """Intersecting Vector of Plane & Segment

    Args:
        matrix(Polygon | Plane): Polygon/Plane to test segment against.
        line(Line): Edge to be tested.
    """

    def __init__(
        self,
        matrix: Polygon | Plane,
        line: Line,
    ):

        p, q = line

        pq = q - p

        # TODO - Sort Maybe?
        # Vertices
        a, b, c, d = MatrixIterator(matrix, length=4)

        # pa, pb, pc, pd | None
        pa = a - p
        pb = b - p
        pc = c - p
        pd = Point() if d.empty else d - p  # type: ignore[assignment]

        m = pc * pq

        # Mutable
        uvw = [-dot_product(*pb, *m), dot_product(*pa, *m), None]

        # Immutable
        u, v, w = uvw

        # Match condition
        matrix_type = type(matrix)

        # Match options
        options = types.SimpleNamespace()
        options.Plane = Plane
        options.Triangle = Triangle
        options.Quadrilateral = Quadrilateral

        match matrix_type:
            case options.Plane:
                intersect = self.__plane(p, pq, matrix)

                super().__init__(*intersect)
            case options.Triangle:

                uvw[1] *= -1

            case options.Triangle:

                if v < 0:

                    u = dot_product(*pd, *m)
                    uvw[0] = u

                    v = -v
                    uvw[1] = v

                    intersecting = self.__dac(
                        uvw, *[pq, pa, pb, pc, pd]
                    )
                    u, v, w = uvw

                    intersect = (
                        (u * a) + (v * d) + (w * c)
                        if intersecting
                        else Point()
                    )

                    super().__init__(*intersect)
                else:
                    u = -dot_product(*pb, *m)
                    uvw[0] = u

        # ABC if Triangle or Quad(edge case)
        if (
            matrix_type == options.Triangle or (v >= 0)
        ) and not matrix_type == options.Plane:

            intersecting = self.__abc(uvw, *[pq, pa, pb, pc, pd])

            u, v, w = uvw

            if intersecting:
                intersect = (u * a) + (v * b) + (w * c)

            else:
                intersect = Point()

            super().__init__(*intersect)

    @classmethod
    def __abc(cls, uvw, *_points) -> bool:

        u, v, w = uvw

        pq, pa, pb, *_rest = _points

        if u < 0:
            return False

        w = scalar_triple(pq, pb, pa)

        if w < 0:
            return False

        denom = 1.0 / sum([u, v, w])

        # u
        uvw[0] = u * denom
        # v
        uvw[1] = v * denom
        # w
        uvw[2] = w * denom

        return True

    @classmethod
    def __dac(cls, uvw, *_points) -> bool:

        u, v, w = uvw

        pq, pa, _pb, _pc, pd = _points

        if u < 0:
            return False

        w = scalar_triple(pq, pa, pd)

        if w < 0:
            return False

        denom = 1.0 / sum([u, v, w])

        # u
        uvw[0] = u * denom
        # v
        uvw[1] = v * denom
        # w
        uvw[2] = w * denom

        return True

    @classmethod
    def __plane(cls, p: Point, pq: Point, plane: Plane) -> Point:

        t = (
            plane.dot_product - dot_product(*plane.normal, *p)
        ) / dot_product(*plane.normal, *pq)

        if t >= 0 and t <= 1:

            return p + t * pq

        return Point()
