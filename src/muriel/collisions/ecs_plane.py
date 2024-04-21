"""ECS Plane Module"""

# pylint: disable=line-too-long
from __future__ import annotations
import os
from types import NoneType
from typing import Any
from functools import reduce
import numpy as np
from numpy.typing import NDArray
from .ecs_constants import Shape, Orientation


class NDarray(np.ndarray):
    """ndArray Base Class"""

    # def __init__(self, *_args, capacity=0) -> None:
    #     super().__init__()  # pylint: disable=no-value-for-parameter

    def __new__(cls, input_array: Any):
        """Construct NDarray Instance

        Args:
            input_array(Any): input

        Raises:
            ValueError: if size or not float or int
        """
        _capacity = getattr(cls, "capacity", 0)

        _vector_like = cls.vectorlike(input_array)

        # Check capacity
        if (
            (
                _vector_like
                and _capacity
                and len(input_array) != _capacity
            )
            or not _vector_like
            and cls.arraylike(input_array)
        ):
            raise ValueError(
                f"{cls} expected list of length {_capacity}, with elements of type int or float but found {cls}."
            )
        # Handle encounters of numpy/native primitives
        if not cls.arraylike(input_array):
            return cls.tobase(input_array)

        return np.asarray(input_array).view(cls)

    @classmethod
    def vectorlike(cls, obj: Any):
        """Check if array contains all floats and/or ints"""
        _flat_list = np.array(obj).tolist()
        if isinstance(_flat_list, list):
            return all(
                [
                    (
                        isinstance(elem, (int, float))
                        if not isinstance(elem, list)
                        else cls.vectorlike(elem)
                    )
                    for elem in _flat_list
                ]
            )
        return False

    @classmethod
    def arraylike(cls, obj: Any):
        """Check if object is array like"""
        return cls.isinstance(obj) or isinstance(obj, (list, tuple))

    @classmethod
    def isinstance(cls, instance: Any) -> bool:
        """Check instance is list, tuple, or numpy array"""
        parents = type(instance).__mro__

        is_numpy = any([np.ndarray == parent for parent in parents])

        return is_numpy

    def __array_finalize__(self, obj) -> None:

        if obj is None:
            return
        # This attribute should be maintained!
        # pylint: disable=attribute-defined-outside-init
        self.__dict__.update(attr=1)  # another way to set attributes

    def tobase(self, _obj=None) -> Any:
        """Convert Subclass to base(NDArray)

        - converts numpy primitives to native primitives
        """
        obj = self if _obj is None else self

        if not NDarray.isinstance(obj):
            primitive = getattr(obj, "tolist", lambda: obj)()
            primitive_cls = primitive.__class__

            return (
                obj
                if any(
                    [
                        primitive_cls == prim
                        for prim in [int, float, NoneType]
                    ]
                )
                else type(primitive.__class__)(
                    obj
                )  # type: ignore[misc]
            )

        # Recursively convert list/tuple to base
        elif isinstance(obj, (list, tuple)):

            return [self.tobase(elem) for elem in obj]

        return np.array(obj)

    def __array_prepare__(self, array, context):
        base_array = np.array(array.tolist())
        ufunc, *_context = context
        _context = [self.tobase(con) for con in _context]

        return super().__array_prepare__(  # pylint: disable=no-member
            base_array,
            (ufunc, *_context),
        )  # type: ignore[reportArgumentType]

    def __eq__(self, other: Any) -> Any:
        """Comparison Override"""
        _other = (
            other if not hasattr(other, "tolist") else other.tolist()
        )

        _self = self.tolist()

        if not self.arraylike(other):
            return False

        if len(_other) != len(_self):
            return False

        return np.all(
            np.isclose(
                _self,
                _other,
                atol=getattr(self, "__thickness", 1e-08),
            )
        )


class Vector3d(NDarray):
    """3-Dimensional Vector

    Args:
         input_array (list[int | float]): list of int(s) and/or float(s).
    """

    capacity = 3

    def __new__(cls, input_array: Any):
        return super().__new__(cls, input_array)


class Line(NDarray):
    """Two 3D Vectors
    - represents a line segment in 3D space

    Args:
         input_array (list[int | float]): list of int(s) and/or float(s) of length 2.
    """

    _a: Vector3d | list[int | float] | tuple[int | float]
    _b: Vector3d | list[int | float] | tuple[int | float]
    cross_product: Vector3d
    difference: Vector3d
    capacity = 2

    def __new__(
        cls,
        input_array: list[
            Vector3d | list[int | float] | tuple[int | float]
        ],
    ):
        return super().__new__(cls, input_array)

    def __array_finalize__(self, obj) -> None:
        super().__array_finalize__(obj)

        obj_list = obj.tolist()
        a = np.array(obj_list[0])
        b = np.array(obj_list[1])

        computed_attrs = {
            "_a": a,
            "_b": b,
            "cross_product": Vector3d(a * b),
            "difference": Vector3d(b - a),
        }

        self.__dict__.update(
            computed_attrs
        )  # another way to set attributes


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
        v1: Vector3d | NDArray,
        v2: Vector3d | NDArray,
        v3: Vector3d | NDArray,
        *_rest: Vector3d | NDArray,
    ):

        normal = self.get_normal(v1, v2, v3)

        normal_gcd = self.gcd(normal)

        # Simplify normal (greatest common denominator)
        simplified_normal = np.divide(
            normal,
            normal_gcd,
            out=np.zeros_like(normal, dtype=float),
            where=normal_gcd != 0.0,
        )

        # Noraml as a unit
        unit_denom = np.sqrt([np.square(p) for p in normal])

        # Handles Division by zero
        orientation = np.divide(
            normal,
            unit_denom,
            out=np.zeros_like(normal, dtype=int),
            where=unit_denom != 0.0,
            casting="unsafe",
        ).tolist()

        self.raw_normal = Vector3d(normal)
        self.normal = Vector3d(simplified_normal)
        self.orientation = Orientation(str(orientation))
        self.unit = orientation

    @classmethod
    def get_normal(
        cls,
        v1: Vector3d | NDArray,
        v2: Vector3d | NDArray,
        v3: Vector3d | NDArray,
    ):
        """Calculate normal + Convert whole numbers to ints"""

        return np.vectorize(
            lambda x: int(x) if x - int(x) == 0 else round(x, 4)
        )(np.cross(v2 - v1, v3 - v1))

    @classmethod
    def gcd(cls, vector: Vector3d | NDArray) -> int:
        """GCD of Vector

        Args:
            vector (Vector3d | NDArray): vector 3

        Returns:
            int: 1 if vector contains non-whole number, else gcd
        """
        g = lambda a, b: (  # pylint: disable=unnecessary-lambda-assignment
            1
            if not float(a).is_integer()
            else int(a) if b == 0 else g(abs(b), abs(a) % abs(b))  # type: ignore[has-type]
        )

        # pylint: disable=unnecessary-lambda
        return reduce(lambda x, y: g(x, y), vector)


class Plane(NDarray):
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
    normal: Vector3d
    unit_normal: Vector3d
    raw_normal: Vector3d
    ab: Vector3d
    bc: Vector3d
    ca: Vector3d
    vertices: int
    name: str
    orientation: Orientation

    def __new__(
        cls,
        input_array: list[Vector3d] | Plane | tuple[Vector3d, ...],
        **_kwargs,
    ):
        _input_vectors = [Vector3d(arr) for arr in input_array]
        return super().__new__(cls, _input_vectors)

    def __init__(  # pylint: disable=super-init-not-called
        self,
        _input_array: list[Vector3d] | Plane | tuple[Vector3d, ...],
        name: str | None = None,
    ):
        self.name = name or f"Unknown:{self.__class__.__name__}"

    def __array_finalize__(self, obj) -> None:

        super().__array_finalize__(obj)
        rows, *__cols = self.shape
        cols = 0 if len(__cols) == 0 else __cols[0]

        # On Mutations return to base if not 3x3
        if rows < 3 or cols != 3:
            self.tobase()
        else:
            a, b, c, *_rest = np.reshape(obj.tolist(), [-1, 3])

            _normal = Normal(a, b, c)

            # Plane is infinite
            vertices = 0 if self.__class__ == Plane else len(obj)

            # This attribute should be maintained!
            computed_attributes = {
                "normal": _normal.normal,
                "unit_normal": _normal.unit,
                "dot_product": np.dot(_normal.normal, a),
                "vertices": vertices,
                "raw_normal": _normal.raw_normal,
                "orientation": _normal.orientation,
                "__thickness": float(
                    os.environ["PLANE_THICKNESS_EPSILON"]
                ),
            }

            self.__dict__.update(
                computed_attributes, ab=b - a, bc=c - b, ca=a - c
            )  # another way to set attributes


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

    Args:
        input_array(list[Vector3d]): list of 3d vectors 2>vector_list>infinity.
        name(str): optional naming for debugging.
    """

    convex: bool
    solid: bool = True

    def __new__(
        cls,
        input_array: list[Vector3d] | Plane | tuple[Vector3d, ...],
        **_kwargs,
    ):
        length = len(input_array)
        if length < 3:
            raise TypeError(
                f"Expected vertices>3, but found {length}."
            )
        return super().__new__(cls, input_array)

    def __array_finalize__(self, obj) -> None:
        """Update dict with convex(bool) attribute"""

        super().__array_finalize__(obj)

        convex = is_convex(obj.tolist())

        self.__dict__.update(convex=convex)

    def __str__(self):
        rows, *__cols = self.shape
        cols = 0 if len(__cols) == 0 else __cols[0]

        match rows * 2 - cols:
            case Shape.QUADRILATERAL:
                _type = "Quadrilateral"

            case Shape.TRIANGLE:
                _type = "Triangle"
            case _:
                return np.ndarray.__str__(self)
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
    _matrix: list[Vector3d | list[int | float] | tuple[int | float]],
):
    """Matrix concavity."""
    match (len(_matrix)):
        # Triangles are always convex
        case 3:
            return True
        case 4:
            _a, _b, _c, _d = _matrix
            d_b = Line([_d, _b])
            a_b = Line([_a, _b])
            c_b = Line([_c, _b])

            bda = np.cross(d_b.difference, a_b.difference)
            bdc = np.cross(d_b.difference, c_b.difference)

            # Non-Convex
            if np.dot(bda, bdc) >= 0.0:
                return False
            # Convex
            else:
                c_a = Line([_c, _a])
                d_a = Line([_d, _a])
                b_a = Line([_b, _a])

                acd = np.cross(c_a.difference, d_a.difference)
                acb = np.cross(c_a.difference, b_a.difference)

                return np.dot(acd, acb) < 0.0

        case _:
            # Polygons with vertices < 3 = convex
            return True


def scalar_triple(u, v, w) -> int | float:
    """Scalar Triple Product

    - formula: u · (v · w)
    """

    idx = [0, 1, 2]

    product = 0
    for i in range(3):
        i, j, k = np.roll(idx, shift=-i)
        _i, _j, _k = np.roll(idx[::-1], shift=i)

        product += u[i] * v[j] * w[k] - u[_i] * v[_j] * w[_k]

    return product


def test_scalar_triple():
    """Test Scalar Triple Product"""

    u, v, w = [[3, 7, 10], [6, 10, 13], [6, 7, 7]]

    assert scalar_triple(u, v, w) == 9


class Intersection(NDarray):
    """Intersecting Vector of Plane & Segment

    Args:
        input_array(Polygon | Plane): Polygon/Plane to test segment against.
        line(Line): Edge to be tested.
    """

    def __new__(
        cls,
        input_array: Polygon | Plane,
        line: Line,
    ):
        # Number of vertices
        num_vertices = input_array.vertices

        p, q = line.tobase()

        pq = q - p

        # Revert child of ndarray to base
        # TODO - Sort Maybe?
        # Vertices
        a, b, c, *rest = input_array.tobase()
        d = len(rest) and rest[0]

        # pa, pb, pc, pd | None
        pa: NDarray = a - p
        pb: NDarray = b - p
        pc: NDarray = c - p
        pd: NDarray = None if not NDarray.arraylike(d) else d - p  # type: ignore[assignment]

        m = np.cross(pc, pq)

        # Mutable
        uvw = [-np.dot(pb, m), np.dot(pa, m), None]

        # Immutable
        u, v, w = uvw

        match num_vertices:
            case Shape.PLANE:
                intersect = cls.__plane(p, pq, input_array)

                return super().__new__(cls, intersect)
            case Shape.TRIANGLE:

                uvw[1] *= -1

            case Shape.QUADRILATERAL:

                if v < 0:

                    u = np.dot(pd, m)
                    uvw[0] = u

                    v = -v
                    uvw[1] = v

                    intersecting = cls.__dac(
                        uvw, *[pq, *[pa, pb, pc, pd]]
                    )
                    u, v, w = uvw

                    intersect = (
                        np.array((u * a) + (v * d) + (w * c))
                        if intersecting
                        else None
                    )

                    return super().__new__(cls, intersect)
                else:
                    u = -np.dot(pb, m)
                    uvw[0] = u

        intersecting = cls.__abc(uvw, *[pq, *[pa, pb, pc, pd]])

        u, v, w = uvw

        intersect = (
            np.array((u * a) + (v * b) + (w * c))
            if intersecting
            else None
        )
        return super().__new__(cls, intersect)

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
    def __plane(cls, p: NDarray, pq: NDarray, plane: Plane):

        t = (plane.dot_product - np.dot(plane.normal, p)) / np.dot(
            plane.normal, pq
        )

        if t >= 0 and t <= 1:

            return p + t * pq

        return None

    def __array_finalize__(self, obj) -> None:
        super().__array_finalize__(obj)

        computed_attributes = {
            "__thickness": float(
                os.environ["PLANE_THICKNESS_EPSILON"]
            ),
        }

        self.__dict__.update(
            computed_attributes
        )  # another way to set attributes
