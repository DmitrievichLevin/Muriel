"""ECS Binary Space Partitioning Tree Module"""

# pylint: disable=line-too-long
from __future__ import annotations
import os
import numpy as np
from numpy.typing import NDArray
from .ecs_plane import (
    Line,
    Polygon,
    Quadrilateral,
    Triangle,
    Vector3d,
    Plane,
    Normal,
    Intersection,
)

from .ecs_constants import PolygonClassification


def split_plane(
    _polygons: list[Quadrilateral | Triangle | Polygon],
) -> tuple[
    Plane,
    Quadrilateral | Triangle | Polygon,
    list[Quadrilateral | Triangle | Polygon],
]:
    """Evaluate Dividing Plane.

    Args:
        _polygons(list[Quadrilateral | Triangle | Polygon]): polygons to choose from

    Returns:
        splitting_plane(Plane): Selected Polygon evaluated as a Plane.
        original(Quadrilateral | Triangle | Polygon): Selected Polygon.
        remaining(list[Quadrilateral | Triangle | Polygon]): Polygons not selected.
    """
    # Tweak to optimize

    best_polygon: Quadrilateral | Triangle | Polygon
    best_score: float = 0

    # Remaining polygons
    remaining = []

    for poly in _polygons:
        _remaining: list[Quadrilateral | Triangle | Polygon] = []
        _infront, _behind, _straddling = [0] * 3

        for test_poly in _polygons:

            # skip self
            if poly == test_poly:
                continue

            _remaining.append(test_poly)

            match (classify_polygon_to_plane(test_poly, poly)):
                case PolygonClassification.COPLANAR:
                    # print(
                    #     f"{poly.name} coplanar/infront of {test_poly.name}"
                    # )
                    # Coplanar polygons treated as being in front of plane
                    _infront += 1

                case PolygonClassification.INFRONT:
                    # print(f"{poly.name} infront of {test_poly.name}")
                    _infront += 1

                case PolygonClassification.BEHIND:
                    # print(f"{poly.name} behind {test_poly.name}")
                    _behind += 1

                case PolygonClassification.STRADDLING:
                    # print(f"{poly.name} straddling {test_poly.name}")
                    _straddling += 1

                case _:
                    raise TypeError(
                        "Expected literal[PolygonClassification], but found none."
                    )

        _blend_factor_env_var = float(os.environ["BLEND_FACTOR"])
        test_score = (
            _blend_factor_env_var * _straddling
            + (1.0 - _blend_factor_env_var)
            + abs(_infront - _behind)
        )

        if test_score > best_score:
            best_score = test_score
            best_polygon = poly
            remaining = _remaining

    _split = Plane(best_polygon.tobase(), name=best_polygon.name)

    return (
        _split,
        best_polygon,
        remaining,
    )


def normalize_vector(vector: Vector3d | list[float | int]):
    """Normalize Vector

    Args:
        vector(Vector3d | list[float | int]) 3D vector to be normalized.

    Returns:
        ||v|| (Vector3d | list[float | int]) where v is vector.
    """
    return np.sqrt(sum(np.square(vector)))


def classify_vector_to_plane(_vector: Vector3d, _plane: Plane):
    """Classify Vector to (Thick)Planes.

    - Based on signed distance.
    """
    # Compute signed distance of point from plane
    dist = np.dot(_plane.normal, _vector) - _plane.dot_product

    vector_norm = normalize_vector(_plane.normal)

    dist = dist / vector_norm

    _plane_thickness_epsilon_env_var = float(
        os.environ["PLANE_THICKNESS_EPSILON"]
    )

    if dist > _plane_thickness_epsilon_env_var:
        return PolygonClassification.INFRONT
    if dist < -_plane_thickness_epsilon_env_var:
        return PolygonClassification.BEHIND
    return PolygonClassification.ON_PLANE


def classify_polygon_to_plane(
    _polygon: Polygon, _split_plane: Plane
) -> PolygonClassification:
    """Classify Polygon

    - Infront
    - Behind
    - Straddling
    """
    _infront, _behind = [0] * 2

    for idx in range(_polygon.vertices):
        match (classify_vector_to_plane(_polygon[idx], _split_plane)):
            case PolygonClassification.INFRONT:
                _infront += 1

            case PolygonClassification.BEHIND:
                _behind += 1

    if _infront and _behind:
        return PolygonClassification.STRADDLING
    if not _behind and _infront:
        return PolygonClassification.INFRONT

    if not _infront and _behind:
        return PolygonClassification.BEHIND

    return PolygonClassification.COPLANAR


def split_polygon(
    _polygon: Polygon, _plane: Plane
) -> tuple[Polygon, Polygon]:
    """Split Polygon by Plane

    Args:
        _polygon (Polygon): polygon to split.
        _plane (Polygon): polygon against which split is calculated.

    Returns:
        (tuple[Polygon, Polygon]): Polygon split in two.
    """
    front_vertices: list[Vector3d] = []
    back_vertices: list[Vector3d] = []

    _a = _polygon[-1]
    a_class = classify_vector_to_plane(_a, _plane)

    # Iterate edges
    for idx in range(_polygon.vertices):

        _b = _polygon[idx]

        b_class = classify_vector_to_plane(_b, _plane)

        match (b_class):
            case PolygonClassification.INFRONT:

                # Straddle Condition: Add intersection to both sides
                if (
                    a_class is PolygonClassification.BEHIND
                    or a_class is PolygonClassification.ON_PLANE
                ):
                    intersect = Intersection(_plane, Line([_b, _a]))

                    if Intersection.arraylike(intersect):
                        front_vertices.append(Vector3d(intersect))
                        back_vertices.append(Vector3d(intersect))

                # Default Condition: Add b to front
                front_vertices.append(Vector3d(_b))

            case PolygonClassification.BEHIND:

                # Straddle Condition: Add intersection to both sides
                if a_class is PolygonClassification.INFRONT:
                    intersect = Intersection(_plane, Line([_a, _b]))

                    if Intersection.arraylike(intersect):
                        front_vertices.append(Vector3d(intersect))
                        back_vertices.append(Vector3d(intersect))
                # A On Plane Condition: Add a to back
                elif a_class is PolygonClassification.ON_PLANE:
                    back_vertices.append(Vector3d(_a))

                # Default Condition: Add b to back
                back_vertices.append(Vector3d(_b))

            case _:
                # B On Plane Condition: Add b to front
                front_vertices.append(Vector3d(_b))

                # A Behind Plane: Add b to back
                if a_class is PolygonClassification.BEHIND:
                    back_vertices.append(Vector3d(_b))

        # Point to b as starting point for next edge
        _a = _b
        a_class = b_class

    front_poly = Polygon(front_vertices, name=f"{_polygon.name}_F")
    back_poly = Polygon(back_vertices, name=f"{_polygon.name}_B")

    return front_poly, back_poly


def point_bounded_by_triangle(
    point: Vector3d | NDArray,
    polygon: Polygon | Triangle,
) -> bool:
    """Determines if point is within the bounds of the Triangle

    Args:
        point (Vector3d | NDArray): 3D Vector.
        polygon (Polygon | Triangle): Polygon with three vertices or Triangle

    Returns:
        (bool): Whether point is within Polygon Segments.
    """
    (
        a,
        b,
        c,
    ) = polygon

    d = polygon.raw_normal

    u = np.around(np.cross((b - a), (point - a)), decimals=4)

    if np.dot(u, d) < 0:
        return False

    v = np.cross(c - b, point - b)

    if np.dot(v, d) < 0:

        return False

    w = np.cross(a - c, point - c)

    if np.dot(w, d) < 0:
        return False

    return True


def point_bounded_by_polygon(
    point: Vector3d | NDArray,
    polygon: Polygon | Triangle | Quadrilateral,
) -> bool:
    """Determines if point is within the bounds of the Polgon

    Args:
        point (Vector3d | NDArray): 3D Vector.
        polygon (Polygon | Triangle | Quadrilateral): Polygon

    Returns:
        (bool): Whether point is within Polygon Segments.
    """
    low = 0
    n = polygon.vertices

    if n == 3:
        return point_bounded_by_triangle(point, polygon)

    high = n

    while low + 1 < high:
        mid = int((low + high) / 2)

        slice_normal = Normal.get_normal(
            polygon[0], polygon[mid], point
        )

        to_right_of = np.dot(slice_normal, polygon.raw_normal)

        if to_right_of > 0:
            low = mid
        else:
            high = mid

    if low == 0 or high == n:
        return False

    slice_normal = Normal.get_normal(
        polygon[low], polygon[high], point
    )

    to_right_of = np.dot(slice_normal, polygon.raw_normal)

    return to_right_of > 0
