"""ECS Binary Space Partitioning Tree Module"""

# pylint: disable=line-too-long
from __future__ import annotations
from math import sqrt
import os
import numpy as np
from .ecs_plane import (
    Line,
    Polygon,
    Quadrilateral,
    Triangle,
    Point,
    Plane,
    Normal,
    Intersection,
)

from .ecs_constants import PolygonClassification


def geo_print(p):
    h = []
    for v in p:
        a, b, c = v
        h.append("Point({%d,%d,%d})" % (a, b, c))
    return f"Polygon({','.join(h)})"


def point_plane_dist(point: Point, plane: Plane | Polygon):
    """Distance of Point from Plane

    Args:
        point (Point): Point
        plane (Plane | Polygon): Plane

    Returns:
        dist (float | int): Distance between Point and Plane in Euclidean space.
    """
    x, y, z = plane.normal
    m1, m2, m3 = point

    top = (m1 * x) + (m2 * y) + (m3 * z) - plane.dot_product
    # print(
    #     "top",
    #     np.dot(plane.normal, point) - plane.dot_product,
    #     normalize_vector(x, y, z),
    #     np.divide(top, normalize_vector(x, y, z)),
    # )
    return np.divide(top, normalize_vector(x, y, z))


def closest_poly_segment(
    point: Point, polygon: Polygon
) -> tuple[Point, Point]:
    """Segment Nearest to Point

    Args:
        point (Point): point of reference
        polygon (Polygon | Point): polygon to compare point against.

    Returns:
        tuple[Point, Point]: Two points representing segment nearest to point.
    """
    # pylint: disable=unnecessary-direct-lambda-call
    i_a, i_b, *_rest = np.argsort(
        [(lambda x: sum(np.abs(x - point)))(p) for p in polygon]
    )

    a, b = polygon[i_a], polygon[i_b]

    return a, b


def manual_perpendicular_split_plane(
    polygons: list[Polygon],
) -> Plane:
    """Manualy Generate optimal splitting plane

    - (1) Find point average from polygon vectors. (2) Find Polygon->Plane closest to point. (3) Find Polygon->Line closest to point. (4) Create Plane from Line on Z-axis.

    Args:
        polygons (list[Polygon]): Polygons to test agains

    Returns:
        plane(Plane): Optimal splitting plane projected on z-axis
    """
    poly_avg = np.average(np.concatenate(polygons), axis=0)

    dists = []
    for p in polygons:
        # un-mark supporting planes
        p.supporting = False
        dists.append(point_plane_dist(poly_avg, p))

    closest_poly = polygons[np.argsort(dists)[0]]

    a, b = closest_poly_segment(poly_avg, closest_poly)

    z_normal = [0, 0, 1]

    optimal_z_plane = Plane(
        a + z_normal,
        b,
        a,
        name=f"{closest_poly.name}_mZ",
    )
    print("average", geo_print(optimal_z_plane))

    return optimal_z_plane


def split_plane(
    _polygons: list[Quadrilateral | Triangle | Polygon],
) -> Plane:
    """Evaluate Dividing Plane.

    Args:
        _polygons(list[Quadrilateral | Triangle | Polygon]): polygons to choose from

    Returns:
        splitting_plane(Plane): Auto-partition plane or manual split(see 'manual_perpendicular_split_plane').
    """

    best_polygon: Quadrilateral | Triangle | Polygon | None = None
    best_score: float = 0

    for poly in _polygons:
        _infront, _behind, _straddling = [0] * 3

        # Check if polygon has already been used in auto-partitioning.
        if poly.supporting:
            continue

        for test_poly in _polygons:

            # skip self
            if poly == test_poly:
                continue

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

    if best_polygon is None:
        print("all supporting")
        manual_split = manual_perpendicular_split_plane(_polygons)

        return manual_split

    print(
        "check best poly",
        [f"{geo_print(p)}, {p.name}" for p in _polygons],
        best_polygon.supporting,
        best_score,
    )
    _split = Plane(*best_polygon, name=best_polygon.name)

    # Flag to prevent reselection during auto-partitioning.
    best_polygon.supporting = True

    return _split


def normalize_vector(x: int | float, y: int | float, z: int | float):
    """Normalize Vector

    Args:
        x(int | float) x value.
        y(int | float) y value.
        z(int | float) z value.

    Returns:
        ||v|| (Point | list[float | int]) where v is vector.
    """

    return sqrt(x**2 + y**2 + z**2)


def classify_vector_to_plane(
    _vector: Point, _plane: Plane | None = None
):
    """Classify Vector to (Thick)Planes.

    - Based on signed distance.
    """
    if _plane is None:
        return PolygonClassification.NO_PLANE

    # Compute signed distance of point from plane
    dist = np.dot(_plane.normal, _vector) - _plane.dot_product

    x, y, z = _plane.normal

    vector_norm = normalize_vector(x, y, z)

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
    _polygon: Polygon, _split_plane: Plane | None
) -> PolygonClassification:
    """Classify Polygon

    - Infront
    - Behind
    - Straddling
    - No Plane
    """
    if _split_plane is None:
        return PolygonClassification.NO_PLANE

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
    front_vertices: list[Point] = []
    back_vertices: list[Point] = []

    _a = _polygon[-1]
    a_class = classify_vector_to_plane(_a, _plane)

    # Iterate edges
    for idx in range(_polygon.vertices):

        _b = _polygon[idx]

        b_class = classify_vector_to_plane(_b, _plane)

        match (b_class):
            case PolygonClassification.INFRONT:

                # Straddle Condition: Add intersection to both sides
                if a_class is PolygonClassification.BEHIND:
                    intersect = Intersection(_plane, Line(_b, _a))

                    if not intersect.empty:
                        front_vertices.append(intersect)
                        back_vertices.append(intersect)

                # Default Condition: Add b to front
                front_vertices.append(_b)

            case PolygonClassification.BEHIND:

                # Straddle Condition: Add intersection to both sides
                if a_class is PolygonClassification.INFRONT:
                    intersect = Intersection(_plane, Line(_a, _b))

                    if not intersect.empty:
                        front_vertices.append(intersect)
                        back_vertices.append(intersect)
                # A On Plane Condition: Add a to back
                elif a_class is PolygonClassification.ON_PLANE:
                    back_vertices.append(_a)

                # Default Condition: Add b to back
                back_vertices.append(_b)

            case _:
                # B On Plane Condition: Add b to front
                front_vertices.append(_b)

                # A Behind Plane: Add b to back
                if a_class is PolygonClassification.BEHIND:
                    back_vertices.append(_b)

        # Point to b as starting point for next edge
        _a = _b
        a_class = b_class

    front_poly = Polygon(*front_vertices, name=_polygon.name)
    back_poly = Polygon(*back_vertices, name=_polygon.name)

    return front_poly, back_poly


def point_bounded_by_triangle(
    point: Point,
    polygon: Polygon | Triangle,
) -> bool:
    """Determines if point is within the bounds of the Triangle

    Args:
        point (Point): 3D Vector.
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
    point: Point,
    polygon: Polygon | Triangle | Quadrilateral,
) -> bool:
    """Determines if point is within the bounds of the Polgon

    Args:
        point (Point): 3D Vector.
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

        slice_normal = Normal(polygon[0], polygon[mid], point).normal

        to_right_of = np.dot(slice_normal, polygon.raw_normal)

        if to_right_of > 0:
            low = mid
        else:
            high = mid

    if low == 0 or high == n:
        return False

    slice_normal = Normal(polygon[low], polygon[high], point).normal

    to_right_of = np.dot(slice_normal, polygon.raw_normal)

    return to_right_of > 0
