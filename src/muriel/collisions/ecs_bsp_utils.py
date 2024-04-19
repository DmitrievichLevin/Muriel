"""ECS Binary Space Partitioning Tree Module"""

from __future__ import annotations
import os
import enum
import numpy as np
from .ecs_plane import (
    Line,
    Polygon,
    Quadrilateral,
    Triangle,
    Vector3d,
    Plane,
    Intersection,
)


class PolygonClassification(enum.StrEnum):
    """Polygon Classifications"""

    COPLANAR = "COPLANAR_WITH_PLANE"
    INFRONT = "INFRONT_OF_PLANE"
    BEHIND = "BEHIND_OF_PLANE"
    STRADDLING = "STRADDLING_PLANE"
    ON_PLANE = "POINT_ON_PLANE"
    PARALLEL = "LINE_PARALLEL_TO_PLANE"
    NON_INTERSECTING = "NON_PLANE_INTERSECTING_LINE"


def split_plane(
    _polygons: list[Quadrilateral | Triangle | Polygon],
) -> Quadrilateral | Triangle | Polygon:
    """Evaluate Dividing Plane."""
    # Tweak to optimize

    best_plane: Quadrilateral | Triangle | Polygon
    best_score: float = 0

    for poly in _polygons:

        _infront, _behind, _straddling = [0] * 3

        for test_poly in _polygons:
            # skip self
            if poly == test_poly:
                continue

            match (classify_polygon_to_plane(test_poly, poly)):
                case PolygonClassification.COPLANAR:
                    print(
                        f"{poly.name} coplanar/infront of {test_poly.name}"
                    )
                    # Coplanar polygons treated as being in front of plane
                    _infront += 1

                case PolygonClassification.INFRONT:
                    print(f"{poly.name} infront of {test_poly.name}")
                    _infront += 1

                case PolygonClassification.BEHIND:
                    print(f"{poly.name} behind {test_poly.name}")
                    _behind += 1

                case PolygonClassification.STRADDLING:
                    print(f"{poly.name} straddling {test_poly.name}")
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
        print(
            "\ntest score",
            test_score,
            poly.name,
            "\n",
            f"infront: {_infront} \n behind: {_behind} \n straddling: {_straddling} \n",
        )
        if test_score > best_score:
            best_score = test_score
            best_plane = poly
    print(
        "best plane: ",
        best_plane.name,
    )
    return best_plane


def classify_vector_to_plane(_vector: Vector3d, _plane: Plane):
    """Classify Vector to (Thick)Planes.

    - Based on signed distance.
    """
    # Compute signed distance of point from plane
    dist = np.dot(_plane.normal, _vector) - _plane.dot_product

    vector_norm = np.sqrt(sum(np.square(_plane.normal)))

    dist = dist / vector_norm

    _plane_thickness_epsilon_env_var = 0

    if dist > _plane_thickness_epsilon_env_var:
        return PolygonClassification.INFRONT
    if dist < _plane_thickness_epsilon_env_var:
        return PolygonClassification.BEHIND
    return PolygonClassification.ON_PLANE


def classify_polygon_to_plane(
    _polygon: Polygon, _split_plane: Polygon
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

    if not _infront and not _behind:
        return PolygonClassification.STRADDLING
    if not _infront:
        return PolygonClassification.INFRONT

    if not _behind:
        return PolygonClassification.BEHIND

    return PolygonClassification.COPLANAR


def split_polygon(
    _polygon: Polygon, _plane: Polygon
) -> tuple[Polygon, Polygon]:
    """Split Polygon by Plane

    Args:
        _polygon (Polygon): polygon to split.
        _plane (Polygon): polygon against which split is calculated.

    Returns:
        (tuple[Polygon, Polygon]): Polygon split in two.
    """
    front_vertices = np.array([])
    back_vertices = np.array([])

    _a = _polygon[-1]

    a_class = classify_vector_to_plane(_a, _plane)

    # Iterate edges
    for idx in range(1, _polygon.vertices - 1):
        _b = _polygon[idx]
        b_class = classify_vector_to_plane(_b, _plane)

        match (b_class):
            case PolygonClassification.INFRONT:

                # Straddle Condition: Add intersection to both sides
                if a_class is PolygonClassification.BEHIND:
                    intersect = Intersection(_plane, Line([_b, _a]))

                    if intersect:
                        front_vertices = np.append(
                            front_vertices, Vector3d(intersect)
                        )
                        back_vertices = np.append(
                            back_vertices, Vector3d(intersect)
                        )

                # Default Condition: Add b to front
                front_vertices = np.append(
                    front_vertices, Vector3d(_b)
                )
                break
            case PolygonClassification.BEHIND:

                # Straddle Condition: Add intersection to both sides
                if a_class is PolygonClassification.INFRONT:
                    intersect = Intersection(_plane, Line([_a, _b]))

                    if intersect:
                        front_vertices = np.append(
                            front_vertices, Vector3d(intersect)
                        )
                        back_vertices = np.append(
                            back_vertices, Vector3d(intersect)
                        )
                # A On Plane Condition: Add a to back
                elif a_class is PolygonClassification.ON_PLANE:
                    back_vertices = np.append(
                        back_vertices, Vector3d(_a)
                    )

                # Default Condition: Add b to back
                back_vertices = np.append(back_vertices, Vector3d(_b))

            case _:
                # B On Plane Condition: Add b to front
                front_vertices = np.append(
                    front_vertices, Vector3d(_b)
                )

                # A Behind Plane: Add b to back
                if a_class is PolygonClassification.BEHIND:
                    back_vertices = np.append(
                        back_vertices, Vector3d(_b)
                    )

        # Point to b as starting point for next edge
        _a = _b
    # TODO - HERE resulting vertices do not make up Vector3s
    print("front and back", front_vertices, back_vertices)
    front_poly = Polygon(front_vertices)
    back_poly = Polygon(back_vertices)

    return front_poly, back_poly
