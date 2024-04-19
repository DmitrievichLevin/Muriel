"""ECS Binary Space Partitioning Tree Module"""

from __future__ import annotations
import numpy as np


from muriel import collisions


class BSPNode:
    """BSP Tree Node

    Raises:
        TypeError: can't find classification for polygon
    """

    _front: BSPNode | None = None
    _back: BSPNode | None = None
    depth: int
    _polygons: list[
        collisions.Quadrilateral
        | collisions.Triangle
        | collisions.Polygon
    ] = []

    def __init__(
        self,
        polygons: list[
            collisions.Quadrilateral
            | collisions.Triangle
            | collisions.Polygon
        ],
        depth: int = 0,
        max_depth=0,
        min_leaf_size=1,
    ):
        self.depth = depth
        if not polygons:
            self._front = None
            self._back = None

        size = len(polygons)

        # print(
        #     "\n depth reached \n",
        #     depth,
        #     max_depth,
        #     size,
        #     min_leaf_size,
        #     "\n",
        #     polygons,
        # )

        if (
            max_depth
            and depth >= max_depth
            or (size <= min_leaf_size)
        ):

            self._polygons = polygons

        else:
            front_list = []
            back_list = []
            _split = collisions.split_plane(polygons)

            for idx, poly in enumerate(polygons):

                cond = collisions.classify_polygon_to_plane(
                    poly, _split
                )
                match (cond):
                    case collisions.PolygonClassification.COPLANAR:

                        pass
                    case collisions.PolygonClassification.INFRONT:
                        print("\n infront \n", idx)
                        front_list.append(poly)

                    case collisions.PolygonClassification.BEHIND:
                        print("\n behind \n")
                        back_list.append(poly)

                    case collisions.PolygonClassification.STRADDLING:
                        print("\n straddling \n")
                        f_part, b_part = collisions.split_polygon(
                            poly, _split
                        )
                        front_list.append(f_part)
                        back_list.append(b_part)

                    case _:
                        raise TypeError(
                            "Expected literal[collisions.PolygonClassification], but found none."
                        )

            self._front = BSPNode(
                front_list,
                depth=depth + 1,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
            )
            self._behind = BSPNode(
                back_list,
                depth=depth + 1,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
            )


def test_polygon_split():
    A, B, C, D, E, F, G, H = (
        collisions.Vector3d([2, 0, -1]),
        collisions.Vector3d([5, 0, -1]),
        collisions.Vector3d([0, 5, -3]),
        collisions.Vector3d([5, 5, -4]),
        collisions.Vector3d([10, 5, 4]),
        collisions.Vector3d([-5, 0, -2]),
        collisions.Vector3d([0, -5, 2]),
        collisions.Vector3d([-2, 8, 3]),
    )

    triangles = (
        collisions.Triangle([A, B, C], name="ABC"),
        collisions.Triangle([C, B, D], name="CBD"),
        collisions.Triangle([B, E, D], name="BED"),
        collisions.Triangle([F, A, C], name="FAC"),
        collisions.Triangle([G, B, A], name="GBA"),
        collisions.Triangle([F, C, H], name="FCH"),
    )

    ABC, CBD, BED, FAC, GBA, FCH = triangles

    # Test Point On Plane ABC
    assert (
        collisions.PolygonClassification.ON_PLANE
        == collisions.classify_vector_to_plane(
            collisions.Vector3d([2, 2, -9 / 5]), ABC
        )
    )

    # Test Point BEHIND Plane ABC
    assert (
        collisions.PolygonClassification.BEHIND
        == collisions.classify_vector_to_plane(
            collisions.Vector3d([12, -8, 1]), ABC
        )
    )

    # Test Point INFRONT Plane ABC
    assert (
        collisions.PolygonClassification.INFRONT
        == collisions.classify_vector_to_plane(
            collisions.Vector3d([12, -8, 5]), ABC
        )
    )

    # COPLANER TEST
    test_split_plane = collisions.Triangle(
        [
            collisions.Vector3d([2, 2, -9 / 5]),
            collisions.Vector3d([12, -8, 1]),
            collisions.Vector3d([12, -8, 5]),
        ],
        name="TESTSPLIT",
    )
    assert (
        collisions.PolygonClassification.COPLANAR
        == collisions.classify_polygon_to_plane(ABC, test_split_plane)
    )

    # Test Picking Split Plane: BED Score == 5.2
    assert BED == collisions.split_plane(triangles)

    result = BSPNode(triangles, max_depth=4)


test_polygon_split()
