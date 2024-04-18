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
    ]

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
                print("poly loop", idx, size, depth, poly.tobase())
                # print(
                #     "classif",
                #     collisions.classify_polygon_to_plane(
                #         poly, _split
                #     ),
                #     _split.tobase(),
                #     poly.tobase(),
                # )
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
    a = collisions.Vector3d([0, 0, 0])
    b = collisions.Vector3d([5, 0, 0])
    c = collisions.Vector3d([0, 5, 0])

    a2 = collisions.Vector3d([0, 5, 0])
    b2 = collisions.Vector3d([5, 0, 0])
    c2 = collisions.Vector3d([5, 5, 0])

    a3 = collisions.Vector3d([5, 0, 0])
    b3 = collisions.Vector3d([10, 5, 0])
    c3 = collisions.Vector3d([5, 5, 0])

    tri = collisions.Triangle([a, b, c])

    tri2 = collisions.Triangle([a2, b2, c2])

    tri3 = collisions.Triangle([a3, b3, c3])

    result = BSPNode([tri, tri2, tri3], max_depth=4)

    front = result._front
    # print(
    #     "\n front \n",
    #     front.__dict__,
    #     "\n back",
    #     result._behind._polygons,
    # )

    # while result:
    #     pass


test_polygon_split()
