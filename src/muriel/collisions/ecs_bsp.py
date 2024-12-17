"""ECS Binary Space Partitioning Tree Module"""

from __future__ import annotations


from .ecs_plane import (
    Polygon,
    Quadrilateral,
    Triangle,
    Point,
    Plane,
)
from .ecs_bsp_utils import (
    split_plane,
    split_polygon,
    classify_polygon_to_plane,
    classify_vector_to_plane,
    point_bounded_by_polygon,
)
from .ecs_constants import PolygonClassification


class BSPNode:
    """BSP Tree Node

    Raises:
        TypeError: can't find classification for polygon
    """

    _front: BSPNode | None = None
    _behind: BSPNode | None = None
    depth: int
    polygons: list[Quadrilateral | Triangle | Polygon] = []
    polygon: Quadrilateral | Triangle | Polygon
    plane: Plane
    leaf: bool = False

    def __init__(
        self,
        polygons: list[Quadrilateral | Triangle | Polygon],
        depth: int = 0,
        parent: BSPNode | None = None,
    ):

        self.parent = parent

        self.depth = depth

        size = len(polygons)

        if size == 1:
            # Leaf Nodes Have no plane
            self.leaf = True
            self.polygon = polygons[0]

        else:
            _split = split_plane(polygons)
            print(
                "one should be splitting",
                [v.supporting for v in polygons],
            )
            front_list = []
            back_list = []
            # TODO - EDGE two polygons left perpendicular to eachother w/ same classification need to separate the two(front,back) and be able to query against the parent
            # Set Node Plane + Polygon
            self.plane = _split
            # self.polygon = selected_polygon

            looping = 0
            for poly in polygons:
                looping += 1
                cond = classify_polygon_to_plane(poly, _split)

                match (cond):
                    case PolygonClassification.COPLANAR:

                        front_list.append(poly)
                    case PolygonClassification.INFRONT:

                        front_list.append(poly)

                    case PolygonClassification.BEHIND:

                        back_list.append(poly)

                    case PolygonClassification.STRADDLING:

                        f_part, b_part = split_polygon(poly, _split)
                        front_list.append(f_part)
                        back_list.append(b_part)

                    case _:
                        raise TypeError(
                            "Expected literal[PolygonClassification], but found none."
                        )
            next_depth = depth + 1

            # Edge Case: Split Plane did not partition polygons of length: 2
            # Manually separate on front & back
            # Set plane to None
            no_partition = (
                front_list
                if (len(front_list) == 2 and len(back_list) == 0)
                else (
                    back_list
                    if (len(back_list) == 2 and len(front_list) == 0)
                    else None
                )
            )
            if no_partition is not None:
                self.plane = None  # type: ignore[reportAttributeAccessIssue]

                (  # pylint: disable=unbalanced-tuple-unpacking
                    _one_of_two,
                    _two_of_two,
                ) = no_partition

                front_list = [_one_of_two]
                back_list = [_two_of_two]

            if front_list:
                self._front = BSPNode(
                    front_list,
                    depth=next_depth,
                    parent=self,
                )

            if back_list:
                self._behind = BSPNode(
                    back_list,
                    depth=next_depth,
                    parent=self,
                )

        # Visual Tracking
        self.__set_key(
            getattr(self, "plane", None),
            getattr(self, "polygon", None),
        )

    def __set_key(self, plane: Plane | None, polygon: Polygon | None):
        """Set Node Key(for Debugging)

        Args:
            plane (Plane | None): if not leaf key == plane.name or if no plane default: 'No_Plane'.
            polygon (Polygon | None): if leaf ket == polygon.name.
        """
        if polygon is not None:
            self.key = self.polygon.name

        elif plane is not None:
            self.key = self.plane.name
        else:
            self.key = "No_P"

    def display(self) -> None:
        """Display BSP Tree in Terminal"""
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    # pylint: disable=protected-access

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""

        # No child.
        if self._front is None and self._behind is None:
            line = self.key
            width = len(line)
            height = 1
            middle = width // 2
            return (
                [line],
                width,
                height,
                middle,
            )

        # Only left child.
        if self._front is None:
            lines, n, p, x = self._behind._display_aux()  # type: ignore[reportOptionalMemberAccess]
            s = self.key
            u = len(s)
            first_line = (x + 1) * " " + (n - x - 1) * "_" + s
            second_line = x * " " + "/" + (n - x - 1 + u) * " "
            shifted_lines = [line + u * " " for line in lines]
            return (
                [first_line, second_line] + shifted_lines,
                n + u,
                p + 2,
                n + u // 2,
            )

        # Only right child.
        if self._behind is None:
            lines, n, p, x = self._front._display_aux()
            s = self.key
            u = len(s)
            first_line = s + x * "_" + (n - x) * " "
            second_line = (u + x) * " " + "\\" + (n - x - 1) * " "
            shifted_lines = [" " * u + line for line in lines]
            return (
                [first_line, second_line] + shifted_lines,
                n + u,
                p + 2,
                u // 2,
            )

        # Two children.
        left, n, p, x = self._behind._display_aux()
        right, m, q, y = self._front._display_aux()
        s = self.key
        u = len(s)

        first_line = "".join(
            [" "] * (x + 1)
            + ["_"] * (n - x - 1)
            + [s]
            + ["_"] * y
            + [" "] * (m - y)
        )

        second_line = "".join(
            [" "] * x
            + ["/"]
            + [" "] * (n - x - 1 + u + y)
            + ["\\"]
            + [" "] * (m - y - 1)
        )
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = list(zip(left, right))

        lines = [first_line, second_line] + [
            a + " " * u + b for a, b in zipped_lines
        ]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    @property
    def info(self):
        return f"key: {self.key} \nleaf:{self.leaf}"

    @classmethod
    def point_collision(cls, node: BSPNode, point: Point):
        """Determine Whether A Given Point Collides With a Node

        Args:
            node(BSPNode): Node of BSPTree containing plane/polygon to test point against.
            point(Point): Point to test against tree.

        Returns:
            (PolygonClassification[Collision | Not_Colliding]): result of test.
        """

        while not node.leaf:
            plane = node.plane

            point_class = classify_vector_to_plane(point, plane)
            match point_class:
                case PolygonClassification.INFRONT:
                    if not node._front:
                        break
                    node = node._front
                case PolygonClassification.BEHIND:
                    if not node._behind:
                        break
                    node = node._behind
                case PolygonClassification.ON_PLANE:

                    front_check, behind_check = None, None

                    if node._front:
                        front_check = cls.point_collision(
                            node._front, point
                        )
                    if node._behind:
                        behind_check = cls.point_collision(
                            node._behind, point
                        )
                    return (
                        PolygonClassification.COLLISION
                        if any(
                            [
                                PolygonClassification.COLLISION
                                == check
                                for check in [
                                    front_check,
                                    behind_check,
                                ]
                            ]
                        )
                        else PolygonClassification.NOT_COLLIDING
                    )

                case PolygonClassification.NO_PLANE:
                    # pylint: disable=pointless-string-statement
                    """Edge Case:
                    - Two Polygons
                    - Unable to auto/manual partition
                    - Split between front & back w/ no plane
                    - Note: No Plane Nodes Should Always have front/back nodes
                    """
                    front = cls.point_collision(node._front, point)  # type: ignore[reportArgumentType]
                    behind = cls.point_collision(node._behind, point)  # type: ignore[reportArgumentType]

                    return (
                        front
                        if behind
                        == PolygonClassification.NOT_COLLIDING
                        else behind
                    )
        if not node.leaf:
            return PolygonClassification.NOT_COLLIDING
        else:
            bounded = point_bounded_by_polygon(point, node.polygon)

            if bounded and node.polygon.solid:
                return PolygonClassification.COLLISION

    @classmethod
    def polygon_collision(
        cls,
        node: BSPNode,
        polygon: Polygon | Quadrilateral | Triangle,
    ) -> Polygon | Quadrilateral | Triangle | None:
        """Determine Whether A Polygon Collides With a Node

        Args:
            node(BSPNode): Node of BSPTree containing plane/polygon to test point against.
            polygon(Polygon | Quadrilateral | Triangle): Polygon to test against tree.

        Returns:
            (Polygon | Quadrilateral | Triangle | None): None or colliding polygon.
        """

        while not node.leaf:

            plane = node.plane

            poly_class = classify_polygon_to_plane(polygon, plane)
            print(
                "polygon collision class",
                poly_class,
                node.leaf,
                node.key,
            )
            match poly_class:
                case PolygonClassification.INFRONT:
                    if not node._front:
                        break
                    node = node._front
                    continue
                case PolygonClassification.BEHIND:
                    if not node._behind:
                        break
                    node = node._behind
                    continue
                case PolygonClassification.STRADDLING:
                    # Split + Chack Front & Back
                    front_check, behind_check = None, None
                    _front, _back = split_polygon(polygon, plane)

                    if node._front:
                        front_check = cls.polygon_collision(
                            node._front, _front
                        )
                    if node._behind:
                        behind_check = cls.polygon_collision(
                            node._behind, _back
                        )

                    # Return Collision Result from split check
                    if any(
                        check is not None
                        for check in [front_check, behind_check]
                    ):
                        return (
                            front_check
                            if front_check is not None
                            else behind_check
                        )
                case PolygonClassification.NO_PLANE:
                    # No plane = Check front & back
                    # Note: Should always have front & back node if no plane

                    front_check = cls.polygon_collision(
                        node._front, polygon  # type: ignore[arguementType]
                    )

                    behind_check = cls.polygon_collision(
                        node._behind, polygon  # type: ignore[arguementType]
                    )

                    # Return Collision Result from split check
                    if any(
                        check is not None
                        for check in [front_check, behind_check]
                    ):
                        return (
                            front_check
                            if front_check is not None
                            else behind_check
                        )

        # No Collision Found
        return None if not node.leaf else node.polygon


def correct_orientation():
    A, B, C, D, E, F, G, H, I, J, K, L = (
        Point(2, 0, -1),
        Point(5, 0, -1),
        Point(0, 5, -3),
        Point(5, 5, -4),
        Point(10, 5, 4),
        Point(-5, 0, -2),
        Point(0, -5, 2),
        Point(-2, 8, 3),
        Point(2, 2, -9 / 5),
        Point(12, -8, 1),
        Point(12, -8, 5),
        Point(12.0, -8.0, 2.2),
    )
    triangles = [
        Triangle(A, B, C, name="ABC"),
        Triangle(C, B, D, name="CBD"),
        Triangle(B, E, D, name="BED"),
        Triangle(F, A, C, name="FAC"),
        Triangle(G, B, A, name="GBA"),
        Triangle(F, C, H, name="FCH"),
        Triangle(I, J, K, name="IJK"),
    ]

    tree = BSPNode(triangles)
    # print("fisrt node check", tree.polygon)
    AABB_bottom = Quadrilateral(
        Point(1, 1, -2),
        Point(3, 1, -2),
        Point(3, 3, -2),
        Point(1, 3, -2),
    )

    tree.display()
    print("collision", BSPNode.polygon_collision(tree, AABB_bottom))


# correct_orientation()
