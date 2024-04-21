"""ECS Binary Space Partitioning Tree Module"""

from __future__ import annotations


from .ecs_plane import (
    Polygon,
    Quadrilateral,
    Triangle,
    Vector3d,
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
            front_list = []
            back_list = []
            _split, selected_polygon, remainingpolygons = split_plane(
                polygons
            )

            # Set Node Plane + Polygon
            self.plane = _split
            self.polygon = selected_polygon
            looping = 0
            for poly in remainingpolygons:
                looping += 1
                cond = classify_polygon_to_plane(poly, _split)

                match (cond):
                    case PolygonClassification.COPLANAR:

                        pass
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
        self.key = self.polygon.name

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

    @classmethod
    def point_collision(cls, node: BSPNode | None, point: Vector3d):
        """Determine Whether A Given Point Collides With a Node

        Args:
            node(BSPNode): Node of BSPTree containing plane/polygon to test point against.
            point(Vector3d): Point to test against tree.

        Returns:
            (PolygonClassification[Collision | Not_Colliding]): result of test.
        """

        while node:
            if not node.leaf:
                plane = node.plane
            else:
                # Leaf Node = check against polygon
                plane = node.polygon

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

                    bounded = point_bounded_by_polygon(
                        point, node.polygon
                    )

                    if bounded and node.polygon.solid:
                        return PolygonClassification.COLLISION

                    front = cls.point_collision(node._front, point)
                    behind = cls.point_collision(node._behind, point)
                    return (
                        front
                        if front == behind
                        else PolygonClassification.COLLISION
                    )

        return PolygonClassification.NOT_COLLIDING


def correct_orientation():
    A, B, C, D, E, F, G, H, I, J, K, L = (
        Vector3d([2, 0, -1]),
        Vector3d([5, 0, -1]),
        Vector3d([0, 5, -3]),
        Vector3d([5, 5, -4]),
        Vector3d([10, 5, 4]),
        Vector3d([-5, 0, -2]),
        Vector3d([0, -5, 2]),
        Vector3d([-2, 8, 3]),
        Vector3d([2, 2, -9 / 5]),
        Vector3d([12, -8, 1]),
        Vector3d([12, -8, 5]),
        Vector3d([12.0, -8.0, 2.2]),
    )
    triangles = [
        Triangle([A, B, C], name="ABC"),
        Triangle([C, B, D], name="CBD"),
        Triangle([B, E, D], name="BED"),
        Triangle([F, A, C], name="FAC"),
        Triangle([G, B, A], name="GBA"),
        Triangle([F, C, H], name="FCH"),
        Triangle([I, J, K], name="IJK"),
    ]

    tree = BSPNode(triangles)

    tree.display()


correct_orientation()
