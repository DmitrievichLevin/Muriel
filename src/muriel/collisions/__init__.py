"""ECS Collisions Module"""

import os

from .ecs_constants import Orientation, PolygonClassification, Shape
from .ecs_plane import (
    Normal,
    Plane,
    Point,
    Matrix,
    Line,
    Polygon,
    Quadrilateral,
    Triangle,
    Intersection,
)
from .ecs_bsp_utils import (
    split_plane,
    split_polygon,
    classify_polygon_to_plane,
    classify_vector_to_plane,
    point_bounded_by_polygon,
    normalize_vector,
    point_plane_dist,
    manual_perpendicular_split_plane,
)

from .ecs_bsp import BSPNode

from .ecs_boundingbox import AABB
