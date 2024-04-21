"""ECS Collisions Module"""

import os

from .ecs_constants import Orientation, PolygonClassification, Shape
from .ecs_plane import (
    Normal,
    Plane,
    Vector3d,
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
)

from .ecs_bsp import BSPNode
