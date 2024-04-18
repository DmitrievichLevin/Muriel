"""ECS Collisions Module"""

import os
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
    PolygonClassification,
    split_plane,
    split_polygon,
    classify_polygon_to_plane,
    classify_vector_to_plane,
)


# TODO - Test that these are interchangable
_thickness, _blend = getattr(
    os.environ, "PLANE_THICKNESS_EPSILON", "0.01"
), getattr(os.environ, "BLEND_FACTOR", "0.8")
# Maximum distance by which the intersection point can deviate (can be tweaked)
os.environ["PLANE_THICKNESS_EPSILON"] = _thickness
# Blend factor for optimizing for balance or splits (should be tweaked)
os.environ["BLEND_FACTOR"] = _blend

# Environment Variable Dependent Imports
from .ecs_bsp import BSPNode  # pylint: disable=wrong-import-position
