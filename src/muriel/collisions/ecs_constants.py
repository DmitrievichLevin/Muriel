"""ECS Collision Constants"""

import os
from enum import IntEnum, StrEnum

# Maximum distance by which the intersection point can deviate (can be tweaked)
os.environ["PLANE_THICKNESS_EPSILON"] = "0.01"
# Blend factor for optimizing for balance or splits (should be tweaked)
os.environ["BLEND_FACTOR"] = "0.8"


# View Normal(Facing up)
class Orientation(StrEnum):
    """Orientation Unit Normals"""

    UP = "[0, 0, 1]"
    DOWN = "[0, 0, -1]"
    EAST = "[1, 0, 0]"
    EAST_UP = "[1, 0, 1]"
    EAST_DOWN = "[1, 0, -1]"
    WEST = "[-1, 0, 0]"
    WEST_UP = "[-1, 0, 1]"
    WEST_DOWN = "[-1, 0, -1]"
    NORTH = "[0, 1, 0]"
    NORTH_UP = "[0, 1, 1]"
    NORTH_DOWN = "[0, 1, -1]"
    NORTH_EAST = "[1, 1, 0]"
    NORTH_EAST_UP = "[1, 1, 1]"
    NORTH_EAST_DOWN = "[1, 1, -1]"
    NORTH_WEST = "[-1, 1, 0]"
    NORTH_WEST_UP = "[-1, 1, 1]"
    NORTH_WEST_DOWN = "[-1, 1, -1]"
    SOUTH = "[0, -1, 0]"
    SOUTH_UP = "[0, -1, 1]"
    SOUTH_DOWN = "[0, -1, -1]"
    SOUTH_EAST = "[1, -1, 0]"
    SOUTH_EAST_UP = "[1, -1, 1]"
    SOUTH_EAST_DOWN = "[1, -1, -1]"
    SOUTH_WEST = "[-1, -1, 0]"
    SOUTH_WEST_UP = "[-1, -1, 1]"
    SOUTH_WEST_DOWN = "[-1, -1, -1]"


class Shape(IntEnum):
    """Supported Polygon Types"""

    PLANE = 0
    QUADRILATERAL = 4
    TRIANGLE = 3


class PolygonClassification(StrEnum):
    """Polygon Classifications"""

    COPLANAR = "COPLANAR_WITH_PLANE"
    INFRONT = "INFRONT_OF_PLANE"
    BEHIND = "BEHIND_OF_PLANE"
    STRADDLING = "STRADDLING_PLANE"
    ON_PLANE = "POINT_ON_PLANE"
    PARALLEL = "LINE_PARALLEL_TO_PLANE"
    NON_INTERSECTING = "NON_PLANE_INTERSECTING_LINE"
    COLLISION = "COLLISION_DETECTED"
    NOT_COLLIDING = "COLLISION_NOT_FOUND"
    NO_PLANE = "NO_PLANE_CHECK_FRONT_AND_BACK"
