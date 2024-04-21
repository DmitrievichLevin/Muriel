"""Muriel."""

import os
from logging import handlers, basicConfig, DEBUG

# Maximum distance by which the intersection point can deviate (can be tweaked)
os.environ["PLANE_THICKNESS_EPSILON"] = "0.01"
# Blend factor for optimizing for balance or splits (should be tweaked)
os.environ["BLEND_FACTOR"] = "0.8"

from muriel.ecs_component import Buffer
from muriel.ecs_events import Observable
from muriel.ecs_events import Observer

from .ecs_attribute import Attribute
from .ecs_compartment import Compartment, ComponentList, Component
from .ecs_builtin import Direction
from .ecs_builtin import Movement
from .ecs_builtin import Position
from .ecs_builtin import Velocity
from .ecs_entity import Entity
from .collisions import (
    Normal,
    Plane,
    Vector3d,
    Line,
    Polygon,
    Quadrilateral,
    Triangle,
    PolygonClassification,
    Intersection,
    split_plane,
    split_polygon,
    classify_polygon_to_plane,
    classify_vector_to_plane,
    BSPNode,
)


basicConfig(
    level=DEBUG,
    handlers=[
        handlers.RotatingFileHandler(
            "tests/debug.log",
        ),
    ],
    format=" \n%(asctime)s \nline %(lineno)s in, %(module)s \nfunction: %(funcName)s() \nmessage: %(message)s",
    force=True,
)


__all__ = (
    "Attribute",
    "Movement",
    "Velocity",
    "Position",
    "Direction",
    "Component",
    "Compartment",
    "ComponentList",
    "Buffer",
    "Observable",
    "Observer",
    "Entity",
    "Normal",
    "Plane",
    "Vector3d",
    "Line",
    "Polygon",
    "Quadrilateral",
    "Triangle",
    "PolygonClassification",
    "Intersection",
    "split_plane",
    "split_polygon",
    "classify_polygon_to_plane",
    "classify_vector_to_plane",
)
