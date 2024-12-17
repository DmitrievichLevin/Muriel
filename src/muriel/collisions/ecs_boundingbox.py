from __future__ import annotations
from .ecs_plane import Point, Triangle
from .ecs_constants import Orientation


class BoundingBox:

    center: Point
    half_widths: Point
    orientation: Orientation

    def __init__(
        self,
        center: Point,
        half_widths: Point,
        orientation: Orientation = Orientation.UP,
    ):
        self.center = center
        self.half_widths = half_widths
        self.orientation = orientation

    def colliding(self, other) -> bool:
        raise NotImplementedError(
            f"{self.__class__} has not implemented method: colliding."
        )


class AABB(BoundingBox):

    def colliding(self, other) -> bool:
        # AABB Collision
        if isinstance(other, self.__class__):
            return self.AABB_collision(other)

        return False

    def AABB_collision(self, other: AABB) -> bool:
        if (
            (
                abs(self.center[0] - other.center[0])
                > (self.half_widths[0] + other.half_widths[0])
            )
            or (
                abs(self.center[1] - other.center[1])
                > (self.half_widths[1] + other.half_widths[1])
            )
            or (
                abs(self.center[2] - other.center[2])
                > (self.half_widths[2] + other.half_widths[2])
            )
        ):
            return False
        return True

    @property
    def on_floor(self):
        pass
