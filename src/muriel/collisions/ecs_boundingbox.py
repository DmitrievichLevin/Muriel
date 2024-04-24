from __future__ import annotations
from .ecs_plane import Vector3d, Triangle
from .ecs_constants import Orientation


class BoundingBox:

    center: Vector3d
    half_widths: Vector3d
    orientation: Orientation

    def __init__(
        self,
        center: Vector3d,
        half_widths: Vector3d,
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


def aabb_collision_test():
    box1 = AABB(Vector3d([2, 2, 1]), Vector3d([1, 1, 1]))
    box2 = AABB(Vector3d([2, 4, 1]), Vector3d([1, 1, 1]))

    print(box1.colliding(box2))


aabb_collision_test()
