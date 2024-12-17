"""ECS BSP Utils Testing Module"""

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
import pytest
from muriel import collisions


def test_distance_point_plane():
    A, B, C = ([2, 0, -1], [5, 0, -1], [0, 5, -3])
    ABC = collisions.Triangle([A, B, C], name="ABC")

    test_point = collisions.Vector3d([4, 6, 6])
    print(
        "plane formula",
        ABC.raw_normal,
        ABC.dot_product,
        ABC.normal,
        collisions.point_plane_dist(test_point, ABC),
    )

    assert True == False
