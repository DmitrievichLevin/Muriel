"""ECS Polygon Test Testing Module"""

import pytest
from muriel import collisions


def test_intersecting_segment():
    """Test Polygon Line Intersection"""

    a = collisions.Vector3d([3, 5, 12])
    b = collisions.Vector3d([6, 5, 9])
    c = collisions.Vector3d([10, 9, 15])
    d = collisions.Vector3d([6, 8, 15])

    quad = collisions.Quadrilateral([a, b, c, d])

    # ABC Test
    line1 = collisions.Line(
        [
            collisions.Vector3d([6, -2, 11]),
            collisions.Vector3d([6, 5, 11]),
        ]
    )

    expected_intersect = [6.000, 5.8, 11.0]

    intersect = collisions.Intersection(quad, line1)

    assert intersect == expected_intersect

    # ADC Test
    line2 = collisions.Line(
        [
            collisions.Vector3d([7, -2, 12.4]),
            collisions.Vector3d([6, 5, 13.8]),
        ]
    )

    expected_intersect = [5.66, 7.37, 14.27]

    intersect = collisions.Intersection(quad, line2)

    assert intersect == expected_intersect

    # Non-Intersecting Test
    line3 = collisions.Line(
        [
            collisions.Vector3d([7, -2, 17]),
            collisions.Vector3d([6, 5, 18]),
        ]
    )

    intersect = collisions.Intersection(quad, line3)

    assert intersect is None
