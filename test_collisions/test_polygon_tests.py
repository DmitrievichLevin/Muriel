"""ECS Polygon Test Testing Module"""

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
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

    assert not collisions.Intersection.arraylike(intersect)


@pytest.fixture
def testing_vectors():
    """Testing Vectors"""
    A, B, C, D, E, F, G, H, I, J, K, L = (
        [2, 0, -1],
        [5, 0, -1],
        [0, 5, -3],
        [5, 5, -4],
        [10, 5, 4],
        [-5, 0, -2],
        [0, -5, 2],
        [-2, 8, 3],
        [2, 2, -9 / 5],
        [12, -8, 1],
        [12, -8, 5],
        [12.0, -8.0, 2.2],
    )

    return [A, B, C, D, E, F, G, H, I, J, K, L]


@pytest.fixture
def testing_triangles(testing_vectors):
    """Testing Triangles"""
    A, B, C, D, E, F, G, H, I, J, K, *_rest = testing_vectors
    return (
        collisions.Triangle([A, B, C], name="ABC"),
        collisions.Triangle([C, B, D], name="CBD"),
        collisions.Triangle([B, E, D], name="BED"),
        collisions.Triangle([F, A, C], name="FAC"),
        collisions.Triangle([G, B, A], name="GBA"),
        collisions.Triangle([F, C, H], name="FCH"),
        collisions.Triangle([I, J, K], name="IJK"),
    )


@pytest.fixture
def testing_polygons(testing_vectors):
    """Testing Polygons"""
    _A, _B, _C, _D, _E, _F, _G, _H, I, J, K, L = testing_vectors
    return (
        collisions.Polygon([I, L, K], name="ILK"),
        collisions.Polygon([I, J, L], name="IJL"),
    )


@pytest.fixture
def test_quad():
    """Quadrilateral"""
    a, b, c, d = (
        collisions.Vector3d([0, 0, 0]),
        collisions.Vector3d([5, 0, 0]),
        collisions.Vector3d([5, 5, 0]),
        collisions.Vector3d([0, 5, 0]),
    )
    return collisions.Quadrilateral([a, b, c, d])


@pytest.fixture
def bsp_tree_tri(testing_triangles):
    """BSP of Triangles"""

    return collisions.BSPNode(testing_triangles)


def test_vector_classifications(testing_vectors, testing_triangles):
    """Test classifying vector in respect to plane."""
    _A, _B, _C, _D, _E, _F, _G, _H, I, J, K, *__rest = testing_vectors

    ABC, *_rest = testing_triangles

    # Test Point On Plane ABC
    assert (
        collisions.PolygonClassification.ON_PLANE
        == collisions.classify_vector_to_plane(I, ABC)
    )

    # Test Point BEHIND Plane ABC
    assert (
        collisions.PolygonClassification.BEHIND
        == collisions.classify_vector_to_plane(J, ABC)
    )

    # Test Point INFRONT Plane ABC
    assert (
        collisions.PolygonClassification.INFRONT
        == collisions.classify_vector_to_plane(
            collisions.Vector3d(K), ABC
        )
    )


def test_classify_polygon_to_plane(
    testing_triangles, testing_vectors
):
    """Test classifying polygon in respect to plane."""
    _ABC, _CBD, _BED, _FAC, _GBA, _FCH, IJK, *_rest = (
        testing_triangles
    )

    A, B, C, *_rest = testing_vectors

    assert (
        collisions.PolygonClassification.STRADDLING
        == collisions.classify_polygon_to_plane(
            IJK, collisions.Plane([A, B, C])
        )
    )


def test_picking_split_plane(testing_triangles):
    """Test selecting splitting plane from polygons"""
    ABC, CBD, BED, FAC, GBA, FCH, _IJK, *_rest = testing_triangles

    # Split Plane: BED Score == 5.2
    split = collisions.split_plane([ABC, CBD, BED, FAC, GBA, FCH])

    assert all(
        [
            BED[idx] == auto_partitioned_point
            for idx, auto_partitioned_point in enumerate(split)
        ]
    )


def test_polygon_split(
    testing_vectors, testing_triangles, testing_polygons
):
    """Test splitting polygon against plane"""
    A, B, C, *_rest = testing_vectors

    _ABC, _CBD, _BED, _FAC, _GBA, _FCH, IJK, *_rest = (
        testing_triangles
    )

    ILK, IJL = testing_polygons

    plane_ABC = collisions.Plane([A, B, C])

    front_split, back_split = collisions.split_polygon(IJK, plane_ABC)

    assert front_split == ILK

    assert back_split == IJL


def test_point_bounded_by_polygon(testing_triangles, test_quad):
    """Test Point Bounded by Polygon"""
    ABC, *_rest = testing_triangles

    point = collisions.Vector3d([2, 2, 3])

    outside_point = collisions.Vector3d([6, 2, 3])

    assert collisions.point_bounded_by_polygon(point, ABC)

    assert collisions.point_bounded_by_polygon(point, test_quad)

    assert not collisions.point_bounded_by_polygon(outside_point, ABC)

    assert not collisions.point_bounded_by_polygon(
        outside_point, test_quad
    )


def test_polygon_orientation(testing_vectors):
    """Test Polygon Orientation"""
    A, B, C, *_rest = testing_vectors

    # Facing View(counter-clock-wise)
    ABC = collisions.Triangle([A, B, C], name="ABC")

    # Facing Opposite View(clock-wise)
    CBA = collisions.Triangle([C, B, A], name="CBA")

    assert ABC.orientation == collisions.Orientation.NORTH_UP

    assert CBA.orientation == collisions.Orientation.SOUTH_DOWN


def test_bsp_tree(bsp_tree_tri):
    """Test Point Colliding With BSPNode"""

    # Visualize BSP Tree in Terminal
    bsp_tree_tri.display()

    point = collisions.Vector3d([2, 2, 3])

    colliding_point = collisions.Vector3d([2, 2, -9 / 5])

    extruding_point = collisions.Vector3d([9.2, -5.2, 2])

    not_collision = collisions.BSPNode.point_collision(
        bsp_tree_tri, point
    )

    _collision = collisions.BSPNode.point_collision(
        bsp_tree_tri, colliding_point
    )

    collision_with_extruding_polygon = (
        collisions.BSPNode.point_collision(
            bsp_tree_tri, extruding_point
        )
    )

    assert (
        not_collision
        == collisions.PolygonClassification.NOT_COLLIDING
    )

    assert _collision == collisions.PolygonClassification.COLLISION

    assert (
        collision_with_extruding_polygon
        == collisions.PolygonClassification.COLLISION
    )
