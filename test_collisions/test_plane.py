"""ECS Plane Module Testing Suite"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np
from muriel.collisions.ecs_plane import (
    Normal,
    Vector3d,
    Polygon,
    Quadrilateral,
    Triangle,
)


@pytest.fixture
def test_matrix():
    """Test Plane Matrix"""
    # 𝐴=(3,5,12) 𝐵=(6,5,9) 𝐶=(6,8,15)
    a = np.array([3, 5, 12])
    b = np.array([6, 5, 9])
    c = np.array([6, 8, 15])
    return [a, b, c]


def test_vector_class():
    """Test Vector3d object"""
    # Vector3d has length of 3

    with pytest.raises(ValueError):
        Vector3d([0, 1])

    # Vector3d elements must be of type float | int
    with pytest.raises(ValueError):
        Vector3d([0, 1, "hello world"])  # type: ignore[reportArgumentType]


def test_normal_obj(test_matrix):
    """Test Normal object"""
    a, b, c = test_matrix

    # n=(9,-18,9) or (1,-2,1)
    _normal = Normal(a, b, c)

    # Non - Simplified normal
    vector = [9, -18, 9]
    assert all([_normal.raw_normal[i] == vector[i] for i in range(3)])

    # Simplified normal
    vector = [1.0, -2.0, 1.0]
    assert all([_normal.normal[i] == vector[i] for i in range(3)])

    # Normal as a unit
    vector = [1.0, -1.0, 1.0]
    assert all([_normal.unit[i] == vector[i] for i in range(3)])


def test_polygon(test_matrix):
    """Test Polygon concavity."""
    a, b, d = test_matrix

    c = Vector3d([10, 9, 15])

    _poly = Polygon([a, b, c, d])

    assert _poly.convex


def test_polygon_subclass_capacity(test_matrix):
    """Test Quadrilateral Object"""
    a, *_rest = test_matrix
    with pytest.raises(ValueError):
        _bad_quad = Quadrilateral([a])

    with pytest.raises(ValueError):
        _bad_tri = Triangle([a])
