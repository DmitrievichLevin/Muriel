"""ECS AABB Testing Module"""

# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
from muriel import collisions


def test_aabb_colliding_aabb():
    """Collision Check on Two AABB(s)"""
    box1 = collisions.AABB(
        collisions.Vector3d([2, 2, 1]), collisions.Vector3d([1, 1, 1])
    )
    box2 = collisions.AABB(
        collisions.Vector3d([2, 4, 1]), collisions.Vector3d([1, 1, 1])
    )

    box3 = collisions.AABB(
        collisions.Vector3d([4, 5, 1]), collisions.Vector3d([1, 1, 1])
    )

    assert box1.colliding(box2)

    assert box2.colliding(box1)

    assert not box1.colliding(box3)

    assert not box3.colliding(box1)
