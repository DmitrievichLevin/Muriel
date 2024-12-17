import pytest
from muriel.ecs_buffers import QBuffer


def test_queue():
    """Test Queue Buffer"""
    buffer = QBuffer()

    assert buffer.dequeue() == None

    buffer.enqueue(18)

    assert buffer.peek() == 18

    buffer.enqueue(99)

    assert dict(buffer) == {
        0: 99,
        1: 18,
    }

    assert buffer.dequeue() == 18

    assert buffer.peek() == 99

    assert dict(buffer) == {0: 99}

    assert buffer.dequeue() == 99

    assert buffer.peek() == None
