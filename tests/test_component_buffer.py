import pytest
from muriel.ecs_component import Buffer

test_buffer = Buffer()


def test_frame_not_found():
    """Test Invalid Frame Request"""
    with pytest.raises(IndexError):
        test_buffer.get(1)


def test_head_update():
    """Test Auto Update Head"""

    # Frame 20 = 4
    test_buffer.put(frame=20, value=4)
    assert test_buffer.head == 20

    # Frame 15 = 2 (Head is still 20)
    test_buffer.put(frame=15, value=2)
    assert test_buffer.head == 20


def test_remove():
    """Test Queue Remove"""
    test_buffer.remove(20)
    assert test_buffer.head == 15

    with pytest.raises(IndexError):
        test_buffer.remove(99)


def test_pop():
    """Test Queue Pop

    - Pop until empty and once more
    """
    assert 2 == test_buffer.pop()
    assert test_buffer.head == 2
    assert test_buffer.pop() == 99
    assert test_buffer.pop() == 15
    assert test_buffer.pop() == 0


def test_peek():
    """Test Buffer Peak"""

    # Empty Queue
    assert test_buffer.peek() is None

    test_buffer.put(value="Hello World", frame=320)

    # Not Empty
    assert test_buffer.peek() == "Hello World"

    test_buffer.put(value="Hola Todo Mundo", frame=999)

    assert test_buffer.peek() == "Hola Todo Mundo"
