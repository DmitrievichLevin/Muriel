import pytest
from muriel.ecs_builtin import Movement, Direction, Compartment


def test_builtin_movement():
    """Test Built-in Movement Compartment."""

    Compartment = Movement()

    # Simulate Arrow Press

    Compartment.input(0, velocity=[1, 0, 0])

    # Assert valid Frame
    assert {
        "frame": 0,
        "position": [0, 0, 0],
        "velocity": [1, 0, 0],
        "direction": 90.0,
        "speed": 24,
    } == Compartment.output(0)

    for i in range(68):
        Compartment.input(frame=i + 1, velocity=[1, 0, 1])

    # Assert valid Frame
    SPEED = 24
    FRAME_RATE = 50
    valid_position = lambda frame: [
        (SPEED * v / 100) / FRAME_RATE for v in [frame, 0, frame - 1]
    ]
    assert valid_position(49) == Compartment.output(49)["position"]

    # Raise Exception for packet without frame
    with pytest.raises(KeyError):
        Compartment.on_data(
            **{
                "position": [10, 0, 0],
                "velocity": [1, 0, 0],
                "direction": 90.0,
                "speed": 24,
            }
        )

    # Simulate Client Validation(Server Packet)
    Compartment.on_data(
        **{
            "frame": 0,
            "position": [10, 0, 0],
            "velocity": [0.0072, 0, 0],
            "direction": 90.0,
            "speed": 24,
        },
    )

    # Assert client reconciliation is correct
    valid_reconciliation = valid_position(36)
    valid_reconciliation[0] += 10

    assert valid_reconciliation == Compartment.output(36)["position"]


def test_observable_component():
    """Observer should register class of type Observable."""

    class LOL:
        pass

    with pytest.raises(TypeError):

        class TestSystem(Compartment):
            direction = Direction
            not_observable = LOL()
            direction.subscribe(not_observable)
