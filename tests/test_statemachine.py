import pytest
from muriel.ecs_input_sequence import SEQUENCE
from muriel.ecs_builtin_states import Walk
from muriel.ecs_state_component import State


def test_statemachine():
    comp = State()
    system = Walk()
    system._setup(comp)

    # default
    assert comp.state == SEQUENCE.IDLE

    # Non-sorted input depth = 2
    comp.update(_input=["S", "L"], frame=0)
    assert comp.state == SEQUENCE.RUN

    # depth = 1
    comp.update(_input=["R"], frame=1)
    assert comp.state == SEQUENCE.WALK

    # Sorted input depth = 2
    comp.update(_input=["L", "S"], frame=2)
    assert comp.state == SEQUENCE.RUN

    # No input
    comp.update(_input=[], frame=3)
    assert comp.state == SEQUENCE.IDLE
    print(comp.state)
