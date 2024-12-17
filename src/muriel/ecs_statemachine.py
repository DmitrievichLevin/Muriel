"""ECS State Machine"""

from __future__ import annotations
import logging
from typing import Any, Callable
import enum
from muriel.ecs_system import System
from muriel.ecs_input_sequence import SEQUENCE, Press


# 4-bit
class NEWTONIAN(enum.IntEnum):
    # idle, walk, run, ascend, descend
    MALLEABLE = 0b0000
    # punch, kick
    FORCE = 0b1010
    # block, counter
    ADAMANT = 0b0001
    # grab
    JUNCTURE = 0b0110


class BaseStateMachine(type):

    def __new__(mcs, name, bases, attrs):

        if name != "StateMachine":
            try:
                attrs["state"]
            except KeyError as e:
                raise KeyError(
                    f"{name} of type StateMachine expected attribute: 'state' of type SEQUENCE, but found None."
                ) from e

        return super().__new__(mcs, name, bases, attrs)


class StateMachine(System, metaclass=BaseStateMachine):
    root = SEQUENCE.IDLE
    state: SEQUENCE

    class StateContext:
        def __init__(self, state_component_update: Callable):
            self.state_component_update = state_component_update

        def __get__(self, state_instance, _state_cls):

            def context_wrapper(
                _input: list[str] | None = None, **kwargs
            ):
                sequence = StateMachine.serialize_input(  # pylint: disable=protected-access
                    _input
                )
                self.state_component_update(
                    state_instance,
                    _input=_input,
                    sequence=sequence,
                    **kwargs,
                )

            return context_wrapper

    @classmethod
    def serialize_input(
        cls, _input: list[str] | None = None
    ) -> SEQUENCE | None:
        """Serialize input using Trie

        Returns:
            SEQUENCE | None: result of input serialization.
        """

        sequence = cls.root.value.find(_input)
        logging.debug(
            "Raw input: %s \nSerialized Input SEQUENCE: %s",
            _input,
            sequence,
        )
        return sequence


# 0b0000: "idle",
# 0b0001: "run",
# 0b0010: "jump",
# 0b0011: "gaurd",
# 0b0100: "counter",
# 0b1000: "combat/p",
# 0b1001: "combat/k",
# 0b1011: "combat/g",
# 0b1111: "hurt",
# 0b11111: "combat/locked",


# check velocity -> check key -> get current state and check precedence


# class State(System):
#     state: int
#     newtonian: int
#     modifier: int

#     def process(self, *args, **kwargs) -> dict[str, Any]:
#         action = kwargs.get("press", None)
#         return super().process(*args, **kwargs)

#     def update(self, frame, *_, **kwargs) -> None:

#         velocity = kwargs.get("prev", {})
#         x, _, z = velocity.get("input", [0] * 3)
#         __, y, ___ = velocity.get("acceleration", [0] * 3)

#         movement = int(max(abs(x), abs(z)))

#         ascend = 0 if y == 0 else y / y * 3

#         logging.debug(
#             f"state update: {frame}: {next_state}. \nvelocity:{velocity} \nkwargs: {kwargs}"
#         )
#         super().update(frame=frame, state=int(next_state))


# class State(Component):
#     # TODO - SEQUENCE DOESN"T Change _Out + state machine system
#     key = "state"
#     default = int(SEQUENCES.IDLE)

#     def update(self, frame, *_, **kwargs) -> None:
#         velocity = kwargs.get("prev", {})
#         x, _, z = velocity.get("input", [0] * 3)
#         __, y, ___ = velocity.get("acceleration", [0] * 3)

#         movement = int(max(abs(x), abs(z)))

#         ascend = 0 if y == 0 else y / y * 3

#         next_state = SEQUENCES(ascend or movement)

#         logging.debug(
#             f"state update: {frame}: {next_state}. \nvelocity:{velocity} \nkwargs: {kwargs}"
#         )
#         super().update(frame=frame, state=int(next_state))

#     def input(
#         self, frame: int | None = None, state=None, **values
#     ) -> None:
#         _state = state or self.output(frame - 1)
#         return super().input(frame, state=int(_state), **values)
