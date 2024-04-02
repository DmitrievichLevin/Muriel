"""ECS Built-in States"""

import logging
from typing import Any
from muriel.ecs_statemachine import StateMachine
from muriel.ecs_input_sequence import SEQUENCE
from muriel.ecs_literals import IDLE


class NotConnected(StateMachine):
    state = SEQUENCE.NOT_CONNECTED

    @classmethod
    def process(cls, **kwargs):
        status = 500
        logging.debug("Attempting to connect... \nPacket: %s", kwargs)
        try:
            status = kwargs["status"]

            if status != 200:
                raise ConnectionError

            return kwargs
        except (ConnectionError, KeyError) as e:
            raise ConnectionError(
                f"User Not Connected; expected packet containing 'status': 200, but found {status}"
            ) from e


class Idle(StateMachine):
    state = SEQUENCE.IDLE

    @classmethod
    def process(cls, **kwargs):
        kwargs.update(velocity=IDLE)
        return kwargs


class Walk(StateMachine):
    state = SEQUENCE.WALK

    @classmethod
    def process(cls, **kwargs):
        return {"hello": "im walking"}


class VelocityVector3(list):
    def __init__(self):
        super().__init__(IDLE)
        self.run = False

    def add(self, direction: str) -> None:
        match (direction):
            case "L":
                self[0] += 1
            case "R":
                self[0] -= 1
            case "U":
                self[2] += 1
            case "D":
                self[2] -= 1
            case "J":
                self[1] += 1
            case "S":
                self.run = True
            case _:
                raise Exception("Invalid Velocity Vector")

    def append(self, __object: Any) -> None:
        raise NotImplementedError

    def __setitem__(self, idx, value):
        return super().__setitem__(
            idx, value * 2 if self.run else 1.0
        )

    def __setattr__(self, __name: str, __value: Any) -> None:
        run = super().__setattr__(__name, __value)
        if __name == "run" and __value:
            self[0] = self[0]
            self[2] = self[2]
        return run


class Run(StateMachine):
    state = SEQUENCE.RUN

    @classmethod
    def process(cls, **kwargs):
        velocity = VelocityVector3()
        for direction in kwargs["_input"]:
            velocity.add(direction=direction)

        return dict(velocity=velocity)


class Ascend(StateMachine):
    state = SEQUENCE.ASCEND

    @classmethod
    def process(cls, **kwargs):
        velocity = VelocityVector3()
        for direction in kwargs["_input"]:
            velocity.add(direction=direction)

        return dict(velocity=velocity)
