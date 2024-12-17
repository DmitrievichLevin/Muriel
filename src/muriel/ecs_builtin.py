"""ECS Built-in Components & Systems"""

import logging
from operator import itemgetter
import typing
from typing import Any
import uuid
import math
from muriel.ecs_literals import IDLE, DELTA
from muriel.ecs_component import Component, IndeterminateComponent
from muriel.ecs_compartment import Compartment
from muriel.ecs_attribute import Attribute
from muriel.ecs_buffers import Buffer
from muriel.ecs_rigidbody import (
    VectorQuantityPhysics,
    four_point_float,
)


class Location(Attribute):
    """Entity Location uuid"""

    data_type = str
    key = "map_id"
    default = None


class ID(Attribute):
    """Entity Id uuid"""

    data_type = str
    key = "reference"
    default = None
    immutable = True


class Speed(Attribute):
    """Entity Speed Attribute
    - for use in tandem built-in VectorQuantityPhysics System
    """

    data_type = int
    key = "speed"
    default = 24


class Velocity(Component):
    """Entity Velocity Component
    - for use in tandem built-in VectorQuantityPhysics System
    """

    key = "velocity"
    default = {"input": [0.0] * 3, "acceleration": [0.0] * 3}
    data_type = [float, float, float]
    acceleration: Buffer
    friction = 0.92
    framerate = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @VectorQuantityPhysics
    def input(self, frame: int = -1, **values) -> None:
        print("putting velocity", frame, values["velocity"])
        super().input(
            frame=frame,
            velocity=values["velocity"],
        )

    def validate(
        self, server: typing.Any, prediction: typing.Any
    ) -> bool:
        logging.debug(
            "Server acceleration: %s\nClient acceleration %s",
            server,
            prediction,
        )
        return (
            all(
                [
                    server[x] == prediction["acceleration"][x]
                    for x in range(3)
                ]
            )
            if prediction
            else False
        )

    def _reconcile(self, **_data) -> None:

        # Get frame in question
        frame = _data["frame"]

        prev = self.client.get(frame - 1)

        velocity_input = VectorQuantityPhysics.normalize_acceleration(
            _data["velocity"]
        )

        _data.update(
            velocity={
                "input": velocity_input,
                "acceleration": _data["velocity"],
            },
            prev=prev,
        )

        # Correct frame in question
        self.client.put(**_data)

        data = _data

        # iteration stops at most recent frame.
        last = self._last

        # Start iteration on succeeding frames
        frame += 1

        # iterate over frames and reconcile
        while frame <= last and data:
            logging.debug(
                "%s reconciliation loop frame: %s \ndata: %s \nstop @ %s",
                self.__class__.__name__,
                frame,
                data,
                last,
            )
            data = self.reconcile(**{**data, "frame": frame})
            frame += 1

    def reconcile(self, frame: int, **data) -> dict:

        last_input, accelr = itemgetter("input", "acceleration")(
            data["velocity"]
        )

        inertia = VectorQuantityPhysics.has_inertia(
            last_input, accelr
        )

        airborne = VectorQuantityPhysics.airborne_duration(accelr[1])

        client_input = self.client.get(frame).get("input")

        data.update(
            inertia=("inertia", inertia),
            airborne=("airborne", airborne),
            velocity=client_input,
            prev={"input": last_input, "acceleration": accelr},
        )
        print("\nphysics processed data %s", data)
        reconciled = VectorQuantityPhysics.process(
            component=self, **data, frame=frame
        )

        self.client.put(**reconciled)
        return reconciled

    def update(self, *args, **kwargs) -> Any:

        return self.input(**kwargs)

    def output(self, frame: int = -1) -> typing.Any:

        acceleration = super().output(frame).get("acceleration")
        return acceleration


class Position(Component):
    """Entity Velocity Component
    - for use in tandem built-in VectorQuantityPhysics System
    """

    key = "position"
    default = [0.0] * 3
    data_type = [float, float, float]

    def input(self, frame: Any = -1, **values) -> None:
        # TODO - Create Message.float subclass that accepts int
        return super().input(frame, **values)

    def move(self, position: list, velocity: list):
        """Move&Slide utility method

        Args:
            position (list): current position
            velocity (list): current acceleration

        Returns:
            (Vector3): new position
        """
        logging.debug(
            "Position acceleration process \nposition:%s \nvelocity: %s",
            position,
            velocity,
        )
        x, y, z = velocity or [0.0] * 3

        cur_x, _, cur_z = position

        # Lateral Movement
        x, y, z = four_point_float(x + cur_x, y, z + cur_z)

        return [x, y, z]

    def _reconcile(self, **data: Any) -> None:
        """Private Reconcile Override

        - Velocity Reconciliation will update
        - Only needs to update current/prev position
        """
        position, frame = data["position"], data["frame"]

        velocity = self.observing["velocity"].output(frame - 1)

        travel_back_to = self.move(
            position=position,
            velocity=[
                -v if idx != 1 else v
                for idx, v in enumerate(velocity)
            ],
        )
        self.client.put(**data)
        data.update(position=travel_back_to, frame=frame - 1)
        self.client.put(**data)

    def update(  # pylint: disable=arguments-differ
        self, frame, *_, **kwargs
    ) -> None:
        logging.debug("Position update args %s \n%s", frame, kwargs)
        position, velocity = (
            self.output(frame - 1),
            kwargs["prev"]["acceleration"],
        )
        travel_to = self.move(
            position=position,
            velocity=velocity,
        )

        logging.debug(
            "position update: %s: %s. \nvelocity:%s \nkwargs: %s",
            frame,
            travel_to,
            velocity,
            kwargs,
        )
        super().update(frame=frame, position=travel_to)

    def validate(
        self, server: typing.Any, prediction: typing.Any
    ) -> bool:
        return (
            all([server[x] == prediction[x] for x in range(3)])
            if prediction
            else False
        )


class Direction(Component):
    """Entity Velocity Component
    - for use in tandem built-in VectorQuantityPhysics System
    """

    key = "direction"
    default = 0.0
    data_type = float

    def update(  # pylint: disable=arguments-differ
        self, frame, *_, **kwargs
    ) -> None:

        velocity = kwargs["velocity"]

        velocity_input, prev_input = (
            velocity["input"],
            velocity.get("prev", {}).get("acceleration", 0.0),
        )

        new_direction = (
            self.angle(velocity_input)
            if any(velocity_input)
            else prev_input
        )

        logging.debug("Direction update: %s", new_direction)

        return super().update(frame, direction=new_direction)

    def angle(self, velocity: list[float]) -> float:
        """Direction Utility Method

        Args:
            velocity(Vector3): current velocity

        Returns:
            direction(float): angle measure
        """

        ang = math.degrees(
            math.atan2(0, -1) - math.atan2(velocity[0], -velocity[2])
        )
        return float(ang + 360 if ang < 0 else ang)


class UserInput(IndeterminateComponent):
    """Input Buffer Component

    - Queue-like linked list
    """

    key = "userinput"
    default = None
    data_type = [str]

    def update(self, *args, **kwargs) -> None:
        pass

    def get(self) -> list:
        return [i for i in self.buffer.clear()]

    def output(self, frame: int = 0) -> dict:
        raise NotImplementedError(
            "UserInput has no attribute 'output'."
        )


class Movement(Compartment):
    """Vector Quantity Component Group"""

    key = "movement"
    position: Component = Position  # type: ignore[assignment]
    velocity: Component = Velocity  # type: ignore[assignment]
    direction: Component = Direction  # type: ignore[assignment]
    velocity.subscribe(position, direction)
    speed = Speed
