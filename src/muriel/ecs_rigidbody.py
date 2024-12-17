"""ECS Vector Quantity Physics System"""

import logging
import math
from operator import itemgetter
from typing import (
    Any,
    Generic,
    ParamSpec,
)
from muriel.util import four_point_float
from muriel.ecs_literals import DELTA
from muriel.ecs_system import SystemDecorator


_K = ParamSpec("_K")


class Newton(int):
    def __str__(self) -> str:
        return super().__str__() + "N(s)"


class Kilogram(int):
    def __str__(self) -> str:
        return super().__str__() + "kg(s)"


class MetersPerSecond(float):
    def __str__(self) -> str:
        return super().__str__() + "m/s^2"


class Second(float):
    def __str__(self) -> str:
        return super().__str__() + "s"


class VectorQuantityPhysics(SystemDecorator, Generic[_K]):
    """Vector Quantity Physics"""

    airborne = 0
    in_motion = 0
    inertia: list[float] | None = None
    delta = DELTA
    # 1 Unit = 1 m(s)
    impulse = 270  # N(s)
    mass = 45  # kg
    gravity = 9.81

    @classmethod
    def process(
        cls,
        component,
        *args,
        frame: int,
        **kwargs,
    ) -> dict[str, Any]:
        """Acceleration

        Rules:
            - Cannot add lateral velocity while is_jumping

        Args:
            func (_type_): _description_
        """
        print("kwargs", kwargs["velocity"], frame)
        (i_name, inertia), (a_name, airborne) = itemgetter(
            "inertia", "airborne"
        )(kwargs)
        delta = cls.delta
        velocity: list[float] = (
            inertia or kwargs["velocity"]  # type: ignore[assignment]
        )

        x, y, z = velocity

        # Sprint = speed*(1 or 1vel)
        # Run = speed/(2 or 0.5vel)
        # Walk = speed/(4 or 0.25vel)
        # in units per second

        speed = component.stats["speed"].value

        # Apply Acceleration to x
        a_x = (speed * x / 100) / delta

        # Apply Acceleration to z
        a_z = (speed * z / 100) / delta

        a_y, inertia, airborne = cls._gravity(
            x, y, z, inertia, airborne
        )

        setattr(component, i_name, inertia)
        setattr(component, a_name, airborne)

        kwargs.update(
            velocity={
                "acceleration": list(four_point_float(a_x, a_y, a_z)),
                "input": list(four_point_float(x, y, z)),
            }
        )
        return kwargs | {"frame": frame}

    @classmethod
    def _gravity(
        cls, x, y, z, inertia, airborne
    ) -> tuple[float, list[float] | None, int]:
        """Gravity Simulation

        Args:
            x (float): Vector3 x
            y (float): Vector3 y
            z (float): Vector3 z
            inertia (Vector3): force of inertia
            airborne (int): duration of not ground_collision
            delta (int): framerate

        Returns:
            tuple[float, list[float] | None, int]: [gravity based Vector3 physics, Inertia-Vector3, time-since-last-ground-collision].
        """
        # Ground Check First TODO - Collision Detection
        if not y and not airborne:
            return 0.0, None, 0

        # 1 Unit = 1 m(s)
        impulse = cls.impulse  # N(s)
        mass = cls.mass  # kg
        gravity = cls.gravity

        velocity = impulse / mass  # m/s

        # Peak
        b = velocity / gravity

        # Max Height
        k = (0.5 * velocity) * (velocity / gravity) * 100

        # Future application?
        _energy = k * mass * gravity

        step = airborne * 1 / cls.delta

        a = -k / (b**2)

        force = (a * (step - b) ** 2) + k  # cm/s -> m/s

        logging.debug(
            "Vertical Acceleration: %sm/s \nHang time: %s \nmax: %s",
            force,
            airborne,
            k,
        )

        # TODO - Will use collision to stop gravity application
        # Currently ground_collision = y_Position <= 0
        if force <= 0 and airborne > 0:
            return 0.0, None, 0
        else:
            return force, inertia or [x, y, z], airborne + 1

    @classmethod
    def lateral_velocity_ms(
        cls, _input: int, speed: float
    ) -> MetersPerSecond:
        return (speed * _input / 100) / cls.delta

    @classmethod
    def vertical_velocity_ms(
        cls, impulse: Newton, mass: Kilogram
    ) -> MetersPerSecond:
        """Velocity in m/s^2

        Args:
            impulse (_type_): _description_
            mass (_type_): _description_

        Returns:
            _type_: _description_
        """
        return MetersPerSecond(impulse / mass)

    @classmethod
    def jump_peak_ms(cls, velocity, gravity) -> MetersPerSecond:
        """Peak of Jump reached in m/s^2

        Args:
            velocity (_type_): _description_
            gravity (_type_): _description_

        Returns:
            _type_: _description_
        """
        return MetersPerSecond(velocity / gravity)

    @classmethod
    def jump_max_height(
        cls, velocity: MetersPerSecond, gravity: MetersPerSecond
    ) -> Second:
        """Max Height of Jump

        Args:
            velocity (_type_): _description_
            gravity (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Max Height
        return Second(
            (0.5 * velocity)
            * cls.jump_peak_ms(velocity, gravity)
            * 100
        )

    @classmethod
    def airborne_duration(cls, y: MetersPerSecond) -> Second:
        """Gravity Simulation

        Args:
            x (float): Vector3 x
            y (float): Vector3 y
            z (float): Vector3 z
            inertia (Vector3): force of inertia
            airborne (int): duration of not ground_collision
            delta (int): framerate

        Returns:
            tuple[float, list[float] | None, int]: [gravity based Vector3 physics, Inertia-Vector3, time-since-last-ground-collision].
        """

        # 1 Unit = 1 m(s)
        impulse = 270  # N(s)
        mass = 45  # kg
        gravity = 9.81

        velocity = cls.vertical_velocity_ms(impulse, mass)  # m/s

        # Peak
        b = cls.jump_peak_ms(velocity, gravity)

        # Max Height
        k = cls.jump_max_height(velocity, gravity) / 100

        a = -k / (b**2)
        inner = (y - k) / -a
        return int((math.sqrt(inner) - b) * 10)

    @classmethod
    def normalize_acceleration(
        cls, acceleration: list[float | int]
    ) -> list[float]:
        vel_x = acceleration[0] / abs(acceleration[0])

        vel_y = (
            1
            if acceleration[1]
            == cls.vertical_velocity_ms(cls.impulse, cls.mass)
            else 0
        )

        vel_z = acceleration[2] / abs(acceleration[2])

        return [vel_x, vel_y, vel_z]

    @classmethod
    def has_inertia(cls, _input, _acceler):
        airborne = VectorQuantityPhysics.airborne_duration(
            _acceler[1]
        )
        movement = any(_in != 0 for _in in [_input[0], _input[2]])

        return _input if movement and airborne else None
