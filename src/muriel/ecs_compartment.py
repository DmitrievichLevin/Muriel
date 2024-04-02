"""ECS Component Group Interface"""

from __future__ import annotations
import threading
from typing import Any, Mapping
from inspect import isclass
import logging
from muriel.ecs_events import Observable
from muriel.ecs_governed import Governed
from muriel.ecs_component import Component
from muriel.ecs_attribute import Attribute


class ComponentList:
    """Component(s) list/Mapping"""

    def __init__(self, **attrs):
        temp_dict = {}
        temp_arr = []
        self._stats = {}
        for name, comp_cls in attrs.items():
            if isclass(comp_cls) and issubclass(
                comp_cls, (Component, Attribute, Compartment)
            ):
                comp = comp_cls()
                comp.key = name
                if isinstance(comp, Attribute):
                    self._stats.update({name: comp})
                comp._stats = self._stats

                temp_dict.update({name: comp})
                temp_arr.append((name, comp))
        self._dict = temp_dict
        self._arr = temp_arr

    def update(
        self, kwargs: dict[str, Component | Attribute | Compartment]
    ):
        # TODO - prevent overwrite + may need overloaded func similar to dict update + type restriction Component | Attribute | Compartment
        for name, component in kwargs.items():
            self._dict.update({"name": component})
            self._arr.append((name, component))

    def __getattribute__(self, __name: str) -> Any:
        # TODO - Better, find root cause for solution
        try:
            return super().__getattribute__("_dict")[__name]
        except Exception:
            return super().__getattribute__(__name)

    def __getitem__(self, idx):
        return self._arr[idx]

    def __iter__(self):
        yield from self._arr

    def __repr__(self) -> str:
        return str({key: val for key, val in self._arr})


class CompartmentMeta(type):
    """Compartment Meta"""

    def __new__(mcs, name, bases, attrs):
        """Construct Compartment Instance

        Args:
            cls: Super Class
            name: Name of subclass
            bases: bases
            attr: instance attributes

        Returns:
            instance(Compartment): Compartment subclass
        """
        if bases[0] is Governed:
            return super().__new__(mcs, name, bases, attrs)

        combined_attrs = {
            bs_key: bs_val
            for base in bases
            for bs_key, bs_val in base.__dict__.items()
        } | attrs

        # Prep Component Classes
        attrs["observables"] = combined_attrs

        # Set class name of Compartment as key
        attrs["key"] = attrs.get("key", name.lower())

        return super().__new__(mcs, name, bases, attrs)


class Compartment(Governed, metaclass=CompartmentMeta):
    """Compartment Interface

    Args:
        Governed (type): Deterministic Base class.
        metaclass (type): CompartmentMeta.

    Raises:
        AttributeError: _description_
        KeyError: _description_

    Returns:
        (Compartment): Grouped Components

    Yields:
        (Mapping): key,component_last_frame
    """

    default: dict
    observables: ComponentList
    data_type: dict

    def __init__(self) -> None:

        self.observables = ComponentList(**self.observables)

        self.default = dict(
            {name: obs.default for name, obs in self.observables}
        )
        self.data_type = dict(
            {name: obs.data_type for name, obs in self.observables}
        )
        super().__init__()

    def input(
        self, frame: int = -1, load: int | None = None, **attributes
    ) -> None:
        self.mutex.acquire()
        logging.debug(
            "%s acquired lock. \nThread: %s",
            self.key,
            threading.get_ident(),
        )
        # Track last processed frame for reconciliation
        if self.last and self.last == frame:
            raise AttributeError(
                f"Attempted to overwrite input frame: {frame}."
            )
        # Update last frame int
        self.last = frame

        # TODO - Threading maybe?
        # Allocate input to relevant components
        for comp_name, comp in self.observables:

            if comp_name in attributes and isinstance(
                comp, Component
            ):
                comp.mutex.acquire()
                try:

                    logging.debug(
                        "%s acquired lock. \nThread: %s \nProcessing <Compartment> %s <Component> %s input...",
                        comp.key,
                        threading.get_ident(),
                        self.key,
                        comp.key,
                    )

                    comp.input(frame=frame, load=load, **attributes)
                finally:
                    comp.mutex.release()
                    logging.debug(
                        "%s released lock. \nThread: %s",
                        comp.key,
                        threading.get_ident(),
                    )
        self.mutex.release()
        logging.debug(
            "%s released lock. \nThread: %s",
            self.key,
            threading.get_ident(),
        )

    def on_data(
        self, **packet: dict[str, int | float | list | str]
    ) -> None:
        """Packet Data Input

        Args:
            packet(dict): Packet contents.

        Raises:
            KeyError: frame not found in packet.
        """
        try:
            frame = packet["frame"]
        except Exception as e:
            raise AttributeError(
                "Compartment expected frame, but found None"
            ) from e

        logging.debug(
            "%s Processing Packet <Frame>: %s \nPacket:%s",
            self.key,
            frame,
            packet,
        )

        for name, obs in self.observables:
            logging.debug("%s -> packet -> %s", self.key, name)
            obs.on_data(**packet)

        logging.debug(
            "%s Successfully processed Packet <Frame>: %s",
            self.key,
            frame,
        )

    def can_update(self, *args, **kwargs) -> bool:
        if any(key in kwargs.keys() for key in self.default.keys()):
            return True
        return False

    def update(self, *args, **kwargs) -> Any:
        self.input(**kwargs)

    def begin_observing(self, *obs: tuple[str, Observable]) -> Any:
        for _, component in self.observables:
            component.begin_observing(*obs)
        return super().begin_observing(*obs)

    def output(self, frame: int | None = None) -> dict:
        """Compilation of Component(s) output."""
        return dict(
            frame=frame or self._last,
            **{
                key: (
                    comp.value
                    if hasattr(comp, "value")
                    else comp.output(frame)
                )
                for key, comp in self.observables
            },
        )

    def __iter__(self):
        yield from (
            (key, comp.output()) for key, comp in self.observables
        )
