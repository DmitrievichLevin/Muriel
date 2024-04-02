"""ECS Attribute Interface"""

from threading import get_ident
import typing
import logging
from muriel.ecs_events import Observer
from muriel.ecs_governed import Governed


class Attribute(Governed, Observer):
    """Attribute Interface: Immutable Entity Attributes

    Yields:
        (tuple): key,value of attribute
    """

    key: str
    default: typing.Any
    __value: typing.Any
    immutable: bool = False

    def __init__(self, value=None):
        key, default = getattr(self, "key"), getattr(self, "default")
        self.value = value or default
        super().__init__(name=key)

    @property
    def value(self):
        """Attribute Value

        Returns:
            (Any): value of attribute
        """
        return self.__value

    @value.setter
    def value(self, v):
        self.__value = v

    def can_update(self, *args, **kwargs) -> bool:
        if (self.value and self.immutable) or not kwargs.get(
            "response", False
        ):
            return False
        return True

    def update(self, *args, **kwargs) -> typing.Any:
        self.on_data(**kwargs)

    def input(
        self,
        frame: int | None = None,
        load: int | None = None,
        **attributes
    ) -> None:
        """Attribute Client Input(Restricted)

        Raises:
            TypeError(Exception): Client Attribute input is not allowed.
        """
        raise TypeError(
            "Client is not allowed to mutate the state of an Attribute."
        )

    def on_data(self, **packet: dict) -> None:
        """Update Attribute Value.

        - Attributes can only be changed by server

        Args:
            packet(dict): Server packet

        Returns:
            true(bool): Attribute was successfully updated by response.
        """
        self.mutex.acquire()
        logging.debug(
            "Attribute %s acquired lock. \nThread: %s",
            self.key,
            get_ident(),
        )

        self.value = packet[self.key]
        logging.debug(
            "updating %s <Attribute> \nvalue: %s",
            self.key,
            self.value,
        )

        self.mutex.release()
        logging.debug(
            "Component %s released lock. \nThread: %s",
            self.key,
            get_ident(),
        )

    def output(self, frame: int = -1) -> typing.Any:
        """Get Attribute"""
        return self.value

    def __iter__(self):
        yield from (self.key, self.value)
