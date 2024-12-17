"""ECS Controller"""

import logging
import typing
from muriel.ecs_component import IndeterminateComponent, Observable


class Frame(IndeterminateComponent):
    """Frame Controller
    - TODO: Implementation for testing.
    """

    key = "frame"
    default = 0

    @property
    def current(self) -> int:
        """Current Frame Node"""
        if self.buffer.root:
            return self.buffer.root.value
        else:
            return -1

    def tick(self) -> int:
        """Advance Frame of Observers"""
        next_frame = self.current + 1
        logging.debug(
            "Advancing frame %s -> %s", self.current, next_frame
        )
        self.input(frame=next_frame, mute=not next_frame)
        return next_frame

    @Observable.notify_observers
    def input(
        self, frame: int | dict | None = None, **values
    ) -> None:
        """Client input processing

        - Enqueue values

        Args:
            frame(int): current
            values(dict): input dict values
        """
        self.buffer.enqueue(frame)

    def on_data(
        self, **data: dict[str, int | float | list[typing.Any] | str]
    ) -> None:
        """Server Input Processing

        Args:
            data(dict): Server data

        Returns:
            valid(bool): always valid
        """

    def output(self, _: int = -1) -> typing.Any:
        return self.buffer.dequeue()
