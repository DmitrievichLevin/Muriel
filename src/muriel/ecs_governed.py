"""ECS Governed Interface"""

from threading import _RLock, RLock, get_ident
from typing import Any
import logging
from muriel.ecs_buffers import Buffer
from muriel.ecs_events import Observer


class Governed(Observer):
    """Governed Interface

    - Specifies methods(optional/required) of objects under Server Authority.
    """

    mutex: _RLock
    key: str
    client: Buffer
    server: Buffer
    _last: int = 0

    def __init__(self, *__, **_) -> None:
        self.mutex = RLock()
        super().__init__()

    @property
    def last(self) -> int:
        """Most recent frame.

        Returns:
            _type_: _description_
        """
        return self._last

    @last.setter
    def last(self, frame: int):
        current_frame: int = self._last
        if current_frame < frame:
            self._last = frame

    def validate(self, server: Any, prediction: Any) -> bool:
        """Validate Component Value

        - Override for validation of non-primitive data types.

        Args:
            server(Any): actual attribute on server from frame.
            predection(Any): client prediction.
        """
        return server == prediction

    def input(
        self,
        frame: int = -1,
        load: int | None = None,
        **values,
    ) -> None:
        """Base Component Input Processing

        - Put value in buffer or prepend
        *Overrides MUST call super to work properly*

        Args:
            frame(int): buffer key
            values(dict): input dict values
        """
        _frame: int = frame if frame > -1 else self.last + 1

        key: str = self.key
        value = values.get(key, None)

        # Mute initial frame & load buffer
        initial_load = not self.last

        # Optional: Mute Observer Notification
        mute = values.get("mute", False)

        # Include previous frame for observers
        _last_frame = self.client.get(self._last)

        logging.debug(
            "<Component> %s Component \nframe: %s \nvalue: %s.",
            self.key,
            _frame,
            values,
        )

        self.last = _frame

        self.client.put(
            frame=_frame,
            load=load,
            mute=initial_load or mute,
            **{key: value, "prev": _last_frame} | values,
        )

    def on_data(
        self, mute=True, **data: dict[str, int | float | list | str]
    ) -> None:
        """Server Input Processing

        - Validates Server determination
        - Invalid Frame triggers Component Reconciliation

        Args:
            data(dict): Server data

        Returns:
            valid(bool): Whether prediction is valid

        Raises:
            KeyError: frame not found in packet.
        """

        # TODO - Handle missing frame on server
        # (increase buffer limit + send packet complete with buffer contents)
        self.mutex.acquire()
        logging.debug(
            "Component %s acquired lock. \nThread: %s",
            self.key,
            get_ident(),
        )
        try:
            frame: int = data["frame"]
            # Server Determined value
            actual = data.get(self.key)
        except Exception as e:
            raise e

        # Client Prediction
        prediction = self.client.get(frame)

        logging.debug(
            "Component %s attempting validation... \n actual: %s prediction: %s",
            self.key,
            actual,
            prediction,
        )

        # Missing frame place directly into client buffer
        if prediction is None:
            self.input(**data)

        # Add server packet to server buffer
        self.server.put(**data, response=True, mute=mute)

        # Validate client predictation against server
        valid = not prediction or self.validate(actual, prediction)

        # Start Reconciliation
        if not valid:
            logging.debug(
                "Component %s failed validation... \nStarting reconciliation...",
                self.key,
            )
            self._reconcile(**data)

        self.mutex.release()
        logging.debug(
            "Component %s released lock. \nThread: %s",
            self.key,
            get_ident(),
        )

    def update(self, *args, **kwargs) -> Any:
        """Observer Update Method

        **Subclasses Must Override**
        """
        raise NotImplementedError()

    def _update(self, *args, response: bool = False, **kwargs):
        """Private Update Controller

        - Update Server Buffers(State Component Controller)
        - Call Interface Update
        - TODO - Maybe better?
        Args:
            response (bool, optional): State Controller Bottleneck. Defaults to False.
        """
        if response:
            self.on_data(**kwargs)
        else:
            self.update(*args, **kwargs)

    def _reconcile(self, **_data) -> None:

        # Get frame in question
        frame = _data["frame"]

        # Correct frame in question
        self.input(**_data)

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
        """Reconcile method(optional)

        - Reconcile invalid prediction of server(actual).

        Args:
            frame(int): frame number

        Returns:
            data(kwargs): valid data.
        """
        logging.debug(
            "Executing %s reconcilation...",
            self.__class__.__name__,
        )
        self.input(frame=frame, **data)
        return data

    def output(self, frame: int = -1) -> Any:
        """Get Processed Input.

        - Get specific frame or last input.

        Args:
            frame(int): frame of attribute
        """
        head = self.client.peek()
        if frame > -1:
            target = self.client.get(frame)
            logging.debug(
                "%s client frame: %s: %s",
                self.key,
                frame,
                target,
            )
            return target
        else:
            logging.debug(
                "%s client frame: %s: %s",
                self.key,
                self.last,
                head,
            )
            return head
