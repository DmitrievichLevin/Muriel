"""ECS Component Interface"""

from typing import Any
from muriel.ecs_events import Observable, Observer
from muriel.ecs_governed import Governed
from muriel.ecs_buffers import Buffer, QBuffer


class Component(Governed, Observable):
    """Component Concrete Class

    Args:
        Governed (_type_): Concrete Deterministic Base Class
        Observer (_type_): Observer Abstract Base Class
        Observable (_type_): Observable Concrete Base Class

    Yields:
        (tuple): key,value of last frame
    """

    key: str
    initial_frame: int
    default: Any
    _stats: dict = {}
    client: Buffer = Buffer  # type: ignore[assignment]
    server: Buffer = Buffer  # type: ignore[assignment]

    def __init__(self, observers=None, **kwargs) -> None:
        """Initialize Component

        - Initialize Buffers
        - Queue Initial Frame
        - Initialize Mutex Lock
        """
        # Get buffer dependencies
        key, initial_frame, default = (
            getattr(self, "key"),
            getattr(self, "initial_frame", 0),
            getattr(self, "default"),
        )

        self.client = Buffer(
            name=f"{key}-client_buffer",
            start=initial_frame,
            default=default,
        )
        self.server = Buffer(
            name=f"{key}-server_buffer",
            start=initial_frame,
            default=None,
        )

        # Initialize Observable name attr
        super().__init__(name=self.key, **kwargs)
        self.client.observers = observers or self.observers
        self.server.observers = observers or self.observers

    # TODO-Remove observer method (remove from component + client-buffer)

    def remove(self, observer: Observer):
        self.client.remove(observer=observer)
        super().remove(observer)

    @property
    def stats(self) -> dict:
        """System Stats Reference

        Returns:
            stats(dict): (name, value) key-pairs
        """
        return self._stats

    def validate(self, server: Any, prediction: Any) -> bool:
        """Validate Component Value

        Args:
            server(Any): actual attribute on server from frame.
            predection(Any): client prediction.

        Returns:
            true(bool): Auto-pass if no implementation exists on subclass.
        """
        return True

    def update(self, *args, **kwargs) -> Any:
        """Observer Update Method

        Default: Component.input
        """

        self.input(**kwargs)

    # def __getattribute__(self, __name: str) -> Any:

    #     observables = super().__getattribute__("observables")
    #     observable = observables.get(__name, None)
    #     if observable:
    #         return observable
    #     else:
    #         return super().__getattribute__(__name)

    def __iter__(self):
        yield from (self.key, self.output())


class IndeterminateComponent(Component):
    """Indeterminate Component

    - Component that functions outside of server authority.
    """

    buffer: QBuffer

    def __init__(self) -> None:
        super().__init__()
        # Remove deterministic buffers
        del self.client
        del self.server

        # Initialize general Queue-like Linked list buffer
        self.buffer = QBuffer(name=self.key)
        self.buffer.observers = self.observers

    @Observable.notify_observers
    def enqueue(self, *values: Any) -> None:
        self.buffer.enqueue(*values)

    def input(
        self,
        frame: int | dict | None = None,
        load: int | None = None,
        **values,
    ) -> None:
        """Client input processing

        - Enqueue values

        Args:
            frame(int): current
            values(dict): input dict values
        """
        raise NotImplementedError(
            "Indeterminate Component has not attribute input."
        )

    def on_data(
        self, **data: dict[str, int | float | list | str]
    ) -> None:
        pass

    def remove(self, observer: Observer):
        self.observers.remove(observer)

    def update(self, *args, **kwargs) -> None:
        """Observer Update Method

        **Subclasses Must Override**
        """

    def output(self, frame: int = 0) -> dict:
        return {self.key: self.buffer.dequeue()} | {"frame": frame}
