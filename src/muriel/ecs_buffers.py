"""ECS Buffers"""

from __future__ import annotations
import typing
import uuid
from typing import Any, Generator, Iterator
import logging
from muriel.ecs_events import Observable


class Buffer(Observable):
    """Observable Buffer

    Args:
        name(str): Identifier e.g. position,health,strength.
    """

    def __init__(
        self,
        name: str = f"{str(uuid.uuid1())}_buffer",
        start: int = 0,
        default: typing.Any = 0,
        component=None,
    ) -> None:
        self.buff: dict = {"head": start, f"{start}": default}
        self.name: str = name

        super().__init__(name=name)
        self.component = component
        if component:
            self.observers = self.component.observers

    @property
    def head(self) -> int:
        """Key of Queue Head

        Returns:
            head: int or None(empty)
        """
        return self.buff["head"]

    @head.setter
    def head(self, val) -> None:
        """Update Head of Queue.

        Args:
            val: new_head if val > current_head or deleted current head prior to call
        """
        cur = self.head
        if (
            self.head is None
            or val > cur
            or self.buff.get(str(cur), None) is None
        ):
            logging.debug(
                "%s updating head %s -> %s",
                self.name,
                self.head,
                val,
            )
            self.buff["head"] = val if val > -1 else None

    def peek(self) -> typing.Any:
        """Queue Head Value

        Returns:
            value: Queue Head w/o mutating Queue.
        """
        return self.buff.get(str(self.head), None)

    def get(self, key: typing.Union[str, int]) -> typing.Any:
        """Get Queue Element

        Args:
            key: frame-number

        Returns:
            value: value corresponding to key or None.
        """
        value = self.buff.get(str(key), None)
        logging.debug(
            "%s fetching frame: %s: %s",
            self.name,
            key,
            value,
        )

        return value

    @Observable.notify_observers
    def put(
        self,
        frame: int,
        value: typing.Any = None,
        **kwargs,
    ) -> None:
        """Put item into Queue

        Args:
            value: value to be placed at frame.
            component: name of encapsulating component
            frame(int | list): frame number(s).
        """

        # Load multiple duplicate frames
        load = kwargs.get("load", None) or 0

        # e.g test-client_buffer
        key = self.name.split("-")[0]

        _value = kwargs.get(key, value)
        for _frame in [f for f in range(frame - load, frame + 1)]:
            self.buff.update({str(_frame): _value})
            logging.debug(
                "%s \nQueue insertion \nframe: %s \nvalue: %s \nkwargs: %s",
                key,
                _frame,
                _value,
                kwargs,
            )
        self.head = frame
        if key == "position":
            logging.debug(
                "positon %s buffer contents \n%s",
                self.name,
                self.buff,
            )

    @Observable.notify_observers
    def pop(
        self, *_args, frame: int | None = None, **_kwargs
    ) -> typing.Any:
        """Pop head of Queue

        - Supports Index

        Returns:
            value(Any): Value at head of queue.
        """
        head = str(frame or self.head)
        value = self.buff.get(head, None)
        if value is not None:
            del self.buff[head]
            self.__find_head()
            logging.debug(
                "%s Queue popped head: %s \nreturning: %s \ncontents: %s.",
                self.name,
                head,
                value,
                self.buff,
            )

        return value

    def __find_head(self) -> None:
        """Find Next Queue Head"""
        # TODO - Optimize
        pointer = self.head - 1
        while (
            self.buff.get(str(pointer), None) is None and pointer > -1
        ):
            pointer -= 1
        self.head = pointer


class QNode:
    """Queue Node"""

    head: QNode | None = None
    tail: QNode | None = None
    root: bool = True

    def __init__(self, value: Any) -> None:
        self.value = value

    @property
    def is_root(self) -> bool:
        """Is Root

        Returns:
            is_root(bool): whether current node is root.
        """
        return self.root

    @property
    def next(self) -> QNode | None:
        """Next node

        Returns:
            node(QNode): next node in linked list.
        """
        return self.tail

    def link(self, node: QNode) -> None:
        """Append Node

        Args:
            node(QNode): next node in list
        """
        node.root = False
        self.tail = node
        node.head = self

    def replace(self, node: QNode) -> None:
        """Replace Node

        Args:
            node(QNode): node to replace in list
        """

        self.head = node.head
        self.tail = node.tail
        node.head = None
        node.tail = None

    def pop(self) -> tuple[Any, QNode | None]:
        """Pop top of Queue

        Returns:
            values(tuple): value, next_node
        """
        head = self.head
        if not self.is_root:
            setattr(head, "tail", None)
        return self.value, (
            None if getattr(head, "is_root", None) else head
        )


class QBuffer(Observable):
    """Observable Queue

    Args:
        name(str): Identifier e.g. position,health,strength.
    """

    __length: int = 0
    root: QNode | None = None
    tail: QNode | None = None

    def __init__(
        self,
        name: str = f"{str(uuid.uuid1())}_queue",
        size: int = 0,
    ):
        # Queue length limit
        self._size = size
        super().__init__(name=name)

    @property
    def empty(self):
        """Is Buffer Empty

        Returns:
            (bool): len(buffer) == 0
        """
        return self.length == 0

    @property
    def length(self):
        """Current size of Buffer

        Returns:
            (int): len(buffer)
        """
        return self.__length

    @length.setter
    def length(self, v):
        size = self._size
        l = v if v > 0 else 0
        self.__length = min(size, l) if size else l

    def peek(self) -> Any:
        """View top of stack w/o dequeue

        Returns:
           (Any): value at top of stack or None
        """
        logging.debug(
            f"Peeking at queue \n{getattr(self.tail, 'value', None)}"
            + f"\n{getattr(self.root, 'value', None)}"
        )
        return getattr(self.tail, "value", None) or getattr(
            self.root, "value", None
        )

    @property
    def end(self):
        return getattr(self.root, "value", None) or getattr(
            self.tail, "value", None
        )

    def enqueue(self, value: Any, *values) -> QNode:
        """Enqueue

        Args:
            value(Any): value to queue

        Returns:
            node(QNode): reference to top of stack.
        """
        node = QNode(value)
        match (self.length):
            case 0:
                self.root = node
            case _:
                root: QNode = getattr(self, "root")
                if self._size and self.length == self._size:
                    node.replace(root)
                else:
                    node.link(root)
                    if not self.tail:
                        self.tail = self.root
                self.root = node

        self.length += 1
        logging.debug(
            "%s enqueue \nvalue: %s \ntail: %s \nroot: %s \nlength: %s",
            self.name,
            value,
            getattr(self.tail, "value", None),
            getattr(self.root, "value", None),
            self.length,
        )

        # Handle multiple values to queue
        if len(values) > 0:
            return self.enqueue(values[0], *values[1:])

        return self.tail or self.root

    def dequeue(self) -> Any:
        """Dequeue Top of stack

        Returns:
            (Any): value at top of stack or None
        """
        l = self.length
        self.length -= 1
        value = None
        match l:
            case 0:
                pass
            case 1:
                # size is 1 and root = only node
                value = self.root.value  # type: ignore[union-attr]
                self.root = None
            case _:
                val, next_tail = self.tail.pop()  # type: ignore[union-attr]
                self.tail = next_tail
                value = val
        logging.debug(
            "%s dequeue \nvalue: %s \ntail: %s \nroot: %s",
            self.name,
            value,
            getattr(self.tail, "value", None),
            getattr(self.root, "value", None),
        )
        return value

    def clear(self) -> Iterator:
        while self.length:
            yield self.dequeue()

    def __iter__(self):
        pointer = self.root
        count = 0
        while pointer:
            yield count, pointer.value
            count += 1
            pointer = pointer.next
