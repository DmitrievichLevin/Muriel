from __future__ import annotations
from inspect import isclass
import threading
import logging
from typing import Any, Callable, Literal
from enum import Enum


class Act(Enum):
    CLOSED = 0
    SEATING = 1


class Observers(list):
    """Observers Stack"""

    def append(self, _observer: Observer) -> None:
        """Append Observers to stack.

        - add observable to observers 'observables' hash.

        Args:
            _observer (Observer): observable subscriber
        """
        # getattr(_observer, "observables", {}).update(
        #     {self.observed.name: self.observed}
        # )
        return super().append(_observer)


class Audience:
    """Hybrid Method Class Wrapper"""

    awaiting_entry: list[type[Observer]] = []
    entered: list[Observer] = []
    __seating: Literal[Act.CLOSED, Act.SEATING] = Act.CLOSED
    observable_cls: type[Observable]

    def __init__(self, queue) -> None:
        self.queue = queue
        self.__doc__ = queue.__doc__
        self.lock = threading.RLock()

    def lock_observable(
        self, observable_cls: type[Observable]
    ) -> None:
        """Place atomic lock on observable initialization.

        - Allow subscribers to initialize prior to initialization,
        so instances are referenced on eachother.

        Args:
            observable_cls (type[Observable]): class of observable.
        """

        logging.debug(
            "%s audience began seating. \nBinding load to %s initialization...",
            observable_cls,
            observable_cls,
        )

        # Set class variable
        self.observable_cls = observable_cls

        # Reference initialization for rebinding to cls
        observable_cls_init = observable_cls.__init__

        # __init__ wrapper
        def atomic_init(
            instance: Observable, *args, **kwargs
        ) -> None:

            def queue_init():

                # Attempt to acquire lock held by Observers
                self.lock.acquire()
                logging.debug(
                    "%s acquired %s lock. \nThread: %s",
                    instance,
                    self,
                    threading.get_ident(),
                )

                # Load Audience into observers list
                logging.debug("Observable %s is staged.", instance)

                # Initialize observable cls
                observable_cls_init(instance, *args, **kwargs)

                # Call instance method on entered observers
                self.queue(instance, *self.entered)

                # Rebind cls __init__
                self.observable_cls.__init__ = observable_cls_init
                logging.debug(
                    "%s & %s rebinding __init__.",
                    observable_cls,
                    instance,
                )

                self.lock.release()
                logging.debug("%s released %s lock.", instance, self)
                self.reset()

            pause = threading.Thread(
                target=queue_init, name="Audience-thread"
            )
            pause.start()

        observable_cls.__init__ = atomic_init  # type: ignore[assignment]

        # Set state of wrapper to 'open'
        self.__seating = Act.SEATING

    def reset(self):
        """Reset method wrapper state"""
        self.awaiting_entry = []
        self.entered = []
        self.__seating = Act.CLOSED
        self.observable_cls = None
        logging.debug("Seating finished: Closed.")

    def queue_observer(self, obs: type[Observer]) -> Callable:
        """Queue observer

        Args:
            obs (type[Observer]): observer class
        """
        # Queue observer class
        self.awaiting_entry.append(obs)

        # Reference cls.__init__ for rebind
        init = obs.__init__

        self.lock.acquire()
        logging.debug(
            "%s has acquired Audience %s lock. \nThread: %s",
            obs,
            self,
            threading.get_ident(),
        )

        def atomic_wrap(member, *args, **kwargs):

            # Queue initialized observer
            self.entered.append(member)

            # Dequeue observer cls
            self.awaiting_entry.pop()

            # call super(Observer).__init__
            init(member, *args, **kwargs)

            logging.debug(
                "1 %s entered %s. %d remain.",
                member,
                self.observable_cls,
                len(self.awaiting_entry),
            )

            # Rebind & release lock
            logging.debug("%s rebinding __init__.", member)
            obs.__init__ = init
            self.lock.release()

        return atomic_wrap

    def __get__(
        self,
        instance: Observable | None = None,
        cls: type[Any] = type,
    ) -> Callable:
        """Hybrid Method

        Args:
            instance (Observable | None, optional): instanced subclass. Defaults to None.
            cls (type[Any], optional): subclass. Defaults to type.

        Returns:
            hybrid_method(Callable): class method or instance method.
        """

        # Class method condition
        if not instance:

            # First call places atomic lock on initialization
            if self.__seating == Act.CLOSED:
                self.lock_observable(cls)
            # Return class method
            return self.__call__
        else:
            # Return instance method
            return lambda obs: self.queue(instance, obs)

    def __call__(self, *obs: type[Observer]) -> None:
        """Observable Class Method

        Raises:
            TypeError: expects Observer subclass
        """
        for o in obs:
            # Bind locking funciton to Observer
            o.__init__ = self.queue_observer(o)  # type: ignore[method-assign]
            if not isclass(o) or not issubclass(o, Observer):
                raise TypeError(
                    f"Expected type[Observer] but found {type(o)}"
                )

            logging.debug(
                "%d Observers: %s awaiting entry to %s.",
                len(self.awaiting_entry),
                self.awaiting_entry,
                self.observable_cls,
            )


class Observable:
    """Observable Concrete Class"""

    observers: Observers

    def __init__(
        self,
        name: str | None = None,
        observers: Observers | None = None,
        **kwargs,
    ) -> None:
        self.mutex = threading.RLock()
        self.__name = name
        self.observers: Observers = observers or Observers()
        super().__init__()

    @property
    def name(self):
        try:
            identifier = self.__name or getattr(self, "key")
        except AttributeError as e:
            raise AttributeError(
                f"Observable {self.__class__.__name__} expected 'name' or 'key' attribute, but found None."
            ) from e
        return identifier

    @name.setter
    def name(self, value: str):
        self.__name = value

    @Audience
    def subscribe(self, *obs: Observer) -> Observers:
        """Hybrid-Method: Subscribe to Observable."""
        for o in obs:
            logging.debug("%s Subscribing -> %s", o, self)
            self.observers.append(o)
            o.begin_observing((getattr(self, "key", self.name), self))
        return self.observers

    def remove(self, observer) -> None:
        """Remove Observer."""
        self.observers.remove(observer)
        logging.debug(
            "Observable: %s \nObserver: %s unsubscribed.",
            self.name,
            observer.__class__.__name__,
        )

    @classmethod
    def notify_observers(cls, method) -> Callable:
        """Notify Observers Decorator

        - Decorate subclass methods
        - Notifies subscribed Observer subclasses when method is called

        Returns:
            function(callable): Thread-Synced method calls observers update methods.
        """

        def notifier(self, *args, mute: bool = False, **kwargs):

            logging.debug(
                "%s attempting observable \nmethod: %s \nargs: %s \nkwargs: %s \nobservers:%s",
                self.name,
                method.__name__,
                args,
                kwargs,
                self.observers,
            )
            self.mutex.acquire()
            logging.debug(
                "%s acquired lock. \nThread: %s",
                self.name,
                threading.get_ident(),
            )

            # Make a local copy in case of synchronous
            # additions of observers:
            clone = self.observers[:]
            try:

                return method(self, *args, **kwargs)
            finally:
                # Mute keyword-argument (bypass observer(s) notification).
                if not mute:
                    for observer in clone:

                        if observer.can_update(*args, **kwargs):
                            logging.debug(
                                "Observer: %s starting update...",
                                observer.__class__.__name__,
                            )
                            observer._update(*args, **kwargs)
                else:
                    logging.debug(
                        "%s muted observer notification(s).",
                        self.name,
                    )
                self.mutex.release()
                logging.debug(
                    "%s released lock. \nThread: %s",
                    self.name,
                    threading.get_ident(),
                )

        return notifier


class Observer:
    """Observer Abstract Class

    Raises:
        NotImplementedError: Must implement abstract method update to subcribe to observable methods decorated with 'Observable.notify_observers'.
    """

    observing: dict = {}

    def __init__(self):
        super().__init__()

    def can_update(self, *args, **kwargs) -> bool:
        """Observer can update

        - Defaults True
        - Reduce redundant calls
        - Optional check

        Returns:
            bool: --gotoline ecs_events.py:325
        """
        return True

    def _update(self, *args, **kwargs) -> Any:
        """Observer Update Method

        **Subclasses Must Override**
        """
        raise NotImplementedError()

    def begin_observing(self, *obs: tuple[str, Observable]) -> Any:
        """Reference Observables on Observer

        Args:
            observables(tuple[str,Observable]): (key,observable)
        """
        self.observing.update(
            {key: observable for key, observable in obs}
        )
