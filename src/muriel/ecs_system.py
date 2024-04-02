"""ECS System Interface"""

# pylint: disable=unnecessary-lambda-assignment
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import (
    Any,
    Callable,
    Generator,
    ParamSpec,
)
from uuid import uuid4
from muriel.ecs_builtin import Component, Compartment, Attribute


_K = ParamSpec("_K")


class BaseSystem:
    """Base System Concrete Class

    - Initilize BaseSystem
    - Initialize system attributes(Prerequisites) on component
    """

    def __init__(self):

        set_property = lambda ins, name: property(
            fget=lambda: getattr(ins, f"_ecs_sys_var_{name}"),
            fset=lambda v: setattr(ins, f"_ecs_sys_var_{name}", v),
        )
        self.flag = str(f"__ecs_system-{uuid4()}")
        self.prerequisites = [
            (
                lambda ins, s_v=sys_var: set_property(ins, s_v),
                f"__{sys_var}",
                getattr(self, sys_var),
            )
            for sys_var in dir(self)
            if "__" not in sys_var
        ]
        logging.debug(
            "System: %s \nSets attribute(s): %s \non components.",
            self.__class__.__name__,
            [sys_var for sys_var in dir(self) if "__" not in sys_var],
        )

    def __sys_vars(self, component) -> Callable[..., dict]:
        def wrap():
            return {
                name.split("_")[-1]: (name, getattr(component, name))
                for _, name, __ in self.prerequisites
            }

        return wrap

    def _setup(self, component) -> Generator:
        """Setup System on component

        - Initialize getter on component (returns sys-prerequisites).

        Args:
            component (Component): decorated component.
        """
        for component_prop, name, value in self.prerequisites:

            sys_var = component_prop(component)
            sys_var = value

            setattr(component, name, sys_var)

        setattr(component, self.flag, self.__sys_vars(component))

        return getattr(component, self.flag)()


class AbstractSystem(BaseSystem):
    """Abstract System Interface"""

    @classmethod
    def process(
        self, component: Component, *args, frame: int, **kwargs
    ) -> dict[str, Any]:
        """System Process

        - Process implementation is responsible for maintaining system-attributes at component level.

        Returns:
            tuple[tuple, dict[str, Any]]: result of system process in the form of args, kwargs.
        """
        raise NotImplementedError()


class System(AbstractSystem):
    """System Interface"""

    def register(
        self, component: Component | Compartment | Attribute
    ):
        """Register Component w/ System

        Args:
            component (Component | Compartment | Attribute): Component using system.
        """
        self._setup(component)


class SystemDecorator(System):
    """System Decorator

    - Decorate methods with System.process: Callable
    """

    __default_process: Callable

    def __init__(self, func):
        self.__default_process = func
        super().__init__()

    def __get__(
        self,
        component,
        _,
    ) -> Callable:
        sys_vars = getattr(
            component, self.flag, lambda: None
        )() or self._setup(component)

        def decorate(*args, **kwargs):

            self.__call__(component, *args, **kwargs | sys_vars)

        return decorate

    def __call__(
        self, component: Component, *args: Any, **kwargs: Any
    ) -> None:
        # Get result of System process
        process_kwargs = self.process(component, *args, **kwargs)

        # Call decorated method with returned process args/kwargs.
        self.__default_process(component, *args, **process_kwargs)
