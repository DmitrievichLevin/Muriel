"ECS State Component"

from typing import Any
import logging
from muriel.ecs_input_sequence import SEQUENCE
from muriel.ecs_component import Component
from muriel.ecs_statemachine import StateMachine
from muriel.ecs_discovery import submodule_walk

# Lazy-load all StateMachine Sub-classes
for name, cls in submodule_walk(StateMachine):
    globals()[name] = cls
    logging.debug(
        "Located StateMachine sub-class \nname:%s \ncls:%s", name, cls
    )


class State(Component):
    key = "state"
    default = SEQUENCE.NOT_CONNECTED.name
    __context: StateMachine

    def __init__(self) -> None:
        super().__init__()
        self.__context = globals()["NotConnected"]

    @property
    def context(self):
        return self.__context

    @property
    def sequence(self):
        return self.context.state

    def __enter(
        self,
        sequence: SEQUENCE,
        previous_state: SEQUENCE,
        frame: int = -1,
        **kwargs,
    ) -> None:

        # Should notify all components
        # Bottle neck client input through state component
        component_updates: dict = self.context.process(
            sequence=sequence,
            frame=frame,
            **kwargs,
        )
        component_updates.update(frame=frame, state=sequence.name)

        if previous_state is SEQUENCE.NOT_CONNECTED:

            self.on_data(
                mute=False,
                **component_updates,
            )
        else:
            self.input(
                **component_updates,
            )

    def input(
        self, frame: int = -1, load: int | None = None, **values
    ) -> None:
        logging.debug("\n calling state input \n")
        return super().input(frame, load, **values)

    def __exit(self, sequence: SEQUENCE, **kwargs) -> Any:
        """
        Exit current state -> Enter next state

        Args:
            new_state (dict): dict with 'action' representing the key of the next state & kwargs(Any)

        Raises:
            KeyError: Invalid state name.

        Returns:
            state (State): Instance of [new_state_name]
        """
        prev_state = self.context.state
        state_cls = sequence.name.capitalize()
        try:
            logging.debug(
                "Exiting State: %s \nEntering State %s",
                prev_state,
                sequence,
            )
            constructor: StateMachine = globals()[state_cls]
        except KeyError as e:
            raise KeyError(
                f"{state_cls} is not an existing State Subclass. Stacktrace: {e}"
            ) from e
        state = constructor
        self.__context = state

        self.__enter(
            sequence=sequence, previous_state=prev_state, **kwargs
        )

    @StateMachine.StateContext
    def update(  # type: ignore[override]
        self,
        sequence: SEQUENCE,
        _input: tuple[str],
        frame: int = -1,
        **kwargs,
    ) -> None:
        # TODO - fix mypy override error
        """
        Update player state with new state or persist current state

        Args:
            new_state (SEQUENCE): Sequence Enum

        Returns:
            state (State): Current State class or next
        """
        if self.sequence != sequence:
            self.__exit(
                sequence=sequence,
                _input=_input,
                frame=frame,
                **kwargs,
            )
        else:
            component_updates: dict = self.context.process(
                sequence=sequence,
                _input=_input,
                frame=frame,
                **kwargs,
            )

            component_updates.update(
                frame=frame, state=self.context.state.name
            )

            self.input(
                **component_updates,
            )
