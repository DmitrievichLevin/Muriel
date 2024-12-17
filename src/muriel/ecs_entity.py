"""ECS Entity Interface"""

from inspect import isclass
import logging
from renity import Message

from muriel.ecs_message_builder import MessageBuilder
from muriel.ecs_compartment import Component, Attribute, Compartment
from muriel.ecs_events import Observer
from muriel.ecs_builtin import (
    Movement,
    Location,
    ID,
    UserInput,
)
from muriel.ecs_input_sequence import SEQUENCE
from muriel.ecs_buffers import QBuffer
from muriel.ecs_state_component import State

message_builder = MessageBuilder()


class FixedStep:
    __frame: QBuffer

    def __init__(self):
        self.__frame = QBuffer(name="frame")

    @property
    def frames(self):
        return self.__frame

    @property
    def current_frame(self):
        return self.__frame.end

    def step(self) -> int:
        next_frame = self.current_frame + 1
        self.__frame.enqueue(next_frame)
        return next_frame


class BaseEntity(Observer, FixedStep):
    """A Base class for Entities

    Attributes:
        type[Attribute | Component | Compartment]: Attributes are instanced on Entity at runtime
    """

    state: Component
    properties: dict
    __entity_message_cls: Message | None = None

    def __init__(self, frame: int = 0, **kwargs):
        """Create Instance

        Args:
            frame (int, optional): starting frame of instanced Entity. Defaults to 0.
        """
        super().__init__()
        self.__init_attributes()

    def __init_attributes(self) -> None:
        """Initialize Attribute Types"""
        message_builder.reset()
        # Initialize State Component
        state = State()
        self.state = state
        _map = dict(state=state)
        for key in dir(self):
            if "__" not in key:
                _attr = getattr(self, key)
                if isclass(_attr) and issubclass(
                    _attr, (Attribute, Component, Compartment)
                ):
                    _attr = _attr()

                    # Subscribe all components to built-in State
                    # TODO - Should only allow Observers
                    _map["state"].subscribe(_attr)

                    _attr.key = key
                    _map.update({key: _attr})
                    # --gotoline ecs_message_builder.py:10
                    if isinstance(_attr.data_type, dict):
                        message_builder.add_fields(**_attr.data_type)
                    else:
                        message_builder.add_field(
                            key=key, data_type=_attr.data_type
                        )

                    # Rebind Attributes
                    setattr(self, key, _attr)
        self.properties = _map

        # Initialize Renity Protocol Buffer
        self.__entity_message_cls = message_builder.message()
        logging.debug(
            "Entity properties %s",
            self.properties,
        )

    @property
    def message(self):
        """__repr__ of Entity

        Returns:
            dict[str, Any]: Entity attributes
        """
        if isinstance(self.__entity_message_cls, Message):
            _current_propeties = dict(self)

            self.__entity_message_cls.message = _current_propeties | {
                "frame": self.current_frame
            }
            logging.debug(
                "Creating <Message> instance from: %s",
                self.__entity_message_cls.message,
            )
            return self.__entity_message_cls.message
        return None

    def update(self, *args, **kwargs) -> None:
        """Advance Frame of Entity"""
        # TODO - Better
        next_step = self.step()

        self.__process(**kwargs | {"frame": next_step})

    def __process(self, response: bool = False, **kwargs):
        """Entity input process

        Args:
            response (bool, optional): whether input is from server packet. Defaults to False.
        """
        # Traverse properties of Entity
        for key, prop in self.properties.items():
            # Compartment property uses compartment_process
            if isinstance(prop, Compartment):
                self.__compartment_process(
                    compartment=prop, response=response, **kwargs
                )
                continue

            # Components/Attributes use component(singleton) process
            if isinstance(prop, (Component, Attribute)):
                # Attributes can only be changed by server
                _response = (
                    True if isinstance(prop, Attribute) else response
                )
                self.__component_process(
                    component=prop,
                    key=key,
                    response=_response,
                    **kwargs,
                )
                continue

    def __get_process(
        self,
        prop: Component | Attribute | Compartment,
        response: bool = False,
    ):
        """Get Process Method

        Args:
            prop(Component, Compartment): requesting component/compartment.
            response(bool): True = Server response

        returns:
            __process(Callable): client/server input process.
        """
        logging.debug(
            "Processing %s data.",
            "server" if response else "input",
        )
        return prop.on_data if response else prop.input

    def __component_process(
        self,
        frame: int,
        component: Component | Attribute,
        key: str,
        response: bool = False,
        **kwargs,
    ):
        """Process Component/Attribute Singleton

        Args:
            frame (int): _description_
            component (Component | Attribute): _description_
            key (str): _description_
            response (bool, optional): _description_. Defaults to False.
        """

        property_value = kwargs.get(key, component.default)
        property_obj = {key: property_value, "frame": frame}
        self.__get_process(component, response)(**property_obj)

        # Build Entity Message Subclass on initial processes
        if not self.__entity_message_cls:
            message_builder.add_field(key=key, value=property_value)
            # Rebind Attribute
            setattr(self, key, component)

    def __compartment_process(
        self,
        frame: int,
        compartment: Compartment,
        response: bool = False,
        **kwargs,
    ):
        """Process Compartment Properties

        Args:
            frame (int): frame to process
            system (System): System currently processing input
            response (bool, optional): non/server packet. Defaults to False.
        """
        self.__get_process(prop=compartment, response=response)(
            frame=frame, **kwargs
        )

        # Build Entity Message Subclass on initial process
        if not self.__entity_message_cls:
            for name, comp in compartment.observables:
                message_builder.add_field(
                    key=name,
                    value=comp.default,
                )

            # Rebind Attribute
            setattr(self, compartment.key, compartment)

    def on_data(self, packet: dict) -> None:
        """On Data

        - TODO - Process bytes
        - TODO - Set Starting frame RTT(Round-trip-time) in future.
        - TODO - load buffer based on rtt.

        Args:
            packet(dict | bytes): data over the wire
        """
        if self.state.sequence == SEQUENCE.NOT_CONNECTED:

            self.frames.enqueue(packet["frame"])
            packet.update(load=3)

            self.state.update(**packet)

        else:
            self.state.on_data(mute=False, **packet)

    def __iter__(self):
        """Entity Generator

        Yields:
            property_values(Generator): values of the most recent processed frame.
        """
        frame = self.current_frame
        for prop_key, prop_arr in self.properties.items():
            if isinstance(prop_arr, Compartment):
                for key, prop in prop_arr.observables:
                    yield key, prop.output(frame)
            elif isinstance(prop_arr, (Component, Attribute)):
                yield prop_key, prop_arr.output(frame)


class Entity(BaseEntity):

    id = ID
    map_id = Location
    movement = Movement

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __bytes__(self, *args):
        pass
        # byte_list = bytes([])
        # for idx, attribute in enumerate(
        #     [
        #         # 16 Bit
        #         self.id_comp,
        #         # 16 Bit
        #         self.location_comp,
        #         # 32 Bit Float
        #         self.rotation_comp,
        #         # 32 bits
        #         self.position_comp,
        #         # *args,
        #     ]
        # ):
        #     byte_list = byte_list + bytes(attribute)
        #     print(
        #         "bytes",
        #         bin(int.from_bytes(bytes(attribute), "big")),
        #         byte_list,
        #     )

        # return byte_list


class User(Entity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.userinput = UserInput()

    def on_input(self, _input: list[str]):

        self.userinput.enqueue(*_input)

    def update(self, *args, **kwargs) -> None:
        __input = self.userinput.output(
            frame=kwargs.get("frame", None)
        )
        return super().update(*args, **__input)

    def step(self) -> int:
        _input = self.userinput.get()

        frame = super().step()

        self.state.update(_input=_input, frame=frame)
