# -*- coding: utf-8 -*-
"""Muriel Message Builder

Class for creating the different parts of the Renity Message objects.

Example:
    >>> from muriel import MessageBuilder
    >>> builder = MessageBuilder()
    ...     hello=StringField(default="World")
    ...     sentence=ListField(StringField(required=True),IntField())
    ...

    >>> builder.add_field(key="test_field", value="str") 

    >>> sub_class = builder.message({"test_field": "Hello World"})

    >>> bytes(sub_class) 
    b"\x92\x88\x0bHello World"
"""

from typing import Dict, Union
import logging
import uuid
from renity import (
    StringField,
    ListField,
    BoolField,
    IntField,
    Message,
)
from renity.fields.constants import I64, FIXED64
from renity.fields.interface import Field


class FloatField(Field):
    """Float or Int Field.

    Renity Field Subclass
    """

    wire = I64
    field = FIXED64
    data_type = (float, int)


class MessageBuilder:
    """Renity Protocol Message Builder

    Builds Buffer Protocol Subclasses dynamically during runtime with a consistent construction process.

    Attributes:
        __message_type(str): Message subclass name.
        __fields(dict): Message fields of subclass under-construction.
        field_types(dict): hash map of fields available for assembly.

    Returns:
        instance(Message): unique Message subclass instance.
    """

    __message_type: str = f"{str(uuid.uuid1())}-message"
    __fields: Dict[
        str,
        Union[Field],
    ]

    field_types = {
        str(f.data_type): f
        for f in [
            StringField,
            ListField,
            BoolField,
            IntField,
        ]
    } | {str(float): FloatField}

    default_fields = {"frame": IntField(required=True)}

    def __init__(self) -> None:
        self.reset()

    def reset(
        self, cls_name: str = f"{str(uuid.uuid1())}-message"
    ) -> None:
        """Generate blank Message object for assembly

        Args:
            cls_name (str, optional): Message subclass name. Defaults to "%s-message"%str(uuid.uuid1()).
        """
        self.__message_type = cls_name
        self.__fields = {}

    def __get_field(
        self, data_type: type | list[type], required: bool = False
    ):
        """Get Buffer Field Class

        - Recursive call to get nested field types

        Args:
            value (Any): default value of field
            required (bool, optional): is field required. Defaults to False.

        Raises:
            TypeError: No field matching value type.
            e: General Exception.

        Returns:
            _type_: _description_
        """
        is_list = isinstance(data_type, list)
        __type = str(data_type) if not is_list else str(list)
        try:
            sub_fields = (
                []
                if not is_list
                else [
                    self.__get_field(v, required=required)
                    for v in data_type
                ]
            )
            print(__type)
            logging.debug(
                "Fetching field %s \nsub_fields%s",
                self.field_types[__type],
                sub_fields,
            )
            return self.field_types[__type](
                required=required, *sub_fields
            )

        except KeyError as e:
            raise TypeError(
                f"Expected value of type {self.field_types.keys()}, but found {type(data_type)}"
            ) from e
        except Exception as e:
            raise e

    def add_fields(self, **fields):
        # TODO - Should be add_field overload
        for key, value in fields.items():
            self.add_field(key=key, data_type=value)

    def add_field(
        self,
        key: str,
        data_type: type | list[type],
        required: bool = False,
    ) -> None:
        """Add field to Message

        Args:
            key (str): name of field attribute.
            value (Any): field_types hashmap key.
            required (bool, optional): whether field is required in Message. Defaults to False.
        """
        logging.debug(
            "Adding <Message><Field> %s: %s ", key, data_type
        )
        self.__fields.update(
            {key: self.__get_field(data_type, required=required)}
        )

    @property
    def message(self) -> type[Message]:
        """Generate Subclass

        Returns:
            type[Message]: Assembled Message subclass.
        """

        return type(
            self.__message_type,
            (Message,),
            self.__fields | self.default_fields,
        )
