"""ECS Built-in User Input Interface"""

from __future__ import annotations
import enum
from muriel.ecs_dataclasses import TrieNode


class Press(TrieNode):
    root: SEQUENCE | None = None
    singleton: bool = False

    def __init__(
        self,
        key: int,
        *inputs: Button,
        modifier: Button | None = None,
    ):
        self.key = key
        self.triggers = [
            [button, modifier] if modifier else [button]
            for button in inputs
        ]

        # singleton = Can't Reach through trie traversal
        if not inputs:
            self.singleton = True

        super().__init__(str(key))

    def set(self, _id: str, new_node: TrieNode) -> None:
        """Set Child Node

        Args:
            _id (str): Trie_id
            new_node (TrieNode): child node

        Raises:
            AttributeError: Attempting to overwrite existing node.
        """
        if self.get(_id):
            raise AttributeError(
                f"Attempting to overwrite existing node: {_id}."
            )
        new_node.prev = self
        self.nodes.update({_id: new_node})

    def find(
        self, _inputs: list[str] | None = None
    ) -> SEQUENCE | None:
        """Public find method

        - Hide recursion dependent args

        Returns:
            SEQUENCE | None: Enum type[Press]
        """
        return self.r_find(_inputs)

    def r_find(
        self,
        _inputs: list[str] | None = None,
        state: SEQUENCE | None = None,
    ) -> SEQUENCE | None:
        """Find SEQUENCE Enum

        - uses all permutations to find enum

        Args:
            state (SEQUENCE | None, optional): used during recursion. Defaults to None.

        Raises:
            e: general exception.

        Returns:
            SEQUENCE | None: Enum type[Press]
        """
        possible = []
        if not _inputs:

            if getattr(self.root, "value", None) == self:
                return self.root
            return state
        for idx, press in enumerate(_inputs):
            child = self.get(press)
            try:
                if child:
                    clone = _inputs[0:idx] + _inputs[idx + 1 :]
                    located = child.value.r_find(*clone, state=child)

                    if located:
                        return located
                    else:
                        possible.append(child)

            except Exception as e:
                raise e

        # Return enum with max int key if multiple
        return (
            max(possible, key=lambda a: a.value.key)
            if possible
            else None
        )


class Button(str, enum.Enum):
    # Buttons
    G = "G"  # B
    P = "P"  # X
    K = "K"  # Y
    J = "J"  # A

    # Toggle
    S = "S"  # e.g. Stick click, left d-pad

    # Direction
    L = "L"
    R = "R"
    U = "U"
    D = "D"


class ConstantMeta(type):
    def __new__(mcs, name, bases, attrs):
        member_names = []
        base = super().__new__(mcs, name, bases, attrs)
        if bases and issubclass(bases[0], Const):
            for _name, value in attrs.items():
                if isinstance(value, Press):
                    first_base = bases[0]
                    interface = (
                        base if first_base == Const else first_base
                    )
                    setattr(
                        base,
                        _name,
                        interface(interface, _name, value),
                    )
                    member_names.append(_name)
        members = getattr(base, "_member_names", [])
        setattr(base, "_member_names", member_names + members)

        return base

    def __repr__(cls):
        return f"<const: '{cls.__name__}'>"

    def __iter__(cls):
        yield from [getattr(cls, name) for name in cls._member_names]


class Const(metaclass=ConstantMeta):
    def __init__(self, cls: Const, name: str, value: Press):
        self.__name = name
        self.__value = value
        self._cls = cls

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, self.__class__):
            return str(self) == str(__value)
        return super().__eq__(__value)

    def __repr__(self):
        return "<%s.%s: %r>" % (
            self._cls.__name__,
            self.name,
            self.value,
        )

    def __str__(self):
        return "%s.%s" % (self._cls.__name__, self.name)


# 6-bit
# Unique Button sequences
class SEQUENCE(Const):
    def __new__(cls, base, *_):
        members = getattr(cls, "_member_names", [])
        if len(cls.__mro__) > 2 and len(members) > 0:
            base._construct()
        return super().__new__(base)

    NOT_CONNECTED = Press(key=-1)
    # 0                 ▲
    IDLE = Press(0b000000)
    # 1                 |
    WALK = Press(
        0b000001,
        Button.L,
        Button.R,
        Button.U,
        Button.D,
    )
    # 2                 |
    RUN = Press(
        0b000010,
        Button.L,
        Button.R,
        Button.U,
        Button.D,
        modifier=Button.S,
    )
    # 3                 |
    GUARD = Press(0b000011, Button.G)
    # 4         order desc by precedence
    COUNTER = Press(0b000100, Button.G, modifier=Button.P)
    # 5                 |
    ASCEND = Press(0b000101, Button.J)
    # -5                |
    DESCEND = Press(0b100101)
    # 6                 |
    COMBAT = Press(
        0b001000,
        Button.P,
        Button.K,
    )
    # 7                 ▼
    HURT = Press(0b001001)

    @classmethod
    def _construct(mcs) -> SEQUENCE:
        __base_presses = list(
            filter(lambda s: not s.value.triggers, mcs)
        )
        root = mcs.IDLE
        SEQUENCE.set_root(root)
        pointer = root
        for status in SEQUENCE:
            _, node = status.name, status.value

            if not node.singleton:
                for combination in node.triggers:
                    for pressed in combination:

                        _next = pointer.value.get(pressed.value)
                        if _next:
                            pointer = _next  # type: ignore[assignment]

                            continue
                        pointer.value.set(pressed.value, status)
                    pointer = root
                pointer = root

        # TODO - Maybe? roots=idle,hurt,descending
        # roots = []
        # for ro in base_presses[1:]:
        #     name, press = ro.name, ro.value

        #     press.nodes = root.value.nodes
        #     roots.append(ro)

        return root

    @classmethod
    def set_root(cls, root):
        root.value.root = root


class s(SEQUENCE):
    RUN = Press(
        0b000010,
        Button.L,
        Button.R,
        Button.U,
        Button.D,
        modifier=Button.S,
    )
