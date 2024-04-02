"""Attribute Unittest"""

import pytest
from muriel.ecs_builtin import Speed


def test_attribute():
    speed = Speed(10)

    assert speed.output() == 10

    assert list(speed) == ["speed", 10]

    with pytest.raises(KeyError):
        speed.on_data(**{"frame": 1})
