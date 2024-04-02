from typing import Any
import pytest
from muriel.ecs_component import Component


def test_component_implementation():
    class TestComponent(Component):
        key = "test"

        def validate(self, server: Any, prediction: Any) -> bool:
            return server == prediction

    # Initialize Component
    comp = TestComponent()

    # Process input
    comp.input(frame=0, test=10)

    # Validate Input
    assert comp.output(0) == 10

    # Test incorrect client reconciliation
    comp.on_data(**{"frame": 0, "test": 11})

    assert comp.output(0) == 11
