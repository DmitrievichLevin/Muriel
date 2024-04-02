import pytest
from muriel.ecs_entity import Entity, User
import uuid

from muriel.ecs_rigidbody import VectorQuantityPhysics


def test_entity():
    """Test Entity initialization"""
    entity = User()
    entity_id = str(uuid.uuid4())
    map_id = str(uuid.uuid4())
    entity.on_data(
        {
            "frame": 1,
            "position": [20, 0, 0],
            "velocity": [1, 0, 1],
            "direction": 45.0,
            "speed": 24,
            "reference": 22,
            "id": entity_id,
            "map_id": map_id,
            "status": 200,
        }
    )

    entity.on_data(
        {
            "frame": 1,
            "position": [20, 0, 0],
            "velocity": [1, 0, 1],
            "direction": 45.0,
            "speed": 24,
            "reference": 22,
            "id": entity_id,
            "map_id": map_id,
            "status": 200,
        }
    )

    # assert {
    #     "type": "6a6718fa-eca7-11ee-9fa4-7cd1c38a3386-message",
    #     "id": entity_id,
    #     "map_id": map_id,
    #     "position": [20, 0, 0],
    #     "velocity": [0.0, 0.0, 0.0],
    #     "direction": 45.0,
    #     "speed": 24,
    #     "frame": 1,
    # } == entity.message
    print("\n\n message:", entity.message)
    entity.on_input(["L", "S"])

    entity.step()
    print("\n\n message:", entity.message)

    entity.step()
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)

    entity.on_data(
        {
            "frame": 3,
            "position": [10, 0, 0],
            "velocity": [1, 11.803799999999995, 1],
            "direction": 45.0,
            "speed": 24,
            "reference": 22,
            "id": entity_id,
            "map_id": map_id,
            "status": 200,
        }
    )
    print("\n\n message:", entity.message)
    entity.step()
    print("\n\n message:", entity.message)
    for x in range(70):
        entity.step()
    # controller = Frame()
    # controller.subscribe(entity)
    # for i in range(11):
    #     entity.handle_input("A")
    #     controller.tick()
    # logging.debug(f"entity1 {entity.movement.output(4)} ")

    # import math

    # def angle(velocity):
    #     normalized = velocity
    #     ang = math.degrees(
    #         math.atan2(0, -1)
    #         - math.atan2(normalized[0], -normalized[2])
    #     )
    #     return ang + 360 if ang < 0 else ang

    # logging.debug(
    #     f"left facing: {angle([0,0,-1])} {angle([0,0,1])}  {angle([-1,0,0])} {angle([1,0,0])}"
    # )
    # assert entity.id.value == 22

    # assert dict(entity) == {
    #     "frame": 1,
    #     "position": [20, 0, 0],
    #     "velocity": [1, 0, 1],
    #     "direction": 45.0,
    #     "speed": 10,
    #     "reference": 22,
    #     "map_id": "5caffa5a-bdfb-4aaf-a143-e2fe074498c9",
    # }


test_entity()
