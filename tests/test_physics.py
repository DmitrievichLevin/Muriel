import pytest
from muriel.ecs_builtin import Velocity, Speed
import logging


def test_physics():
    # If 7.9 units are 1/4 mile, then one unit would be 0.0316 miles or 167 feet.
    # Average human walking speed = 0.00086992mps or 1.4(meters per second)
    # Usain Bold Top Speed = 27.33mph/60**2 = 0.00759166666mps
    # Bolt Acceleration = 9.5mps or 0.00590303mps
    # Acceleration Units per/s = 0.00590303/0.0316 = 0.18680474683ups
    # Units per/s =  0.00086992/0.0316 = 0.02752911392ups = Walking Speed
    # Units per/s = 0.00759166666/0.0316 = 0.24024261582ups = Maxed Running Speed
    # Player height = 0.001982 Units ~ 5,4 Human Height
    # DELTA = 1/50

    # vel_comp = Velocity()

    # vel_comp._stats = {"speed": Speed()}

    # idle = [0, 0, 0]

    # top_speed = 0
    # while vel_comp.client.peek()[0] < (24 / 100):
    #     vel_comp.input(frame=top_speed, velocity=[1, 0, 0])
    #     top_speed += 1

    # assert top_speed == 67

    # frame = top_speed
    # while vel_comp.client.peek()[0] > 0:
    #     vel_comp.input(frame=frame, velocity=idle)
    #     frame += 1

    # assert frame - top_speed == 11

    # logging.info(
    #     f"\n Max Speed in {top_speed/50} seconds {top_speed+1} frames \n {(frame - top_speed)/50} seconds {frame-top_speed} frames to stop"
    # )
    pass
