import pygame
import os
from math import radians, cos, sin
from muriel.ecs_entity import User
from muriel.ecs_controller import Frame


"""
BLACK = (0, 0, 0) GRAY = (127, 127, 127) WHITE = (255, 255, 255)
RED = (255, 0, 0) GREEN = (0, 255, 0) BLUE = (0, 0, 255)
YELLOW = (255, 255, 0) CYAN = (0, 255, 255) MAGENTA = (255, 0, 255)
"""
ACTION_COLORS = {
    31: (255, 0, 255),
    8: (255, 0, 255),
    9: (255, 0, 255),
    0: (127, 127, 127),
    15: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 255),
}

MAP_SIZE = 50
SCREEN_SIZE = 640
OFFSET = SCREEN_SIZE / 2
CELL_SIZE = int((25 * SCREEN_SIZE) / MAP_SIZE)
TRANSLATION = SCREEN_SIZE / MAP_SIZE


class Display:
    def __init__(self, user):
        self.player = user
        pygame.display.init()
        self.surface = pygame.display.set_mode(
            (SCREEN_SIZE, SCREEN_SIZE + 200), pygame.RESIZABLE
        )
        self.surface.fill((255, 255, 255))
        self.drawGrid()
        pygame.display.update()
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)

    def add_player(self, p):
        self.players += p

    def drawGrid(self):
        # Set the size of the grid block
        for x in range(0, SCREEN_SIZE, CELL_SIZE):
            for y in range(0, SCREEN_SIZE, CELL_SIZE):
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.surface, (0, 0, 0), rect, 1)

    def drawPlayer(
        self, position=None, direction=None, state=1, **kwargs
    ):
        translate = [
            position[0] * TRANSLATION + OFFSET,
            -position[2] * TRANSLATION + OFFSET,
        ]
        self.direction_marker(translate, direction, state)
        pygame.draw.circle(
            self.surface,
            ACTION_COLORS[state],
            translate,
            10,
            5,
        )

        floor = pygame.Rect(
            5, SCREEN_SIZE + 200 - 5, SCREEN_SIZE - 10, 5
        )
        pygame.draw.rect(
            self.surface,
            ACTION_COLORS[15],
            floor,
        )

        player_height = pygame.Rect(
            5, SCREEN_SIZE + 200 - 5 - 161, SCREEN_SIZE - 10, 5
        )
        pygame.draw.rect(
            self.surface,
            ACTION_COLORS[15],
            player_height,
        )

        player_rect = pygame.Rect(
            position[0] + 50,
            SCREEN_SIZE + 200,
            50,
            161,
        )
        player_rect.bottom = SCREEN_SIZE + 200 - 5 - position[1]

        pygame.draw.rect(
            self.surface,
            ACTION_COLORS[state],
            player_rect,
        )

    def direction_marker(self, origin, _rot, action):
        pos = [[0, -25], [10, -15], [-10, -15]]

        rads = radians(_rot)

        position = []

        for x, z in pos:
            t_x = (x * cos(rads)) - (z * sin(rads)) + origin[0]
            t_z = (x * sin(rads)) + (z * cos(rads)) + origin[1]
            position.append((t_x, t_z))

        pygame.draw.polygon(
            self.surface, ACTION_COLORS[action], position, 5
        )

    def refresh(self, *players):

        self.surface.fill((255, 255, 255))
        self.drawGrid()
        for player in players:
            packet = player.message
            self.drawPlayer(**packet)
        pygame.display.update()


player = User(
    **{
        "frame": 1,
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "direction": 45.0,
        "speed": 240,
        "reference": 22,
        "map_id": "5caffa5a-bdfb-4aaf-a143-e2fe074498c9",
    }
)


class PyController:

    def handle_input(self):

        key = pygame.key.get_pressed()
        x = 0
        y = 0
        z = 0

        if key[pygame.K_LEFT]:
            x += -1

        if key[pygame.K_RIGHT]:
            x += 1
        if key[pygame.K_DOWN]:
            z += -1

        if key[pygame.K_UP]:
            z += 1

        if key[pygame.K_SPACE]:
            y += 1

        if any(v != 0 for v in [x, y, z]):

            return {"velocity": [x, y, z]}


class PyController2:

    def handle_input(self):

        key = pygame.key.get_pressed()
        x = 0
        y = 0
        z = 0

        if key[pygame.K_a]:
            x += -1

        if key[pygame.K_d]:
            x += 1
        if key[pygame.K_s]:
            z += -1

        if key[pygame.K_w]:
            z += 1

        if key[pygame.K_f]:
            y += 1

        if any(v != 0 for v in [x, y, z]):

            return {"velocity": [x, y, z]}


display = Display(player)
running = True
controller = Frame()
controller.subscribe(player)
game_controller = PyController()
game_controller2 = PyController2()

import time

while running:
    f = controller.tick()
    display.refresh(player)
    pygame.event.get()
    if f == 50:
        print(f"{f}: {player.message['position']}")
    _input = game_controller.handle_input()
    if _input:
        player.input(_input)

    if f == 250:
        print(f"{f} {player.message['position']}")
        player.process(
            response=True,
            **{
                "frame": 50,
                "position": [-10.0, 0.0, 0.0],
                "velocity": [-1.0, 0, 1.0],
                "direction": 0,
                "speed": 240,
                "reference": 22,
                "map_id": "5caffa5a-bdfb-4aaf-a143-e2fe074498c9",
            },
        )
        print(f"{f} {player.message['position']}")

    # _input = game_controller2.handle_input()
    # if _input:
    #     player2.input(_input)


# delta = 1 / 50


# def jump(frame):
#     impulse = 245  # 256
#     mass = 100  # 45
#     gravity = 9.81

#     velocity = impulse / mass

#     # Peak
#     b = velocity / gravity

#     # Max Height
#     k = (0.5 * velocity) * (velocity / gravity) * 100
#     energy = k * mass * gravity

#     x = frame * delta

#     a = -(k**2) / 2

#     print(f"velocity: {velocity} \n -k^2/2:{(a*(x-b)**2)+k}")

#     return -a * (x - b) ** 2 + k


# print(jump(20))

# position = []
# for x in range(30):
#     p = jump(20 + x)
#     position.append(p)
#     t = x * delta
#     print(f"\n Position at {20+x}s: {p}")
